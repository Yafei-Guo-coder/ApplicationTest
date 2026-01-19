
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


captured: Dict[str, torch.Tensor] = {}

@torch.no_grad()
def attn_column_sums_streaming(q, k, causal=True, block_cols=1024):
    """
    返回 [B, L]：所有头平均后的 attention 矩阵按列求和（不构造 LxL）。
    q,k: [B,H,L,D]
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = q.new_zeros((B, L), dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            Q = q[b, h].to(torch.float32)          # [L, D]
            K = k[b, h].to(torch.float32)          # [L, D]

            m = torch.full((L,), -float('inf'), dtype=torch.float32, device=Q.device)
            l = torch.zeros((L,), dtype=torch.float32, device=Q.device)
            col_sum = torch.zeros((L,), dtype=torch.float32, device=Q.device)

            for j0 in range(0, L, block_cols):
                j1 = min(j0 + block_cols, L)
                Kb = K[j0:j1]                                  # [B,D] 这里B是块宽
                S = (Q @ Kb.t()) * scale                       # [L,B]

                if causal:
                    row_idx = torch.arange(L, device=Q.device).unsqueeze(1)       # [L,1]
                    col_idx = torch.arange(j0, j1, device=Q.device).unsqueeze(0)  # [1,B]
                    S = S.masked_fill(col_idx > row_idx, float('-inf'))

                block_row_max = torch.max(S, dim=1).values
                new_m = torch.maximum(m, block_row_max)
                l *= torch.exp(m - new_m)

                exp_scores = torch.exp(S - new_m.unsqueeze(1))   # [L,B]
                l += torch.sum(exp_scores, dim=1)

                probs_block = exp_scores / l.unsqueeze(1)        # [L,B]
                col_sum[j0:j1] += probs_block.sum(dim=0)
                m = new_m

            out[b] += col_sum / H
            
    return out

def _get(module, names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None

def _get_heads_from_model(model, last_attn_module):
    # 优先从 config
    H_q = getattr(getattr(model, "config", object()), "num_attention_heads", None)
    H_kv = getattr(getattr(model, "config", object()), "num_key_value_heads", None)
    return int(H_q), int(H_kv)

def _attach_qk_hooks(last_attn_module, model):
    """
    从 config 读 H_q/H_kv；Q 按 H_q reshape，K 按 H_kv reshape，再把 K 扩展到 H_q。
    捕获到的 q/k 形状最终都为 [B, H_q, L, D]。
    """
    q_linear = _get(last_attn_module, ["q_proj", "wq", "query", "q"])
    k_linear = _get(last_attn_module, ["k_proj", "wk", "key", "k"])
    if q_linear is None or k_linear is None:
        return []

    H_q, H_kv = _get_heads_from_model(model, last_attn_module)
    group = H_q // H_kv
    captured["__H_q__"] = torch.tensor(H_q)
    captured["__H_kv__"] = torch.tensor(H_kv)

    hooks = []

    def _grab_q(module, inp, out):
        # out: [B, L, D_all_q] 其中 D_all_q = H_q * D
        B, L, Dall_q = out.shape
        assert Dall_q % H_q == 0, f"Q proj dim {Dall_q} not divisible by H_q={H_q}"
        D = Dall_q // H_q
        q = out.view(B, L, H_q, D).permute(0, 2, 1, 3).contiguous()  # [B,H_q,L,D]
        captured["q_linear"] = q.detach()

    def _grab_k(module, inp, out):
        # out: [B, L, D_all_k] 其中 D_all_k = H_kv * D
        B, L, Dall_k = out.shape
        assert Dall_k % H_kv == 0, f"K proj dim {Dall_k} not divisible by H_kv={H_kv}"
        D = Dall_k // H_kv
        k = out.view(B, L, H_kv, D).permute(0, 2, 1, 3).contiguous()  # [B,H_kv,L,D]
        if H_kv != H_q:
            k = k.repeat_interleave(group, dim=1)  # → [B,H_q,L,D]
        captured["k_linear"] = k.detach()

    hooks.append(q_linear.register_forward_hook(_grab_q))
    hooks.append(k_linear.register_forward_hook(_grab_k))
    return hooks

def _apply_rope_if_possible(q, k, last_attn_module, seq_len):
    """
    尝试用模块自带的 rotary_emb/rope 对 q/k 施加 RoPE。
    失败则直接返回原 q/k（会有偏差但不影响运行）。
    """
    try:
        rotary = getattr(last_attn_module, "rotary_emb", None) or getattr(last_attn_module, "rope", None)
        if rotary is None:
            return q, k

        B, H, L, D = q.shape
        if hasattr(rotary, "forward"):
            cos, sin = rotary(torch.empty(B*H, L, D, device=q.device, dtype=q.dtype), seq_len=L)
        elif hasattr(rotary, "get_cos_sin"):
            cos, sin = rotary.get_cos_sin(L, device=q.device, dtype=q.dtype)
        else:
            return q, k

        # 形状对齐到 [B,H,L,D]
        if cos.dim() == 2:  # [L, D]
            cos = cos.unsqueeze(0).unsqueeze(0).expand(B, H, L, D)
            sin = sin.unsqueeze(0).unsqueeze(0).expand(B, H, L, D)
        elif cos.dim() == 3:  # [H?, L, D] 或 [B?, L, D]
            cos = cos.unsqueeze(0).expand(B, -1, -1, -1)
            sin = sin.unsqueeze(0).expand(B, -1, -1, -1)

        def _rope(a, cos, sin):
            D2 = a.shape[-1] // 2
            a1, a2 = a[..., :D2], a[..., D2:]
            rot = torch.cat([-a2, a1], dim=-1)
            return a * cos.to(a.dtype) + rot * sin.to(a.dtype)

        return _rope(q, cos, sin), _rope(k, cos, sin)
    except Exception:
        return q, k
    
def calc_attentions(seq: str, model, tokenizer, block_cols=1024) -> List[float]:
    """
    只使用 Q/K 路径：
    - 在最后一层 self-attn 的 q_proj/k_proj 上挂钩抓取 Q/K（包含线性投影，尽量补 RoPE）
    - 用分块 softmax 计算所有头平均后的“按列求和”向量
    返回：长度为 L 的 Python list（float）
    """
    captured.clear()

    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(seq, return_tensors="pt", truncation=False)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}

    last_attn = model.layers[-1].self_attn
    if last_attn is None:
        raise ValueError("未找到最后一层 self-attention 模块，无法挂钩 Q/K。")

    hooks = _attach_qk_hooks(last_attn, model)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)   # 触发前向以捕获 q/k
    finally:
        for h in hooks:
            try: h.remove()
            except Exception: pass

    q = captured.get("q_linear", None)
    k = captured.get("k_linear", None)
    if q is None or k is None:
        raise ValueError("未捕获到 Q/K（可能实现名不匹配或优化融合过深）。")

    # 尝试应用与前向一致的 RoPE
    q, k = q.to(device), k.to(device)
    q, k = _apply_rope_if_possible(q, k, last_attn, q.shape[2])

    # 因果 mask：Causal LM 通常为 True
    col_sums = attn_column_sums_streaming(q, k, causal=True, block_cols=block_cols)  # [B,L]
    vec = col_sums[0]  # 单样本
    return vec.detach().cpu().float().numpy().tolist()