#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流式注意力分数提取（内存高效版）
基于Q/K手动计算，支持超长序列
"""

import argparse
import os
import json
import math
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==================== 核心函数（来自文档6）====================

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
                Kb = K[j0:j1]                                  # [B,D]
                S = (Q @ Kb.t()) * scale                       # [L,B]

                if causal:
                    row_idx = torch.arange(L, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j0, j1, device=Q.device).unsqueeze(0)
                    S = S.masked_fill(col_idx > row_idx, float('-inf'))

                block_row_max = torch.max(S, dim=1).values
                new_m = torch.maximum(m, block_row_max)
                l *= torch.exp(m - new_m)

                exp_scores = torch.exp(S - new_m.unsqueeze(1))
                l += torch.sum(exp_scores, dim=1)

                probs_block = exp_scores / l.unsqueeze(1)
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
    H_q = getattr(getattr(model, "config", object()), "num_attention_heads", None)
    H_kv = getattr(getattr(model, "config", object()), "num_key_value_heads", None)
    return int(H_q), int(H_kv)


def _attach_qk_hooks(last_attn_module, model):
    """
    从 config 读 H_q/H_kv；Q 按 H_q reshape，K 按 H_kv reshape，再把 K 扩展到 H_q。
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
        B, L, Dall_q = out.shape
        assert Dall_q % H_q == 0
        D = Dall_q // H_q
        q = out.view(B, L, H_q, D).permute(0, 2, 1, 3).contiguous()
        captured["q_linear"] = q.detach()

    def _grab_k(module, inp, out):
        B, L, Dall_k = out.shape
        assert Dall_k % H_kv == 0
        D = Dall_k // H_kv
        k = out.view(B, L, H_kv, D).permute(0, 2, 1, 3).contiguous()
        if H_kv != H_q:
            k = k.repeat_interleave(group, dim=1)
        captured["k_linear"] = k.detach()

    hooks.append(q_linear.register_forward_hook(_grab_q))
    hooks.append(k_linear.register_forward_hook(_grab_k))
    return hooks


def _apply_rope_if_possible(q, k, last_attn_module, seq_len):
    """尝试应用RoPE"""
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

        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0).expand(B, H, L, D)
            sin = sin.unsqueeze(0).unsqueeze(0).expand(B, H, L, D)
        elif cos.dim() == 3:
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


def calc_attentions_streaming(seq: str, model, tokenizer, device, block_cols=1024, causal=False) -> List[float]:
    """
    使用流式方法计算注意力分数
    
    参数:
        causal: True=因果mask（只看前面），False=双向注意力（看全部）
    """
    captured.clear()

    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(seq, return_tensors="pt", truncation=False)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}

    last_attn = model.model.layers[-1].self_attn
    if last_attn is None:
        raise ValueError("未找到最后一层 self-attention 模块")

    hooks = _attach_qk_hooks(last_attn, model)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in hooks:
            try: h.remove()
            except Exception: pass

    q = captured.get("q_linear", None)
    k = captured.get("k_linear", None)
    if q is None or k is None:
        raise ValueError("未捕获到 Q/K")

    q, k = q.to(device), k.to(device)
    q, k = _apply_rope_if_possible(q, k, last_attn, q.shape[2])

    # 使用传入的causal参数
    col_sums = attn_column_sums_streaming(q, k, causal=causal, block_cols=block_cols)
    vec = col_sums[0]
    return vec.detach().cpu().float().numpy().tolist()


# ==================== 主程序（批量处理JSON）====================

def parse_arguments():
    parser = argparse.ArgumentParser(description="流式注意力分数提取（内存高效版）")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入JSON目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出JSON目录")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--file_pattern", type=str, default="*.json", help="文件匹配模式，如 block_*.json")
    parser.add_argument("--output_suffix", type=str, default="_streaming", help="输出文件后缀")
    parser.add_argument("--block_cols", type=int, default=1024, help="流式计算的块大小（越小越省内存）")
    parser.add_argument("--keep_original_sequence", action="store_true", help="保留原始序列字段")
    parser.add_argument("--original_seq_field", type=str, default="original_sequence", help="原始序列字段名")
    parser.add_argument("--causal", action="store_true", 
                       help="使用因果mask（只看前面的token），默认为双向注意力")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("流式注意力分数提取（内存高效版）")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"块大小: {args.block_cols}")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # 加载模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("✅ 模型加载成功")
    
    # 获取JSON文件
    json_files = list(Path(args.input_dir).glob(args.file_pattern))
    print(f"\n找到 {len(json_files)} 个文件")
    
    total_samples = 0
    
    for json_file in json_files:
        output_file = Path(args.output_dir) / f"{json_file.stem}{args.output_suffix}.json"
        
        print(f"\n{'='*60}")
        print(f"处理文件: {json_file.name}")
        
        dataset = load_dataset("json", data_files=str(json_file), split="all")
        samples = list(dataset)
        
        print(f"样本数: {len(samples)}")
        
        processed = []
        
        for sample in tqdm(samples, desc="处理样本"):
            seq = sample["sequence"]
            
            try:
                # 🔥 使用参数控制是否causal
                attn_scores = calc_attentions_streaming(
                    seq, model, tokenizer, device, 
                    block_cols=args.block_cols,
                    causal=args.causal  # ← 新增参数
                )
                
                new_sample = sample.copy()
                if args.keep_original_sequence:
                    new_sample[args.original_seq_field] = seq
                new_sample["sequence"] = attn_scores
                
                processed.append(new_sample)
                
            except Exception as e:
                print(f"\n⚠️ 处理样本失败: {e}")
                processed.append(sample)
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        total_samples += len(processed)
        print(f"✅ 保存到: {output_file.name}")
    
    print("\n" + "=" * 80)
    print("🎉 处理完成!")
    print(f"总样本数: {total_samples}")
    print("=" * 80)


if __name__ == "__main__":
    main()
