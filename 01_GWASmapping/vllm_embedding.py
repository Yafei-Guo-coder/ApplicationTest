import os
import json
import torch
import time
from tqdm import tqdm
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.pooling_params import PoolingParams

# =========================
# 配置区
# =========================
model_path = "/mnt/zzb/default/Workspace/xz/hf/rice_1B_stage2_8k_hf"

train_path = "/mnt/zzb/default/Workspace/zhushenjun/Rice-Genome/basic_benchmark/vllm_test/train.jsonl"
test_path  = "/mnt/zzb/default/Workspace/zhushenjun/Rice-Genome/basic_benchmark/vllm_test/test.jsonl"
# eval_path  = "/mnt/zzb/default/Workspace/zhushenjun/Rice-Genome/basic_benchmark/varieties_classification_128k/eval.jsonl"

output_dir = "/mnt/zzb/default/Workspace/zhushenjun/benchmarks/vllm_embedding/test"
dataset_name = "test"

os.makedirs(output_dir, exist_ok=True)

seq_length = 128 * 1024
gpu_num = 2
batch_size = 4

# =========================
# 初始化 vLLM
# =========================
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=gpu_num,
    block_size=128,
    enable_prefix_caching=True,
    enforce_eager=False,
    gpu_memory_utilization=0.85,
    dtype=torch.bfloat16,
    max_model_len=seq_length,
    max_num_batched_tokens=seq_length*batch_size,
    override_pooler_config=PoolerConfig(pooling_type="MEAN", normalize=False),
    task="embedding",
    enable_chunked_prefill=True
)

tokenizer = llm.get_tokenizer()

# =========================
# 读取 jsonl
# =========================
def load_jsonl(path):
    seqs, labels = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            seqs.append(obj["sequence"])
            labels.append(obj["label"])
    return seqs, torch.tensor(labels)

# =========================
# vLLM batch 编码
# =========================
def encode_batch(seqs):
    token_ids = tokenizer(seqs, add_special_tokens=False)["input_ids"]

    outputs = llm.encode(
        prompt_token_ids=token_ids,
        pooling_params=PoolingParams()   # ✅ 必须传
    )
    print("Real batch:", len(outputs))

    pooled_embeddings = []
    for out in outputs:
        emb = out.outputs.data          # [hidden]
        pooled_embeddings.append(emb.float().cpu())

    return torch.stack(pooled_embeddings, dim=0)  # [B, H]

# =========================
# 主流程
# =========================
def run_split(seqs, labels, split="train"):
    print(f"\n🚀 Processing {split} set...")

    all_embeddings = []

    start = time.time()
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch_seqs = seqs[i:i+batch_size]
        emb = encode_batch(batch_seqs)
        all_embeddings.append(emb)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    data = {
        "embeddings": all_embeddings,
        "labels": labels
    }

    out_path = f"{output_dir}/{dataset_name}-lastlayer_{split}.pt"
    torch.save(data, out_path)

    print(f"✅ Saved: {out_path}")
    print("Shape:", all_embeddings.shape)
    print("Time used:", time.time() - start, "s")

# =========================
# 跑三份数据
# =========================
train_seqs, train_labels = load_jsonl(train_path)
test_seqs,  test_labels  = load_jsonl(test_path)
# eval_seqs,  eval_labels  = load_jsonl(eval_path)

print("Train:", len(train_seqs))
print("Test :", len(test_seqs))
# print("Eval :", len(eval_seqs))

run_split(train_seqs, train_labels, "train")
run_split(test_seqs,  test_labels,  "test")
# run_split(eval_seqs,  eval_labels,  "eval")

print("\n🎉 All done.")
