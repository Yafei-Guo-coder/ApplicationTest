import json

# 读入 json 文件
with open("block_10_attn.json", "r") as f:
    data = json.load(f)

# 统计每个样本的 sequence 长度
for item in data:
    spec = item.get("spec")
    seq_len = len(item.get("sequence", []))
    print(spec, seq_len)
