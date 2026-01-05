import json
import pysam

# ----------------------
# 配置
# ----------------------
fasta_file = "/mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/GCA_001433935.1_IRGSP-1.0_genomic.fna.gz"
json_file = "json_blocks/block_1.json"  # 只检查第一个 block

# 对应染色体映射
chrom_map = {
    '1': 'AP014957.1',
    '2': 'AP014958.1',
    '3': 'AP014959.1',
    '4': 'AP014960.1',
    '5': 'AP014961.1',
    '6': 'AP014962.1',
    '7': 'AP014963.1',
    '8': 'AP014964.1',
    '9': 'AP014965.1',
    '10': 'AP014966.1',
    '11': 'AP014967.1',
    '12': 'AP014968.1'
}

# ----------------------
# 打开 FASTA
# ----------------------
fasta = pysam.FastaFile(fasta_file)
print("染色体列表:", fasta.references)

# ----------------------
# 读取 JSON
# ----------------------
with open(json_file) as f:
    data = json.load(f)

print(f"JSON 中样本数: {len(data)}")

# ----------------------
# 检查第一个样本的序列
# ----------------------
sample_info = data[0]
sample_id = sample_info["spec"]
block_name = sample_info["loc"]
sequence = sample_info["sequence"]

print(f"检查样本: {sample_id}, block: {block_name}")

# 假设 block_name 格式为 block_数字，对应 bed_df 中的行号
block_idx = int(block_name.split("_")[1]) - 1

# 读取 BED
import pandas as pd
bed_file = "awn.p3.merge.expand.bed"
bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
block_row = bed_df.iloc[block_idx]
chrom = str(block_row.chrom)
start = int(block_row.start)
end = int(block_row.end)

# 获取参考序列
ref_seq = list(fasta.fetch(chrom_map[chrom], start, end).upper())

# ----------------------
# 对比 SNP 替换
# ----------------------
diff_positions = []
for i, (r, s) in enumerate(zip(ref_seq, sequence)):
    if r != s:
        diff_positions.append((i + start, r, s))  # 输出基因组位置和替换信息

print(f"共有 {len(diff_positions)} 个 SNP 被替换:")
for pos, ref, alt in diff_positions[:20]:  # 只显示前20个
    print(f"位置: {pos}, REF: {ref} -> ALT: {alt}")

fasta.close()
