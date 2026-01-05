import os
import json
import numpy as np
import pandas as pd
import pysam
from pandas_plink import read_plink
from tqdm import tqdm

# =====================
# 路径配置
# =====================
# bed_file = "awn.p3.merge.expand.bed"

# bed_file = "GAD1.bed"
bed_file = "GS3.bed"
pheno_file = "/mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/2_3K_Rice_pheno"
# bfile_prefix = "RICE_RP_382_region"
# bfile_prefix = "RICE_RP_GAD1"
bfile_prefix = "RICE_RP_GS3"
# bfile_prefix = "/mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/genotyping_data/2_3K_rice_and_7_RiceDiversityPanel/RICE_RP_mLIDs"
fasta_file = "/mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/GCA_001433935.1_IRGSP-1.0_genomic.fna.gz"

# target_pheno_col = "awn_presence"
# out_dir = "json_blocks_awnTrue"
target_pheno_col = "grain_length"
out_dir = "json_blocks_GLTrue"
os.makedirs(out_dir, exist_ok=True)

# =====================
# 染色体映射
# =====================
chrom_map = {
    '1': 'AP014957.1', '2': 'AP014958.1', '3': 'AP014959.1',
    '4': 'AP014960.1', '5': 'AP014961.1', '6': 'AP014962.1',
    '7': 'AP014963.1', '8': 'AP014964.1', '9': 'AP014965.1',
    '10': 'AP014966.1', '11': 'AP014967.1', '12': 'AP014968.1'
}

# =====================
# 表型
# =====================
pheno = pd.read_csv(pheno_file, sep="\t")
pheno = pheno.dropna(subset=[target_pheno_col])
pheno = pheno.set_index("SampleID")
print(f"有效表型样本数: {len(pheno)}")

# =====================
# BED blocks
# =====================
bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
print(f"BED 区间数: {len(bed_df)}")

# =====================
# FASTA
# =====================
fasta = pysam.FastaFile(fasta_file)

# =====================
# PLINK（一次性加载）
# =====================
print("➡️ 读取 PLINK 文件 ...")
bim, fam, G = read_plink(bfile_prefix, verbose=True)
# G: (n_snps, n_samples)
G = G.compute().astype(np.int8)
print(f"✅ genotype matrix loaded: {G.shape}")

# 样本顺序映射
sample_ids = fam.iid.to_list()
sample_to_col = {sid: i for i, sid in enumerate(sample_ids)}

# 表型样本在 genotype 中的列号
pheno_samples = [s for s in pheno.index if s in sample_to_col]
pheno_idx = np.array([sample_to_col[s] for s in pheno_samples])
labels = pheno.loc[pheno_samples, target_pheno_col].astype(int).values

print(f"PLINK 样本数（表型过滤后）: {len(pheno_samples)}")
print("示例样本:", pheno_samples[:5])

# =====================
# 主循环：block 级
# =====================
for block_id, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Processing blocks"):
    chrom = str(row.chrom)
    start = int(row.start)
    end = int(row.end)
    block_name = f"block_{block_id+1}"

    if chrom not in chrom_map:
        continue

    try:
        ref_seq = fasta.fetch(chrom_map[chrom], start, end).upper()
    except Exception:
        continue

    # ---- SNP 子集
    mask = (
        (bim.chrom.astype(str) == chrom) &
        (bim.pos.values >= start + 1) &
        (bim.pos.values <= end)
    )
    snp_idx = np.where(mask)[0]

    if snp_idx.size == 0:
        continue

    # ---- 子矩阵 (n_snps × n_samples) → 转置为 (n_samples × n_snps)
    G_block = G[snp_idx[:, None], pheno_idx].T

    # ---- SNP 信息
    snp_pos = bim.pos.values[snp_idx]
    ref_allele = bim.a0.values[snp_idx]
    alt_allele = bim.a1.values[snp_idx]

    json_list = []

    for i, sample in enumerate(pheno_samples):
        seq = list(ref_seq)
        for j, pos in enumerate(snp_pos):
            offset = pos - start - 1
            if offset < 0 or offset >= len(seq):
                continue
            g = G_block[i, j]
            if g == 0:
                seq[offset] = ref_allele[j]
            elif g in (1, 2):
                seq[offset] = alt_allele[j]
        json_list.append({
            "label": int(labels[i]),
            "spec": sample,
            "loc": block_name,
            "sequence": "".join(seq)
        })

    # 输出 JSON
    out_path = os.path.join(out_dir, f"{block_name}.json")
    with open(out_path, "w") as f:
        json.dump(json_list, f, indent=2)

    print(f"✔ {block_name} 输出完成 | SNPs={len(snp_idx)} | 样本数={len(pheno_samples)}")

fasta.close()
print("🎉 全部完成")
