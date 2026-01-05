#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从PLINK2通过VCF中间格式生成样本一致性序列
避免pgenlib的segfault问题
"""

import os
import json
import subprocess
import pandas as pd
import pysam
from tqdm import tqdm
import tempfile
import shutil

# =====================
# 路径配置
# =====================
bed_file = "GAD1.bed"
pheno_file = "/mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/2_3K_Rice_pheno"
pgen_prefix = "RICE_RP_GAD1"
fasta_file = "/mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/GCA_001433935.1_IRGSP-1.0_genomic.fna.gz"

target_pheno_col = "awn_presence"
out_dir = "json_blocks_APTrue_with_indel_vcf"
os.makedirs(out_dir, exist_ok=True)

# 创建临时目录
temp_dir = tempfile.mkdtemp(prefix="plink2_vcf_")
print(f"临时目录: {temp_dir}")

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
# 1. 读取表型
# =====================
print("➡️ 读取表型数据 ...")
pheno = pd.read_csv(pheno_file, sep="\t")
pheno = pheno.dropna(subset=[target_pheno_col])
pheno = pheno.set_index("SampleID")
print(f"✅ 有效表型样本数: {len(pheno)}")

# =====================
# 2. 读取BED区间
# =====================
print("➡️ 读取BED文件 ...")
bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
print(f"✅ BED 区间数: {len(bed_df)}")

# =====================
# 3. 读取FASTA参考序列
# =====================
print("➡️ 打开FASTA文件 ...")
fasta = pysam.FastaFile(fasta_file)

# =====================
# 4. 读取样本ID映射
# =====================
print("➡️ 读取PSAM获取样本列表 ...")
psam = pd.read_csv(f"{pgen_prefix}.psam", sep="\t")
psam.columns = psam.columns.str.replace('#', '')

# 找到样本ID列
sample_col = None
for col in ['IID', 'iid']:
    if col in psam.columns:
        sample_col = col
        break
if sample_col is None:
    sample_col = psam.columns[1] if len(psam.columns) >= 2 else psam.columns[0]

all_sample_ids = psam[sample_col].tolist()
print(f"✅ 基因型样本数: {len(all_sample_ids)}")

# 样本匹配
pheno_samples = [s for s in pheno.index if s in all_sample_ids]
sample_to_idx = {sid: i for i, sid in enumerate(pheno_samples)}
labels = pheno.loc[pheno_samples, target_pheno_col].values

print(f"✅ 共同样本数: {len(pheno_samples)}")

if len(pheno_samples) == 0:
    print("❌ 错误: 没有共同样本")
    exit(1)

# =====================
# 5. 辅助函数
# =====================
def classify_variant(ref, alt):
    """判断变异类型"""
    ref = str(ref).upper()
    alt = str(alt).upper()
    
    len_ref = len(ref)
    len_alt = len(alt)
    
    if len_ref == 1 and len_alt == 1:
        return 'SNP'
    elif len_ref < len_alt:
        return 'INS'
    elif len_ref > len_alt:
        return 'DEL'
    else:
        return 'COMPLEX'


def apply_variant_to_sequence(ref_seq, variant_list):
    """将变异应用到参考序列"""
    variant_list = sorted(variant_list, key=lambda x: x[0], reverse=True)
    seq = list(ref_seq)
    
    for offset, var_type, ref, alt in variant_list:
        if offset < 0 or offset >= len(seq):
            continue
        
        if var_type == 'SNP':
            seq[offset] = alt
            
        elif var_type == 'INS':
            seq[offset] = alt[0]
            insert_bases = alt[1:]
            for i, base in enumerate(insert_bases):
                seq.insert(offset + 1 + i, base)
            
        elif var_type == 'DEL':
            seq[offset] = alt[0] if len(alt) > 0 else 'N'
            for i in range(1, len(ref)):
                if offset + i < len(seq):
                    seq[offset + i] = 'N'
        
        elif var_type == 'COMPLEX':
            seq[offset] = alt[0] if len(alt) > 0 else 'N'
            if len(ref) > 1:
                for i in range(1, len(ref)):
                    if offset + i < len(seq):
                        seq[offset + i] = 'N'
    
    return ''.join(seq)


def extract_region_to_vcf(pgen_prefix, chrom, start, end, output_vcf):
    """
    使用PLINK2将指定区间导出为VCF
    """
    cmd = [
        'plink2',
        '--pfile', pgen_prefix,
        '--chr', str(chrom),
        '--from-bp', str(start + 1),
        '--to-bp', str(end),
        '--export', 'vcf',
        '--out', output_vcf.replace('.vcf', '').replace('.vcf.gz', '')
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # PLINK2输出的是 xxx.vcf
        expected_vcf = output_vcf.replace('.vcf', '').replace('.vcf.gz', '') + '.vcf'
        
        if os.path.exists(expected_vcf):
            # 如果需要压缩
            if output_vcf.endswith('.gz'):
                subprocess.run(['bgzip', '-f', expected_vcf], check=True)
                subprocess.run(['tabix', '-f', '-p', 'vcf', expected_vcf + '.gz'], check=True)
                return expected_vcf + '.gz'
            return expected_vcf
        else:
            print(f"  ⚠️  VCF文件未生成: {expected_vcf}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"  ✗ PLINK2错误: {e.stderr}")
        return None
    except FileNotFoundError:
        print("  ✗ 错误: 未找到plink2命令")
        print("    请安装: conda install -c bioconda plink2")
        exit(1)


# =====================
# 6. 主循环
# =====================
print("\n" + "="*60)
print("开始处理blocks...")
print("="*60)

for block_id, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Processing blocks"):
    chrom = str(row['chrom'])
    start = int(row['start'])
    end = int(row['end'])
    block_name = f"block_{block_id+1}"
    
    if chrom not in chrom_map:
        print(f"⚠️  跳过 {block_name}: 染色体 {chrom} 不在映射表中")
        continue
    
    # 提取参考序列
    try:
        ref_seq = fasta.fetch(chrom_map[chrom], start, end).upper()
    except Exception as e:
        print(f"⚠️  跳过 {block_name}: FASTA提取失败 - {e}")
        continue
    
    print(f"\n📍 {block_name}: chr{chrom}:{start}-{end}")
    
    # 导出该区间的VCF
    vcf_path = os.path.join(temp_dir, f"{block_name}.vcf")
    vcf_file = extract_region_to_vcf(pgen_prefix, chrom, start, end, vcf_path)
    
    if vcf_file is None or not os.path.exists(vcf_file):
        print(f"  ⚠️  跳过 {block_name}: VCF导出失败")
        continue
    
    # 读取VCF
    try:
        vcf = pysam.VariantFile(vcf_file)
    except Exception as e:
        print(f"  ✗ VCF读取失败: {e}")
        continue
    
    # 获取VCF中的样本列表
    vcf_samples = list(vcf.header.samples)
    
    # 匹配样本
    common_in_vcf = [s for s in pheno_samples if s in vcf_samples]
    
    if len(common_in_vcf) == 0:
        print(f"  ⚠️  跳过 {block_name}: VCF中无共同样本")
        vcf.close()
        continue
    
    print(f"  VCF样本数: {len(vcf_samples)}, 共同样本: {len(common_in_vcf)}")
    
    # 收集该区间的所有变异
    variants_list = []
    for record in vcf.fetch():
        variants_list.append(record)
    
    print(f"  变异数: {len(variants_list)}")
    
    if len(variants_list) == 0:
        vcf.close()
        continue
    
    # 逐样本生成序列
    json_list = []
    
    for sample in tqdm(common_in_vcf, desc=f"  {block_name} 样本", leave=False):
        sample_variants = []
        
        for record in variants_list:
            pos = record.pos
            ref = str(record.ref).upper()
            alts = [str(a).upper() for a in record.alts] if record.alts else []
            
            # 过滤特殊标记
            alts = [a for a in alts if a not in ['*', '<DEL>', 'DEL']]
            
            if len(alts) == 0:
                continue
            
            # 获取该样本的基因型
            try:
                gt = record.samples[sample]['GT']
            except:
                gt = (None, None)
            
            # 解析基因型
            if gt is None or None in gt:
                # Missing
                applied_allele = 'N' * len(ref)
            elif gt == (0, 0):
                # REF/REF
                continue
            elif 0 in gt:
                # 杂合: 使用ALT
                alt_idx = max(gt) - 1
                applied_allele = alts[alt_idx] if alt_idx < len(alts) else alts[0]
            else:
                # 纯合ALT
                alt_idx = gt[0] - 1
                applied_allele = alts[alt_idx] if alt_idx < len(alts) else alts[0]
            
            # 计算offset
            offset = pos - start - 1
            
            # 变异类型
            var_type = classify_variant(ref, applied_allele)
            
            sample_variants.append((offset, var_type, ref, applied_allele))
        
        # 生成序列
        try:
            consensus_seq = apply_variant_to_sequence(ref_seq, sample_variants)
        except Exception as e:
            print(f"\n  ✗ 样本 {sample} 失败: {e}")
            consensus_seq = ref_seq
        
        # 获取标签
        label = int(pheno.loc[sample, target_pheno_col])
        
        json_list.append({
            "label": label,
            "spec": sample,
            "loc": block_name,
            "sequence": consensus_seq
        })
    
    vcf.close()
    
    # 保存JSON
    out_path = os.path.join(out_dir, f"{block_name}.json")
    with open(out_path, "w") as f:
        json.dump(json_list, f, indent=2)
    
    print(f"  ✅ {block_name} 完成 | 样本数={len(json_list)}")

# =====================
# 7. 清理
# =====================
fasta.close()

print("\n清理临时文件...")
shutil.rmtree(temp_dir)

print("\n" + "="*60)
print("🎉 全部完成!")
print(f"输出目录: {out_dir}")
print("="*60)
