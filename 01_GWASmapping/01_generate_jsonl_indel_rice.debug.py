#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从VCF格式生成样本一致性序列
支持SNP、Indel(插入/缺失)、多等位变异、DEL标记
"""

import os
import json
import numpy as np
import pandas as pd
import pysam
from cyvcf2 import VCF
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="从VCF生成block级样本一致性序列(JSON)"
    )

    parser.add_argument(
        "--bed",
        required=True,
        help="BED 文件（block/window 区间）"
    )
    parser.add_argument(
        "--pheno",
        required=True,
        help="表型文件（必须包含 SampleID 列）"
    )
    parser.add_argument(
        "--vcf",
        required=True,
        help="VCF/VCF.GZ/BCF 文件"
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="参考基因组 FASTA(.fa/.fna/.gz)"
    )
    parser.add_argument(
        "--pheno-col",
        required=True,
        help="表型列名，例如 Trait"
    )
    parser.add_argument(
        "--out",
        default="json_blocks",
        help="输出目录（默认: json_blocks）"
    )

    return parser.parse_args()

# =====================
# 参数读取
# =====================
args = parse_args()

bed_file = args.bed
pheno_file = args.pheno
vcf_file = args.vcf
fasta_file = args.fasta
target_pheno_col = args.pheno_col
out_dir = args.out

os.makedirs(out_dir, exist_ok=True)

# =====================
# 染色体映射
# =====================
chrom_map = {
    '1': '1', '2': '2', '3': '3',
    '4': '4', '5': '5', '6': '6',
    '7': '7', '8': '8', '9': '9',
    '10': '10', '11': '11', '12': '12'
}

# 反向映射：从FASTA染色体名到简化名
chrom_map_reverse = {v: k for k, v in chrom_map.items()}

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
# 4. 读取VCF文件
# =====================
print("➡️ 读取VCF文件 ...")
vcf = VCF(vcf_file)
vcf_samples = vcf.samples
print(f"✅ VCF样本数: {len(vcf_samples)}")
print(f"  示例样本: {vcf_samples[:5]}")

# =====================
# 5. 样本过滤和映射
# =====================
# 找到既在表型中又在VCF中的样本
pheno_samples = [s for s in pheno.index if s in vcf_samples]
sample_to_idx = {sid: i for i, sid in enumerate(vcf_samples)}
pheno_idx = [sample_to_idx[s] for s in pheno_samples]
labels = pheno.loc[pheno_samples, target_pheno_col].values

print(f"✅ 共同样本数: {len(pheno_samples)}")
print(f"示例样本: {pheno_samples[:5]}")

# =====================
# 6. 辅助函数
# =====================
def classify_variant(ref, alt):
    """
    判断变异类型
    返回: 'SNP', 'INS' (insertion), 'DEL' (deletion), 'COMPLEX'
    """
    ref = str(ref).upper()
    alt = str(alt).upper()

    len_ref = len(ref)
    len_alt = len(alt)

    if len_ref == 1 and len_alt == 1:
        return 'SNP'
    elif len_ref < len_alt:
        return 'INS'  # 插入
    elif len_ref > len_alt:
        return 'DEL'  # 缺失
    else:
        return 'COMPLEX'


def apply_variant_to_sequence(ref_seq, variant_list):
    """
    将变异应用到参考序列

    参数:
        ref_seq: 参考序列(字符串)
        variant_list: [(offset, variant_type, ref, alt), ...]
                     offset: 相对于ref_seq起始的位置(0-based)

    返回:
        修改后的序列(字符串)
    """
    # 按位置从后往前排序,避免插入/删除影响后续位置
    variant_list = sorted(variant_list, key=lambda x: x[0], reverse=True)

    seq = list(ref_seq)

    for offset, var_type, ref, alt in variant_list:
        if offset < 0 or offset >= len(seq):
            continue

        if var_type == 'SNP':
            # 简单替换
            seq[offset] = alt

        elif var_type == 'INS':
            # 插入: 在offset位置后插入额外碱基
            seq[offset] = alt[0]
            insert_bases = alt[1:]
            for i, base in enumerate(insert_bases):
                seq.insert(offset + 1 + i, base)

        elif var_type == 'DEL':
            # 缺失: 替换为N
            seq[offset] = alt[0] if len(alt) > 0 else 'N'
            for i in range(1, len(ref)):
                if offset + i < len(seq):
                    seq[offset + i] = 'N'

        elif var_type == 'COMPLEX':
            # 复杂变异: 简单处理为替换+填充N
            seq[offset] = alt[0] if len(alt) > 0 else 'N'
            if len(ref) > 1:
                for i in range(1, len(ref)):
                    if offset + i < len(seq):
                        seq[offset + i] = 'N'

    return ''.join(seq)


# =====================
# 7. 主循环: 逐block处理
# =====================
print("\n" + "="*60)
print("开始处理blocks...")
print("="*60)

for block_id, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Processing blocks"):
    chrom = str(row['chrom'])
    start = int(row['start'])
    end = int(row['end'])
    block_name = f"block_{block_id+1}"

    # 跳过不在映射表中的染色体
    if chrom not in chrom_map:
        print(f"⚠️  跳过 {block_name}: 染色体 {chrom} 不在映射表中")
        continue

    # 提取参考序列
    try:
        ref_seq = fasta.fetch(chrom_map[chrom], start, end).upper()
    except Exception as e:
        print(f"⚠️  跳过 {block_name}: FASTA提取失败 - {e}")
        continue

    print(f"\n📍 {block_name}: 染色体{chrom}:{start}-{end}")

    # 读取该区间的所有变异
    variants_in_block = []
    
    # VCF染色体名可能是"4"或"AP014960.1"，都试一下
    vcf_chrom_names = [chrom, chrom_map.get(chrom, chrom)]
    
    for vcf_chrom in vcf_chrom_names:
        try:
            for variant in vcf(f"{vcf_chrom}:{start}-{end}"):
                variants_in_block.append(variant)
            if len(variants_in_block) > 0:
                break  # 找到变异就停止
        except Exception as e:
            continue
    
    if len(variants_in_block) == 0:
        print(f"⚠️  跳过 {block_name}: 无变异")
        continue

    print(f"  变异数: {len(variants_in_block)}")

    # 逐样本生成序列
    json_list = []

    for i, sample in enumerate(tqdm(pheno_samples, desc=f"  {block_name} 样本", leave=False)):
        sample_idx = pheno_idx[i]
        sample_variants = []

        for variant in variants_in_block:
            # 获取该样本的基因型
            gt = variant.genotypes[sample_idx]
            
            # gt是一个列表: [allele1, allele2, phased]
            # allele: 0=REF, 1=第1个ALT, 2=第2个ALT, -1=缺失
            allele1, allele2, phased = gt[0], gt[1], gt[2]
            
            # 🔥 调试：只看前5个样本
            if i < 10:
                print(f"\n样本 {sample} (索引{i}):")
                print(f"  位置: {variant.POS}")
                print(f"  基因型: {allele1}/{allele2}")
                print(f"  REF: {variant.REF}, ALT: {variant.ALT}")
                print(f"  BED: {start}-{end}, 参考序列长度: {len(ref_seq)}")
            
            # 处理缺失基因型
            if allele1 == -1 or allele2 == -1:
                applied_allele = 'N' * len(variant.REF)
                var_type = 'COMPLEX'
                offset = variant.POS - start - 1
                sample_variants.append((offset, var_type, variant.REF, applied_allele))
                if i < 10:
                    print(f"  → 缺失基因型，添加: offset={offset}, type={var_type}")
                continue
            
            # 纯合参考型 (0/0)
            elif allele1 == 0 and allele2 == 0:
                if i < 10:
                    print(f"  → 纯合参考型，跳过")
                continue  # 不修改参考序列
            
            # 其他情况：使用等位基因
            else:
                if i < 10:
                    print(f"  → 非参考基因型，开始处理...")
                
                # 对于杂合子，选择非参考等位基因
                # 对于纯合子，使用该等位基因
                if allele1 == 0:
                    allele_idx = allele2  # 0/1 -> 使用ALT1
                elif allele2 == 0:
                    allele_idx = allele1  # 1/0 -> 使用ALT1
                else:
                    # 两个都是ALT，优先使用较大的（更后面的ALT）
                    allele_idx = max(allele1, allele2)
                
                if i < 10:
                    print(f"  选择等位基因索引: {allele_idx}")
                
                # 获取对应的ALT序列
                if allele_idx == 0:
                    if i < 10:
                        print(f"  → 等位基因索引为0，跳过")
                    continue  # 参考等位基因（不应该到这里）
                else:
                    # allele_idx-1 对应到ALT列表的索引
                    # 例如：allele_idx=1 对应 ALT[0]，allele_idx=2 对应 ALT[1]
                    alt_list = variant.ALT
                    alt_index = allele_idx - 1
                    
                    if i < 10:
                        print(f"  ALT列表: {alt_list}, alt_index={alt_index}")
                    
                    if alt_index < len(alt_list):
                        applied_allele = alt_list[alt_index]
                        
                        if i < 10:
                            print(f"  获取到applied_allele: {applied_allele}")
                        
                        # 🔥 关键：处理DEL标记（可能是 "DEL", "<DEL>", "*" 等）
                        if applied_allele in ['DEL', '<DEL>', '*', '.']:
                            # DEL表示该位置完全缺失，用N填充REF长度
                            applied_allele = 'N' * len(variant.REF)
                            var_type = 'DEL'
                            if i < 10:
                                print(f"  → DEL标记，转换为N")
                        else:
                            # 正常变异，判断类型
                            var_type = classify_variant(variant.REF, applied_allele)
                            if i < 10:
                                print(f"  → 正常变异，类型: {var_type}")
                    else:
                        # 索引越界，标记为缺失
                        applied_allele = 'N' * len(variant.REF)
                        var_type = 'COMPLEX'
                        offset = variant.POS - start - 1
                        sample_variants.append((offset, var_type, variant.REF, applied_allele))
                        if i < 10:
                            print(f"  → 索引越界，添加: offset={offset}, type={var_type}")
                        continue

            # 计算相对参考序列的offset (0-based)
            # VCF的POS是1-based
            offset = variant.POS - start - 1

            if i < 10:
                print(f"  计算offset: {variant.POS} - {start} - 1 = {offset}")
                print(f"  添加到sample_variants: offset={offset}, type={var_type}, ref={variant.REF}, alt={applied_allele}")

            sample_variants.append((offset, var_type, variant.REF, applied_allele))

        # 🔥 应用变异前后对比
        if i < 10:
            print(f"\n样本 {sample} 应用变异:")
            print(f"  sample_variants 数量: {len(sample_variants)}")
            if len(sample_variants) > 0:
                for sv in sample_variants:
                    offset_sv = sv[0]
                    print(f"  变异 offset={offset_sv}: 原始ref_seq[{offset_sv}]={ref_seq[offset_sv] if offset_sv < len(ref_seq) else '越界'}")
        
        # 应用所有变异到参考序列
        consensus_seq = apply_variant_to_sequence(ref_seq, sample_variants)

        if i < 10:
            if len(sample_variants) > 0:
                for sv in sample_variants:
                    offset_sv = sv[0]
                    print(f"  变异 offset={offset_sv}: 替换后consensus_seq[{offset_sv}]={consensus_seq[offset_sv] if offset_sv < len(consensus_seq) else '越界'}")

        json_list.append({
            "label": int(labels[i]),
            "spec": sample,
            "loc": block_name,
            "sequence": consensus_seq
        })

    # 输出JSON
    out_path = os.path.join(out_dir, f"{block_name}.json")
    with open(out_path, "w") as f:
        json.dump(json_list, f, indent=2)

    print(f"✅ {block_name} 完成 | 变异数={len(variants_in_block)} | 样本数={len(pheno_samples)}")

# =====================
# 8. 清理
# =====================
fasta.close()
vcf.close()

print("\n" + "="*60)
print("🎉 全部完成!")
print(f"输出目录: {out_dir}")
print("="*60)
