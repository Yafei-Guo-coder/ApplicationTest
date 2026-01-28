#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化attention分数长度
将因indel导致的不同长度序列的分数标准化为参考序列长度
- 插入(INS): 多个碱基的分数平均为1个
- 缺失(DEL): 保持N位置的分数
- 支持多等位基因（ALT可能是 "T,DEL" 或 "A,T,DEL"）
"""

import os
import json
import numpy as np
import pandas as pd
import pysam
from cyvcf2 import VCF
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='标准化attention分数长度')
    parser.add_argument('--json_dir', type=str, required=True,
                       help='包含原始JSON文件的目录 (block_*.json)')
    parser.add_argument('--seq_json_dir', type=str, required=True,
                       help='包含序列JSON的目录 (用于获取实际序列)')
    parser.add_argument('--bed_file', type=str, required=True,
                       help='BED文件,定义参考区间')
    parser.add_argument('--vcf_file', type=str, required=True,
                       help='VCF文件路径（支持.vcf, .vcf.gz, .bcf）')
    parser.add_argument('--fasta_file', type=str, required=True,
                       help='参考基因组FASTA')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    return parser.parse_args()


# 染色体映射
chrom_map = {
    '1': 'AP014957.1', '2': 'AP014958.1', '3': 'AP014959.1',
    '4': 'AP014960.1', '5': 'AP014961.1', '6': 'AP014962.1',
    '7': 'AP014963.1', '8': 'AP014964.1', '9': 'AP014965.1',
    '10': 'AP014966.1', '11': 'AP014967.1', '12': 'AP014968.1'
}


def classify_variant(ref, alt):
    """
    判断单个等位基因的变异类型
    
    注意：这里的alt是单个等位基因，不是"T,DEL"这种多等位格式
    """
    # 处理DEL标记
    if alt in ['DEL', '<DEL>', '*', '.']:
        return 'DEL'
    
    ref = str(ref).upper()
    alt = str(alt).upper()
    
    if len(ref) == 1 and len(alt) == 1:
        return 'SNP'
    elif len(ref) < len(alt):
        return 'INS'
    elif len(ref) > len(alt):
        return 'DEL'
    else:
        return 'COMPLEX'


def get_variant_info(ref, alt):
    """
    获取变异信息
    
    返回: (variant_type, ins_length)
    """
    var_type = classify_variant(ref, alt)
    
    if alt in ['DEL', '<DEL>', '*', '.']:
        ins_length = 0
    else:
        ins_length = max(0, len(alt) - len(ref))
    
    return var_type, ins_length


def load_variants_in_region_from_vcf(vcf_file, chrom, start, end):
    """
    从VCF文件读取指定区间的变异信息
    
    返回: DataFrame with columns [POS, REF, ALT, variant_type, ins_length]
    注意：这里只记录每个位置可能出现的最大插入长度
    """
    vcf = VCF(vcf_file)
    
    variants_list = []
    
    # VCF染色体名可能是"4"或"AP014960.1"
    vcf_chrom_names = [str(chrom), chrom_map.get(str(chrom), str(chrom))]
    
    for vcf_chrom in vcf_chrom_names:
        try:
            for variant in vcf(f"{vcf_chrom}:{start}-{end}"):
                pos = variant.POS
                ref = variant.REF
                alts = variant.ALT
                
                # 分析所有ALT等位基因，找出最极端的情况
                max_ins_len = 0
                variant_types = set()
                
                for alt in alts:
                    var_type, ins_len = get_variant_info(ref, alt)
                    variant_types.add(var_type)
                    max_ins_len = max(max_ins_len, ins_len)
                
                # 如果有任何一个ALT是INS，就标记为INS
                if 'INS' in variant_types:
                    final_type = 'INS'
                elif 'DEL' in variant_types:
                    final_type = 'DEL'
                elif 'SNP' in variant_types:
                    final_type = 'SNP'
                else:
                    final_type = 'COMPLEX'
                
                variants_list.append({
                    'POS': pos,
                    'REF': ref,
                    'ALT': ','.join(alts),
                    'variant_type': final_type,
                    'ins_length': max_ins_len
                })
            
            if len(variants_list) > 0:
                break
        except Exception as e:
            continue
    
    vcf.close()
    
    if len(variants_list) == 0:
        return pd.DataFrame(columns=['POS', 'REF', 'ALT', 'variant_type', 'ins_length'])
    
    return pd.DataFrame(variants_list).sort_values('POS')


def get_sample_variants_from_vcf(vcf_file, chrom, start, end, sample_id, all_samples):
    """
    从VCF获取特定样本实际携带的变异
    
    返回: list of (position, variant_type, ins_length)
    """
    vcf = VCF(vcf_file)
    
    # 找到样本索引
    if sample_id not in all_samples:
        vcf.close()
        return []
    
    sample_idx = all_samples.index(sample_id)
    
    sample_variants = []
    
    # VCF染色体名可能是"4"或"AP014960.1"
    vcf_chrom_names = [str(chrom), chrom_map.get(str(chrom), str(chrom))]
    
    for vcf_chrom in vcf_chrom_names:
        try:
            for variant in vcf(f"{vcf_chrom}:{start}-{end}"):
                # 获取该样本的基因型
                gt = variant.genotypes[sample_idx]
                allele1, allele2 = gt[0], gt[1]
                
                # 跳过参考型 0/0
                if allele1 == 0 and allele2 == 0:
                    continue
                
                # 跳过缺失基因型
                if allele1 == -1 or allele2 == -1:
                    continue
                # 🔥 正确逻辑：逐个等位基因判断，只记录 INS
                alleles = [allele1, allele2]

                for allele_idx in alleles:
                    if allele_idx <= 0:
                        continue

                    if allele_idx - 1 >= len(variant.ALT):
                        continue

                    alt = variant.ALT[allele_idx - 1]
                    ref = variant.REF

                    var_type, ins_len = get_variant_info(ref, alt)

                    # 只有当该样本真的携带 INS 时才记录
                    if var_type == 'INS' and ins_len > 0:
                        sample_variants.append({
                            'POS': variant.POS,
                            'ins_length': ins_len
                        })

            
            if len(sample_variants) > 0 or vcf_chrom == vcf_chrom_names[-1]:
                break
        except Exception as e:
            continue
    
    vcf.close()
    return sample_variants


def build_position_mapping_with_sample_variants(ref_seq, sample_variants, ref_start):
    """
    使用样本实际携带的变异构建映射
    
    参数:
        ref_seq: 参考序列
        sample_variants: 该样本实际携带的变异列表
        ref_start: 参考序列起始位置
    
    返回:
        mapping: 样本序列位置 -> 参考序列位置
    """
    ref_len = len(ref_seq)
    
    # 如果没有插入变异，长度相同
    if not sample_variants:
        return list(range(ref_len))
    
    # 构建变异字典：ref_offset -> ins_length
    var_dict = {}
    for var in sample_variants:
        ref_offset = var['POS'] - ref_start - 1
        if 0 <= ref_offset < ref_len:
            var_dict[ref_offset] = var['ins_length']
    
    # 构建映射
    mapping = []
    ref_pos = 0
    
    while ref_pos < ref_len:
        # 当前参考位置
        mapping.append(ref_pos)
        
        # 检查该位置是否有插入
        if ref_pos in var_dict:
            ins_len = var_dict[ref_pos]
            # 添加插入的碱基（标记为-1）
            for _ in range(ins_len):
                mapping.append(-1)
        
        ref_pos += 1
    
    return mapping


def normalize_attention_scores(scores, mapping, ref_len):
    """
    根据映射关系标准化attention分数
    
    参数:
        scores: 原始分数列表 (长度可能不等于ref_len)
        mapping: 位置映射 (由build_position_mapping生成)
        ref_len: 参考序列长度
    
    返回:
        normalized_scores: 标准化后的分数列表 (长度=ref_len)
    """
    if len(scores) == ref_len and len(mapping) == ref_len:
        # 长度已经一致,直接返回
        return scores
    
    # 初始化输出
    normalized = [0.0] * ref_len
    counts = [0] * ref_len  # 记录每个位置累积了多少个分数
    
    # 遍历样本序列的每个位置
    for sample_pos, score in enumerate(scores):
        if sample_pos >= len(mapping):
            break
        
        ref_pos = mapping[sample_pos]
        
        if ref_pos == -1:
            # 这是插入的碱基,需要平均到前一个位置
            # 找到前一个非插入位置
            prev_pos = sample_pos - 1
            while prev_pos >= 0 and mapping[prev_pos] == -1:
                prev_pos -= 1
            
            if prev_pos >= 0:
                target_ref_pos = mapping[prev_pos]
                if 0 <= target_ref_pos < ref_len:
                    normalized[target_ref_pos] += score
                    counts[target_ref_pos] += 1
        else:
            # 正常位置
            if 0 <= ref_pos < ref_len:
                normalized[ref_pos] += float(score)
                counts[ref_pos] += 1
    
    # 计算平均值
    for i in range(ref_len):
        if counts[i] > 0:
            normalized[i] /= counts[i]
        # 如果counts[i]=0,保持0.0
    
    return normalized


def process_block(block_name, json_file, seq_json_file, bed_row, 
                  vcf_file, fasta, output_dir, all_samples):
    """
    处理单个block
    """
    chrom = str(bed_row['chrom'])
    start = int(bed_row['start'])
    end = int(bed_row['end'])
    
    print(f"\n📍 {block_name}: chr{chrom}:{start}-{end}")
    
    # 1. 读取参考序列
    try:
        ref_seq = fasta.fetch(chrom_map[chrom], start, end).upper()
        ref_len = len(ref_seq)
    except Exception as e:
        print(f"  ✗ 参考序列提取失败: {e}")
        return None
    
    print(f"  参考序列长度: {ref_len}")
    
    # 2. 读取变异信息（仅用于统计显示）
    variants = load_variants_in_region_from_vcf(vcf_file, chrom, start, end)
    print(f"  VCF中变异数: {len(variants)}")
    
    if len(variants) > 0:
        n_ins = (variants['variant_type'] == 'INS').sum()
        n_del = (variants['variant_type'] == 'DEL').sum()
        n_snp = (variants['variant_type'] == 'SNP').sum()
        print(f"    (统计: SNP: {n_snp}, INS: {n_ins}, DEL: {n_del})")
    
    # 3. 读取原始attention分数
    with open(json_file, 'r') as f:
        attention_data = json.load(f)
    
    # 4. 读取样本序列(用于验证)
    with open(seq_json_file, 'r') as f:
        seq_data = json.load(f)
    
    # 构建样本到序列的映射
    seq_dict = {item['spec']: item['sequence'] for item in seq_data}
    
    # 5. 逐样本标准化
    normalized_data = []
    
    for item in tqdm(attention_data, desc=f"  {block_name} 标准化", leave=False):
        sample_id = item['spec']
        original_scores = item['sequence']
        
        # 获取该样本的序列
        if sample_id not in seq_dict:
            print(f"  ⚠️  样本 {sample_id} 未找到序列,跳过")
            continue
        
        sample_seq = seq_dict[sample_id]
        
        # 检查长度是否匹配
        if len(original_scores) != len(sample_seq):
            print(f"  ⚠️  样本 {sample_id} 分数长度({len(original_scores)}) != 序列长度({len(sample_seq)})")
            # 尝试截断或填充
            if len(original_scores) > len(sample_seq):
                original_scores = original_scores[:len(sample_seq)]
            else:
                original_scores = original_scores + [0.0] * (len(sample_seq) - len(original_scores))
        
        # 🔥 关键修改：从VCF读取该样本实际携带的变异
        sample_variants = get_sample_variants_from_vcf(
            vcf_file, chrom, start, end, sample_id, all_samples
        )
        
        # 构建位置映射（只使用该样本实际的插入变异）
        mapping = build_position_mapping_with_sample_variants(
            ref_seq, sample_variants, start
        )
        # 🔐 安全校验：样本长度 ≠ mapping 长度 → 禁止使用插入映射
        if len(mapping) != len(sample_seq):
            print(
                f"⚠️ 回退为无插入映射: {sample_id} "
                f"(seq={len(sample_seq)}, map={len(mapping)})"
            )
            mapping = list(range(len(ref_seq)))



        # 标准化分数
        normalized_scores = normalize_attention_scores(original_scores, mapping, ref_len)
        
        normalized_data.append({
            'label': item['label'],
            'spec': sample_id,
            'loc': block_name,
            'sequence': normalized_scores
        })
    
    # 6. 保存结果
    output_file = os.path.join(output_dir, f"{block_name}_normalized.json")
    with open(output_file, 'w') as f:
        json.dump(normalized_data, f, indent=2)
    
    print(f"  ✅ 完成 | 样本数: {len(normalized_data)}, 标准化长度: {ref_len}")
    
    return {
        'block': block_name,
        'samples': len(normalized_data),
        'ref_len': ref_len,
        'output': output_file
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Attention分数长度标准化（支持多等位基因）")
    print("="*60)
    print(f"原始JSON目录: {args.json_dir}")
    print(f"序列JSON目录: {args.seq_json_dir}")
    print(f"VCF文件: {args.vcf_file}")
    print(f"输出目录: {args.output_dir}")
    
    # 读取BED
    bed_df = pd.read_csv(args.bed_file, sep="\t", header=None, 
                        names=["chrom", "start", "end"])
    print(f"\nBED区间数: {len(bed_df)}")
    
    # 打开FASTA
    fasta = pysam.FastaFile(args.fasta_file)
    
    # 检查VCF文件
    if not os.path.exists(args.vcf_file):
        print(f"错误: VCF文件不存在: {args.vcf_file}")
        return
    
    # 读取VCF样本列表
    print("\n读取VCF样本列表...")
    vcf = VCF(args.vcf_file)
    all_samples = vcf.samples
    vcf.close()
    print(f"VCF包含 {len(all_samples)} 个样本")
    
    # 处理每个block
    results = []
    
    for block_id, row in bed_df.iterrows():
        block_name = f"block_{block_id + 1}"
        
        json_file = os.path.join(args.json_dir, f"{block_name}_attn.json")
        seq_json_file = os.path.join(args.seq_json_dir, f"{block_name}.json")
        
        if not os.path.exists(json_file):
            print(f"\n⚠️  跳过 {block_name}: 未找到 {json_file}")
            continue
        
        if not os.path.exists(seq_json_file):
            print(f"\n⚠️  跳过 {block_name}: 未找到 {seq_json_file}")
            continue
        
        result = process_block(
            block_name, json_file, seq_json_file, row,
            args.vcf_file, fasta, args.output_dir, all_samples
        )
        
        if result:
            results.append(result)
    
    # 关闭文件
    fasta.close()
    
    # 生成总结
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    
    if len(results) > 0:
        summary = pd.DataFrame(results)
        summary_file = os.path.join(args.output_dir, "normalization_summary.csv")
        summary.to_csv(summary_file, index=False)
        
        print(f"\n总计处理: {len(results)} 个blocks")
        print(f"总结文件: {summary_file}")
    
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
