#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准化attention分数长度
将因indel导致的不同长度序列的分数标准化为参考序列长度
- 插入(INS): 多个碱基的分数平均为1个
- 缺失(DEL): 保持N位置的分数
"""

import os
import json
import numpy as np
import pandas as pd
import pysam
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
    parser.add_argument('--pvar_prefix', type=str, required=True,
                       help='PLINK2 pvar文件前缀')
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


def load_variants_in_region(pvar_file, chrom, start, end):
    """
    读取PVAR文件,获取指定区间的变异信息
    
    返回: DataFrame with columns [POS, REF, ALT, variant_type]
    """
    # 计算需要跳过的元数据行
    skip_rows = 0
    with open(pvar_file, 'r') as f:
        for line in f:
            if line.startswith('##'):
                skip_rows += 1
            else:
                break
    
    # 读取PVAR
    pvar = pd.read_csv(pvar_file, sep="\t", skiprows=skip_rows)
    pvar.columns = pvar.columns.str.replace('#', '')
    
    # 筛选该区间的变异
    mask = (
        (pvar['CHROM'].astype(str) == str(chrom)) &
        (pvar['POS'] >= start + 1) &
        (pvar['POS'] <= end)
    )
    
    variants = pvar[mask].copy()
    
    # 判断变异类型
    def classify_variant(ref, alt):
        # 只看第一个ALT
        alt = str(alt).split(',')[0]
        # 过滤特殊标记
        if alt in ['DEL', '<DEL>', '*']:
            return 'DEL'
        
        ref = str(ref).upper()
        alt = alt.upper()
        
        if len(ref) == 1 and len(alt) == 1:
            return 'SNP'
        elif len(ref) < len(alt):
            return 'INS'
        elif len(ref) > len(alt):
            return 'DEL'
        else:
            return 'COMPLEX'
    
    variants['variant_type'] = variants.apply(
        lambda row: classify_variant(row['REF'], row['ALT']), 
        axis=1
    )
    
    # 计算插入长度
    def get_ins_length(ref, alt):
        alt = str(alt).split(',')[0]
        if alt in ['DEL', '<DEL>', '*']:
            return 0
        return max(0, len(alt) - len(ref))
    
    variants['ins_length'] = variants.apply(
        lambda row: get_ins_length(row['REF'], row['ALT']),
        axis=1
    )
    
    return variants[['POS', 'REF', 'ALT', 'variant_type', 'ins_length']].sort_values('POS')


def build_position_mapping(ref_seq, sample_seq, variants, ref_start):
    """
    构建样本序列位置到参考序列位置的映射
    
    参数:
        ref_seq: 参考序列
        sample_seq: 样本序列
        variants: 该区间的变异信息 DataFrame
        ref_start: 参考序列起始位置(0-based in genome)
    
    返回:
        mapping: list, mapping[sample_pos] = ref_pos
                如果mapping[i] = j, 表示样本序列第i个碱基对应参考序列第j个碱基
                如果mapping[i] = -1, 表示这是插入的碱基,需要与前后平均
    """
    ref_len = len(ref_seq)
    sample_len = len(sample_seq)
    
    # 初始化映射: 默认一一对应
    mapping = list(range(ref_len))
    
    # 如果长度相同,直接返回
    if sample_len == ref_len:
        return mapping
    
    # 构建每个参考位置的变异信息
    var_dict = {}  # {ref_offset: variant_info}
    
    for _, var in variants.iterrows():
        ref_offset = var['POS'] - ref_start - 1  # 0-based offset
        if 0 <= ref_offset < ref_len:
            var_dict[ref_offset] = {
                'type': var['variant_type'],
                'ins_len': var['ins_length']
            }
    
    # 重建映射
    new_mapping = []
    sample_pos = 0
    ref_pos = 0
    
    while ref_pos < ref_len and sample_pos < sample_len:
        # 检查该位置是否有变异
        if ref_pos in var_dict:
            var_info = var_dict[ref_pos]
            
            if var_info['type'] == 'INS' and var_info['ins_len'] > 0:
                # 插入: 
                # 第1个碱基对应ref_pos
                new_mapping.append(ref_pos)
                sample_pos += 1
                
                # 后续插入的碱基标记为-1(需要平均)
                for _ in range(var_info['ins_len']):
                    if sample_pos < sample_len:
                        new_mapping.append(-1)  # 插入标记
                        sample_pos += 1
                
                ref_pos += 1
            
            elif var_info['type'] == 'DEL':
                # 缺失: 样本序列该位置是N,保持对应
                new_mapping.append(ref_pos)
                sample_pos += 1
                ref_pos += 1
            
            else:
                # SNP或其他
                new_mapping.append(ref_pos)
                sample_pos += 1
                ref_pos += 1
        else:
            # 无变异,正常对应
            new_mapping.append(ref_pos)
            sample_pos += 1
            ref_pos += 1
    
    # 处理剩余部分
    while sample_pos < sample_len:
        new_mapping.append(ref_len - 1)  # 超出部分映射到最后
        sample_pos += 1
    
    return new_mapping


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
                  pvar_file, fasta, output_dir):
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
    
    # 2. 读取变异信息
    variants = load_variants_in_region(pvar_file, chrom, start, end)
    print(f"  变异数: {len(variants)}")
    
    n_ins = (variants['variant_type'] == 'INS').sum()
    n_del = (variants['variant_type'] == 'DEL').sum()
    n_snp = (variants['variant_type'] == 'SNP').sum()
    print(f"    SNP: {n_snp}, INS: {n_ins}, DEL: {n_del}")
    
    # 3. 读取原始attention分数
    with open(json_file, 'r') as f:
        attention_data = json.load(f)
    
    # 4. 读取样本序列(用于构建映射)
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
        
        # 构建位置映射
        mapping = build_position_mapping(ref_seq, sample_seq, variants, start)
        
        # 标准化分数
        normalized_scores = normalize_attention_scores(original_scores, mapping, ref_len)
        
        normalized_data.append({
            'label': item['label'],
            'spec': sample_id,
            'loc': block_name,
            'sequence': normalized_scores
        })
    
    # 6. 保存结果
    output_file = os.path.join(output_dir, f"{block_name}_processed.json")
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
    print("Attention分数长度标准化")
    print("="*60)
    print(f"原始JSON目录: {args.json_dir}")
    print(f"序列JSON目录: {args.seq_json_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 读取BED
    bed_df = pd.read_csv(args.bed_file, sep="\t", header=None, 
                        names=["chrom", "start", "end"])
    print(f"\nBED区间数: {len(bed_df)}")
    
    # 打开FASTA
    fasta = pysam.FastaFile(args.fasta_file)
    
    # PVAR文件
    pvar_file = f"{args.pvar_prefix}.pvar"
    if not os.path.exists(pvar_file):
        print(f"错误: PVAR文件不存在: {pvar_file}")
        return
    
    # 处理每个block
    results = []
    
    for block_id, row in bed_df.iterrows():
        block_name = f"block_{block_id + 1}"
        
        json_file = os.path.join(args.json_dir, f"{block_name}_processed.json")
        seq_json_file = os.path.join(args.seq_json_dir, f"{block_name}.json")
        
        if not os.path.exists(json_file):
            print(f"\n⚠️  跳过 {block_name}: 未找到 {json_file}")
            continue
        
        if not os.path.exists(seq_json_file):
            print(f"\n⚠️  跳过 {block_name}: 未找到 {seq_json_file}")
            continue
        
        result = process_block(
            block_name, json_file, seq_json_file, row,
            pvar_file, fasta, args.output_dir
        )
        
        if result:
            results.append(result)
    
    # 关闭文件
    fasta.close()
    
    # 生成总结
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    
    summary = pd.DataFrame(results)
    summary_file = os.path.join(args.output_dir, "normalization_summary.csv")
    summary.to_csv(summary_file, index=False)
    
    print(f"\n总计处理: {len(results)} 个blocks")
    print(f"总结文件: {summary_file}")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
