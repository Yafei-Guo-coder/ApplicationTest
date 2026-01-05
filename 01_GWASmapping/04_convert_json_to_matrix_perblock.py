#!/usr/bin/env python3
"""
将attention JSON文件转换为差异分析所需的矩阵格式
每个block单独保存到一个文件夹
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True,
                        help="包含注意力JSON文件的目录")
    parser.add_argument("--bed_file", type=str, required=True,
                        help="BED文件,包含block位置信息")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--phenotype_file", type=str, default=None,
                        help="表型文件(可选),用于过滤和标注样本")
    parser.add_argument("--file_pattern", type=str, default="block_*_processed.json",
                        help="JSON文件匹配模式")
    parser.add_argument("--merge_all", action="store_true",
                        help="是否也生成合并所有block的矩阵")
    return parser.parse_args()

def load_bed_positions(bed_file):
    """
    从BED文件读取每个block的染色体位置信息
    返回: dict {block_name: {'chrom': chr, 'start': pos, 'end': pos}}
    """
    bed_df = pd.read_csv(bed_file, sep="\t", header=None,
                         names=["chrom", "start", "end"])
    
    block_positions = {}
    for idx, row in bed_df.iterrows():
        block_name = f"block_{idx+1}"
        block_positions[block_name] = {
            'chrom': str(row['chrom']),
            'start': int(row['start']),
            'end': int(row['end'])
        }
    return block_positions

def process_single_json(json_file, block_name, pos_info):
    """
    处理单个JSON文件,提取该block的注意力分数
    """
    start = pos_info['start']
    end = pos_info['end']
    
    # 加载JSON
    with open(json_file, 'r') as f:
        samples = json.load(f)
    
    # 存储该block的数据
    block_data = []
    
    # 处理每个样本
    for sample in samples:
        sample_id = sample['spec']
        label = sample['label']
        attention_scores = sample['sequence']
        
        # 检查长度
        expected_len = end - start
        if len(attention_scores) != expected_len:
            print(f"  警告: {block_name} 样本 {sample_id} 长度不匹配: "
                  f"{len(attention_scores)} != {expected_len}")
            # 截断或填充
            if len(attention_scores) > expected_len:
                attention_scores = attention_scores[:expected_len]
            else:
                attention_scores = attention_scores + [np.nan] * (expected_len - len(attention_scores))
        
        # 为每个碱基位置创建记录
        for i, score in enumerate(attention_scores):
            position = start + i + 1  # 1-based坐标
            block_data.append({
                'sample_id': sample_id,
                'label': label,
                'chrom': pos_info['chrom'],
                'position': position,
                'attention': score
            })
    
    return pd.DataFrame(block_data)

def create_attention_matrix(df, phenotype_df=None):
    """
    从DataFrame创建注意力矩阵
    """
    # 如果提供了表型文件,过滤样本
    if phenotype_df is not None:
        valid_samples = set(phenotype_df['SampleID'].tolist())
        df = df[df['sample_id'].isin(valid_samples)]
    
    # 数据透视:样本 x 位置
    pivot_df = df.pivot_table(
        index='sample_id',
        columns='position',
        values='attention',
        aggfunc='first'
    )
    
    # 列名转为 "pos_XXXX" 格式
    pivot_df.columns = [f"pos_{col}" for col in pivot_df.columns]
    
    return pivot_df

def create_metadata(df, phenotype_df=None):
    """
    创建metadata文件
    """
    # 获取每个样本的label
    sample_labels = df.groupby('sample_id')['label'].first()
    
    metadata = pd.DataFrame({
        'sample_id': sample_labels.index,
        'sample_type': sample_labels.values
    })
    metadata = metadata.set_index('sample_id')
    
    # 如果有表型文件,合并额外信息
    if phenotype_df is not None:
        phenotype_df = phenotype_df.set_index('SampleID')
        metadata = metadata.join(phenotype_df, how='left')
    
    return metadata

def save_block_results(block_name, block_df, phenotype_df, output_dir):
    """
    保存单个block的结果到独立文件夹
    """
    # 创建block专属文件夹
    block_dir = os.path.join(output_dir, block_name)
    os.makedirs(block_dir, exist_ok=True)
    
    # 创建注意力矩阵
    attention_matrix = create_attention_matrix(block_df, phenotype_df)
    
    # 创建metadata
    metadata = create_metadata(block_df, phenotype_df)
    
    # 保存文件
    hap1_file = os.path.join(block_dir, "hap1_attention_collapsed.csv")
    attention_matrix.to_csv(hap1_file)
    
    # 创建hap2 (空矩阵或复制hap1)
    hap2_file = os.path.join(block_dir, "hap2_attention_collapsed.csv")
    hap2_matrix = pd.DataFrame(np.nan, index=attention_matrix.index, 
                               columns=attention_matrix.columns)
    hap2_matrix.to_csv(hap2_file)
    
    # 保存metadata
    metadata_file = os.path.join(block_dir, "metadata.csv")
    metadata.to_csv(metadata_file)
    
    # 生成block摘要
    summary = {
        'block_name': block_name,
        'samples': int(len(attention_matrix)),
        'positions': int(len(attention_matrix.columns)),
        'position_range': {
            'min': int(block_df['position'].min()),
            'max': int(block_df['position'].max())
        },
        'chrom': block_df['chrom'].iloc[0],
        'label_distribution': block_df.groupby('label')['sample_id'].nunique().to_dict(),
        'missing_rate': float(attention_matrix.isna().mean().mean())
    }
    
    summary_file = os.path.join(block_dir, "block_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        'block': block_name,
        'output_dir': block_dir,
        'samples': len(attention_matrix),
        'positions': len(attention_matrix.columns),
        'summary': summary
    }

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("注意力矩阵转换流程 (每个block单独保存)")
    print("=" * 60)
    
    # 1. 加载BED文件
    print("\n1. 加载BED文件...")
    block_positions = load_bed_positions(args.bed_file)
    print(f"   加载了 {len(block_positions)} 个block的位置信息")
    
    # 2. 加载表型文件(如果提供)
    phenotype_df = None
    if args.phenotype_file:
        print("\n2. 加载表型文件...")
        phenotype_df = pd.read_csv(args.phenotype_file, sep="\t")
        print(f"   表型文件包含 {len(phenotype_df)} 个样本")
    
    # 3. 查找所有JSON文件
    print("\n3. 查找JSON文件...")
    json_path = Path(args.json_dir)
    json_files = sorted(json_path.glob(args.file_pattern))
    
    if not json_files:
        raise ValueError(f"未找到匹配的JSON文件: {args.file_pattern}")
    
    print(f"   找到 {len(json_files)} 个JSON文件")
    
    # 4. 逐个处理JSON文件并保存
    print("\n4. 处理JSON文件并保存到各自文件夹...")
    
    all_blocks_data = []  # 用于可选的合并输出
    block_results = []
    
    for json_file in tqdm(json_files, desc="处理blocks"):
        # 从文件名提取block名称
        block_name = json_file.stem.replace("_processed", "")
        
        if block_name not in block_positions:
            print(f"  警告: {block_name} 不在BED文件中,跳过")
            continue
        
        # 获取位置信息
        pos_info = block_positions[block_name]
        
        # 处理该JSON文件
        block_df = process_single_json(json_file, block_name, pos_info)
        
        # 保存该block的结果
        result = save_block_results(block_name, block_df, phenotype_df, args.output_dir)
        block_results.append(result)
        
        # 如果需要合并所有block
        if args.merge_all:
            all_blocks_data.append(block_df)
    
    # 5. 生成总体摘要
    print("\n5. 生成总体摘要...")
    
    total_summary = {
        'total_blocks': len(block_results),
        'blocks': block_results,
        'output_structure': f"{args.output_dir}/block_X/"
    }
    
    summary_file = os.path.join(args.output_dir, "conversion_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(total_summary, f, indent=2)
    print(f"   ✓ 总体摘要: {summary_file}")
    
    # 6. 可选:合并所有block
    if args.merge_all and all_blocks_data:
        print("\n6. 生成合并所有block的矩阵...")
        merged_dir = os.path.join(args.output_dir, "merged_all_blocks")
        os.makedirs(merged_dir, exist_ok=True)
        
        all_df = pd.concat(all_blocks_data, ignore_index=True)
        
        # 创建合并矩阵
        merged_matrix = create_attention_matrix(all_df, phenotype_df)
        merged_metadata = create_metadata(all_df, phenotype_df)
        
        # 保存
        merged_matrix.to_csv(os.path.join(merged_dir, "hap1_attention_collapsed.csv"))
        
        hap2_merged = pd.DataFrame(np.nan, index=merged_matrix.index, 
                                   columns=merged_matrix.columns)
        hap2_merged.to_csv(os.path.join(merged_dir, "hap2_attention_collapsed.csv"))
        
        merged_metadata.to_csv(os.path.join(merged_dir, "metadata.csv"))
        
        print(f"   ✓ 合并矩阵大小: {merged_matrix.shape}")
        print(f"   ✓ 保存到: {merged_dir}")
    
    # 7. 打印完成信息
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"每个block的结果保存在各自的文件夹中:")
    
    for result in block_results[:5]:  # 显示前5个
        print(f"  - {result['block']}/")
        print(f"      hap1_attention_collapsed.csv ({result['samples']} x {result['positions']})")
        print(f"      hap2_attention_collapsed.csv")
        print(f"      metadata.csv")
        print(f"      block_summary.json")
    
    if len(block_results) > 5:
        print(f"  ... 还有 {len(block_results) - 5} 个blocks")
    
    if args.merge_all:
        print(f"\n合并所有blocks的结果:")
        print(f"  - merged_all_blocks/")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
