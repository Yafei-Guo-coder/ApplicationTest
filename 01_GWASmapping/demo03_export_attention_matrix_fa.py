#!/usr/bin/env python3
"""
Extract raw per-position attention scores for each haplotype without any
coordinate compression and export multiple CSVs:
  1. metadata.csv  -> sample + 原始标签
  2. hap1_attention.csv -> sample + 每个位点的注意力
  3. hap2_attention.csv -> sample + 每个位点的注意力
  4. hap1_attention_collapsed.csv -> 将插入重复列按 VCF 长度折叠求均值
  5. hap2_attention_collapsed.csv -> 同上

注意：所有位点使用统一的列顺序，插入导致的重复坐标会生成
`pos_<位置>_dupN` 列；缺失则表现为 NaN。这样可方便后续逐位点分析，
同时保留 hap1 / hap2 各自的序列信息。
"""

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from calc_flash_attention import calc_attentions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vcf_indel_info(vcf_path: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return (insertion_lengths, position_presence) from VCF.

    insertion_lengths[pos] -> 最大插入碱基数 (ALT 长度 - REF 长度) 的最大值；
    position_presence[pos] -> 该位置在 VCF 中出现（含 DEL 范围）
    """
    insertion_lengths: Dict[int, int] = defaultdict(int)
    position_presence: Dict[int, int] = defaultdict(int)

    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            pos = int(parts[1])
            ref = parts[3]
            alts = parts[4].split(',')

            position_presence[pos] = max(position_presence[pos], 1)

            for alt in alts:
                if len(alt) > len(ref):
                    ins_len = len(alt) - len(ref)
                    if ins_len > insertion_lengths[pos]:
                        insertion_lengths[pos] = ins_len
                elif len(alt) < len(ref):
                    del_len = len(ref) - len(alt)
                    for offset in range(del_len + 1):
                        position_presence[pos + offset] = max(position_presence[pos + offset], 1)

    return insertion_lengths, position_presence


def determine_position_columns(
    df: pd.DataFrame,
    insertion_lengths: Dict[int, int],
    position_presence: Dict[int, int],
) -> Dict[int, int]:
    """Return mapping of position -> max occurrences across haplotypes, incorporating VCF indels."""
    occurrences: Dict[int, int] = defaultdict(int)

    # counts from data
    for _, row in df.iterrows():
        for col in ['hap1_pos', 'hap2_pos']:
            pos_str = row.get(col)
            if not isinstance(pos_str, str) or not pos_str:
                continue
            pos_list = [int(p) for p in pos_str.split(';')]
            counts = Counter(pos_list)
            for pos, cnt in counts.items():
                if cnt > occurrences[pos]:
                    occurrences[pos] = cnt

    # ensure positions from VCF exist at least once
    for pos in position_presence.keys():
        occurrences[pos] = max(occurrences[pos], 1)

    # ensure insertion positions have enough duplicate columns (anchor + inserted bases)
    for pos, ins_len in insertion_lengths.items():
        required = 1 + ins_len  # anchor + inserted
        occurrences[pos] = max(occurrences[pos], required)

    return occurrences


def build_column_list(pos_occurrences: Dict[int, int]) -> List[str]:
    columns: List[str] = []
    for pos in sorted(pos_occurrences.keys()):
        max_occ = pos_occurrences[pos]
        columns.append(f"pos_{pos}")
        for dup_idx in range(1, max_occ):
            columns.append(f"pos_{pos}_dup{dup_idx}")
    return columns


def compute_haplotype_tables(
    df: pd.DataFrame,
    model,
    tokenizer,
    pos_columns: List[str],
    block_cols: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    print(f"K分块大小: {block_cols}")

    hap1_rows: List[Dict[str, Optional[float]]] = []
    hap2_rows: List[Dict[str, Optional[float]]] = []
    metadata_rows: List[Dict[str, int]] = []
    all_position_columns = set(pos_columns)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring attention"):
        sample = row['sample']
        original_label = int(row['sample_type'])
        metadata_rows.append({'sample': sample, 'sample_type': original_label})

        for hap_name, seq_col, pos_col, container in [
            ('hap1', 'hap1_seq', 'hap1_pos', hap1_rows),
            ('hap2', 'hap2_seq', 'hap2_pos', hap2_rows),
        ]:
            seq = row.get(seq_col)
            pos_str = row.get(pos_col)
            if not isinstance(seq, str) or len(seq) == 0:
                container.append({'sample': sample})
                continue
            if not isinstance(pos_str, str) or len(pos_str) == 0:
                container.append({'sample': sample})
                continue

            pos_list = [int(p) for p in pos_str.split(';')]
            attn = calc_attentions(seq, model, tokenizer, block_cols=block_cols)
            length = min(len(pos_list), len(attn))

            row_dict: Dict[str, Optional[float]] = {'sample': sample}

            counters: Dict[int, int] = defaultdict(int)
            for pos, att in zip(pos_list[:length], attn[:length]):
                idx = counters[pos]
                counters[pos] += 1
                if idx == 0:
                    col_name = f"pos_{pos}"
                else:
                    col_name = f"pos_{pos}_dup{idx}"
                row_dict[col_name] = float(att)
                all_position_columns.add(col_name)

            container.append(row_dict)

    position_cols_sorted = ['sample'] + sorted(all_position_columns)

    def build_df(rows: List[Dict[str, Optional[float]]]) -> pd.DataFrame:
        df_rows = pd.DataFrame(rows)
        return df_rows.reindex(columns=position_cols_sorted)

    hap1_df = build_df(hap1_rows)
    hap2_df = build_df(hap2_rows)
    metadata_df = pd.DataFrame(metadata_rows).drop_duplicates(subset='sample').reset_index(drop=True)

    return metadata_df, hap1_df, hap2_df


def collapse_haplotype_df(hap_df: pd.DataFrame, pos_occurrences: Dict[int, int]) -> pd.DataFrame:
    base_positions = sorted(pos_occurrences.keys())
    collapsed = {'sample': hap_df['sample']}

    for pos in base_positions:
        base_col = f"pos_{pos}"
        cols = [c for c in hap_df.columns if c == base_col or c.startswith(f"{base_col}_dup")]
        if cols:
            collapsed[base_col] = hap_df[cols].mean(axis=1, skipna=True)
        else:
            collapsed[base_col] = np.nan

    return pd.DataFrame(collapsed)


def main():
    parser = argparse.ArgumentParser(description="Export raw per-position attention matrix for HBB haplotypes")
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--vcf_file', required=True, help='用于判定INDEL的VCF文件')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--block_cols', type=int, required=True, help='block size of K')

    args = parser.parse_args()

    print("HBB per-position attention export")
    print("=" * 60)
    print(f"输入: {args.input_csv}")
    print(f"模型: {args.model_path}")
    print(f"VCF: {args.vcf_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {device}")

    df = pd.read_csv(args.input_csv)
    insertion_lengths, position_presence = load_vcf_indel_info(args.vcf_file)
    pos_occurrences = determine_position_columns(df, insertion_lengths, position_presence)
    pos_columns = build_column_list(pos_occurrences)
    print(f"检测到 {len(pos_occurrences)} 个独立参考坐标，生成基础位置列 {len(pos_columns)} 个")

    print("加载模型与分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = int(1e9)
    model = AutoModel.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device)

    metadata_df, hap1_df, hap2_df = compute_haplotype_tables(df, model, tokenizer, pos_columns, args.block_cols)
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'metadata.csv')
    hap1_path = os.path.join(args.output_dir, 'hap1_attention.csv')
    hap2_path = os.path.join(args.output_dir, 'hap2_attention.csv')
    metadata_df.to_csv(meta_path, index=False)
    hap1_df.to_csv(hap1_path, index=False)
    hap2_df.to_csv(hap2_path, index=False)
    hap1_collapsed = collapse_haplotype_df(hap1_df, pos_occurrences)
    hap2_collapsed = collapse_haplotype_df(hap2_df, pos_occurrences)
    hap1_collapsed_path = os.path.join(args.output_dir, 'hap1_attention_collapsed.csv')
    hap2_collapsed_path = os.path.join(args.output_dir, 'hap2_attention_collapsed.csv')
    hap1_collapsed.to_csv(hap1_collapsed_path, index=False)
    hap2_collapsed.to_csv(hap2_collapsed_path, index=False)
    print("已保存:\n"
            f"  {meta_path}\n"
            f"  {hap1_path}\n"
            f"  {hap2_path}\n"
            f"  {hap1_collapsed_path}\n"
            f"  {hap2_collapsed_path}")
    

if __name__ == '__main__':
    main()
