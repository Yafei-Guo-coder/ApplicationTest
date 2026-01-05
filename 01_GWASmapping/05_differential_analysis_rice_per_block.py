#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理多个block的Attention差异分析
自动扫描所有block文件夹并分别进行分析
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(attn_dir, haplotype='hap1'):
    """加载attention数据和metadata"""
    # 读取attention矩阵
    attn_file = os.path.join(attn_dir, f'{haplotype}_attention_collapsed.csv')
    if not os.path.exists(attn_file):
        raise FileNotFoundError(f"未找到文件: {attn_file}")
    
    attn = pd.read_csv(attn_file, index_col=0)
    
    # 读取metadata
    meta_file = os.path.join(attn_dir, 'metadata.csv')
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"未找到文件: {meta_file}")
    
    metadata = pd.read_csv(meta_file, index_col=0)
    
    # 确保样本顺序一致
    common_samples = attn.index.intersection(metadata.index)
    attn = attn.loc[common_samples]
    metadata = metadata.loc[common_samples]
    
    return attn, metadata


def differential_analysis(attn, metadata, group_col='sample_type', 
                         group_a=1, group_b=2, min_samples=10):
    """执行差异分析"""
    # 提取两组数据
    mask_a = metadata[group_col] == group_a
    mask_b = metadata[group_col] == group_b
    
    group_a_data = attn.loc[mask_a]
    group_b_data = attn.loc[mask_b]
    
    n_a = len(group_a_data)
    n_b = len(group_b_data)
    
    if n_a < min_samples or n_b < min_samples:
        raise ValueError(f"样本数不足! Group {group_a}: {n_a}, Group {group_b}: {n_b}")
    
    # 存储结果
    results = []
    positions = attn.columns.tolist()
    
    for pos in positions:
        # 提取该位点的数据(去除NaN)
        vals_a = group_a_data[pos].dropna().values
        vals_b = group_b_data[pos].dropna().values
        
        # 如果某组有效数据不足,跳过
        if len(vals_a) < min_samples or len(vals_b) < min_samples:
            results.append({
                'position': pos,
                'n_a': len(vals_a),
                'n_b': len(vals_b),
                'mean_a': np.nan,
                'mean_b': np.nan,
                'median_a': np.nan,
                'median_b': np.nan,
                'std_a': np.nan,
                'std_b': np.nan,
                'delta': np.nan,
                'log2fc': np.nan,
                'pvalue': np.nan,
                'statistic': np.nan
            })
            continue
        
        # 计算统计量
        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        median_a = np.median(vals_a)
        median_b = np.median(vals_b)
        std_a = np.std(vals_a, ddof=1)
        std_b = np.std(vals_b, ddof=1)
        
        delta = mean_b - mean_a
        
        # log2FC (防止除零)
        if mean_a > 0 and mean_b > 0:
            log2fc = np.log2(mean_b / mean_a)
        elif mean_a == 0 and mean_b > 0:
            log2fc = np.inf
        elif mean_a > 0 and mean_b == 0:
            log2fc = -np.inf
        else:
            log2fc = 0
        
        # Mann-Whitney U 检验
        try:
            statistic, pvalue = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        except:
            statistic, pvalue = np.nan, np.nan
        
        results.append({
            'position': pos,
            'n_a': len(vals_a),
            'n_b': len(vals_b),
            'mean_a': mean_a,
            'mean_b': mean_b,
            'median_a': median_a,
            'median_b': median_b,
            'std_a': std_a,
            'std_b': std_b,
            'delta': delta,
            'log2fc': log2fc,
            'pvalue': pvalue,
            'statistic': statistic
        })
    
    # 转为DataFrame
    results_df = pd.DataFrame(results)
    
    # FDR校正
    valid_pvals = results_df['pvalue'].notna()
    if valid_pvals.sum() > 0:
        results_df.loc[valid_pvals, 'padj'] = multipletests(
            results_df.loc[valid_pvals, 'pvalue'], 
            method='fdr_bh'
        )[1]
    else:
        results_df['padj'] = np.nan
    
    # 添加显著性标记
    results_df['significant'] = (results_df['padj'] < 0.05) & (results_df['log2fc'].abs() > 1)
    
    return results_df


def plot_volcano(results, output_file, title='Volcano Plot', 
                 padj_thresh=0.05, fc_thresh=1):
    """绘制火山图"""
    df = results.copy()
    df = df[df['pvalue'].notna()].copy()
    
    if len(df) == 0:
        return
    
    df['color'] = 'gray'
    df.loc[(df['padj'] < padj_thresh) & (df['log2fc'] > fc_thresh), 'color'] = 'red'
    df.loc[(df['padj'] < padj_thresh) & (df['log2fc'] < -fc_thresh), 'color'] = 'blue'
    
    df['log2fc_plot'] = df['log2fc'].replace([np.inf, -np.inf], [10, -10])
    df['-log10p'] = -np.log10(df['padj'] + 1e-300)
    
    n_up = (df['color'] == 'red').sum()
    n_down = (df['color'] == 'blue').sum()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for color, label in [('gray', 'NS'), ('red', f'Up ({n_up})'), ('blue', f'Down ({n_down})')]:
        subset = df[df['color'] == color]
        ax.scatter(subset['log2fc_plot'], subset['-log10p'], 
                  c=color, alpha=0.6, s=20, label=label, edgecolors='none')
    
    ax.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('log2(Fold Change)', fontsize=12)
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_manhattan(results, output_file, title='Manhattan Plot', 
                   padj_thresh=0.05, fc_thresh=1):
    """绘制曼哈顿图"""
    df = results.copy()
    df = df[df['pvalue'].notna()].copy()
    
    if len(df) == 0:
        return
    
    df['pos_int'] = pd.to_numeric(df['position'].astype(str).str.replace('pos_', ''), errors='coerce')
    if df['pos_int'].isna().all():
        df['pos_int'] = range(len(df))
    
    df = df.sort_values('pos_int')
    df['-log10p'] = -np.log10(df['padj'] + 1e-300)
    
    df['color'] = 'gray'
    df.loc[(df['padj'] < padj_thresh) & (df['log2fc'].abs() > fc_thresh), 'color'] = 'red'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    for color in ['gray', 'red']:
        subset = df[df['color'] == color]
        ax1.scatter(subset['pos_int'], subset['-log10p'], 
                   c=color, s=15, alpha=0.7, edgecolors='none')
    ax1.axhline(-np.log10(padj_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('-log10(Adjusted P)', fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.scatter(df['pos_int'], df['log2fc'], c=df['color'], s=15, alpha=0.7, edgecolors='none')
    ax2.axhline(fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(-fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(0, color='black', linewidth=1.5, alpha=0.3)
    ax2.set_xlabel('Genomic Position', fontsize=11)
    ax2.set_ylabel('log2(Fold Change)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_positions(results, attn, metadata, output_file, 
                       group_col='sample_type', group_a=1, group_b=2, top_n=20):
    """绘制Top显著位点的箱线图"""
    sig = results[results['significant'] == True].copy()
    if len(sig) == 0:
        print(f"⚠ 没有显著位点，跳过绘图: {output_file}")
        return
    
    sig = sig.sort_values('padj').head(top_n)
    
    mask_a = metadata[group_col] == group_a
    mask_b = metadata[group_col] == group_b
    
    n_cols = 5
    n_rows = int(np.ceil(len(sig) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    
    # 统一 axes 为一维可迭代
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    
    for idx, (_, row) in enumerate(sig.iterrows()):
        pos = row['position']
        ax = axes[idx]
        
        vals_a = attn.loc[mask_a, pos].dropna()
        vals_b = attn.loc[mask_b, pos].dropna()
        
        data_plot = pd.DataFrame({
            'Attention': list(vals_a) + list(vals_b),
            'Group': [f'Group {group_a}'] * len(vals_a) + [f'Group {group_b}'] * len(vals_b)
        })
        
        sns.boxplot(data=data_plot, x='Group', y='Attention', ax=ax, palette=['lightblue', 'lightcoral'])
        sns.stripplot(data=data_plot, x='Group', y='Attention', ax=ax, 
                      color='black', alpha=0.3, size=2)
        
        ax.set_title(f"{pos}\npadj={row['padj']:.2e}, log2FC={row['log2fc']:.2f}", fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('Attention Score', fontsize=9)
        ax.tick_params(labelsize=8)
    
    # 关闭多余 subplot
    for idx in range(len(sig), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def process_single_block(block_dir, block_name, args):
    """处理单个block"""
    print(f"\n{'='*60}")
    print(f"处理: {block_name}")
    print(f"{'='*60}")
    
    # 创建该block的输出目录
    block_output = os.path.join(args.output_dir, block_name)
    os.makedirs(block_output, exist_ok=True)
    os.makedirs(os.path.join(block_output, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(block_output, 'tables'), exist_ok=True)
    
    result_summary = {
        'block': block_name,
        'status': 'failed',
        'error': None,
        'n_samples': 0,
        'n_positions': 0,
        'n_significant': 0
    }
    
    try:
        # 加载数据
        attn, metadata = load_data(block_dir, args.haplotype)
        
        print(f"  ✓ 矩阵: {attn.shape} (样本×位点)")
        print(f"  ✓ 样本类型分布: {metadata['sample_type'].value_counts().to_dict()}")
        
        result_summary['n_samples'] = len(attn)
        result_summary['n_positions'] = len(attn.columns)
        
        # 差异分析
        results = differential_analysis(
            attn, metadata,
            group_a=args.group_a,
            group_b=args.group_b,
            min_samples=args.min_samples
        )
        
        n_sig = results['significant'].sum()
        result_summary['n_significant'] = int(n_sig)
        
        print(f"  ✓ 显著位点: {n_sig} / {len(results)}")
        
        # 保存结果表格
        prefix = f"{args.haplotype}_group{args.group_a}vs{args.group_b}"
        
        results.to_csv(
            os.path.join(block_output, 'tables', f'{prefix}_all_results.csv'),
            index=False
        )
        
        sig_results = results[results['significant'] == True].sort_values('padj')
        if len(sig_results) > 0:
            sig_results.to_csv(
                os.path.join(block_output, 'tables', f'{prefix}_significant.csv'),
                index=False
            )
        
        # 绘图
        plot_volcano(
            results,
            os.path.join(block_output, 'figures', f'{prefix}_volcano.png'),
            title=f'{block_name} - Volcano Plot',
            padj_thresh=args.padj_thresh,
            fc_thresh=args.fc_thresh
        )
        
        plot_manhattan(
            results,
            os.path.join(block_output, 'figures', f'{prefix}_manhattan.png'),
            title=f'{block_name} - Manhattan Plot',
            padj_thresh=args.padj_thresh,
            fc_thresh=args.fc_thresh
        )
        
        if len(sig_results) > 0:
            plot_top_positions(
                results, attn, metadata,
                os.path.join(block_output, 'figures', f'{prefix}_top20_boxplots.png'),
                group_a=args.group_a,
                group_b=args.group_b,
                top_n=20
            )
        
        result_summary['status'] = 'success'
        print(f"  ✓ 完成!")
        
    except FileNotFoundError as e:
        result_summary['error'] = f"文件未找到: {e}"
        print(f"  ✗ 跳过: {result_summary['error']}")
    except ValueError as e:
        result_summary['error'] = f"样本数不足: {e}"
        print(f"  ✗ 跳过: {result_summary['error']}")
    except Exception as e:
        result_summary['error'] = f"{type(e).__name__}: {e}"
        print(f"  ✗ 错误: {result_summary['error']}")
    
    return result_summary


def find_block_directories(input_dir):
    """查找所有block文件夹"""
    block_dirs = []
    
    input_path = Path(input_dir)
    
    # 查找所有包含 hap1_attention_collapsed.csv 的文件夹
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            hap1_file = subdir / "hap1_attention_collapsed.csv"
            metadata_file = subdir / "metadata.csv"
            
            if hap1_file.exists() and metadata_file.exists():
                block_dirs.append((subdir, subdir.name))
    
    return sorted(block_dirs)


def main():
    parser = argparse.ArgumentParser(description='批量处理多个block的差异分析')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='包含所有block子文件夹的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--haplotype', type=str, default='hap1',
                       choices=['hap1', 'hap2'],
                       help='分析的单倍型')
    parser.add_argument('--group_a', type=int, default=1,
                       help='对照组编号')
    parser.add_argument('--group_b', type=int, default=2,
                       help='处理组编号')
    parser.add_argument('--min_samples', type=int, default=10,
                       help='最小样本数')
    parser.add_argument('--padj_thresh', type=float, default=0.05,
                       help='FDR阈值')
    parser.add_argument('--fc_thresh', type=float, default=1,
                       help='log2FC阈值')
    parser.add_argument('--block_pattern', type=str, default=None,
                       help='只处理匹配的block,如 "block_1,block_5,block_10"')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# 批量Block差异分析")
    print(f"{'#'*60}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"单倍型: {args.haplotype}")
    print(f"比较组: Group {args.group_a} vs Group {args.group_b}")
    
    # 查找所有block目录
    print(f"\n扫描block文件夹...")
    block_dirs = find_block_directories(args.input_dir)
    
    if not block_dirs:
        print(f"错误: 在 {args.input_dir} 中未找到任何block文件夹")
        return
    
    print(f"找到 {len(block_dirs)} 个block文件夹")
    
    # 如果指定了特定block,过滤
    if args.block_pattern:
        selected_blocks = set(args.block_pattern.split(','))
        block_dirs = [(d, n) for d, n in block_dirs if n in selected_blocks]
        print(f"根据模式过滤后: {len(block_dirs)} 个block")
    
    # 逐个处理
    all_results = []
    
    for block_dir, block_name in block_dirs:
        result = process_single_block(block_dir, block_name, args)
        all_results.append(result)
    
    # 生成总结报告
    print(f"\n{'#'*60}")
    print(f"# 处理完成!")
    print(f"{'#'*60}")
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    fail_count = len(all_results) - success_count
    
    print(f"\n总计: {len(all_results)} 个blocks")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    # 保存总结
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(args.output_dir, 'batch_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n总结报告: {summary_file}")
    
    # 显示成功处理的blocks
    if success_count > 0:
        print(f"\n成功处理的blocks:")
        for r in all_results:
            if r['status'] == 'success':
                print(f"  ✓ {r['block']}: {r['n_significant']} 显著位点 "
                      f"({r['n_samples']} 样本, {r['n_positions']} 位点)")
    
    # 显示失败的blocks
    if fail_count > 0:
        print(f"\n失败的blocks:")
        for r in all_results:
            if r['status'] == 'failed':
                print(f"  ✗ {r['block']}: {r['error']}")
    
    print(f"\n结果保存在: {args.output_dir}")
    print(f"  每个block的结果在各自的子文件夹中")


if __name__ == '__main__':
    main()
