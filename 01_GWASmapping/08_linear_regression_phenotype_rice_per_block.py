import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import argparse
import os
import argparse

import pandas as pd
import numpy as np

import matplotlib
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data_with_phenotype(attn_dir, phenotype_col, haplotype='hap1'):
    """加载attention数据和表型数据"""
    # 读取attention矩阵
    attn_file = os.path.join(attn_dir, f'{haplotype}_attention_collapsed.csv')
    if not os.path.exists(attn_file):
        raise FileNotFoundError(f"未找到文件: {attn_file}")
    
    attn = pd.read_csv(attn_file, index_col=0)
    
    # 读取metadata (包含表型)
    meta_file = os.path.join(attn_dir, 'metadata.csv')
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"未找到文件: {meta_file}")
    
    metadata = pd.read_csv(meta_file, index_col=0)
    
    # 检查表型列是否存在
    if phenotype_col not in metadata.columns:
        raise ValueError(f"表型列 '{phenotype_col}' 不在metadata中! "
                        f"可用列: {metadata.columns.tolist()}")
    
    # 确保样本顺序一致
    common_samples = attn.index.intersection(metadata.index)
    attn = attn.loc[common_samples]
    metadata = metadata.loc[common_samples]
    
    # 提取表型值(去除NaN)
    phenotype = metadata[phenotype_col].dropna()
    
    # 只保留有表型的样本
    common_with_pheno = attn.index.intersection(phenotype.index)
    attn = attn.loc[common_with_pheno]
    phenotype = phenotype.loc[common_with_pheno]
    
    return attn, phenotype, metadata.loc[common_with_pheno]


def linear_regression_analysis(attn, phenotype, min_samples=20):
    """
    对每个位点进行线性回归分析
    
    模型: phenotype = β0 + β1 * attention + ε
    
    返回每个位点的:
    - beta: 回归系数(效应大小)
    - se: 标准误
    - t_stat: t统计量
    - pvalue: P值
    - r_squared: R²(解释方差比例)
    """
    
    n_samples = len(attn)
    positions = attn.columns.tolist()
    
    print(f"\n开始线性回归分析...")
    print(f"  样本数: {n_samples}")
    print(f"  位点数: {len(positions)}")
    print(f"  表型范围: [{phenotype.min():.2f}, {phenotype.max():.2f}]")
    print(f"  表型均值±标准差: {phenotype.mean():.2f} ± {phenotype.std():.2f}")
    
    results = []
    y = phenotype.values  # 因变量(表型)
    
    for i, pos in enumerate(positions):
        if (i + 1) % 1000 == 0:
            print(f"  处理进度: {i+1}/{len(positions)}")
        
        # 提取该位点的attention值(自变量)
        x = attn[pos].values
        
        # 去除NaN
        valid_mask = ~np.isnan(x)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < min_samples:
            results.append({
                'position': pos,
                'n_samples': len(x_valid),
                'beta': np.nan,
                'se': np.nan,
                't_stat': np.nan,
                'pvalue': np.nan,
                'r_squared': np.nan,
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'mean_attention': np.nan,
                'std_attention': np.nan
            })
            continue
        
        # 添加截距项
        X = sm.add_constant(x_valid)
        
        try:
            # 拟合OLS模型
            model = sm.OLS(y_valid, X).fit()
            
            # 提取结果
            beta = model.params[1]      # 斜率(attention的系数)
            se = model.bse[1]           # 标准误
            t_stat = model.tvalues[1]   # t统计量
            pvalue = model.pvalues[1]   # P值
            r_squared = model.rsquared  # R²
            
            # 计算Pearson相关系数(作为参考)
            pearson_r, pearson_p = stats.pearsonr(x_valid, y_valid)
            
            results.append({
                'position': pos,
                'n_samples': len(x_valid),
                'beta': beta,
                'se': se,
                't_stat': t_stat,
                'pvalue': pvalue,
                'r_squared': r_squared,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'mean_attention': np.mean(x_valid),
                'std_attention': np.std(x_valid)
            })
            
        except Exception as e:
            results.append({
                'position': pos,
                'n_samples': len(x_valid),
                'beta': np.nan,
                'se': np.nan,
                't_stat': np.nan,
                'pvalue': np.nan,
                'r_squared': np.nan,
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'mean_attention': np.mean(x_valid),
                'std_attention': np.std(x_valid)
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
    
    # 多层显著性标记
    # 1. 宽松标准 (仅FDR)
    results_df['sig_loose'] = (results_df['padj'] < 0.05)
    
    # 2. 中等标准 (FDR + 效应大小)
    results_df['sig_moderate'] = (results_df['padj'] < 0.01) & (results_df['r_squared'] > 0.05)
    
    # 3. 严格标准 (FDR + 效应大小 + beta显著性)
    results_df['sig_strict'] = (
        (results_df['padj'] < 0.001) & 
        (results_df['r_squared'] > 0.10) &
        (results_df['beta'].abs() > results_df['beta'].abs().quantile(0.90))
    )
    
    # 4. 极严格标准 (Bonferroni + 强效应)
    bonferroni_thresh = 0.05 / valid_pvals.sum() if valid_pvals.sum() > 0 else 0
    results_df['sig_bonferroni'] = (
        (results_df['pvalue'] < bonferroni_thresh) &
        (results_df['r_squared'] > 0.15)
    )
    
    # 默认使用中等标准
    results_df['significant'] = results_df['sig_moderate']
    
    print(f"\n分析完成!")
    print(f"  有效位点: {valid_pvals.sum()} / {len(results_df)}")
    print(f"\n显著位点统计 (不同标准):")
    print(f"  宽松 (padj<0.05): {results_df['sig_loose'].sum()}")
    print(f"  中等 (padj<0.01 & R²>0.05): {results_df['sig_moderate'].sum()}")
    print(f"  严格 (padj<0.001 & R²>0.10 & top 10% beta): {results_df['sig_strict'].sum()}")
    print(f"  极严格 (Bonferroni & R²>0.15): {results_df['sig_bonferroni'].sum()}")
    
    return results_df


def plot_manhattan_regression(results, output_file, title='Association Manhattan Plot',
                              padj_thresh=0.05):
    """绘制关联分析的曼哈顿图，用颜色区分不同显著性级别"""
    df = results.copy()
    df = df[df['pvalue'].notna()].copy()
    
    if len(df) == 0:
        print("  [WARNING] 没有有效数据")
        return
    
    # 提取位置坐标
    df['pos_int'] = pd.to_numeric(df['position'].astype(str).str.replace('pos_', ''), errors='coerce')
    if df['pos_int'].isna().all():
        df['pos_int'] = range(len(df))
    
    df = df.sort_values('pos_int')
    df['-log10p'] = -np.log10(df['padj'] + 1e-300)
    
    # 计算Bonferroni阈值
    n_tests = df['pvalue'].notna().sum()
    bonferroni_thresh = 0.05 / n_tests if n_tests > 0 else 0.05
    bonferroni_line = -np.log10(bonferroni_thresh)
    
    # 根据显著性级别分层着色
    df['sig_level'] = 'Non-significant'
    df.loc[df['sig_loose'], 'sig_level'] = 'Loose'
    df.loc[df['sig_moderate'], 'sig_level'] = 'Moderate'
    df.loc[df['sig_strict'], 'sig_level'] = 'Strict'
    df.loc[df['sig_bonferroni'], 'sig_level'] = 'Bonferroni'
    
    # 颜色映射
    color_map = {
        'Non-significant': 'lightgray',
        'Loose': '#FFB6C1',      # 浅粉色
        'Moderate': '#FFA500',   # 橙色
        'Strict': '#FF4500',     # 红橙色
        'Bonferroni': '#8B0000'  # 深红色
    }
    
    # 双图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # 上图: -log10(padj) - 分层绘制，确保显著点在上层
    plot_order = ['Non-significant', 'Loose', 'Moderate', 'Strict', 'Bonferroni']
    for level in plot_order:
        subset = df[df['sig_level'] == level]
        if len(subset) > 0:
            size = 15 if level == 'Non-significant' else 25
            alpha = 0.4 if level == 'Non-significant' else 0.8
            zorder = 1 if level == 'Non-significant' else 3
            ax1.scatter(subset['pos_int'], subset['-log10p'],
                       c=color_map[level], s=size, alpha=alpha, 
                       edgecolors='none', label=f'{level} ({len(subset)})',
                       zorder=zorder)
    
    # FDR阈值线
    ax1.axhline(-np.log10(padj_thresh), color='orange', linestyle='--', 
                linewidth=1.5, alpha=0.8, label=f'FDR={padj_thresh}', zorder=2)
    
    # Bonferroni阈值线
    ax1.axhline(bonferroni_line, color='darkred', linestyle='--', 
                linewidth=1.5, alpha=0.8, label=f'Bonferroni', zorder=2)
    
    ax1.set_ylabel('-log10(Adjusted P)', fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 下图: beta (效应大小) - 同样分层着色
    for level in plot_order:
        subset = df[df['sig_level'] == level]
        if len(subset) > 0:
            size = 15 if level == 'Non-significant' else 25
            alpha = 0.4 if level == 'Non-significant' else 0.8
            zorder = 1 if level == 'Non-significant' else 3
            ax2.scatter(subset['pos_int'], subset['beta'],
                       c=color_map[level], s=size, alpha=alpha,
                       edgecolors='none', zorder=zorder)
    
    ax2.axhline(0, color='black', linewidth=1.5, alpha=0.3, zorder=2)
    ax2.set_xlabel('Genomic Position', fontsize=11)
    ax2.set_ylabel('Beta (Effect Size)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_file}")


def plot_qq(results, output_file, title='Q-Q Plot'):
    """绘制Q-Q图检验P值分布"""
    df = results.copy()
    observed_p = df['pvalue'].dropna().values
    
    if len(observed_p) == 0:
        return
    
    # 排序
    observed_p = np.sort(observed_p)
    n = len(observed_p)
    
    # 期望P值(均匀分布)
    expected_p = np.arange(1, n + 1) / (n + 1)
    
    # -log10转换
    observed_log = -np.log10(observed_p + 1e-300)
    expected_log = -np.log10(expected_p)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 散点
    ax.scatter(expected_log, observed_log, alpha=0.6, s=20, edgecolors='none')
    
    # 对角线(y=x)
    max_val = max(expected_log.max(), observed_log.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
    
    ax.set_xlabel('Expected -log10(P)', fontsize=12)
    ax.set_ylabel('Observed -log10(P)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_file}")


def plot_top_associations(results, attn, phenotype, output_file, top_n=20):
    """绘制Top关联位点的散点图"""
    sig = results[results['significant'] == True].copy()
    if len(sig) == 0:
        print("  [WARNING] 没有显著位点")
        return
    
    sig = sig.sort_values('padj').head(top_n)
    
    n_cols = 5
    n_rows = int(np.ceil(len(sig) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    
    for idx, (_, row) in enumerate(sig.iterrows()):
        pos = row['position']
        ax = axes[idx]
        
        x = attn[pos].dropna()
        y = phenotype.loc[x.index]
        
        # 散点图
        ax.scatter(x, y, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        
        # 拟合线
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)
        
        ax.set_title(f"{pos}\nβ={row['beta']:.3f}, R²={row['r_squared']:.3f}\npadj={row['padj']:.2e}",
                    fontsize=9)
        ax.set_xlabel('Attention Score', fontsize=8)
        ax.set_ylabel('Phenotype', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(sig), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_file}")


def plot_significance_thresholds(results, output_file):
    """可视化不同显著性阈值下的结果"""
    df = results[results['pvalue'].notna()].copy()
    
    if len(df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. R² vs -log10(padj) 散点图
    ax = axes[0, 0]
    df['-log10p'] = -np.log10(df['padj'] + 1e-300)
    
    # 限制y轴范围避免极端值
    df['-log10p_capped'] = df['-log10p'].clip(upper=50)
    
    colors = []
    for _, row in df.iterrows():
        if row['sig_bonferroni']:
            colors.append('darkred')
        elif row['sig_strict']:
            colors.append('red')
        elif row['sig_moderate']:
            colors.append('orange')
        elif row['sig_loose']:
            colors.append('lightcoral')
        else:
            colors.append('gray')
    
    ax.scatter(df['r_squared'], df['-log10p_capped'], c=colors, alpha=0.6, s=15)
    ax.axhline(8, color='orange', linestyle='--', linewidth=1, label='padj=1e-8')
    ax.axvline(0.05, color='blue', linestyle='--', linewidth=1, label='R²=0.05')
    ax.axvline(0.10, color='green', linestyle='--', linewidth=1, label='R²=0.10')
    ax.set_xlabel('R² (Variance Explained)', fontsize=11)
    ax.set_ylabel('-log10(padj) [capped at 50]', fontsize=11)
    ax.set_title('Significance vs Effect Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. 显著位点数统计
    ax = axes[0, 1]
    sig_counts = {
        'Loose\n(padj<0.05)': df['sig_loose'].sum(),
        'Moderate\n(padj<0.01\n& R²>0.05)': df['sig_moderate'].sum(),
        'Strict\n(padj<0.001\n& R²>0.10\n& top beta)': df['sig_strict'].sum(),
        'Bonferroni\n(& R²>0.15)': df['sig_bonferroni'].sum()
    }
    
    bars = ax.bar(range(len(sig_counts)), list(sig_counts.values()), 
                  color=['lightcoral', 'orange', 'red', 'darkred'], edgecolor='black')
    ax.set_xticks(range(len(sig_counts)))
    ax.set_xticklabels(list(sig_counts.keys()), fontsize=9)
    ax.set_ylabel('Number of Significant Positions', fontsize=11)
    ax.set_title('Significant Positions by Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数量
    for bar, count in zip(bars, sig_counts.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. R²分布直方图（分层着色）
    ax = axes[1, 0]
    r2_bins = np.linspace(0, df['r_squared'].max(), 50)
    
    ax.hist(df[df['sig_bonferroni']]['r_squared'], bins=r2_bins, 
            alpha=0.8, label='Bonferroni', color='darkred')
    ax.hist(df[df['sig_strict']]['r_squared'], bins=r2_bins,
            alpha=0.7, label='Strict', color='red')
    ax.hist(df[df['sig_moderate']]['r_squared'], bins=r2_bins,
            alpha=0.6, label='Moderate', color='orange')
    ax.hist(df[~df['sig_loose']]['r_squared'], bins=r2_bins,
            alpha=0.4, label='Non-sig', color='gray')
    
    ax.axvline(0.05, color='blue', linestyle='--', linewidth=2)
    ax.axvline(0.10, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('R² (Variance Explained)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('R² Distribution by Significance Level', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Beta vs P值
    ax = axes[1, 1]
    ax.scatter(df['beta'], df['-log10p_capped'], c=colors, alpha=0.6, s=15)
    ax.axhline(8, color='orange', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Beta (Effect Size)', fontsize=11)
    ax.set_ylabel('-log10(padj) [capped]', fontsize=11)
    ax.set_title('Effect Size vs Significance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_file}")
    """绘制效应大小(beta)和R²的分布"""
    df = results[results['pvalue'].notna()].copy()
    
    if len(df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Beta分布
    ax = axes[0, 0]
    ax.hist(df['beta'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Beta (Effect Size)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Effect Sizes', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # R²分布
    ax = axes[0, 1]
    ax.hist(df['r_squared'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('R² (Variance Explained)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of R²', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Beta vs R² (显著vs不显著)
    ax = axes[1, 0]
    sig = df[df['significant'] == True]
    non_sig = df[df['significant'] == False]
    
    ax.scatter(non_sig['beta'], non_sig['r_squared'], 
              alpha=0.3, s=20, label='Non-significant', color='gray')
    ax.scatter(sig['beta'], sig['r_squared'],
              alpha=0.8, s=30, label='Significant', color='red', edgecolors='k')
    ax.set_xlabel('Beta', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Effect Size vs Variance Explained', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # P值分布
    ax = axes[1, 1]
    ax.hist(df['pvalue'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax.set_xlabel('P-value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of P-values', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_file}")


def process_single_block(block_dir, block_name, args):
    """处理单个block"""
    print(f"\n{'='*60}")
    print(f"处理: {block_name}")
    print(f"{'='*60}")
    
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
        'n_significant': 0,
        'top_r_squared': 0
    }
    
    try:
        # 加载数据
        attn, phenotype, metadata = load_data_with_phenotype(
            block_dir, args.phenotype_col, args.haplotype
        )
        
        print(f"  ✓ 矩阵: {attn.shape}")
        print(f"  ✓ 表型: {len(phenotype)} 样本")
        
        result_summary['n_samples'] = len(attn)
        result_summary['n_positions'] = len(attn.columns)
        
        # 线性回归分析
        results = linear_regression_analysis(
            attn, phenotype,
            min_samples=args.min_samples
        )
        
        # 根据用户选择的标准设置 significant 列
        sig_col_map = {
            'loose': 'sig_loose',
            'moderate': 'sig_moderate',
            'strict': 'sig_strict',
            'bonferroni': 'sig_bonferroni'
        }
        results['significant'] = results[sig_col_map[args.sig_level]]
        
        n_sig = results['significant'].sum()
        result_summary['n_significant'] = int(n_sig)
        
        if n_sig > 0:
            result_summary['top_r_squared'] = float(results[results['significant']==True]['r_squared'].max())
        
        print(f"  ✓ 显著位点: {n_sig}")
        
        # 保存结果
        prefix = f"{args.haplotype}_{args.phenotype_col}"
        
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
        plot_manhattan_regression(
            results,
            os.path.join(block_output, 'figures', f'{prefix}_manhattan.png'),
            title=f'{block_name} - Association Analysis',
            padj_thresh=args.padj_thresh
        )
        
        plot_qq(
            results,
            os.path.join(block_output, 'figures', f'{prefix}_qqplot.png'),
            title=f'{block_name} - Q-Q Plot'
        )
        
        plot_significance_thresholds(
            results,
            os.path.join(block_output, 'figures', f'{prefix}_significance_thresholds.png')
        )
        
        if len(sig_results) > 0:
            plot_top_associations(
                results, attn, phenotype,
                os.path.join(block_output, 'figures', f'{prefix}_top20_scatter.png'),
                top_n=20
            )
        
        result_summary['status'] = 'success'
        print(f"  ✓ 完成!")
        
    except FileNotFoundError as e:
        result_summary['error'] = f"文件未找到: {e}"
        print(f"  ✗ 跳过: {result_summary['error']}")
    except ValueError as e:
        result_summary['error'] = str(e)
        print(f"  ✗ 跳过: {result_summary['error']}")
    except Exception as e:
        result_summary['error'] = f"{type(e).__name__}: {e}"
        print(f"  ✗ 错误: {result_summary['error']}")
    
    return result_summary


def find_block_directories(input_dir):
    """查找所有block文件夹"""
    block_dirs = []
    input_path = Path(input_dir)
    
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            hap1_file = subdir / "hap1_attention_collapsed.csv"
            metadata_file = subdir / "metadata.csv"
            
            if hap1_file.exists() and metadata_file.exists():
                block_dirs.append((subdir, subdir.name))
    
    return sorted(block_dirs)


def main():
    parser = argparse.ArgumentParser(description='Attention与表型的线性回归关联分析')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='包含所有block子文件夹的目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--phenotype_col', type=str, required=True,
                       help='metadata中的表型列名(数量性状)')
    parser.add_argument('--haplotype', type=str, default='hap1',
                       choices=['hap1', 'hap2'],
                       help='分析的单倍型')
    parser.add_argument('--min_samples', type=int, default=20,
                       help='最小样本数')
    parser.add_argument('--padj_thresh', type=float, default=0.05,
                       help='FDR阈值')
    parser.add_argument('--sig_level', type=str, default='moderate',
                       choices=['loose', 'moderate', 'strict', 'bonferroni'],
                       help='显著性标准: loose(宽松), moderate(中等), strict(严格), bonferroni(极严格)')
    parser.add_argument('--block_pattern', type=str, default=None,
                       help='只处理匹配的block,如 "block_1,block_5,block_10"')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# 线性回归关联分析")
    print(f"{'#'*60}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"表型列名: {args.phenotype_col}")
    print(f"单倍型: {args.haplotype}")
    
    # 查找blocks
    print(f"\n扫描block文件夹...")
    block_dirs = find_block_directories(args.input_dir)
    
    if not block_dirs:
        print(f"错误: 未找到block文件夹")
        return
    
    print(f"找到 {len(block_dirs)} 个block")
    
    if args.block_pattern:
        selected_blocks = set(args.block_pattern.split(','))
        block_dirs = [(d, n) for d, n in block_dirs if n in selected_blocks]
        print(f"过滤后: {len(block_dirs)} 个block")
    
    # 处理
    all_results = []
    for block_dir, block_name in block_dirs:
        result = process_single_block(block_dir, block_name, args)
        all_results.append(result)
    
    # 总结
    print(f"\n{'#'*60}")
    print(f"# 完成!")
    print(f"{'#'*60}")
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    
    print(f"\n总计: {len(all_results)} blocks")
    print(f"成功: {success_count}")
    
    # 保存总结
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(args.output_dir, 'regression_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n总结: {summary_file}")
    
    if success_count > 0:
        print(f"\n成功blocks:")
        for r in all_results:
            if r['status'] == 'success':
                print(f"  ✓ {r['block']}: {r['n_significant']} 显著位点, "
                      f"最高R²={r['top_r_squared']:.3f}")


if __name__ == '__main__':
    main()
