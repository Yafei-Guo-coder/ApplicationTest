#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# 设置 Seaborn 风格
sns.set(style="whitegrid")

def plot_trait_distribution(df, file_name, output_dir):
    """
    对每个列（表型）绘制分布图
    """
    # 去掉第一列 'ID'
    traits = df.columns[1:]
    
    for trait in traits:
        plt.figure(figsize=(6,4))
        # 绘制直方图和 KDE 曲线
        sns.histplot(df[trait], kde=True, bins=30, color="skyblue", edgecolor="black")
        plt.title(f"{trait} distribution in {file_name}")
        plt.xlabel(trait)
        plt.ylabel("Count")
        plt.tight_layout()
        
        # 保存图像
        out_path = output_dir / f"{file_name}_{trait}_dist.png"
        plt.savefig(out_path)
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_phenotypes.py file1 file2 ...")
        sys.exit(1)
    
    files = sys.argv[1:]
    output_dir = Path("output_plots")
    output_dir.mkdir(exist_ok=True)
    
    for f in files:
        fpath = Path(f)
        print(f"处理文件: {fpath.name}")
        try:
            df = pd.read_csv(fpath, sep="\t", na_values=["NA"])
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
            continue
        
        plot_trait_distribution(df, fpath.stem, output_dir)
        print(f"  ✓ 已生成 {len(df.columns)-1} 张图到 {output_dir}")

if __name__ == "__main__":
    main()

