import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import os

# =====================
# 配置参数
# =====================
pheno_file = "/mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/2_3K_Rice_pheno"
group_file = "sampleNameMap.withGroup.txt"

pheno_col = "grain_length"     # 要分析的表型
subpop_col = "Subpopulation"   # 细分群
pop_col = "Population"         # 大群（Indica / Japonica / Intermediate）

output_dir = "phenotype_plots_grain_length"
os.makedirs(output_dir, exist_ok=True)

# =====================
# 读取 & 合并数据
# =====================
pheno = pd.read_csv(pheno_file, sep="\t")
group_map = pd.read_csv(group_file, sep="\t")

df = pd.merge(
    pheno,
    group_map,
    left_on="SampleID",
    right_on="Cultivar ID",
    how="inner"
)

df = df[df[pheno_col].notna()]
print(f"有效样本数: {len(df)}")

# =====================
# 1. Overall 分布（Hist + KDE）
# =====================
plt.figure(figsize=(8,5))
values = df[pheno_col].values

plt.hist(values, bins=25, density=True, alpha=0.5, color="steelblue")
kde = gaussian_kde(values)
x = np.linspace(values.min(), values.max(), 300)
plt.plot(x, kde(x), lw=2, color="steelblue")
plt.fill_between(x, kde(x), alpha=0.3, color="steelblue")

plt.xlabel(pheno_col)
plt.ylabel("Density")
plt.title(f"{pheno_col} - Overall Distribution")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/{pheno_col}_overall_hist_kde.png", dpi=300)
plt.close()

# =====================
# 2. Subpopulation KDE（一张图）
# =====================
plt.figure(figsize=(10,6))
subpops = sorted(df[subpop_col].dropna().unique())
colors = sns.color_palette("tab10", len(subpops))

for c, sp in zip(colors, subpops):
    vals = df[df[subpop_col] == sp][pheno_col].values
    if len(vals) < 5:
        continue
    kde = gaussian_kde(vals)
    x = np.linspace(vals.min(), vals.max(), 300)
    plt.plot(x, kde(x), lw=2, label=sp, color=c)
    plt.fill_between(x, kde(x), alpha=0.3, color=c)

plt.xlabel(pheno_col)
plt.ylabel("Density")
plt.title(f"{pheno_col} - Subpopulation KDE")
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/{pheno_col}_subpopulation_kde.png", dpi=300)
plt.close()

# =====================
# 3. Population 分布（Hist + KDE）
# =====================
plt.figure(figsize=(8,5))
pops = sorted(df[pop_col].dropna().unique())
colors = sns.color_palette("Set2", len(pops))

for c, p in zip(colors, pops):
    vals = df[df[pop_col] == p][pheno_col].values
    if len(vals) < 5:
        continue
    plt.hist(vals, bins=25, density=True, alpha=0.4, color=c)
    kde = gaussian_kde(vals)
    x = np.linspace(vals.min(), vals.max(), 300)
    plt.plot(x, kde(x), lw=2, label=p, color=c)

plt.xlabel(pheno_col)
plt.ylabel("Density")
plt.title(f"{pheno_col} - Population Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/{pheno_col}_population_hist_kde.png", dpi=300)
plt.close()

# =====================
# 4. Subpopulation 箱线图
# =====================
plt.figure(figsize=(12,6))
sns.boxplot(
    data=df,
    x=subpop_col,
    y=pheno_col,
    palette="tab10",
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.title(f"{pheno_col} - Subpopulation Boxplot")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/{pheno_col}_subpopulation_boxplot.png", dpi=300)
plt.close()

# =====================
# 5. Population 箱线图
# =====================
plt.figure(figsize=(6,6))
sns.boxplot(
    data=df,
    x=pop_col,
    y=pheno_col,
    palette="Set2",
    showfliers=False
)
plt.title(f"{pheno_col} - Population Boxplot")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/{pheno_col}_population_boxplot.png", dpi=300)
plt.close()

# =====================
# 6. Subpopulation 显著性检验
# =====================
groups, labels = [], []
for sp in subpops:
    vals = df[df[subpop_col] == sp][pheno_col].values
    if len(vals) >= 5:
        groups.append(vals)
        labels.append(sp)

H, p_kw = kruskal(*groups)
print(f"[Subpopulation] Kruskal-Wallis p = {p_kw:.3e}")

pair_p, pair_name = [], []
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        _, p = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
        pair_p.append(p)
        pair_name.append(f"{labels[i]} vs {labels[j]}")

rej, p_adj, _, _ = multipletests(pair_p, method="fdr_bh")

pd.DataFrame({
    "Comparison": pair_name,
    "p_value": pair_p,
    "p_adj_FDR": p_adj,
    "Significant": rej
}).to_csv(
    f"{output_dir}/{pheno_col}_subpopulation_stats.tsv",
    sep="\t",
    index=False
)

# =====================
# 7. Population 显著性检验
# =====================
groups, labels = [], []
for p in pops:
    vals = df[df[pop_col] == p][pheno_col].values
    if len(vals) >= 5:
        groups.append(vals)
        labels.append(p)

H, p_kw = kruskal(*groups)
print(f"[Population] Kruskal-Wallis p = {p_kw:.3e}")

pair_p, pair_name = [], []
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        _, p = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
        pair_p.append(p)
        pair_name.append(f"{labels[i]} vs {labels[j]}")

rej, p_adj, _, _ = multipletests(pair_p, method="fdr_bh")

pd.DataFrame({
    "Comparison": pair_name,
    "p_value": pair_p,
    "p_adj_FDR": p_adj,
    "Significant": rej
}).to_csv(
    f"{output_dir}/{pheno_col}_population_stats.tsv",
    sep="\t",
    index=False
)
