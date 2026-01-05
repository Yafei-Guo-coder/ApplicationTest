import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. 读取数据
# ------------------------------
gwas_file = "/mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/GWAS_summary_statistics/2_3kRice/MAF.3K_Rice_pheno.awn_presence.fastGWA"  # 替换成你的文件路径
df = pd.read_csv(gwas_file, sep="\t")

# ------------------------------
# 2. 准备绘图数据
# ------------------------------
df['CHR'] = df['CHR'].astype(str)
df['-log10(P)'] = -np.log10(df['P'])

# 为了画曼哈顿图，需要累计位置
df['BP_cum'] = 0
chrom_sizes = {}

cum_size = 0
chroms = df['CHR'].unique()

for chrom in chroms:
    chrom_df = df[df['CHR'] == chrom]
    chrom_sizes[chrom] = chrom_df['POS'].max()
    df.loc[df['CHR'] == chrom, 'BP_cum'] = chrom_df['POS'] + cum_size
    cum_size += chrom_sizes[chrom]

# ------------------------------
# 3. 设置染色体颜色
# ------------------------------
colors = ['#4E79A7', '#F28E2B'] * 6  # 两种颜色交替
df['color'] = df['CHR'].map(lambda x: colors[int(x) % 2])

# ------------------------------
# 4. 绘制曼哈顿图
# ------------------------------
plt.figure(figsize=(15,6))
plt.scatter(df['BP_cum'], df['-log10(P)'], c=df['color'], s=5)
plt.xlabel('Chromosome')
plt.ylabel('-log10(P)')
plt.title('Manhattan plot')

# 设置染色体刻度在中心位置
ticks = []
labels = []
for chrom in chroms:
    chrom_df = df[df['CHR'] == chrom]
    ticks.append((chrom_df['BP_cum'].min() + chrom_df['BP_cum'].max()) / 2)
    labels.append(chrom)
plt.xticks(ticks, labels)

plt.tight_layout()
plt.savefig('manhattan_plot.png', dpi=300)
plt.show()
