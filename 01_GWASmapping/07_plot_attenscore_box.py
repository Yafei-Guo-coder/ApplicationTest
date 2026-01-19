import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. 参数设置
# ===============================
# input_file = "/mnt/zzb/default/Workspace/guoyafei/appTest/attention_matrices_per_block_APTrue_with_indel_rice_normalized/block_1/hap1_attention_collapsed.csv"   # CSV 文件路径
# out_dir = "attention_matrices_per_block_box_indel_rice_normalized"

input_file = "/mnt/zzb/default/Workspace/guoyafei/appTest/attention_matrices_per_block_GL/block_1/hap1_attention_collapsed.csv"   # CSV 文件路径
out_dir = "attention_matrices_per_block_box_GL"

group_size = 750              # 每张图多少个 position

# ===============================
# 1. 参数
# ===============================

fig_width = 40     # inch
fig_height = 5     # inch
dpi = 200

point_size = 4
point_alpha = 0.35
jitter_sd = 0.08

os.makedirs(out_dir, exist_ok=True)

# ===============================
# 2. 读 CSV
# ===============================
df = pd.read_csv(input_file)

sample_col = df.columns[0]
position_cols = df.columns[1:]

# ===============================
# 3. position 排序
# ===============================
pos_numeric = position_cols.str.replace("pos_", "", regex=False).astype(int)
position_order = position_cols[np.argsort(pos_numeric)]

df = df[[sample_col] + position_order.tolist()]

# ===============================
# 4. 分组画图
# ===============================
num_positions = len(position_order)
num_groups = int(np.ceil(num_positions / group_size))

for g in range(num_groups):
    start = g * group_size
    end = min((g + 1) * group_size, num_positions)
    group_positions = position_order[start:end]

    print(f"Plotting group {g + 1}: positions {start + 1}–{end}")

    data = df[group_positions]

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        dpi=dpi
    )

    # boxplot
    ax.boxplot(
        data.values,
        positions=np.arange(len(group_positions)),
        widths=0.6,
        showfliers=False
    )

    # jitter scatter
    for i, col in enumerate(group_positions):
        y = data[col].values
        x = np.random.normal(i, jitter_sd, size=len(y))
        ax.scatter(
            x,
            y,
            s=point_size,
            alpha=point_alpha
        )

    ax.set_xticks(np.arange(len(group_positions)))
    ax.set_xticklabels(group_positions, rotation=90, fontsize=5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")
    ax.set_title(f"Positions {start + 1}–{end}")

    plt.tight_layout()

    out_file = os.path.join(out_dir, f"boxplot_group_{g + 1}.png")
    plt.savefig(out_file)
    plt.close()

print("All PNG plots finished.")
