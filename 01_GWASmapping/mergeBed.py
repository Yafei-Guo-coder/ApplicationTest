import pandas as pd

# -----------------------------
# 参数设置
# -----------------------------
input_bed = "awn.p3.bed"   # 输入文件
output_bed = "awn.p3.merge.bed" # 输出文件
max_distance = 8000       # 合并阈值，8kb

# -----------------------------
# 读取BED文件
# -----------------------------
# 假设 BED 格式为 chr start end，可根据需要调整
df = pd.read_csv(input_bed, sep="\t", header=None, names=["chr","start","end"])

# 如果只有一个位置列，假设 end = start + 1
if df.shape[1] == 2:
    df['end'] = df['start'] + 1

df = df.sort_values(['chr', 'start']).reset_index(drop=True)

# -----------------------------
# 合并区间
# -----------------------------
merged = []
for chrom, group in df.groupby('chr'):
    group = group.reset_index(drop=True)
    start = group.loc[0, 'start']
    end = group.loc[0, 'end']
    count = 1
    value_sum = end - start  # 如果你有数值列，可以改成 group.loc[i,'value']

    for i in range(1, len(group)):
        cur_start = group.loc[i,'start']
        cur_end = group.loc[i,'end']

        if cur_start - end <= max_distance:
            # 合并区间
            end = max(end, cur_end)
            count += 1
            value_sum += cur_end - cur_start
        else:
            # 保存上一个区间
            merged.append([chrom, start, end, count, value_sum])
            # 开启新区间
            start = cur_start
            end = cur_end
            count = 1
            value_sum = cur_end - cur_start

    # 保存最后一个区间
    merged.append([chrom, start, end, count, value_sum])

# -----------------------------
# 输出文件
# -----------------------------
merged_df = pd.DataFrame(merged, columns=['chr','start','end','num_points','sum_value'])
merged_df.to_csv(output_bed, sep="\t", index=False)
print(f"合并完成，共 {len(merged_df)} 个区间，输出到 {output_bed}")
