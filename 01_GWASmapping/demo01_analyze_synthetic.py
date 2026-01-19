import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import quantile_transform
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import math
from scipy.stats import mannwhitneyu, wasserstein_distance, pearsonr, spearmanr
from itertools import combinations

import pyBigWig
HG38_SIZES = {"chr11": 135_086_622,}


# 合并两条单倍型的拷贝数数据
def combine_h1_h2(hap1, hap2):
    combined = pd.DataFrame(index=hap1.index, columns=hap1.columns, dtype=float)
    del_dosage = hap1.isna().astype(int) + hap2.isna().astype(int)  # 0/1/2
    
    for col in hap1.columns:
        a = hap1[col].values
        b = hap2[col].values
        # 可用均值：两条都NA->NA；一条NA->取另一条；都在->均值
        with np.errstate(invalid='ignore'):
            both_na = np.isnan(a) & np.isnan(b)
            one_na  = np.isnan(a) ^ np.isnan(b)
            both_ok = (~np.isnan(a)) & (~np.isnan(b))
            out = np.full_like(a, np.nan, dtype=float)
            out[both_ok] = (a[both_ok] + b[both_ok]) / 2.0
            out[one_na]  = np.where(np.isnan(a[one_na]), b[one_na], a[one_na])
            combined[col] = out
        
    return combined, del_dosage


# 计算组内组间差异效应，包括多种方法，后面都用conhen_d
def compute_separability_scores(matrix, groups, min_n=10, dropna=True):
    uniq = sorted(groups.unique().tolist())
    pairs = [(a,b) for i,a in enumerate(uniq) for b in uniq[i+1:]]
    
    positions = matrix.columns
    out_rows = []
    
    group_to_df = {g: matrix.loc[groups == g] for g in uniq}
    
    for pos in positions:
        # 提前取出每组该位点向量（含 NaN）；按需清洗
        col_per_group = {}
        for g in uniq:
            v = group_to_df[g][pos].to_numpy()
            if dropna:
                v = v[np.isfinite(v)]
            col_per_group[g] = v

        for gA, gB in pairs:
            a = col_per_group[gA]
            b = col_per_group[gB]

            nA, nB = a.size, b.size
            if nA < min_n or nB < min_n:
                out_rows.append({
                    "position": pos, "groupA": gA, "groupB": gB,
                    "n_A": nA, "n_B": nB,
                    "mean_A": np.nan, "mean_B": np.nan,
                    "std_A": np.nan, "std_B": np.nan,
                    "auc": np.nan, "cliffs_delta": np.nan,
                    "wasserstein": np.nan,
                    "cohen_d": np.nan, "hedges_g": np.nan,
                })
                continue

            meanA = float(np.mean(a)); meanB = float(np.mean(b))
            stdA  = float(np.std(a, ddof=1)) if nA > 1 else np.nan
            stdB  = float(np.std(b, ddof=1)) if nB > 1 else np.nan

            # Mann–Whitney U -> AUC
            U, _ = mannwhitneyu(a, b, alternative="two-sided")
            auc = U / (nA * nB)  # P(a < b) + 1/2 P(a == b)

            # Cliff's delta
            cliffs = 2.0 * auc - 1.0

            # Wasserstein distance
            wdist = wasserstein_distance(a, b)

            # Cohen's d (B - A) / pooled std
            if (nA > 1) and (nB > 1) and np.isfinite(stdA) and np.isfinite(stdB):
                sp_num = (nA - 1) * (stdA ** 2) + (nB - 1) * (stdB ** 2)
                sp_den = (nA + nB - 2)
                s_pool = np.sqrt(sp_num / sp_den) if sp_den > 0 and sp_num >= 0 else np.nan
            else:
                s_pool = np.nan
            cohen_d = (meanB - meanA) / s_pool if (s_pool is not None and np.isfinite(s_pool) and s_pool > 0) else np.nan

            # Hedges' g：对 Cohen's d 的无偏修正
            # 修正因子 J = 1 - 3/(4*(nA+nB) - 9)
            N = nA + nB
            if np.isfinite(cohen_d) and N > 3:
                J = 1.0 - 3.0 / (4.0 * N - 9.0)
                hedges_g = cohen_d * J
            else:
                hedges_g = np.nan

            out_rows.append({
                "position": pos, "groupA": gA, "groupB": gB,
                "n_A": nA, "n_B": nB,
                "mean_A": meanA, "mean_B": meanB,
                "std_A": stdA, "std_B": stdB,
                "auc": float(auc),
                "cliffs_delta": float(cliffs),
                "wasserstein": float(wdist),
                "cohen_d": float(cohen_d),
                "hedges_g": float(hedges_g),
            })

    return pd.DataFrame(out_rows)


# 差异attention分析主要逻辑
def bh(p):
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if mask.sum():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q

def differential_attention(matrix, groups, group_pairs=None, min_n=10, skip_del=False):
    if group_pairs is None:
        uniq = sorted(groups.unique().tolist())
        group_pairs = [(a,b) for i,a in enumerate(uniq) for b in uniq[i+1:]]
    cols = matrix.columns
    # 删除剂量（单倍型级别：NA=1, 非NA=0）
    del_flag = matrix.isna().astype(int)
    
    # 1) 连续注意力：rank test（忽略NA）
    rows = []
    for gA, gB in group_pairs:
        print(f"Compare {gA} vs. {gB} for continuous value rank test ...")
        maskA = (groups == gA).values
        maskB = (groups == gB).values
        XA = matrix.values[maskA, :]
        XB = matrix.values[maskB, :]

        meanA = np.nanmean(XA, axis=0)
        meanB = np.nanmean(XB, axis=0)
        baseMean = np.nanmean(matrix.values, axis=0)
        
        global_min = np.nanmin([np.nanmin(meanA), np.nanmin(meanB)])
        shift = (-global_min + 1e-8) if np.isfinite(global_min) and global_min <= 0 else 0.0
        log2FC = np.log2((meanB + shift) / (meanA + shift + 1e-8))
        delta = meanB - meanA

        p_cont = np.empty(len(cols), dtype=float)
        nA_eff = np.sum(~np.isnan(XA), axis=0)
        nB_eff = np.sum(~np.isnan(XB), axis=0)

        for j in range(len(cols)):
            a = XA[:, j]; a = a[np.isfinite(a)]
            b = XB[:, j]; b = b[np.isfinite(b)]
            if len(a) >= min_n and len(b) >= min_n:
                _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            else:
                p = np.nan
            p_cont[j] = p
        q_cont = bh(p_cont)

        part_cont = pd.DataFrame({
            "hap_part": "continuous",
            "groupA": gA, "groupB": gB,
            "position": cols,
            "baseMean": baseMean,
            "mean_A": meanA, "mean_B": meanB,
            "log2FC": log2FC, "delta": delta,
            "nA_nonNA": nA_eff, "nB_nonNA": nB_eff,
            "pvalue": p_cont, "padj": q_cont
        })
        rows.append(part_cont)
        
    # 2) 删除富集（比较 NA 比例）
    if not skip_del:
        rows_del = []
        uniq = sorted(groups.unique().tolist())
        for gA, gB in group_pairs:
            print(f"Compare {gA} vs. {gB} for missing value test ...")
            maskA = (groups == gA).values
            maskB = (groups == gB).values
            if np.sum(maskA) < min_n or np.sum(maskB) < min_n:
                continue
            
            dA = del_flag.values[maskA, :]  # 1=deleted, 0=present
            dB = del_flag.values[maskB, :]

            p_del = np.empty(len(cols), dtype=float)
            or_est = np.empty(len(cols), dtype=float)
            for j in range(len(cols)):
                a_del = int(dA[:, j].sum()); a_pres = int((~np.isnan(matrix.values[maskA, j])).sum()) - (len(dA[:, j]) - (len(dA[:, j]) - a_del))
                # 更安全：直接用样本计数
                nA = dA.shape[0]; a_pres = nA - a_del

                b_del = int(dB[:, j].sum()); nB = dB.shape[0]; b_pres = nB - b_del

                table = np.array([[a_del, a_pres],
                                [b_del, b_pres]])
                # 若计数都>=5，用卡方；否则 Fisher
                if (table < 5).any():
                    _, p = stats.fisher_exact(table)
                else:
                    _, p = stats.chi2_contingency(table)[:2]
                p_del[j] = p
                # 粗略OR（加0.5连续性修正）
                or_est[j] = ((a_del + 0.5)/(a_pres + 0.5)) / ((b_del + 0.5)/(b_pres + 0.5))

            q_del = bh(p_del)
            part_del = pd.DataFrame({
                "hap_part": "deletion_enrichment",
                "groupA": gA, "groupB": gB,
                "position": cols,
                "OR_deleted": or_est,
                "pvalue": p_del, "padj": q_del,
                "nA": dA.shape[0], "nB": dB.shape[0]
            })
            rows_del.append(part_del)
        
    out = pd.concat(rows, ignore_index=True)
    out_del = None
    if not skip_del:
        out_del = pd.concat(rows_del, ignore_index=True)
    return out, out_del


# 得到的结果与真值位点的对比
def diff_attention_with_truth(
    diff_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    score_df: pd.DataFrame = None,  # 若提供则合并cohen_d作为额外筛选条件
    groupA=0,
    groupB=1,
    adjp_threshold=0.05,
    lfc_quantile=0.5,
    ignore_pos=None,         # 需要忽略的碱基位置，暂时为chunk连接处的处理
    cohen_threshold=None,    # 若提供则以cohen_d作为额外筛选条件
    sort_col="log2FC",
    pos_col="position",      # 差异结果里的坐标列名，可为 "pos_123" 这种字符串或 int
    truth_pos_col="pos",     # 真值里的坐标列名（1-based）
    window=10,               # 与真值匹配的半径（bp），即 |position - pos| <= window
    topk=20                  # 返回 Top-K
) -> pd.DataFrame:
    """
    在差异注意力结果中筛选 (padj < adjp_threshold & groupA==... & groupB==... & |log2FC|>=...),
    再将其与真值位点按 ±window bp 匹配，返回按 |log2FC| 排序后的 Top-K（带真值注释）。

    需要列（diff_df）: [groupA, groupB, position/baseMean/log2FC/delta/padj]
    需要列（truth_df）: [variant_id, AF, pos] （pos 为 1-based；若列名不同用 truth_pos_col 指定）
    """
    df = diff_df.copy()
    if score_df is not None:
        df = df.merge(score_df[["position", "groupA", "groupB", "cohen_d"]], how="inner", 
                      on=["position", "groupA", "groupB"])  
    # --- 规整 position 列为 int --
    if df[pos_col].dtype == object:
        # 兼容 "pos_123" 格式
        df[pos_col] = df[pos_col].str.replace("^pos_", "", regex=True)
    df[pos_col] = df[pos_col].astype(int)
    
    if ignore_pos is not None:
        if not isinstance(ignore_pos, list):
            raise ValueError(f"ignore_pos should be a list, but received {ignore_pos}")
        df = df[~df[pos_col].isin(ignore_pos)]
        
    df = df[(df["groupA"] == groupA) & (df["groupB"] == groupB)]
        
    # --- 基本筛选 ---
    mask = pd.Series(True, index=df.index)
    if adjp_threshold is not None:
        mask = (mask & (df["padj"] < adjp_threshold))
    if lfc_quantile is not None:
        mask = (mask & (df["log2FC"].abs() >= df["log2FC"].abs().quantile(lfc_quantile)))
    if score_df is not None and cohen_threshold is not None:
        mask = (mask & (df["cohen_d"].abs() > cohen_threshold))
    
    filt = df.loc[mask].copy()
    if filt.empty:
        # 空结果就直接返回空框架
        cols = ["groupA","groupB",pos_col,"baseMean","log2FC","delta","padj",
                "ID","cpra","vcf_position","distance"]
        return pd.DataFrame(columns=cols)

    # --- 真值准备 ---
    truth = truth_df.copy()
    truth[truth_pos_col] = truth[truth_pos_col].astype(int)

    roi_rows = []
    # 先索引加速范围筛选
    filt = filt.sort_values(pos_col).reset_index(drop=True)
    pos_vals = filt[pos_col].to_numpy()

    for _, r in truth.iterrows():
        pos = int(r[truth_pos_col])
        left, right = pos - window, pos + window
        subset_mask = (pos_vals >= left) & (pos_vals <= right)
        if not subset_mask.any():
            continue
        sub = filt.loc[subset_mask, [pos_col]].copy()
        sub["ID"]   = r["ID"]
        sub["vcf_position"] = pos
        sub["cpra"]         = r["cpra"]
        sub["distance"]     = sub[pos_col] - pos
        sub["abs_dist"]     = sub["distance"].abs()
        roi_rows.append(sub)

    # 把最近真值“去重”到每个 position 仅一条
    if roi_rows:
        all_hits = pd.concat(roi_rows, ignore_index=True)

        sort_df = all_hits.copy()
        sort_df = sort_df.sort_values(
            by=[pos_col, "abs_dist"],
            ascending=[True, True],
            kind="mergesort"  # 稳定排序，保留原始相对次序
        )
        # 每个 position 取最优一条
        nearest_hit = sort_df.drop_duplicates(subset=[pos_col], keep="first")
        # 合并回差异表（左连接，未命中位置将为 NaN）
        annotated = filt.merge(nearest_hit[[pos_col, "ID", "cpra", "vcf_position", "distance"]],
                               on=pos_col, how="left")
    else:
        # 所有位置均无命中 -> 全 NaN
        annotated = filt.copy()
        annotated["ID"]   = np.nan
        annotated["vcf_position"] = np.nan
        annotated["cpra"]           = np.nan
        annotated["distance"]     = np.nan

    # 排序并取 Top-K
    annotated[f"abs_{sort_col}"] = annotated[sort_col].abs()
    annotated = annotated.sort_values(f"abs_{sort_col}", ascending=False)

    cols_out = ["groupA","groupB",pos_col,"baseMean","log2FC","delta","padj",
                "ID","vcf_position","cpra","distance"]
    cols_out = [c for c in cols_out if c in annotated.columns]
    return annotated.loc[:, cols_out].head(topk)


# 筛选显著位点上的logFC来画与效应量的相关关系图
def filter_by_groupwise_threshold(res_df, padj_thresh=0.01, lfc_q=0.95):
    df = res_df.copy()
    df["abs_lfc"] = df["log2FC"].abs()
    # 每个对比单独计算分位数阈值，并广播回行
    df["lfc_thr"] = df.groupby(["groupA", "groupB"])["abs_lfc"].transform(
        lambda s: s.quantile(lfc_q)
    )
    filt = (df["padj"] < padj_thresh) & (df["abs_lfc"] >= df["lfc_thr"])
    return df.loc[filt]

# 绘制两个指标的关系散点图，并计算相关系数
def plot_metric_relationship(
    df,
    x_metric: str,
    y_metric: str,
    group_pair: tuple = None,
    cutoff: float = None,              
    figsize=(6, 5),
    alpha=0.7,
    s=35,
    highlight_color="red", 
    fig_save_to="figures/plot.png"            
):
    """
    df: merged DataFrame, must contain x_metric, y_metric, groupA, groupB
    group_pair: tuple, e.g., (0,1) or (1,3)
    cutoff: abs cutoff for y_metric, draw horizontal dash lines and highlight points exceeding cutoff
    """

    # --- filtering by group pair ---
    if group_pair is None:
        df_sub = df.copy()
    else:
        gA, gB = group_pair
        df_sub = df[(df["groupA"] == gA) & (df["groupB"] == gB)].copy()

    # --- compute correlation ---
    x = df_sub[x_metric].values
    y = df_sub[y_metric].values
    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)

    if group_pair is None:
        title = (f"{x_metric} vs {y_metric} | All samples\n"
             f"Pearson r={pear_r:.3f} (p={pear_p:.2e}), "
             f"Spearman r={spear_r:.3f} (p={spear_p:.2e})")
    else:
        title = (f"{x_metric} vs {y_metric} | Group {gA} vs {gB}\n"
                f"Pearson r={pear_r:.3f} (p={pear_p:.2e}), "
                f"Spearman r={spear_r:.3f} (p={spear_p:.2e})")

    # --- plot base scatter ---
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df_sub, x=x_metric, y=y_metric,
                    s=s, alpha=alpha, edgecolor="k", color="gray")

    # --- highlight cutoff exceeding points ---
    if cutoff is not None:
        high_mask = df_sub[y_metric].abs() >= cutoff

        # highlight points
        sns.scatterplot(
            data=df_sub[high_mask],
            x=x_metric, y=y_metric,
            s=s*1.4, alpha=1.0, edgecolor="k",
            color=highlight_color, label=f"|{y_metric}| ≥ {cutoff}"
        )

        # draw horizontal cutoff lines
        plt.axhline(+cutoff, linestyle="--", color=highlight_color, alpha=0.8)
        plt.axhline(-cutoff, linestyle="--", color=highlight_color, alpha=0.8)

    # --- final cosmetics ---
    plt.title(title, fontsize=12)
    plt.xlabel(x_metric, fontsize=11)
    plt.ylabel(y_metric, fontsize=11)
    plt.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(fig_save_to, dpi=300)
    

# 绘制attention log2FC随位置变化曲线，并标注真值位点
def plot_lfc_with_truth_by_pairs(
    diff_df: pd.DataFrame,
    score_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    group_pairs,
    lfc_quantile_thresh: float = 0.90,
    padj_thresh: float = 0.05,
    cohen_thresh: float = 0.5,
    ignore_pos: list = None,
    pos_col: str = "position",
    groupA_col: str = "groupA",
    groupB_col: str = "groupB",
    truth_pos_col: str = "pos",
    truth_id_col: str = "ID",
    figsize=(18, 4),
    truth_fontsize: int = 10,
    unify_ylim: bool = False,
    show_name = True,
):
    """
    对指定的 group 对（group_pairs）逐个绘制 log2FC 随位置的曲线，标注真值位点并高亮显著点。
    - 显著性规则：|log2FC| ≥ quantile(|log2FC|, lfc_quantile_thresh) 且 padj < padj_thresh
    - 标题会显示显著点计数
    - 每个对比单独一张图（返回 [(fig, ax), ...]）

    参数
    ----
    diff_df: 差异分析结果，至少包含列 [position, log2FC, padj, groupA, groupB]
             position 可为 'pos_XXXX' 字符串或整数坐标
    truth_df: 真值表，至少包含 [truth_pos_col]；如果有 truth_id_col 会显示 ID
    group_pairs: 需要绘图的组对列表，例如 [(0,1), (0,2), (1,2)]
    lfc_quantile_thresh: 显著性使用的 |log2FC| 分位阈值（0~1）
    padj_thresh: 显著性使用的 FDR 阈值
    truth_fontsize: 真值 ID 标注字体大小
    unify_ylim: 若为 True，会在所有子图绘制前先扫描各对比的 log2FC 取全局 y 轴范围

    返回
    ----
    figs_axes: list of (fig, ax)
    """
    df = diff_df.copy()
    df = df.merge(score_df[["position", "groupA", "groupB", "cohen_d"]], how="inner", 
                  on=["position", "groupA", "groupB"],)

    # --- position 统一为 int（兼容 "pos_XXXX"） ---
    if df[pos_col].dtype == object:
        df[pos_col] = df[pos_col].str.replace("^pos_", "", regex=True)
    df[pos_col] = df[pos_col].astype(int)
    
    if ignore_pos is not None:
        if not isinstance(ignore_pos, list):
            raise ValueError(f"ignore_pos should be a list, but received {ignore_pos}")
        df = df[~df[pos_col].isin(ignore_pos)]

    # --- 全域 x 轴范围（统一 x 轴，便于比较） ---
    min_pos = int(df[pos_col].min())
    max_pos = int(df[pos_col].max())
    all_positions = pd.DataFrame({pos_col: np.arange(min_pos, max_pos + 1, dtype=int)})

    # --- 真值 ---
    truth = truth_df.copy()
    truth[truth_pos_col] = truth[truth_pos_col].astype(int)
    has_id = truth_id_col in truth.columns
    in_range_truth = truth[(truth[truth_pos_col] >= min_pos) & (truth[truth_pos_col] <= max_pos)]

    # --- 若需要统一 y 轴范围，先预扫 ---
    global_ymin, global_ymax = None, None
    if unify_ylim:
        ymin_list = []
        ymax_list = []
        for gA, gB in group_pairs:
            sub = df[(df[groupA_col] == gA) & (df[groupB_col] == gB)][[pos_col, "log2FC", "padj", "cohen_d"]]
            if sub.empty:
                continue
            merged = all_positions.merge(sub, on=pos_col, how="left")
            lfc = merged["log2FC"].fillna(0.0).values
            if lfc.size:
                ymin_list.append(np.nanmin(lfc))
                ymax_list.append(np.nanmax(lfc))
        if ymin_list and ymax_list:
            # 留点边距
            global_ymin = min(ymin_list) - 0.05 * (max(ymax_list) - min(ymin_list))
            global_ymax = max(ymax_list) + 0.05 * (max(ymax_list) - min(ymin_list))

    figs_axes = []

    for (gA, gB) in group_pairs:
        sub = df[(df[groupA_col] == gA) & (df[groupB_col] == gB)][[pos_col, "log2FC", "padj", "cohen_d"]]
        y_cur_min = np.nanmin(sub["log2FC"])
        y_cur_max = np.nanmax(sub["log2FC"])
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if sub.empty:
            ax.set_title(f"Comparison {gA} vs {gB} | No data", fontsize=12)
            ax.set_xlabel("Position"); ax.set_ylabel("log2FC")
            figs_axes.append((fig, ax))
            continue

        merged = all_positions.merge(sub, on=pos_col, how="left")
        merged["log2FC"] = merged["log2FC"].fillna(0.0)
        merged["padj"] = merged["padj"].fillna(1.0)

        # 显著性：分位阈值 + FDR
        sig_count = 0
        if lfc_quantile_thresh is not None and padj_thresh is not None:
            abs_lfc = merged["log2FC"].abs()
            q_thr = abs_lfc.quantile(lfc_quantile_thresh) if abs_lfc.max() > 0 else 0.0
            abs_cohen = merged["cohen_d"].abs()
            sig_mask = (abs_lfc >= q_thr) & (merged["padj"] < padj_thresh) & (abs_cohen > cohen_thresh)
            sig_count = int(sig_mask.sum())

        # 主曲线
        ax.plot(merged[pos_col], merged["log2FC"], color="tab:blue", linewidth=1.0, alpha=0.9)
        # 填充正/负区域
        ax.fill_between(merged[pos_col], merged["log2FC"], 0, where=(merged["log2FC"] > 0), alpha=0.25)
        ax.fill_between(merged[pos_col], merged["log2FC"], 0, where=(merged["log2FC"] < 0), alpha=0.25)

        # 高亮显著点
        if sig_count > 0:
            ax.scatter(
                merged.loc[sig_mask, pos_col],
                merged.loc[sig_mask, "log2FC"],
                s=14, marker="o", edgecolors="k", linewidths=0.6, zorder=3
            )

        # 真值标注（竖线 + ID）
        for _, r in in_range_truth.iterrows():
            xpos = int(r[truth_pos_col])
            ax.axvline(x=xpos, color="darkred", linestyle="--", linewidth=0.9, alpha=0.85, zorder=1)
            if has_id and show_name:
                ymin, ymax = ax.get_ylim()
                if unify_ylim:
                    ymin, ymax = global_ymin, global_ymax
                ax.text(
                    xpos,
                    ymin + 0.90 * (ymax - ymin),  # 顶部 90% 处
                    str(r[truth_id_col]),
                    rotation=90, fontsize=truth_fontsize, ha="right", va="top",
                    color="darkred"
                )

        # 统一 y 轴范围（可选）
        if unify_ylim and (global_ymin is not None) and (global_ymax is not None):
            ax.set_ylim(global_ymin, global_ymax)

        ax.set_xlim(min_pos, max_pos)
        ax.set_xlabel("Position")
        ax.set_ylabel("log2FC")
        ax.grid(alpha=0.2, linestyle=":")
        
        if lfc_quantile_thresh is not None and padj_thresh is not None:
            ax.set_title(
                f"Comparison {gA} vs {gB} | significant: {sig_count} "
                f"( |lfc|≥Q{lfc_quantile_thresh:.2f}, FDR<{padj_thresh} )",
                fontsize=16
            )
        else:
            ax.set_title(
                f"Comparison {gA} vs {gB}", fontsize=16
            )
            
        plt.tight_layout()
        figs_axes.append((fig, ax))

    return figs_axes


# 生成bigwig文件用于IGV, UCSC浏览器查看
def export_attention_bigwig(
    res_df: pd.DataFrame,
    group_pair=(0, 3),
    value_col: str = "log2FC",
    chrom: str = "chr11",
    genome_sizes: dict = HG38_SIZES,
    out_path: str = "attention_log2fc.bw",
    fill_value: float = 0.0,
    abs: bool = False,
):
    """
    将差异分析结果（hap1/hap2 的原始 res_*_cont）导出为 UCSC 可用的 bigWig。
    步骤：按 group 选择行 -> 规范 position -> 补齐连贯区间 -> 取 value_col -> 写 bigWig

    参数
    ----
    res_df: DataFrame，需包含列：["position", "groupA", "groupB", value_col]
            position 可为 "pos_XXXX" 或整数坐标（1-based）
    group_pair: (groupA, groupB)，例如 (0,3)
    value_col: 信号列名，默认 "log2FC"
    chrom: 染色体名（UCSC 风格），默认 "chr11"
    genome_sizes: dict，提供 chr 长度（默认提供 hg38 的 chr11）
    out_path: 输出 bigWig 路径
    fill_value: 对缺失位置用何值填充（bigWig 连续信号常用 0）

    返回
    ----
    写出 bigWig 文件到 out_path
    """
    if chrom not in genome_sizes:
        raise ValueError(f"chrom '{chrom}' 不在提供的基因组大小表中，请在 genome_sizes 中加入它的长度。")

    gA, gB = group_pair
    sub = res_df[(res_df["groupA"] == gA) & (res_df["groupB"] == gB)].copy()
    if sub.empty:
        raise ValueError(f"在 res_df 中没有找到 group {gA} vs {gB} 的记录。")

    # 规范 position：支持 "pos_XXXX" 或 int
    if sub["position"].dtype == object:
        sub["position"] = sub["position"].str.replace("^pos_", "", regex=True)
    sub["position"] = sub["position"].astype(int)

    # 区间范围并补齐（1-based 全覆盖）
    min_pos = int(sub["position"].min())
    max_pos = int(sub["position"].max())
    all_positions = pd.DataFrame({"position": np.arange(min_pos, max_pos + 1, dtype=int)})

    # 合并并只保留需要的列
    merged = all_positions.merge(sub[["position", value_col]], on="position", how="left")
    merged[value_col] = merged[value_col].astype(float).fillna(fill_value)
    if abs:
        merged[value_col] = merged[value_col].abs()

    # 转为 bigWig 所需的 0-based 半开区间
    merged["start"] = merged["position"] - 1
    merged["end"]   = merged["position"]
    merged = merged.sort_values("start", kind="mergesort")

    # 写 bigWig
    bw = pyBigWig.open(out_path, "w")
    bw.addHeader([(chrom, int(genome_sizes[chrom]))])
    bw.addEntries(
        chroms=[chrom] * len(merged),
        starts=merged["start"].tolist(),
        ends=merged["end"].tolist(),
        values=merged[value_col].tolist()
    )
    bw.close()
    print(f"✅ bigWig written: {out_path}")


# 控制参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HBB attention differential analysis pipeline."
    )

    # ====== 基础路径设置 ======
    parser.add_argument(
        "--hbb_attn_dir",
        type=str,
        required=True,
        help="Path to directory containing HBB attention matrices (hap1/hap2 collapsed).",
    )
    parser.add_argument(
        "--hbb_homo_attn_dir",
        type=str,
        default=None,
        help="Path to homozygous HBB attention matrix directory.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save intermediate and final results.",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        required=True,
        help="Directory to save plots and diagnostic figures.",
    )
    parser.add_argument(
        "--ucsc_dir",
        type=str,
        required=True,
        help="Directory to store UCSC-compatible output files (bedGraph/bigWig etc.).",
    )

    # ====== 参数控制 ======
    parser.add_argument(
        "--ignore_pos",
        type=str,
        default=None,
        help="Comma-separated positions to ignore, e.g. '5226248:5226268,5229150:5229170,5229177'. "
             "If None, all positions are included."
    )
    
    parser.add_argument(
        "--min_n",
        type=int,
        default=10,
        help="Minimum sample number to do significant test and deletion test"
    )

    args = parser.parse_args()

    # ====== 目录初始化 ======
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(args.ucsc_dir, exist_ok=True)

    # ====== 解析 ignore_pos ======
    if args.ignore_pos is not None:
        try:
            parts = []
            for token in args.ignore_pos.split(","):
                if ":" in token:
                    s, e = token.split(":")
                    parts.extend(range(int(s), int(e)))
                else:
                    parts.append(int(token))
            args.ignore_pos = parts
        except Exception:
            print("[WARNING] Failed to parse ignore_pos; keep all pos instead.")
            args.ignore_pos = None
    else:
        args.ignore_pos = None

    return args
    
    
if __name__ == "__main__":
    
    args = parse_args()
    
    HBB_ATTN_DIR = args.hbb_attn_dir
    HBB_HOMO_ATTN_DIR = args.hbb_homo_attn_dir
    RESULTS_DIR = args.results_dir
    FIGURE_DIR = args.figure_dir
    UCSC_DIR = args.ucsc_dir
    
    IGNORE_POS = args.ignore_pos
    MIN_N = args.min_n
    
    SKIP_DEL = False
    PAIRS = [(1,2)]
    LFC_QUANTILE_THRESH = 0.95
    PADJ_THRESH = 0.01
    
    # --------------------------- 准备真值 --------------------------- #
    print("准备真值 ...")
    # 8个真值位点
    data = [
        ("1", 7653710, "C", "T", None),
        ("2", 7654900, "G", "A", None),
        ("3", 7656001, "G", "C", None),
        ("4", 7657740, "A", "ACCG", None),
        ("5", 7659340, "T", "C", None),
        ("6", 7661001, "T", "A", None),
        ("7", 7663600, "T", "G", None),
        ("8", 7665230, "ACAAAA", "A", None),
        ("9", 7666001, "T", "G", None),
        ("10", 7669450, "C", "A", None),
    ]
    true_roi = pd.DataFrame(data, columns=["ID", "pos", "ref", "alt", "AF"])
    true_roi["cpra"] = "chr11" + "-" + true_roi["pos"].astype(str) + "-" + true_roi["ref"] + "-" + true_roi["alt"]
    true_roi = true_roi[["ID", "cpra", "pos"]]
    
    
    # --------------------------- 加载注意力分数 --------------------------- #
    print("加载注意力分数 ...")
    # 1400例样本的
    samples = pd.read_csv(f"{HBB_ATTN_DIR}/metadata.csv", index_col=0)
    hap1_attn_matrix_collapsed = pd.read_csv(f"{HBB_ATTN_DIR}/hap1_attention_collapsed.csv", index_col=0)
    hap2_attn_matrix_collapsed = pd.read_csv(f"{HBB_ATTN_DIR}/hap2_attention_collapsed.csv", index_col=0)
    
    valid_cols = [col for col in hap1_attn_matrix_collapsed.columns if not hap1_attn_matrix_collapsed[col].isnull().all()]
    hap1_attn_matrix_collapsed = hap1_attn_matrix_collapsed[valid_cols]
    hap2_attn_matrix_collapsed = hap2_attn_matrix_collapsed[valid_cols]

    # 纯合样本的
    samples_homo = pd.DataFrame()
    hap1_attn_matrix_collapsed_homo = pd.DataFrame()
    hap2_attn_matrix_collapsed_homo = pd.DataFrame()
    if HBB_HOMO_ATTN_DIR is not None:
        samples_homo = pd.read_csv(f"{HBB_HOMO_ATTN_DIR}/metadata.csv", index_col=0)
        hap1_attn_matrix_collapsed_homo = pd.read_csv(f"{HBB_HOMO_ATTN_DIR}/hap1_attention_collapsed.csv", index_col=0)
        hap2_attn_matrix_collapsed_homo = pd.read_csv(f"{HBB_HOMO_ATTN_DIR}/hap2_attention_collapsed.csv", index_col=0)
        
    valid_cols = [col for col in hap1_attn_matrix_collapsed_homo.columns if not hap1_attn_matrix_collapsed_homo[col].isnull().all()]
    hap1_attn_matrix_collapsed_homo = hap1_attn_matrix_collapsed_homo[valid_cols]
    hap2_attn_matrix_collapsed_homo = hap2_attn_matrix_collapsed_homo[valid_cols]

    samples = pd.concat([samples, samples_homo], ignore_index=False)
    hap1 = pd.concat([hap1_attn_matrix_collapsed, hap1_attn_matrix_collapsed_homo], ignore_index=False)
    hap2 = pd.concat([hap2_attn_matrix_collapsed, hap2_attn_matrix_collapsed_homo], ignore_index=False)

    # reorder the samples 
    # 纯合 3->0, 没病携带 0->1, 轻症 1->2, 重症 2->3
    labelmap = {3:0, 0:1, 1:2, 2:3}
    samples["sample_type_reorder"] = samples["sample_type"].apply(lambda x: labelmap[x])
    
    print(samples)
    has_enough_sample = (samples.value_counts("sample_type_reorder") > args.min_n).all()
    if not has_enough_sample:
        SKIP_DEL = True
        LFC_QUANTILE_THRESH = None
        PADJ_THRESH = None
    
    # h1, h2混合
    # combined, del_dosage = combine_h1_h2(hap1, hap2)
    
    
    # --------------------------- 计算组间差异对组内差异的大小 --------------------------- #
    print("计算组间差异对组内差异的大小 ...")
    samples = samples.loc[hap1.index]
    groups = samples["sample_type_reorder"]
    hap1_scores = compute_separability_scores(hap1, groups, min_n=MIN_N, dropna=True)

    samples = samples.loc[hap2.index]
    groups = samples["sample_type_reorder"]
    hap2_scores = compute_separability_scores(hap2, groups, min_n=MIN_N, dropna=True)

    # samples = samples.loc[combined.index]
    # groups = samples["sample_type_reorder"]
    # combined_scores = compute_separability_scores(combined, groups, min_n=10, dropna=True)
    
    
    # --------------------------- 差异attention分析 --------------------------- #
    print("差异attention分析 ...")
    samples = samples.loc[hap1.index]
    hap2 = hap2.loc[hap1.index]
    groups = samples["sample_type_reorder"]
    res_hap1_cont, res_hap1_del = differential_attention(hap1, groups, group_pairs=PAIRS, skip_del=SKIP_DEL)
    res_hap2_cont, res_hap2_del = differential_attention(hap2, groups, group_pairs=PAIRS, skip_del=SKIP_DEL)
    
    # samples = samples.loc[combined.index]
    # groups = samples["sample_type_reorder"]
    # res_combined_cont, res_combined_del = differential_attention(combined, groups)
    
    # 显著删除位点的结果
    if not SKIP_DEL:
        res_hap1_del = res_hap1_del[(res_hap1_del.padj < 0.01) & (res_hap1_del.groupA == 1)][["groupA", "groupB", "position", "OR_deleted", "padj"]]
        res_hap2_del = res_hap2_del[(res_hap2_del.padj < 0.01) & (res_hap2_del.groupA == 1)][["groupA", "groupB", "position", "OR_deleted", "padj"]]
        res_hap1_del.to_csv(f"{RESULTS_DIR}/hap1_deletion_enrichment_significant.csv", index=False)
        res_hap2_del.to_csv(f"{RESULTS_DIR}/hap2_deletion_enrichment_significant.csv", index=False)
    
    
    # --------------------------- 差异attention分析结果与真值对比 --------------------------- #
    print("差异attention分析结果与真值对比 ...")
    hap1_align_results = pd.DataFrame()
    for gA, gB in PAIRS:
        res_combined_cont_roi = diff_attention_with_truth(
            res_hap1_cont,
            true_roi,
            groupA=gA,
            groupB=gB,
            adjp_threshold=PADJ_THRESH,
            lfc_quantile=LFC_QUANTILE_THRESH,
            ignore_pos=IGNORE_POS,
            sort_col="log2FC",
            pos_col="position",
            truth_pos_col="pos",
            window=10,
            topk=20
        ).reset_index(drop=True)
        res_combined_cont_roi["order"] = res_combined_cont_roi.index
        res_combined_cont_roi = res_combined_cont_roi[res_combined_cont_roi.ID.notnull()]
        res_combined_cont_roi = res_combined_cont_roi[["groupA", "groupB", "order", "position", "log2FC", "delta", "padj", "ID", "cpra", "distance"]]
        hap1_align_results = pd.concat([hap1_align_results, res_combined_cont_roi], ignore_index=False)
        
    hap2_align_results = pd.DataFrame()
    for gA, gB in PAIRS:
        res_combined_cont_roi = diff_attention_with_truth(
            res_hap2_cont,
            true_roi,
            groupA=gA,
            groupB=gB,
            adjp_threshold=PADJ_THRESH,
            lfc_quantile=LFC_QUANTILE_THRESH,
            ignore_pos=IGNORE_POS,
            sort_col="log2FC",
            pos_col="position",
            truth_pos_col="pos",
            window=10,
            topk=20
        ).reset_index(drop=True)
        res_combined_cont_roi["order"] = res_combined_cont_roi.index
        res_combined_cont_roi = res_combined_cont_roi[res_combined_cont_roi.ID.notnull()]
        res_combined_cont_roi = res_combined_cont_roi[["groupA", "groupB", "order", "position", "log2FC", "delta", "padj", "ID", "cpra", "distance"]]
        hap2_align_results = pd.concat([hap2_align_results, res_combined_cont_roi], ignore_index=False)
            
    pd.merge(
        hap1_align_results[["groupA", "groupB", "position", "log2FC", "ID", "cpra", "distance"]],
        hap2_align_results[["groupA", "groupB", "position", "log2FC", "ID", "cpra", "distance"]],
        how="outer", on=["groupA", "groupB", "position", "ID", "cpra", "distance"],
        suffixes=["_hap1", "_hap2"]
    )[["groupA", "groupB", "position", "ID", "cpra", "distance", "log2FC_hap1", "log2FC_hap2"]]\
        .to_csv(f"{RESULTS_DIR}/hap1_hap2_differential_attention_8true_alignment.csv", index=False)
    
    # --------------------------- 绘制各个碱基位置上的log2FC与真值 --------------------------- #
    print("绘制各个碱基位置上的log2FC与真值 ...")
    pairs = PAIRS  
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap1_cont,
        score_df=hap1_scores,
        truth_df=true_roi,                 # true_roi 需含列 pos，若含 ID 将显示
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          # 绝对log2FC分位阈值
        padj_thresh=PADJ_THRESH,                  # FDR阈值
        ignore_pos=IGNORE_POS,
        cohen_thresh=0,
        truth_fontsize=10,                 # 真值ID更大字号
        unify_ylim=False,                    # 可选：所有图统一y轴范围
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap1_log2FC_with_8truth_{gA}vs{gB}.png", dpi=300)
        
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap2_cont,
        score_df=hap2_scores,
        truth_df=true_roi,                 
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          
        padj_thresh=PADJ_THRESH,     
        ignore_pos=IGNORE_POS,             
        cohen_thresh=0,
        truth_fontsize=10,                 
        unify_ylim=False,                    
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap2_log2FC_with_8truth_{gA}vs{gB}.png", dpi=300)
        
        
    # --------------------------- 准备用于UCSC的数据 --------------------------- #
    print("准备用于UCSC的数据 ...")
    for group_pair in PAIRS:
        gA, gB = group_pair
        export_attention_bigwig(res_hap1_cont, group_pair=group_pair, out_path=f"{UCSC_DIR}/hap1_log2fc_{gA}vs{gB}.bw", abs=False)
        export_attention_bigwig(res_hap2_cont, group_pair=group_pair, out_path=f"{UCSC_DIR}/hap2_log2fc_{gA}vs{gB}.bw", abs=False)