#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从PLINK2格式(pgen/psam/pvar)生成样本一致性序列
支持SNP、Indel(插入/缺失)、多等位变异
"""

import os
import json
import numpy as np
import pandas as pd
import pysam
from pgenlib import PgenReader
from tqdm import tqdm

# =====================
# 路径配置
# =====================
bed_file = "GAD1.bed"
pheno_file = "/mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/2_3K_Rice_pheno"
# pgen_prefix = "/mnt/zzb/default/Workspace/guoyafei/riceData/rice4k_geno_add_del"  # .pgen/.psam/.pvar的共同前缀
pgen_prefix = "RICE_RP_GAD1"
fasta_file = "/mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/GCA_001433935.1_IRGSP-1.0_genomic.fna.gz"

target_pheno_col = "awn_presence"
out_dir = "json_blocks_APTrue_with_indel"
os.makedirs(out_dir, exist_ok=True)

# =====================
# 染色体映射
# =====================
chrom_map = {
    '1': 'AP014957.1', '2': 'AP014958.1', '3': 'AP014959.1',
    '4': 'AP014960.1', '5': 'AP014961.1', '6': 'AP014962.1',
    '7': 'AP014963.1', '8': 'AP014964.1', '9': 'AP014965.1',
    '10': 'AP014966.1', '11': 'AP014967.1', '12': 'AP014968.1'
}

# =====================
# 1. 读取表型
# =====================
print("➡️ 读取表型数据 ...")
pheno = pd.read_csv(pheno_file, sep="\t")
pheno = pheno.dropna(subset=[target_pheno_col])
pheno = pheno.set_index("SampleID")
print(f"✅ 有效表型样本数: {len(pheno)}")

# =====================
# 2. 读取BED区间
# =====================
print("➡️ 读取BED文件 ...")
bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
print(f"✅ BED 区间数: {len(bed_df)}")

# =====================
# 3. 读取FASTA参考序列
# =====================
print("➡️ 打开FASTA文件 ...")
fasta = pysam.FastaFile(fasta_file)

# =====================
# 4. 读取PLINK2文件
# =====================
print("➡️ 读取PLINK2文件 ...")

# 4.1 读取.psam (样本信息)
psam_file = f"{pgen_prefix}.psam"
print(f"  读取: {psam_file}")

# 先检查文件格式
with open(psam_file, 'r') as f:
    first_line = f.readline().strip()
    print(f"  PSAM首行: {first_line}")

# 读取PSAM，保留#开头的列名
psam = pd.read_csv(psam_file, sep="\t")
print(f"  原始列名: {psam.columns.tolist()}")

# 灵活处理列名（可能是IID, #IID, 或FID IID格式）
possible_id_cols = ['IID', '#IID', 'iid', '#iid']
sample_col = None

for col in possible_id_cols:
    if col in psam.columns:
        sample_col = col
        break

# 如果还是找不到，尝试第一列或第二列
if sample_col is None:
    if len(psam.columns) >= 2:
        # PLINK2格式通常是: FID IID 或 #FID IID
        # 使用第二列作为样本ID
        sample_col = psam.columns[1]
        print(f"  ⚠️  未找到标准IID列，使用第2列: {sample_col}")
    else:
        # 使用第一列
        sample_col = psam.columns[0]
        print(f"  ⚠️  未找到标准IID列，使用第1列: {sample_col}")

sample_ids = psam[sample_col].tolist()
print(f"✅ PSAM样本数: {len(sample_ids)}")
print(f"  使用列: {sample_col}")
print(f"  示例样本: {sample_ids[:5]}")

# 4.2 读取.pvar (变异信息)
pvar_file = f"{pgen_prefix}.pvar"
print(f"  读取: {pvar_file}")

# 计算需要跳过的元数据行数（##开头的行）
skip_rows = 0
with open(pvar_file, 'r') as f:
    for line in f:
        if line.startswith('##'):
            skip_rows += 1
        else:
            print(f"  PVAR数据首行: {line.strip()}")
            break

print(f"  跳过元数据行: {skip_rows}")

# 读取PVAR，跳过##开头的行
pvar = pd.read_csv(pvar_file, sep="\t", skiprows=skip_rows)
print(f"  原始列名: {pvar.columns.tolist()}")

# 标准化列名（移除#）
pvar.columns = pvar.columns.str.replace('#', '')
print(f"  标准化后列名: {pvar.columns.tolist()}")

# 检查必需列
required_cols = ['CHROM', 'POS', 'REF', 'ALT']
missing_cols = [col for col in required_cols if col not in pvar.columns]
if missing_cols:
    raise ValueError(f"PVAR文件缺少必需列: {missing_cols}\n可用列: {pvar.columns.tolist()}")

print(f"✅ PVAR变异数: {len(pvar)}")

# 显示前几行
print("\n  PVAR前3行:")
display_cols = [c for c in ['CHROM', 'POS', 'ID', 'REF', 'ALT'] if c in pvar.columns]
print(pvar.head(3)[display_cols].to_string())

# 4.3 打开.pgen (基因型数据)
pgen_file = f"{pgen_prefix}.pgen"
print(f"  打开: {pgen_file}")

# 安全地打开pgen
try:
    pgen_reader = PgenReader(bytes(pgen_file, 'utf8'))
    n_variants = pgen_reader.get_variant_ct()
    n_samples = pgen_reader.get_raw_sample_ct()
    print(f"✅ PGEN: {n_variants} 变异, {n_samples} 样本")
except Exception as e:
    print(f"❌ 无法打开PGEN文件: {e}")
    print("\n可能的解决方案:")
    print("1. 检查pgen文件是否完整")
    print("2. 尝试重新生成pgen文件: plink2 --bfile xxx --make-pgen --out xxx")
    print("3. 更新pgenlib: pip install --upgrade pgenlib")
    exit(1)

# 验证PVAR和PGEN一致
if len(pvar) != n_variants:
    print(f"⚠️  警告: PVAR变异数({len(pvar)}) != PGEN变异数({n_variants})")
    print("   使用较小的数值")
    n_variants = min(len(pvar), n_variants)

# =====================
# 5. 样本过滤和映射
# =====================
print("\n" + "="*60)
print("样本匹配检查")
print("="*60)

# 表型样本
pheno_sample_set = set(pheno.index)
print(f"表型样本数: {len(pheno_sample_set)}")
print(f"表型样本示例: {list(pheno_sample_set)[:5]}")

# 基因型样本
geno_sample_set = set(sample_ids)
print(f"\n基因型样本数: {len(geno_sample_set)}")
print(f"基因型样本示例: {sample_ids[:5]}")

# 检查匹配
common_samples_set = pheno_sample_set & geno_sample_set
print(f"\n直接匹配的样本数: {len(common_samples_set)}")

# 如果没有匹配,尝试诊断原因
if len(common_samples_set) == 0:
    print("\n⚠️  警告: 没有匹配的样本!")
    print("\n可能的原因:")
    print("1. 样本ID格式不同")
    
    # 检查ID格式差异
    pheno_sample = list(pheno_sample_set)[0]
    geno_sample = sample_ids[0]
    
    print(f"\n表型样本ID示例: '{pheno_sample}' (类型: {type(pheno_sample).__name__})")
    print(f"基因型样本ID示例: '{geno_sample}' (类型: {type(geno_sample).__name__})")
    
    # 尝试各种转换
    print("\n尝试ID转换...")
    
    # 尝试1: 去除前缀/后缀
    if any('_' in str(s) for s in list(pheno_sample_set)[:10]):
        print("  表型ID包含下划线,可能需要分割")
    
    if any('_' in str(s) for s in sample_ids[:10]):
        print("  基因型ID包含下划线,可能需要分割")
    
    # 尝试2: 字符串vs数字
    try:
        pheno_as_int = set(int(s) if str(s).isdigit() else s for s in pheno_sample_set)
        geno_as_int = set(int(s) if str(s).isdigit() else s for s in geno_sample_set)
        match_as_int = len(pheno_as_int & geno_as_int)
        if match_as_int > 0:
            print(f"  ✓ 转换为整数后匹配: {match_as_int} 个样本")
            # 应用转换
            pheno.index = pheno.index.map(lambda x: int(x) if str(x).isdigit() else x)
            common_samples_set = set(pheno.index) & geno_sample_set
    except:
        pass
    
    # 尝试3: 去除空格
    pheno_stripped = set(str(s).strip() for s in pheno_sample_set)
    geno_stripped = set(str(s).strip() for s in geno_sample_set)
    match_stripped = len(pheno_stripped & geno_stripped)
    if match_stripped > 0:
        print(f"  ✓ 去除空格后匹配: {match_stripped} 个样本")
        pheno.index = pheno.index.map(lambda x: str(x).strip())
        common_samples_set = pheno_stripped & geno_stripped
    
    # 尝试4: 检查是否有共同前缀/后缀模式
    if len(common_samples_set) == 0:
        print("\n  建议:")
        print("  1. 检查表型文件的'SampleID'列格式")
        print("  2. 检查PSAM文件的IID列格式")
        print("  3. 确保两者使用相同的样本命名规则")
        print("\n  退出程序,请修正样本ID不匹配问题")
        exit(1)

# 找到既在表型中又在基因型中的样本
pheno_samples = [s for s in pheno.index if s in geno_sample_set]
sample_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
pheno_idx = np.array([sample_to_idx[s] for s in pheno_samples])
labels = pheno.loc[pheno_samples, target_pheno_col].values

print(f"\n✅ 最终匹配样本数: {len(pheno_samples)}")
if len(pheno_samples) > 0:
    print(f"示例样本: {pheno_samples[:5]}")
    print(f"标签示例: {labels[:5]}")
else:
    print("❌ 错误: 没有匹配的样本,无法继续")
    exit(1)

# =====================
# 6. 辅助函数
# =====================
def classify_variant(ref, alt):
    """
    判断变异类型
    返回: 'SNP', 'INS' (insertion), 'DEL' (deletion), 'COMPLEX'
    """
    ref = str(ref).upper()
    alt = str(alt).upper()
    
    len_ref = len(ref)
    len_alt = len(alt)
    
    if len_ref == 1 and len_alt == 1:
        return 'SNP'
    elif len_ref < len_alt:
        return 'INS'  # 插入
    elif len_ref > len_alt:
        return 'DEL'  # 缺失
    else:
        return 'COMPLEX'


def apply_variant_to_sequence(ref_seq, variant_list):
    """
    将变异应用到参考序列
    
    参数:
        ref_seq: 参考序列(字符串)
        variant_list: [(offset, variant_type, ref, alt), ...]
                     offset: 相对于ref_seq起始的位置(0-based)
    
    返回:
        修改后的序列(字符串)
    """
    # 按位置从后往前排序,避免插入/删除影响后续位置
    variant_list = sorted(variant_list, key=lambda x: x[0], reverse=True)
    
    seq = list(ref_seq)
    
    for offset, var_type, ref, alt in variant_list:
        if offset < 0 or offset >= len(seq):
            continue
        
        if var_type == 'SNP':
            # 简单替换
            seq[offset] = alt
            
        elif var_type == 'INS':
            # 插入: 在offset位置后插入额外碱基
            # 例: REF=A, ALT=ATT, 插入TT
            # offset位置的碱基替换为alt[0], 然后插入alt[1:]
            seq[offset] = alt[0]
            insert_bases = alt[1:]
            for i, base in enumerate(insert_bases):
                seq.insert(offset + 1 + i, base)
            
        elif var_type == 'DEL':
            # 缺失: 替换为N
            # 例: REF=ATT, ALT=A, 缺失2个碱基
            del_len = len(ref) - len(alt)
            # 保留alt[0]在offset位置, 后续del_len个位置用N替换
            seq[offset] = alt[0] if len(alt) > 0 else 'N'
            for i in range(1, len(ref)):
                if offset + i < len(seq):
                    seq[offset + i] = 'N'
        
        elif var_type == 'COMPLEX':
            # 复杂变异: 简单处理为替换+填充N
            seq[offset] = alt[0] if len(alt) > 0 else 'N'
            if len(ref) > 1:
                for i in range(1, len(ref)):
                    if offset + i < len(seq):
                        seq[offset + i] = 'N'
    
    return ''.join(seq)


# =====================
# 7. 主循环: 逐block处理
# =====================
print("\n" + "="*60)
print("开始处理blocks...")
print("="*60)

for block_id, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Processing blocks"):
    chrom = str(row['chrom'])
    start = int(row['start'])
    end = int(row['end'])
    block_name = f"block_{block_id+1}"
    
    # 跳过不在映射表中的染色体
    if chrom not in chrom_map:
        print(f"⚠️  跳过 {block_name}: 染色体 {chrom} 不在映射表中")
        continue
    
    # 提取参考序列
    try:
        ref_seq = fasta.fetch(chrom_map[chrom], start, end).upper()
    except Exception as e:
        print(f"⚠️  跳过 {block_name}: FASTA提取失败 - {e}")
        continue
    
    # 筛选该block内的变异
    mask = (
        (pvar['CHROM'].astype(str) == chrom) &
        (pvar['POS'] >= start + 1) &
        (pvar['POS'] <= end)
    )
    var_idx = np.where(mask)[0]
    
    if len(var_idx) == 0:
        print(f"⚠️  跳过 {block_name}: 无变异")
        continue
    
    # 获取变异信息
    variants_in_block = pvar.iloc[var_idx].copy()
    var_positions = variants_in_block['POS'].values
    var_refs = variants_in_block['REF'].values
    var_alts = variants_in_block['ALT'].values
    
    print(f"\n📍 {block_name}: {len(var_idx)} 变异")
    
    # 逐样本生成序列
    json_list = []
    
    # 预分配基因型数组（避免重复分配）
    geno_array = np.empty(n_samples, dtype=np.int32)
    
    for i, sample in enumerate(tqdm(pheno_samples, desc=f"  {block_name} 样本", leave=False)):
        sample_idx = pheno_idx[i]
        
        # 该样本在此block的所有变异列表
        sample_variants = []
        
        for j, global_var_idx in enumerate(var_idx):
            try:
                # 读取该变异的基因型
                # 重要: 确保索引在有效范围内
                if global_var_idx < 0 or global_var_idx >= n_variants:
                    print(f"\n  ⚠️  变异索引越界: {global_var_idx} (总变异数: {n_variants})")
                    continue
                
                if sample_idx < 0 or sample_idx >= n_samples:
                    print(f"\n  ⚠️  样本索引越界: {sample_idx} (总样本数: {n_samples})")
                    continue
                
                # 读取基因型
                # 使用read_alleles_range可能更安全
                pgen_reader.read(global_var_idx, geno_array)
                geno = geno_array[sample_idx]
                
            except Exception as e:
                print(f"\n  ✗ 读取基因型出错 (变异{j}, 全局索引{global_var_idx}): {e}")
                geno = -9  # 设为missing
            
            # 获取该变异的REF和ALT
            ref = str(var_refs[j]).upper()
            alt_str = str(var_alts[j]).upper()
            
            # 跳过特殊标记
            if alt_str in ['DEL', '<DEL>', '*']:
                # 这是纯缺失标记,不是实际序列
                alt_str = ref[0] if len(ref) > 0 else 'N'
            
            # 处理多等位基因 (ALT可能是 "A,T,G" 或 "A,DEL" 这种格式)
            alts = [a.strip() for a in alt_str.split(',')]
            # 过滤掉DEL标记
            alts = [a for a in alts if a not in ['DEL', '<DEL>', '*']]
            
            if len(alts) == 0:
                # 没有有效的ALT
                continue
            
            # 根据基因型选择等位基因
            if geno == -9 or geno < 0:
                # missing genotype: 用N填充
                applied_allele = 'N' * len(ref)
            elif geno == 0:
                # REF/REF: 不修改(参考序列已经是REF)
                continue
            elif geno == 1:
                # REF/ALT: 使用第一个ALT (杂合一般显示ALT)
                applied_allele = alts[0] if len(alts) > 0 else ref
            elif geno == 2:
                # ALT/ALT: 使用第一个ALT
                applied_allele = alts[0] if len(alts) > 0 else ref
            else:
                # 其他编码(如多等位的复杂编码),简化处理
                allele_idx = min(geno - 1, len(alts) - 1)
                applied_allele = alts[allele_idx] if allele_idx >= 0 else ref
            
            # 计算相对参考序列的offset (0-based)
            offset = var_positions[j] - start - 1
            
            # 判断变异类型
            var_type = classify_variant(ref, applied_allele)
            
            sample_variants.append((offset, var_type, ref, applied_allele))
        
        # 应用所有变异到参考序列
        try:
            consensus_seq = apply_variant_to_sequence(ref_seq, sample_variants)
        except Exception as e:
            print(f"\n  ✗ 样本 {sample} 序列生成失败: {e}")
            consensus_seq = ref_seq  # 使用参考序列
        
        json_list.append({
            "label": int(labels[i]),
            "spec": sample,
            "loc": block_name,
            "sequence": consensus_seq
        })
    
    # 输出JSON
    out_path = os.path.join(out_dir, f"{block_name}.json")
    with open(out_path, "w") as f:
        json.dump(json_list, f, indent=2)
    
    print(f"✅ {block_name} 完成 | 变异数={len(var_idx)} | 样本数={len(pheno_samples)}")

# =====================
# 8. 清理
# =====================
fasta.close()
pgen_reader.close()

print("\n" + "="*60)
print("🎉 全部完成!")
print(f"输出目录: {out_dir}")
print("="*60)
