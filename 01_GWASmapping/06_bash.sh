# python plot_phenotype.py /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/3_XieWB_pheno/1_Panicle_Architecture_2013HN 
# python plot_phenotype.py /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/3_XieWB_pheno/1_Panicle_Architecture_2014WH  
# python plot_phenotype.py /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/3_XieWB_pheno/3_Yield_Traits
# python plot_phenotype.py /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/3_XieWB_pheno/1_Panicle_Architecture_2013WH  
# python plot_phenotype.py /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/3_XieWB_pheno/2_Stigma_Traits                
# python plot_phenotype.py /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/3_XieWB_pheno/4_FlagLeafAngle
# awk '{if($10 < 0.001) print $1"\t"$3"\t"$10}' /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/GWAS_summary_statistics/2_3kRice/MAF.3K_Rice_pheno.awn_presence.fastGWA | awk '{print $1"\t"$2-1"\t"$2"\t"$3}' > awn.p3.bed
# bedtools merge -i awn.p3.bed -d 8000 -c 4 -o count,collapse> awn.p3.merge.bed
# python split_expand_bed.py awn.p3.merge.bed awn.p3.merge.expand.bed
# awk '{if($5 !="NA") print $1"\t"$1}' /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/phenotyping_data/2_3K_Rice_pheno | sed '1d' > keep_382.txt

###################################### 处理基因型文件 #####################################

# plink2 \
#   --bfile /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/genotyping_data/2_3K_rice_and_7_RiceDiversityPanel/RICE_RP_mLIDs \
#   --keep keep_382.txt \
#   --maf 0.0001 \
#    --extract bed1 awn.p3.merge.expand.bed \
#   --make-bed \
#   --out RICE_RP_382

# plink2 \
#   --bfile /mnt/zzb/default/Workspace/Rice-Genome/application/GWAS_fine_mapping/RiceGWAScohort/genotyping_data/2_3K_rice_and_7_RiceDiversityPanel/RICE_RP_mLIDs \
#   --extract bed1 GAD1.bed \
#   --make-bed \
#   --out RICE_RP_GAD1

# plink2 \
#   --pfile /mnt/zzb/default/Workspace/guoyafei/riceData/rice4k_geno_add_del \
#   --extract bed0 GAD1.bed \
#   --make-pgen \
#   --out RICE_RP_GAD1  
# sed  -i 's/,DEL//' RICE_RP_GAD1.pvar

# plink2 --pfile RICE_RP_GAD1 \
#        --export vcf \
#        --out RICE_RP_GAD1


################################## 提取注意力分数 ############################################
#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# export PATH=/root/miniconda3/bin:$PATH

# conda activate vllm


# python 02_atten_score_extract.py \
#   --model_path "/mnt/zzb/default/Workspace/zhangjunyang/GeneRice/output/HuggingFace/AgriGenome_4n8a_1.2b_8k_pt_stage1_iter9000_hf_tp4pp1/" \
#   --input_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/json_blocks_awnTrue" \
#   --output_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/output_awnTrue" \
#   --batch_size 1


# python 02_atten_score_extract.py \
#   --model_path "/mnt/zzb/default/Workspace/zhangjunyang/GeneRice/output/HuggingFace/AgriGenome_4n8a_1.2b_8k_pt_stage1_iter9000_hf_tp4pp1/" \
#   --input_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/json_blocks_GLTrue" \
#   --output_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/output_GLTrue" \
#   --batch_size 1

# python 02_atten_score_extract.py \
#   --model_path "/mnt/zzb/default/Workspace/xz/hf/rice_1B_1M_hf" \
#   --input_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/json_blocks_APTrue_with_indel_vcf" \
#   --output_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/output_APTrue_with_indel" \
#   --batch_size 1

# python 02_atten_score_extract_plink2_indel_normalize.py \
#     --json_dir /mnt/zzb/default/Workspace/guoyafei/appTest/output_APTrue_with_indel \   # 原始attention JSON
#     --seq_json_dir  /mnt/zzb/default/Workspace/guoyafei/appTest/json_blocks_APTrue_with_indel_vcf\   # 样本序列JSON
#     --bed_file GAD1.bed \
#     --pvar_prefix RICE_RP_GAD1 \
#     --fasta_file /mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/GCA_001433935.1_IRGSP-1.0_genomic.fna.gz \
#     --output_dir /mnt/zzb/default/Workspace/guoyafei/appTest/output_APTrue_with_indel_normalized

################################## 做注意力分数的结果汇总 #################################

# 基本用法 - 每个block独立输出
# conda activate opencompass
# python convert_json_to_matrix_perblock.py \
#     --json_dir ./output \
#     --bed_file awn.p3.merge.expand.bed \
#     --output_dir ./attention_matrices_per_block


python 04_convert_json_to_matrix_perblock.py \
    --json_dir ./output_APTrue_with_indel\
    --bed_file GAD1.bed \
    --output_dir ./attention_matrices_per_block_APTrue_with_indel

# python 04_convert_json_to_matrix_perblock.py \
#     --json_dir ./output_GLTrue \
#     --bed_file GS3.bed \
#     --output_dir ./attention_matrices_per_block_GL

# 同时生成合并所有block的矩阵
# python convert_json_to_matrix_perblock.py \
#     --json_dir ./attention_jsons \
#     --bed_file awn.p3.merge.expand.bed \
#     --output_dir ./attention_matrices_per_block \
#     --merge_all

# 带表型文件
# python differential_analysis_rice_per_block.py \
#     --json_dir ./attention_jsons \
#     --bed_file awn.p3.merge.expand.bed \
#     --output_dir ./attention_matrices_per_block \
#     --phenotype_file phenotype.txt \
#     --merge_all

# python convert_json_to_matrix_perblock.py \
#     --json_dir ./output_awnTrue \
#     --bed_file GAD1.bed \
#     --output_dir ./attention_matrices_per_block_awnTrue

#################################### 做分组差异检测 ################################

# 基本用法
# python differential_analysis_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block \
#     --output_dir ./diff_results_all_blocks \
#     --haplotype hap1 \
#     --group_a 1 \
#     --group_b 2

# 完整参数
# python differential_analysis_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block \
#     --output_dir ./diff_results_all_blocks \
#     --haplotype hap1 \
#     --group_a 1 \
#     --group_b 5 \
#     --min_samples 5 \
#     --padj_thresh 0.05 \
#     --fc_thresh 1

# python 05_differential_analysis_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./diff_results_all_blocks_GL_Japonica \
#     --haplotype hap1 \
#     --group_a 1 \
#     --group_b 2 \
#     --min_samples 5 \
#     --padj_thresh 0.05 \
#     --fc_thresh 1


# python 05_differential_analysis_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_APTrue_with_indel_normalized \
#     --output_dir ./diff_results_all_blocks_APTrue_with_indel_normalized \
#     --haplotype hap1 \
#     --group_a 1 \
#     --group_b 2 \
#     --min_samples 5 \
#     --padj_thresh 0.05 \
#     --fc_thresh 1

# python differential_analysis_rice_per_block_V2.py \
#     --input_dir ./attention_matrices_per_block_awnTrue \
#     --output_dir ./diff_results_all_blocks_awnTrue_4 \
#     --haplotype hap1 \
#     --group_a 1 \
#     --group_b 2 \
#     --min_samples 5 \
#     --padj_thresh 0.05 \
#     --fc_thresh 1

# python 05_differential_analysis_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./diff_results_all_blocks_awnTrue_4 \
#     --haplotype hap1 \
#     --group_a 1 \
#     --group_b 2 \
#     --min_samples 5 \
#     --padj_thresh 0.05 \
#     --fc_thresh 1

# 只处理特定blocks
# python batch_differential_analysis.py \
#     --input_dir ./attention_matrices_per_block \
#     --output_dir ./diff_results_all_blocks \
#     --block_pattern "block_1,block_5,block_10"

###################################### 表型分组处理 ################################

# awk -F'\t' 'BEGIN{OFS="\t"}
# {
#     group = $3
#     if ($3=="Indica I" || $3=="Indica II" || $3=="Indica III")
#         group = "Indica"
#     else if ($3=="Temperate Japonica" || $3=="Tropical Japonica")
#         group = "Japonica"
#     print $1, $2, $3, $4, $5, group
# }' sampleNameMap.txt > sampleNameMap.withGroup.txt

# awk -F'\t' 'BEGIN{OFS="\t"}
# {
#     group = $3
#     if ($3=="Indica I" || $3=="Indica II" || $3=="Indica III")
#         group = "Indica"
#     else if ($3=="Temperate Japonica" || $3=="Tropical Japonica")
#         group = "Japonica"
#     else if ($3=="Indica Intermediate" || $3=="Intermediate" || $3=="Japonica Intermediate")
#         group = "Intermediate"

#     print $1, $2, $3, $4, $5, group
# }' sampleNameMap.txt > sampleNameMap.withGroup.txt

################################# linear regression analysis #############################

# 分析产量(yield_kg)与attention的关联
# python 08_linear_regression_phenotype_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./regression_results_GL \
#     --phenotype_col sample_type \
#     --haplotype hap1 \
#     --min_samples 10

# 分析株高(height_cm)
# python linear_regression_phenotype.py \
#     --input_dir ./attention_matrices_per_block \
#     --output_dir ./regression_results_height \
#     --phenotype_col height_cm \
#     --haplotype hap1

# 只分析特定blocks
# python linear_regression_phenotype.py \
#     --input_dir ./attention_matrices_per_block \
#     --output_dir ./regression_results \
#     --phenotype_col yield_kg \
#     --block_pattern "block_1,block_5,block_10"

# python 08_linear_regression_phenotype_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./regression_moderate_GL_Japonica \
#     --phenotype_col sample_type \
#     --sig_level moderate

# 使用严格标准(筛选关键位点)
# python 08_linear_regression_phenotype_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./regression_strict_GL_Japonica \
#     --phenotype_col sample_type \
#     --sig_level strict

# 使用Bonferroni(最保守)
# python 08_linear_regression_phenotype_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./regression_bonferroni_GL_Japonica \
#     --phenotype_col sample_type \
#     --sig_level bonferroni \
#     --min_samples 10 

# 完整参数示例
# python 08_linear_regression_phenotype_rice_per_block.py \
#     --input_dir ./attention_matrices_per_block_GL \
#     --output_dir ./regression_results_GL \
#     --phenotype_col sample_type \
#     --haplotype hap1 \
#     --min_samples 10 \
#     --padj_thresh 0.05 \
#     --sig_level strict \
#     --block_pattern "block_1,block_2,block_3"
