# for i in ALK.20k ALK.8k wx.20k wx.8k
# do
for i in ALK.8k
do
# for j in first middle last
# do
for j in last
do
# for i in wx.8k
# do
# for j in short long
# do
# for rep in {1..6}
# do
source /root/miniforge3/etc/profile.d/conda.sh
conda activate opencompass
python /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/01_generate_jsonl_indel_rice.py \
   --bed /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/simulation/genotype/${i}.bed \
   --pheno /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/simulation/genotype/pheno.200.txt \
   --vcf /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/simulation/genotype/${i}.${j}.vcf.gz \
   --fasta /mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/osa1_r7.asm.ch.fa.gz \
   --pheno-col Trait \
   --out json_${i}/snp${j}
   
echo "step 1 is finished!!!!!!!!!!!!"

conda activate vllm 

python /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/demo02_calc_flash_attention_run.py  \
    --model_path /mnt/zzb/default/Workspace/xz/hf/rice_1B_stage2_8k_hf \
    --input_dir json_${i}/snp${j} \
    --output_dir output_${i}/snp${j} \
    --block_cols 256 \
    --output_suffix _attn \
    --file_pattern "block_*.json"

echo  "step2 is ok!!!!!!!!!!!!!!!!!!"
conda activate opencompass

python /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/03_atten_score_indel_normalizeV2.py   \
        --json_dir  output_${i}/snp${j}  \
        --seq_json_dir json_${i}/snp${j} \
        --bed_file /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/simulation/genotype/${i}.bed  \
        --vcf_file /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/simulation/genotype/${i}.${j}.vcf.gz \
        --fasta_file /mnt/zzb/default/Public/OsGenos/Oryza_sativa/chromosome/GCA_001433935.1_IRGSP-1.0_genomic.fna.gz \
        --output_dir output_${i}/snp${j}/norm
echo "step 3 is ok !!!!!!!!!!!!"

python /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/04_convert_json_to_matrix_perblock.py  \
    --json_dir  output_${i}/snp${j}/norm \
    --bed_file  /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/0119Test/simulation/genotype/${i}.bed \
    --output_dir attention_${i}/snp${j}

echo "step 4 is ok !!!!!!!!!!!!"

python /mnt/zzb/default/Workspace/guoyafei/appTest/Waxy/05_differential_analysis_rice_per_block.py \
    --input_dir attention_${i}/snp${j} \
    --output_dir diff_${i}/snp${j} \
    --haplotype hap1 \
    --group_a 0 \
    --group_b 1

done 
done
# done


