#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# export PATH=/root/miniconda3/bin:$PATH

# conda activate vllm


# python 02_atten_score_extract.py \
#   --model_path "/mnt/zzb/default/Workspace/zhangjunyang/GeneRice/output/HuggingFace/AgriGenome_4n8a_1.2b_8k_pt_stage1_iter9000_hf_tp4pp1/" \
#   --input_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/json_blocks_awnTrue" \
#   --output_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/output_awnTrue" \
#   --batch_size 1


python 02_atten_score_extract.py \
  --model_path "/mnt/zzb/default/Workspace/zhangjunyang/GeneRice/output/HuggingFace/AgriGenome_4n8a_1.2b_8k_pt_stage1_iter9000_hf_tp4pp1/" \
  --input_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/json_blocks_GLTrue" \
  --output_dir "/mnt/zzb/default/Workspace/guoyafei/appTest/output_GLTrue" \
  --batch_size 1
