import torch
import json
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量处理JSON文件，计算注意力权重并替换sequence字段")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入JSON文件目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出JSON文件目录路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小，默认为1")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--file_pattern", type=str, default="*.json", help="文件匹配模式")
    parser.add_argument("--output_suffix", type=str, default="_processed", help="输出文件后缀")
    parser.add_argument("--keep_original_sequence", action="store_true", help="保留原始sequence字段")
    parser.add_argument("--original_seq_field", type=str, default="original_sequence", help="原始序列字段名")
    parser.add_argument("--debug", action="store_true", help="开启详细调试信息")
    parser.add_argument("--debug_samples", type=int, default=3, help="详细打印前N个样本的处理过程")
    
    return parser.parse_args()

def print_debug(msg, level="INFO", debug_mode=True):
    """打印调试信息"""
    if debug_mode:
        print(f"[{level}] {msg}")

def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("注意力分数提取流程")
    print("=" * 80)
    print("参数配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob(args.file_pattern))
    
    if not json_files:
        print(f"❌ 错误: 在 {args.input_dir} 中未找到匹配文件")
        return
    
    print(f"\n✅ 找到 {len(json_files)} 个文件")
    
    # 全局变量用于存储注意力权重
    captured_attentions = {}
    
    def get_attention_hook(name: str):
        """返回一个 hook 函数，用于捕获指定模块的输出"""
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) == 2:
                attn_weights = outputs[1]  # [B, H, L, L]
                captured_attentions[name] = attn_weights.detach().cpu()
                if args.debug:
                    print_debug(f"  🎯 捕获注意力权重: shape={attn_weights.shape}")
        return hook
    
    # 加载模型
    print("\n" + "="*80)
    print("📦 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        device = torch.device(args.device)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        
        print(f"✅ 模型加载成功")
        print(f"   设备: {device}")
        print(f"   模型层数: {len(model.model.layers)}")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    # 注册钩子（捕获最后一层的注意力）
    try:
        target_layer = model.model.layers[-1].self_attn
        hook_handle = target_layer.register_forward_hook(get_attention_hook("last_self_attn"))
        print(f"✅ 注意力钩子已注册到最后一层 (layer {len(model.model.layers)-1})")
    except Exception as e:
        print(f"❌ 注册钩子失败: {e}")
        return
    
    def process_single_with_debug(sample, device, sample_idx=0, is_debug=False):
        """单样本处理（带调试信息）"""
        ref_seq = sample["sequence"]
        
        if is_debug:
            print("\n" + "="*80)
            print(f"🔍 详细处理第 {sample_idx+1} 个样本")
            print("="*80)
            print(f"📝 原始数据:")
            print(f"   样本ID: {sample.get('spec', 'N/A')}")
            print(f"   标签: {sample.get('label', 'N/A')}")
            print(f"   位置: {sample.get('loc', 'N/A')}")
            print(f"   序列长度: {len(ref_seq)}")
            print(f"   序列前50个字符: {ref_seq[:50]}...")
        
        # Step 1: Tokenization
        inputs = tokenizer(ref_seq, return_tensors="pt")
        
        if is_debug:
            print(f"\n📊 Step 1: Tokenization")
            print(f"   input_ids shape: {inputs['input_ids'].shape}")
            print(f"   序列被分成 {inputs['input_ids'].shape[1]} 个tokens")
            print(f"   前10个token IDs: {inputs['input_ids'][0, :10].tolist()}")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Step 2: 模型前向传播
        if is_debug:
            print(f"\n🧠 Step 2: 模型前向传播")
            print(f"   将tokens输入模型...")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        if is_debug:
            print(f"   ✓ 模型输出完成")
            print(f"   logits shape: {outputs.logits.shape}")
        
        # Step 3: 获取注意力权重
        attn_weights = captured_attentions["last_self_attn"]
        
        if is_debug:
            print(f"\n🎯 Step 3: 注意力权重")
            print(f"   原始注意力矩阵 shape: {attn_weights.shape}")
            print(f"   解释:")
            print(f"     - 维度0 (Batch): {attn_weights.shape[0]} (批次大小)")
            print(f"     - 维度1 (Heads): {attn_weights.shape[1]} (注意力头数)")
            print(f"     - 维度2 (Query): {attn_weights.shape[2]} (查询序列长度)")
            print(f"     - 维度3 (Key): {attn_weights.shape[3]} (键序列长度)")
            print(f"\n   注意力矩阵含义:")
            print(f"     attn[i,j,k,l] = 第i个样本、第j个注意力头、")
            print(f"                      第k个token对第l个token的注意力分数")
            
            # 🔥 新增：展示单个注意力头的矩阵
            print(f"\n   📊 第1个注意力头的矩阵预览 (前10x10):")
            first_head = attn_weights[0, 0, :10, :10].cpu().float().numpy()
            print(f"   行=Query位置, 列=Key位置, 值=注意力分数")
            print(f"   (每行之和=1.0，因为是softmax后的概率分布)")
            print()
            print("        ", end="")
            for col in range(10):
                print(f"  K{col:2d}  ", end="")
            print()
            for row in range(10):
                print(f"   Q{row:2d}  ", end="")
                for col in range(10):
                    print(f"{first_head[row, col]:6.3f}", end=" ")
                print()
            
            # 🔥 验证每行和
            row_sums = attn_weights[0, 0].sum(dim=1)[:10].cpu().float().numpy()
            print(f"\n   ✓ 验证: 前10行的行和 (应该都约等于1.0):")
            print(f"   {row_sums}")
            
            # 🔥 上三角 vs 下三角分析
            print(f"\n   🔺 上三角 vs 下三角分析:")
            print(f"   在自回归模型(如GPT)中:")
            print(f"     - 下三角 (包括对角线): token只能看到自己和之前的token")
            print(f"     - 上三角: token看未来的token (通常被mask掉=0)")
            
            full_matrix = attn_weights[0, 0].cpu().float().numpy()
            # 下三角和 (包括对角线)
            import numpy as np
            lower_tri = np.tril(full_matrix)
            upper_tri = np.triu(full_matrix, k=1)  # k=1排除对角线
            
            lower_sum = lower_tri.sum()
            upper_sum = upper_tri.sum()
            total_sum = full_matrix.sum()
            
            print(f"     - 下三角和 (包括对角线): {lower_sum:.3f} ({lower_sum/total_sum*100:.1f}%)")
            print(f"     - 上三角和 (不含对角线): {upper_sum:.3f} ({upper_sum/total_sum*100:.1f}%)")
            print(f"     - 总和: {total_sum:.3f}")
            
            if upper_sum < 0.01:
                print(f"   ✓ 检测到因果注意力mask (上三角≈0)")
            else:
                print(f"   ✓ 双向注意力 (无mask)")
            
            # 🔥 对角线分析
            diagonal = np.diag(full_matrix)
            print(f"\n   📍 对角线分析 (token对自己的注意力):")
            print(f"     - 对角线平均值: {diagonal.mean():.4f}")
            print(f"     - 对角线最大值: {diagonal.max():.4f}")
            print(f"     - 对角线最小值: {diagonal.min():.4f}")
            print(f"     - 前10个对角线值: {diagonal[:10]}")
        
        # Step 4: 平均注意力头
        attn_avg_heads = attn_weights.mean(dim=1)  # [B, L, L] -> [1, L, L]
        
        if is_debug:
            print(f"\n📈 Step 4: 平均所有注意力头")
            print(f"   操作: attn_weights.mean(dim=1)")
            print(f"   平均后 shape: {attn_avg_heads.shape}")
            print(f"   解释: 将 {attn_weights.shape[1]} 个注意力头的结果取平均")
            print(f"   现在矩阵[k,l] = 第k个token对第l个token的平均注意力")
            
            # 🔥 平均后的矩阵预览
            print(f"\n   📊 平均后的矩阵预览 (前10x10):")
            avg_matrix = attn_avg_heads[0, :10, :10].cpu().float().numpy()
            print("        ", end="")
            for col in range(10):
                print(f"  K{col:2d}  ", end="")
            print()
            for row in range(10):
                print(f"   Q{row:2d}  ", end="")
                for col in range(10):
                    print(f"{avg_matrix[row, col]:6.3f}", end=" ")
                print()
        
        # Step 5: 对每个token求和（获得每个token接收的总注意力）
        ref_attn = attn_avg_heads[0].sum(dim=0)  # [L, L] -> [L]
        
        if is_debug:
            print(f"\n➕ Step 5: 对每个token求和")
            print(f"   操作: attn_avg_heads[0].sum(dim=0)")
            print(f"   求和后 shape: {ref_attn.shape}")
            print(f"   解释: 对每一列求和，得到每个token接收的总注意力")
            print(f"\n   📐 理解这个求和:")
            print(f"   原矩阵: 每行代表一个Query token看其他token的注意力")
            print(f"   列求和: 统计每个Key token被多少个Query关注")
            print(f"   结果: 一个一维向量，表示每个token的'重要性'")
            
            # 🔥 手动展示第0列的求和过程
            col_0_values = attn_avg_heads[0, :, 0].cpu().numpy()
            print(f"\n   🔍 示例: 第0个token的总注意力 = 第0列之和")
            print(f"   第0列的值 (所有token对第0个token的注意力):")
            print(f"   {col_0_values[:10]}... (前10个)")
            print(f"   求和 = {col_0_values.sum():.6f}")
            print(f"   验证: ref_attn[0] = {ref_attn[0].item():.6f} ✓")
            
            print(f"\n   🎯 最终注意力分数:")
            print(f"     - 向量长度: {len(ref_attn)}")
            print(f"     - 最小值: {ref_attn.min().item():.6f}")
            print(f"     - 最大值: {ref_attn.max().item():.6f}")
            print(f"     - 平均值: {ref_attn.mean().item():.6f}")
            print(f"     - 前10个值: {ref_attn[:10].tolist()}")
            
            # 可视化注意力分布
            print(f"\n   📊 注意力分数分布:")
            scores = ref_attn.cpu().float().numpy()
            import numpy as np
            percentiles = [0, 25, 50, 75, 100]
            for p in percentiles:
                val = np.percentile(scores, p)
                print(f"     {p}th percentile: {val:.6f}")
        
        # Step 6: 创建新样本
        new_sample = sample.copy()
        
        if args.keep_original_sequence:
            new_sample[args.original_seq_field] = new_sample["sequence"]
        
        # 替换sequence字段为注意力分数
        new_sample["sequence"] = ref_attn.cpu().float().numpy().tolist()
        
        if is_debug:
            print(f"\n✅ Step 6: 保存结果")
            print(f"   原始序列长度: {len(ref_seq)}")
            print(f"   注意力向量长度: {len(new_sample['sequence'])}")
            if args.keep_original_sequence:
                print(f"   原始序列保存在字段: '{args.original_seq_field}'")
            print("="*80)
        
        return new_sample
    
    def process_batch(samples, device):
        """批量处理样本"""
        sequences = [sample["sequence"] for sample in samples]
        
        print(f"\n  批量处理 {len(sequences)} 个样本...")
        
        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if args.debug:
            print(f"  Tokenized shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_attentions = captured_attentions["last_self_attn"]
        
        if args.debug:
            print(f"  批量注意力 shape: {batch_attentions.shape}")
        
        processed_samples = []
        
        for i, sample in enumerate(samples):
            attn_weights = batch_attentions[i:i+1]
            
            try:
                ref_attn = attn_weights.mean(dim=1)[0].sum(dim=0)
                new_sample = sample.copy()
                
                if args.keep_original_sequence:
                    new_sample[args.original_seq_field] = new_sample["sequence"]
                
                new_sample["sequence"] = ref_attn.cpu().float().numpy().tolist()
                processed_samples.append(new_sample)
                
            except Exception as e:
                print(f"  ❌ 处理批次中样本 {i} 时出错: {e}")
                processed_samples.append(sample)
        
        return processed_samples
    
    # 处理每个文件
    total_samples = 0
    processed_files = 0
    
    for file_idx, json_file in enumerate(json_files):
        output_file = Path(args.output_dir) / f"{json_file.stem}{args.output_suffix}.json"
        
        print(f"\n{'='*80}")
        print(f"📁 处理文件 [{file_idx+1}/{len(json_files)}]: {json_file.name}")
        print(f"📤 输出到: {output_file.name}")
        print("="*80)
        
        try:
            dataset = load_dataset("json", data_files=str(json_file), split="all")
            all_samples = list(dataset)
            
            print(f"✅ 加载成功，包含 {len(all_samples)} 个样本")
            
            processed_samples = []
            
            if args.batch_size > 1:
                # 批量处理
                for i in tqdm(range(0, len(all_samples), args.batch_size), 
                            desc="批次处理", unit="batch"):
                    batch = all_samples[i:i + args.batch_size]
                    processed_batch = process_batch(batch, device)
                    processed_samples.extend(processed_batch)
                    captured_attentions.clear()
            else:
                # 单样本处理
                for i, sample in enumerate(tqdm(all_samples, desc="处理样本", unit="sample")):
                    # 前N个样本详细打印
                    is_debug = args.debug and i < args.debug_samples
                    processed_sample = process_single_with_debug(sample, device, i, is_debug)
                    processed_samples.append(processed_sample)
                    captured_attentions.clear()
            
            # 保存结果
            print(f"\n💾 保存处理结果...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_samples, f, ensure_ascii=False, indent=2)
            
            samples_processed = len(processed_samples)
            total_samples += samples_processed
            processed_files += 1
            
            print(f"✅ 完成! 处理了 {samples_processed} 个样本")
            
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 清理
    hook_handle.remove()
    
    print("\n" + "="*80)
    print("🎉 全部处理完成!")
    print("="*80)
    print(f"✅ 成功处理文件: {processed_files}/{len(json_files)}")
    print(f"✅ 总共处理样本: {total_samples}")
    print(f"📂 输出目录: {args.output_dir}")
    print("="*80)
    
    # 保存摘要
    summary = {
        "total_files_found": len(json_files),
        "files_processed": processed_files,
        "total_samples_processed": total_samples,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    
    summary_file = Path(args.output_dir) / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"📊 处理摘要保存到: {summary_file}")

if __name__ == "__main__":
    main()
