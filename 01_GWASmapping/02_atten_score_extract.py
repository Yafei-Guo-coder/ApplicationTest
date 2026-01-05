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
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入JSON文件目录路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出JSON文件目录路径"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小，默认为1（单样本处理）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备，默认为cuda"
    )
    
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.json",
        help="文件匹配模式，默认为*.json"
    )
    
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_processed",
        help="输出文件后缀，默认为_processed"
    )
    
    parser.add_argument(
        "--keep_original_sequence",
        action="store_true",
        help="保留原始sequence字段作为新字段"
    )
    
    parser.add_argument(
        "--original_seq_field",
        type=str,
        default="original_sequence",
        help="原始序列字段名（如果启用keep_original_sequence）"
    )
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_arguments()
    
    print("=" * 60)
    print("参数配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  输入目录: {args.input_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  计算设备: {args.device}")
    print(f"  文件模式: {args.file_pattern}")
    print(f"  输出后缀: {args.output_suffix}")
    print(f"  保留原始序列: {args.keep_original_sequence}")
    if args.keep_original_sequence:
        print(f"  原始序列字段名: {args.original_seq_field}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob(args.file_pattern))
    
    if not json_files:
        print(f"错误: 在 {args.input_dir} 中未找到匹配 {args.file_pattern} 的文件")
        return
    
    print(f"找到 {len(json_files)} 个匹配的文件")
    for file in json_files[:5]:  # 显示前5个文件
        print(f"  - {file.name}")
    if len(json_files) > 5:
        print(f"  ... 以及 {len(json_files) - 5} 个其他文件")
    
    # 全局变量用于存储注意力权重
    captured_attentions = {}
    
    def get_attention_hook(name: str):
        """返回一个 hook 函数，用于捕获指定模块的输出"""
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) == 2:
                attn_weights = outputs[1]  # [B, H, L, L]
                captured_attentions[name] = attn_weights.detach().cpu()
        return hook
    
    # 加载模型和tokenizer
    print("\n加载模型和tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        
        # 设置设备
        device = torch.device(args.device)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # model = torch.nn.DataParallel(model)
        model.to(device)
        model.eval()
        
        print(f"模型加载成功，已移至 {args.device}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 注册钩子
    try:
        target_layer = model.model.layers[-1].self_attn
        hook_handle = target_layer.register_forward_hook(get_attention_hook("last_self_attn"))
        print("注意力钩子注册成功")
    except Exception as e:
        print(f"注册钩子失败: {e}")
        return
    
    def process_batch(samples, device):
        """批量处理样本"""
        sequences = [sample["sequence"] for sample in samples]
        
        # 批量tokenize
        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192  # 可根据需要调整
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 处理注意力权重
        batch_attentions = captured_attentions["last_self_attn"]
        processed_samples = []
        
        for i, sample in enumerate(samples):
            # 获取对应样本的注意力权重
            attn_weights = batch_attentions[i:i+1]
            print(attn_weights.shape)
            try:
                # average among heads and summed to each base
                ref_attn = attn_weights.mean(dim=1)[0].sum(dim=0)
                
                # 创建新样本
                new_sample = sample.copy()
                
                # 如果要求保留原始序列
                if args.keep_original_sequence:
                    new_sample[args.original_seq_field] = new_sample["sequence"]
                
                # 用注意力权重替换sequence字段
                new_sample["sequence"] = ref_attn.cpu().float().numpy().tolist()
                processed_samples.append(new_sample)
                
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                # 保留原始样本（不替换sequence）
                processed_samples.append(sample)
        
        return processed_samples
    
    def process_single(sample, device):
        """单样本处理"""
        ref_seq = sample["sequence"]
        
        inputs = tokenizer(ref_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # average among heads and summed to each base
        ref_attn = captured_attentions["last_self_attn"].mean(dim=1)[0].sum(dim=0)
        
        # 创建新样本
        new_sample = sample.copy()
        
        # 如果要求保留原始序列
        if args.keep_original_sequence:
            new_sample[args.original_seq_field] = new_sample["sequence"]
        
        # 用注意力权重替换sequence字段
        new_sample["sequence"] = ref_attn.cpu().float().numpy().tolist()
        
        return new_sample
    
    # 处理每个文件
    total_samples = 0
    processed_files = 0
    
    for json_file in json_files:
        output_file = Path(args.output_dir) / f"{json_file.stem}{args.output_suffix}.json"
        
        print(f"\n{'='*40}")
        print(f"处理文件: {json_file.name}")
        print(f"输出到: {output_file.name}")
        
        try:
            # 加载数据集
            dataset = load_dataset("json", data_files=str(json_file), split="all")
            all_samples = list(dataset)
            
            print(f"文件包含 {len(all_samples)} 个样本")
            
            processed_samples = []
            
            # 根据批处理大小选择处理方式
            if args.batch_size > 1:
                # 批量处理
                for i in tqdm(range(0, len(all_samples), args.batch_size), 
                            desc="批次处理", unit="batch"):
                    batch = all_samples[i:i + args.batch_size]
                    processed_batch = process_batch(batch, device)
                    processed_samples.extend(processed_batch)
                    captured_attentions.clear()  # 清空以释放内存
            else:
                # 单样本处理
                for i, sample in enumerate(tqdm(all_samples, desc="处理样本", unit="sample")):
                    processed_sample = process_single(sample, device)
                    processed_samples.append(processed_sample)
                    captured_attentions.clear()
            
            # 保存结果
            print(f"保存处理结果...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_samples, f, ensure_ascii=False, indent=2)
            
            samples_processed = len(processed_samples)
            total_samples += samples_processed
            processed_files += 1
            
            print(f"✓ 完成处理 {json_file.name}, 处理了 {samples_processed} 个样本")
            
        except Exception as e:
            print(f"✗ 处理文件 {json_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 清理钩子
    hook_handle.remove()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"成功处理文件: {processed_files}/{len(json_files)}")
    print(f"总共处理样本: {total_samples}")
    print(f"输出目录: {args.output_dir}")
    
    # 保存处理摘要
    summary = {
        "model_path": args.model_path,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "total_files_found": len(json_files),
        "files_processed": processed_files,
        "total_samples_processed": total_samples,
        "batch_size": args.batch_size,
        "device": args.device,
        "output_suffix": args.output_suffix,
        "keep_original_sequence": args.keep_original_sequence,
        "original_seq_field": args.original_seq_field if args.keep_original_sequence else None,
        "processed_files": [f.name for f in json_files[:10]],  # 只记录前10个文件
    }
    
    summary_file = Path(args.output_dir) / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"处理摘要保存到: {summary_file}")

if __name__ == "__main__":
    main()
