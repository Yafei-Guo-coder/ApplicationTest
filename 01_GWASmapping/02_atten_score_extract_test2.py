import torch
import json
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="批量处理JSON文件，计算注意力权重并替换sequence字段")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入JSON文件目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出JSON文件目录路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--file_pattern", type=str, default="*.json", help="文件匹配模式")
    parser.add_argument("--output_suffix", type=str, default="_processed", help="输出文件后缀")
    parser.add_argument("--keep_original_sequence", action="store_true", help="保留原始sequence字段")
    parser.add_argument("--original_seq_field", type=str, default="original_sequence", help="原始序列字段名")
    parser.add_argument("--debug", action="store_true", help="开启详细调试信息")
    parser.add_argument("--debug_samples", type=int, default=3, help="详细打印前N个样本的处理过程")
    
    return parser.parse_args()

def print_debug(msg, level="INFO", debug_mode=True):
    if debug_mode:
        print(f"[{level}] {msg}")

def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("注意力分数提取流程")
    print("=" * 80)
    
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取JSON文件
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob(args.file_pattern))
    
    if not json_files:
        print(f"❌ 错误: 在 {args.input_dir} 中未找到匹配文件")
        return
    
    print(f"\n✅ 找到 {len(json_files)} 个文件")
    
    captured_attentions = {}

    def get_attention_hook(name: str):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) == 2:
                attn_weights = outputs[1]  # [B, H, L, L]
                captured_attentions[name] = attn_weights.detach().cpu()
                if args.debug:
                    print_debug(f"🎯 捕获注意力权重: shape={attn_weights.shape}")
        return hook

    # 加载模型
    print("\n📦 加载模型...")
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
    print(f"✅ 模型加载成功, 层数: {len(model.model.layers)}")

    # 注册最后一层注意力钩子
    target_layer = model.model.layers[-1].self_attn
    hook_handle = target_layer.register_forward_hook(get_attention_hook("last_self_attn"))
    print(f"✅ 注意力钩子已注册到最后一层 (layer {len(model.model.layers)-1})")

    def process_single_with_debug(sample, device, sample_idx=0, is_debug=False):
        ref_seq = sample["sequence"]
        
        if is_debug:
            print("\n" + "="*80)
            print(f"🔍 处理第 {sample_idx+1} 个样本")
            print(f"📝 原始序列长度: {len(ref_seq)}")
            print(f"序列前50个字符: {ref_seq[:50]}...")
        
        inputs = tokenizer(ref_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        attn_weights = captured_attentions["last_self_attn"]
        attn_weights = attn_weights.float()  # ✅ 统一转换为float32

        if is_debug:
            print(f"�� 原始注意力矩阵 shape: {attn_weights.shape}")
            # 展示第一个头的前10x10矩阵
            first_head = attn_weights[0, 0, :10, :10].numpy()
            print("第1个注意力头前10x10矩阵:")
            print(first_head)
        
        # 平均所有head
        attn_avg_heads = attn_weights.mean(dim=1)  # [B, L, L]
        
        # 对每个token求和，得到每个token的“重要性”
        ref_attn = attn_avg_heads[0].sum(dim=0)  # [L]
        
        new_sample = sample.copy()
        if args.keep_original_sequence:
            new_sample[args.original_seq_field] = sample["sequence"]
        new_sample["sequence"] = ref_attn.cpu().numpy().tolist()
        
        return new_sample

    def process_batch(samples, device):
        sequences = [s["sequence"] for s in samples]
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_attn = captured_attentions["last_self_attn"].float()
        processed_samples = []
        for i, sample in enumerate(samples):
            attn_weights = batch_attn[i:i+1]
            ref_attn = attn_weights.mean(dim=1)[0].sum(dim=0)
            new_sample = sample.copy()
            if args.keep_original_sequence:
                new_sample[args.original_seq_field] = sample["sequence"]
            new_sample["sequence"] = ref_attn.cpu().numpy().tolist()
            processed_samples.append(new_sample)
        return processed_samples

    # 处理每个文件
    total_samples = 0
    processed_files = 0

    for file_idx, json_file in enumerate(json_files):
        output_file = Path(args.output_dir) / f"{json_file.stem}{args.output_suffix}.json"
        print(f"\n📁 处理文件 [{file_idx+1}/{len(json_files)}]: {json_file.name}")

        dataset = load_dataset("json", data_files=str(json_file), split="all")
        all_samples = list(dataset)
        print(f"✅ 加载成功，包含 {len(all_samples)} 个样本")
        
        processed_samples = []
        if args.batch_size > 1:
            for i in tqdm(range(0, len(all_samples), args.batch_size), desc="批次处理"):
                batch = all_samples[i:i+args.batch_size]
                processed_samples.extend(process_batch(batch, device))
                captured_attentions.clear()
        else:
            for i, sample in enumerate(tqdm(all_samples, desc="处理样本")):
                is_debug = args.debug and i < args.debug_samples
                processed_samples.append(process_single_with_debug(sample, device, i, is_debug))
                captured_attentions.clear()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_samples, f, ensure_ascii=False, indent=2)
        
        total_samples += len(processed_samples)
        processed_files += 1
        print(f"✅ 完成! 处理 {len(processed_samples)} 个样本")

    hook_handle.remove()
    print(f"\n🎉 全部处理完成! 共处理 {total_samples} 个样本, 输出 {processed_files} 个文件")

    summary = {
        "total_files_found": len(json_files),
        "files_processed": processed_files,
        "total_samples_processed": total_samples,
        "batch_size": args.batch_size,
        "device": args.device
    }

    summary_file = Path(args.output_dir) / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📊 处理摘要保存到: {summary_file}")


if __name__ == "__main__":
    main()
