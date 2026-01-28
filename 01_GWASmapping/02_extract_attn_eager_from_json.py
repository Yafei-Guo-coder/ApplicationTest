import torch
import json
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Use eager attention + hook to extract attention scores from JSON files"
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--file_pattern", type=str, default="*.json")
    parser.add_argument("--output_suffix", type=str, default="_attn")

    parser.add_argument("--keep_original_sequence", action="store_true")
    parser.add_argument("--original_seq_field", type=str, default="original_sequence")

    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- load model ----------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="eager",   # 👈 第一种方法的关键
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # ---------- attention hook ----------
    captured_attentions = {}

    def attn_hook(module, inputs, outputs):
        # outputs = (attn_output, attn_weights)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            captured_attentions["attn"] = outputs[1].detach().cpu()

    # 默认抓最后一层 self-attention
    target_layer = model.model.layers[-1].self_attn
    hook_handle = target_layer.register_forward_hook(attn_hook)

    # ---------- process ----------
    json_files = list(Path(args.input_dir).glob(args.file_pattern))
    print(f"Found {len(json_files)} files")

    for jf in json_files:
        print(f"\nProcessing {jf.name}")
        out_path = Path(args.output_dir) / f"{jf.stem}{args.output_suffix}.json"

        dataset = load_dataset("json", data_files=str(jf), split="all")
        samples = list(dataset)

        processed = []

        for i in tqdm(range(0, len(samples), args.batch_size)):
            batch = samples[i:i + args.batch_size]
            seqs = [s["sequence"] for s in batch]

            inputs = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            attn = captured_attentions["attn"]  # [B, H, L, L]

            for b_idx, sample in enumerate(batch):
                # mean over heads, then sum over source positions
                # shape: [L]
                attn_vec = attn[b_idx].mean(dim=0).sum(dim=0)

                new_sample = sample.copy()

                if args.keep_original_sequence:
                    new_sample[args.original_seq_field] = sample["sequence"]

                new_sample["sequence"] = attn_vec.float().tolist()
                processed.append(new_sample)

            captured_attentions.clear()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        print(f"Saved -> {out_path.name}")

    hook_handle.remove()
    print("\nAll done!")


if __name__ == "__main__":
    main()
