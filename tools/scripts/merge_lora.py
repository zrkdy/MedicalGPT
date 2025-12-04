#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并LoRA权重到基础模型
使用场景：将训练好的LoRA适配器合并到基础模型，便于部署
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_path, lora_model_path, output_dir):
    """合并LoRA权重到基础模型"""
    print("="*60)
    print("LoRA 权重合并工具")
    print("="*60)
    
    print(f"\n1. 加载基础模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   ✓ 基础模型加载完成")
    
    print(f"\n2. 加载LoRA适配器: {lora_model_path}")
    model = PeftModel.from_pretrained(model, lora_model_path)
    print(f"   ✓ LoRA适配器加载完成")
    
    print(f"\n3. 合并权重...")
    model = model.merge_and_unload()
    print(f"   ✓ 权重合并完成")
    
    print(f"\n4. 保存合并后的模型: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"   ✓ 模型保存完成")
    
    print("\n" + "="*60)
    print("合并完成！")
    print(f"合并后的模型位置: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="合并LoRA权重到基础模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/merge_lora.py \\
        --base_model Qwen/Qwen2.5-3B-Instruct \\
        --lora_model outputs-sft-qwen2.5-3b/checkpoint-best \\
        --output_dir outputs-sft-qwen2.5-3b-merged
        """
    )
    parser.add_argument("--base_model", required=True, help="基础模型路径")
    parser.add_argument("--lora_model", required=True, help="LoRA模型路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    merge_lora(args.base_model, args.lora_model, args.output_dir)
