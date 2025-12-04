#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测模型在 CEval 医疗指标上的表现
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch


def load_model_and_tokenizer(model_path: str):
    """加载模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model.eval()
    
    return model, tokenizer


def format_question(item: Dict) -> str:
    """格式化问题"""
    question = item.get('question', '')
    options = []
    
    for key in ['A', 'B', 'C', 'D', 'E']:
        if key in item and item[key]:
            options.append(f"{key}. {item[key]}")
    
    formatted = f"{question}\n" + "\n".join(options) + "\n请选择正确答案(A/B/C/D/E):"
    return formatted


def extract_answer(response: str) -> str:
    """从模型回复中提取答案"""
    response = response.strip().upper()
    
    # 直接匹配单个字母
    for char in response:
        if char in ['A', 'B', 'C', 'D', 'E']:
            return char
    
    # 匹配 "答案是A" 类型
    for option in ['A', 'B', 'C', 'D', 'E']:
        if f"答案{option}" in response or f"选{option}" in response or f"是{option}" in response:
            return option
    
    return ""


def evaluate_dataset(
    model,
    tokenizer,
    eval_file: str,
    max_samples: int = None
) -> Dict:
    """评测单个数据集"""
    
    # 加载数据
    data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"\n评测数据集: {Path(eval_file).stem}")
    print(f"样本数: {len(data)}")
    
    # 评测
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(data, desc="评测进度"):
        question = format_question(item)
        correct_answer = item.get('answer', '').upper()
        
        # 构造提示
        messages = [
            {"role": "system", "content": "你是一个医学专家，请根据问题选择正确答案。"},
            {"role": "user", "content": question}
        ]
        
        # 生成回答
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.95
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            predicted_answer = extract_answer(response)
            
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": item.get('question', ''),
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response
            })
            
        except Exception as e:
            print(f"\n评测失败: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "dataset": Path(eval_file).stem,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser(description="评测模型")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="data/eval_benchmark",
        help="评测集目录"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results.json",
        help="结果输出文件"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个数据集的最大样本数"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # 评测所有数据集
    eval_dir = Path(args.eval_dir)
    eval_files = list(eval_dir.glob("*.jsonl"))
    
    all_results = []
    
    for eval_file in eval_files:
        result = evaluate_dataset(
            model,
            tokenizer,
            str(eval_file),
            args.max_samples
        )
        all_results.append(result)
    
    # 汇总结果
    summary = {
        "model_path": args.model_path,
        "results": all_results,
        "overall": {
            "total_correct": sum(r["correct"] for r in all_results),
            "total_samples": sum(r["total"] for r in all_results),
            "average_accuracy": sum(r["accuracy"] for r in all_results) / len(all_results) if all_results else 0
        }
    }
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)
    
    for result in all_results:
        print(f"\n{result['dataset']}:")
        print(f"  准确率: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    print(f"\n总体平均准确率: {summary['overall']['average_accuracy']:.2%}")
    print(f"总计: {summary['overall']['total_correct']}/{summary['overall']['total_samples']}")
    
    print(f"\n详细结果已保存: {args.output_file}")


if __name__ == "__main__":
    main()
