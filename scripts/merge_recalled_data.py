#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并召回数据为训练集
支持 ShareGPT 和 Alpaca 格式
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_recalled_data(recalled_dir: str) -> List[Dict]:
    """加载所有召回的数据"""
    recalled_path = Path(recalled_dir)
    all_data = []
    
    for file_path in recalled_path.glob("recalled_*.jsonl"):
        print(f"加载: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = [json.loads(line) for line in f if line.strip()]
            all_data.extend(file_data)
            print(f"  样本数: {len(file_data)}")
    
    return all_data


def convert_to_sharegpt(item: Dict) -> Dict:
    """
    转换为 ShareGPT 格式
    """
    # 如果已经是 ShareGPT 格式
    if "conversations" in item:
        return {
            "conversations": item["conversations"],
            "source": item.get("recall_source", "recalled"),
            "similarity": item.get("similarity_score", 0)
        }
    
    # Alpaca 格式转换
    if "instruction" in item:
        conversations = []
        
        # Human 消息
        human_msg = item["instruction"]
        if "input" in item and item["input"]:
            human_msg = f"{item['instruction']}\n{item['input']}"
        
        conversations.append({
            "from": "human",
            "value": human_msg
        })
        
        # GPT 消息
        if "output" in item:
            conversations.append({
                "from": "gpt",
                "value": item["output"]
            })
        
        return {
            "conversations": conversations,
            "source": item.get("recall_source", "recalled"),
            "similarity": item.get("similarity_score", 0)
        }
    
    # 问答格式转换
    if "question" in item:
        conversations = [
            {"from": "human", "value": item["question"]}
        ]
        
        answer_key = None
        for key in ["response", "answer", "response_chosen"]:
            if key in item:
                answer_key = key
                break
        
        if answer_key:
            conversations.append({
                "from": "gpt",
                "value": item[answer_key]
            })
        
        return {
            "conversations": conversations,
            "source": item.get("recall_source", "recalled"),
            "similarity": item.get("similarity_score", 0)
        }
    
    # 无法转换
    return None


def load_additional_dataset(dataset_spec: str) -> List[Dict]:
    """
    加载额外数据集
    格式: dataset_name:num_samples
    例如: shibing624/sharegpt_gpt4:10000
    """
    if ":" in dataset_spec:
        dataset_name, num_samples = dataset_spec.split(":")
        num_samples = int(num_samples)
    else:
        dataset_name = dataset_spec
        num_samples = None
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("警告: 无法导入 datasets，跳过额外数据集")
        return []
    
    print(f"加载额外数据集: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    if num_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    return list(dataset)


def merge_recalled_data(
    input_dir: str,
    output_file: str,
    format_type: str = "sharegpt",
    add_original: bool = False,
    additional_datasets: List[str] = None,
    shuffle: bool = True,
    deduplicate: bool = True
):
    """
    合并召回数据为训练集
    """
    # 加载召回数据
    print("加载召回数据...")
    recalled_data = load_recalled_data(input_dir)
    print(f"召回数据总数: {len(recalled_data)}")
    
    # 转换格式
    print(f"\n转换为 {format_type} 格式...")
    converted_data = []
    
    for item in tqdm(recalled_data, desc="格式转换"):
        if format_type == "sharegpt":
            converted = convert_to_sharegpt(item)
            if converted:
                converted_data.append(converted)
        else:
            # 保持原格式
            converted_data.append(item)
    
    print(f"转换成功: {len(converted_data)} 条")
    
    # 添加额外数据集
    if additional_datasets:
        print("\n添加额外数据集...")
        for dataset_spec in additional_datasets:
            additional_data = load_additional_dataset(dataset_spec)
            
            # 转换格式
            for item in tqdm(additional_data, desc=f"转换 {dataset_spec}"):
                if format_type == "sharegpt":
                    converted = convert_to_sharegpt(item)
                    if converted:
                        converted["source"] = "additional"
                        converted_data.append(converted)
                else:
                    converted_data.append(item)
    
    # 去重
    if deduplicate:
        print("\n去重处理...")
        seen = set()
        deduplicated_data = []
        
        for item in converted_data:
            # 使用对话内容作为去重键
            if "conversations" in item:
                key = json.dumps(item["conversations"], ensure_ascii=False, sort_keys=True)
            else:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            
            if key not in seen:
                seen.add(key)
                deduplicated_data.append(item)
        
        print(f"去重: {len(converted_data)} -> {len(deduplicated_data)}")
        converted_data = deduplicated_data
    
    # 打乱
    if shuffle:
        print("\n打乱数据...")
        random.seed(42)
        random.shuffle(converted_data)
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计信息
    print(f"\n✓ 合并完成！")
    print(f"  最终样本数: {len(converted_data)}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 按来源统计
    if format_type == "sharegpt":
        source_stats = {}
        for item in converted_data:
            source = item.get("source", "unknown")
            source_stats[source] = source_stats.get(source, 0) + 1
        
        print("\n来源统计:")
        for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
    
    # 样本预览
    print("\n数据样本预览:")
    for i, item in enumerate(converted_data[:2], 1):
        print(f"\n--- 样本 {i} ---")
        print(json.dumps(item, ensure_ascii=False, indent=2)[:500])
        print("...")


def main():
    parser = argparse.ArgumentParser(description="合并召回数据为训练集")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="召回数据目录"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "alpaca", "original"],
        help="输出格式"
    )
    parser.add_argument(
        "--add_original",
        type=bool,
        default=False,
        help="是否添加原始医疗数据"
    )
    parser.add_argument(
        "--additional_datasets",
        type=str,
        nargs="+",
        default=None,
        help="额外数据集 (格式: dataset_name:num_samples)"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="是否打乱数据"
    )
    parser.add_argument(
        "--deduplicate",
        type=bool,
        default=True,
        help="是否去重"
    )
    
    args = parser.parse_args()
    
    merge_recalled_data(
        input_dir=args.input_dir,
        output_file=args.output_file,
        format_type=args.format,
        add_original=args.add_original,
        additional_datasets=args.additional_datasets,
        shuffle=args.shuffle,
        deduplicate=args.deduplicate
    )
    
    print("\n✅ 合并完成！")


if __name__ == "__main__":
    main()
