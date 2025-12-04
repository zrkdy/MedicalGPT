#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量化训练数据集
支持从 HuggingFace 或本地文件加载
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time


def get_embedding_model(model_name: str, api_key: str = None):
    """获取向量化模型"""
    if model_name == "glm-embedding-3":
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError("请安装 zhipuai: pip install zhipuai")
        
        if not api_key:
            api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 ZHIPUAI_API_KEY 环境变量")
        
        client = ZhipuAI(api_key=api_key)
        
        def embed_func(texts: List[str]) -> List[List[float]]:
            embeddings = []
            for text in texts:
                try:
                    response = client.embeddings.create(
                        model="embedding-3",
                        input=text[:8000],  # API限制
                    )
                    embeddings.append(response.data[0].embedding)
                    time.sleep(0.05)
                except Exception as e:
                    print(f"API错误: {e}")
                    embeddings.append([0.0] * 1024)
            return embeddings
        
        return embed_func, 1024
    
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers")
        
        model = SentenceTransformer(model_name)
        
        def embed_func(texts: List[str]) -> List[List[float]]:
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        
        return embed_func, model.get_sentence_embedding_dimension()


def load_dataset_from_hf(dataset_name: str, max_samples: Optional[int] = None) -> List[Dict]:
    """从 HuggingFace 加载数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")
    
    print(f"从 HuggingFace 加载数据集: {dataset_name}")
    
    # 特殊处理 shibing624/medical 数据集
    if dataset_name == "shibing624/medical":
        print("使用 Parquet 格式加载...")
        try:
            # 尝试从 parquet 文件加载
            dataset = load_dataset(
                "parquet",
                data_files={
                    "train": "hf://datasets/shibing624/medical/train-*.parquet"
                },
                split="train"
            )
        except Exception as e:
            print(f"Parquet 加载失败: {e}")
            print("尝试降级方法...")
            # 降级到旧版本方法
            try:
                dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
            except:
                # 最后尝试不带 trust_remote_code
                dataset = load_dataset(dataset_name, split="train")
    else:
        # 处理不同格式
        if "/" in dataset_name:
            dataset = load_dataset(dataset_name, split="train")
        else:
            dataset = load_dataset(dataset_name)
            if hasattr(dataset, 'train'):
                dataset = dataset['train']
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return list(dataset)


def load_dataset_from_file(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """从本地文件加载数据集"""
    print(f"从本地文件加载: {file_path}")
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    
    return data


def extract_text_from_item(item: Dict) -> str:
    """
    从数据项中提取文本
    支持多种格式: ShareGPT, Alpaca, 纯文本
    """
    # ShareGPT 格式
    if "conversations" in item:
        texts = []
        for conv in item["conversations"]:
            role = conv.get("from", conv.get("role", ""))
            value = conv.get("value", conv.get("content", ""))
            if value:
                texts.append(f"{role}: {value}")
        return " ".join(texts)
    
    # Alpaca 格式
    elif "instruction" in item:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            return f"{instruction} {input_text} {output}"
        return f"{instruction} {output}"
    
    # 问答格式
    elif "question" in item and "response" in item:
        return f"{item['question']} {item['response']}"
    
    elif "question" in item and "answer" in item:
        return f"{item['question']} {item['answer']}"
    
    # 纯文本格式
    elif "text" in item:
        return item["text"]
    
    # 其他格式
    else:
        # 尝试拼接所有文本字段
        return " ".join(str(v) for v in item.values() if isinstance(v, str))


def vectorize_training_dataset(
    dataset_source: str,
    output_file: str,
    model_name: str = "glm-embedding-3",
    api_key: str = None,
    max_samples: Optional[int] = None,
    batch_size: int = 100,
    resume_from: Optional[str] = None
):
    """
    向量化训练数据集
    """
    # 加载数据
    if dataset_source.endswith('.jsonl') or dataset_source.endswith('.json'):
        data = load_dataset_from_file(dataset_source, max_samples)
    else:
        data = load_dataset_from_hf(dataset_source, max_samples)
    
    print(f"总样本数: {len(data)}")
    
    # 获取向量化模型
    embed_func, dim = get_embedding_model(model_name, api_key)
    print(f"使用模型: {model_name}, 向量维度: {dim}")
    
    # 准备输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 断点续传
    start_idx = 0
    if resume_from and Path(resume_from).exists():
        print(f"从断点文件恢复: {resume_from}")
        with open(resume_from, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"已处理 {start_idx} 条，继续处理...")
        mode = 'a'
    else:
        mode = 'w'
    
    # 批量处理
    with open(output_file, mode, encoding='utf-8') as f:
        for i in tqdm(range(start_idx, len(data), batch_size), desc="向量化进度"):
            batch = data[i:i+batch_size]
            
            # 提取文本
            texts = []
            valid_items = []
            for item in batch:
                text = extract_text_from_item(item)
                if text.strip():
                    texts.append(text[:2000])  # 截断过长文本
                    valid_items.append(item)
            
            if not texts:
                continue
            
            # 获取向量
            try:
                embeddings = embed_func(texts)
            except Exception as e:
                print(f"\n批次 {i} 向量化失败: {e}")
                continue
            
            # 保存结果
            for item, text, embedding in zip(valid_items, texts, embeddings):
                vectorized_item = {
                    **item,
                    "extracted_text": text,
                    "embedding": embedding
                }
                f.write(json.dumps(vectorized_item, ensure_ascii=False) + '\n')
            
            # 定期保存进度
            if (i // batch_size) % 10 == 0:
                f.flush()
    
    print(f"\n✓ 向量化完成: {output_file}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="向量化训练数据集")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="HuggingFace数据集名称 或 本地文件路径"
    )
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument(
        "--model_name",
        type=str,
        default="glm-embedding-3",
        help="向量化模型"
    )
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--batch_size", type=int, default=100, help="批处理大小")
    parser.add_argument("--resume_from", type=str, default=None, help="断点续传文件")
    
    args = parser.parse_args()
    
    vectorize_training_dataset(
        dataset_source=args.dataset_name,
        output_file=args.output_file,
        model_name=args.model_name,
        api_key=args.api_key,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        resume_from=args.resume_from
    )
    
    print("\n✅ 向量化完成！")


if __name__ == "__main__":
    main()
