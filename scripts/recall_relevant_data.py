#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量召回相关数据
基于评测集向量，从训练数据中召回最相关的样本
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict


def load_vectorized_data(file_path: str) -> List[Dict]:
    """加载向量化数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def recall_top_k(
    query_embedding: List[float],
    candidate_data: List[Dict],
    top_k: int = 50,
    similarity_threshold: float = 0.0
) -> List[Tuple[Dict, float]]:
    """
    召回 Top-K 最相关的数据
    返回: [(data_item, similarity_score), ...]
    """
    similarities = []
    
    for item in candidate_data:
        if "embedding" not in item:
            continue
        
        sim = cosine_similarity(query_embedding, item["embedding"])
        
        if sim >= similarity_threshold:
            similarities.append((item, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def recall_from_eval_set(
    eval_vectors_dir: str,
    train_vectors_file: str,
    output_dir: str,
    top_k: int = 50,
    similarity_threshold: float = 0.75,
    max_similarity: float = 0.99,
    dedup: bool = True
):
    """
    基于评测集向量召回训练数据
    """
    eval_dir = Path(eval_vectors_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载训练数据向量
    print(f"加载训练数据向量: {train_vectors_file}")
    train_data = load_vectorized_data(train_vectors_file)
    print(f"训练数据总数: {len(train_data)}")
    
    # 处理每个评测集
    eval_files = list(eval_dir.glob("*_vectorized.jsonl"))
    
    statistics = {}
    all_recalled_ids = set()  # 用于跨评测集去重
    
    for eval_file in eval_files:
        eval_name = eval_file.stem.replace("_vectorized", "")
        print(f"\n处理评测集: {eval_name}")
        
        # 加载评测集向量
        eval_data = load_vectorized_data(eval_file)
        print(f"评测问题数: {len(eval_data)}")
        
        # 召回数据
        recalled_data = []
        recalled_ids = set()
        similarity_scores = []
        
        for eval_item in tqdm(eval_data, desc=f"召回 {eval_name}"):
            if "embedding" not in eval_item:
                continue
            
            # 召回 Top-K
            top_k_results = recall_top_k(
                query_embedding=eval_item["embedding"],
                candidate_data=train_data,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            for train_item, sim_score in top_k_results:
                # 过滤过高相似度（可能是评测集泄露）
                if sim_score > max_similarity:
                    continue
                
                # 生成唯一ID用于去重
                item_id = json.dumps(train_item.get("extracted_text", ""), ensure_ascii=False)
                
                # 去重
                if dedup and item_id in recalled_ids:
                    continue
                
                if dedup:
                    recalled_ids.add(item_id)
                    all_recalled_ids.add(item_id)
                
                # 添加召回信息
                recalled_item = {
                    **train_item,
                    "recall_source": eval_name,
                    "similarity_score": float(sim_score),
                    "eval_question": eval_item.get("question", "")
                }
                
                # 移除向量（节省空间）
                recalled_item.pop("embedding", None)
                
                recalled_data.append(recalled_item)
                similarity_scores.append(sim_score)
        
        # 保存召回结果
        output_file = output_path / f"recalled_{eval_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in recalled_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 统计信息
        statistics[eval_name] = {
            "eval_questions": len(eval_data),
            "recalled_samples": len(recalled_data),
            "unique_samples": len(recalled_ids),
            "avg_similarity": float(np.mean(similarity_scores)) if similarity_scores else 0,
            "min_similarity": float(np.min(similarity_scores)) if similarity_scores else 0,
            "max_similarity": float(np.max(similarity_scores)) if similarity_scores else 0,
            "output_file": str(output_file)
        }
        
        print(f"✓ 召回完成: {len(recalled_data)} 条")
        print(f"  平均相似度: {statistics[eval_name]['avg_similarity']:.4f}")
        print(f"  相似度范围: [{statistics[eval_name]['min_similarity']:.4f}, {statistics[eval_name]['max_similarity']:.4f}]")
    
    # 保存统计信息
    statistics["summary"] = {
        "total_eval_sets": len(eval_files),
        "total_unique_recalled": len(all_recalled_ids),
        "parameters": {
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "max_similarity": max_similarity,
            "dedup": dedup
        }
    }
    
    stats_file = output_path / "recall_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 统计信息已保存: {stats_file}")
    print(f"\n总召回样本数（去重后）: {len(all_recalled_ids)}")


def main():
    parser = argparse.ArgumentParser(description="向量召回相关数据")
    parser.add_argument(
        "--eval_vectors",
        type=str,
        required=True,
        help="评测集向量目录"
    )
    parser.add_argument(
        "--train_vectors",
        type=str,
        required=True,
        help="训练数据向量文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="每个评测问题召回的样本数"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.75,
        help="相似度阈值（0-1）"
    )
    parser.add_argument(
        "--max_similarity",
        type=float,
        default=0.99,
        help="最大相似度（过滤评测集泄露）"
    )
    parser.add_argument(
        "--dedup",
        type=bool,
        default=True,
        help="是否去重"
    )
    
    args = parser.parse_args()
    
    recall_from_eval_set(
        eval_vectors_dir=args.eval_vectors,
        train_vectors_file=args.train_vectors,
        output_dir=args.output_dir,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        max_similarity=args.max_similarity,
        dedup=args.dedup
    )
    
    print("\n✅ 召回完成！")


if __name__ == "__main__":
    main()
