#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化步骤2: Top-K平均分筛选高质量数据
参考 HealthAI-2025 的质量排序策略
"""

import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


def calculate_topk_average_scores(
    eval_vectors_file: str,
    train_vectors_file: str,
    output_file: str,
    top_k: int = 5,
    batch_size: int = 1000
):
    """
    计算每个训练样本与评测集的Top-K平均相似度
    
    策略：
    1. 对每个训练样本，找出与它最相似的Top-K个评测样本
    2. 计算这Top-K个相似度的平均值作为该训练样本的质量分数
    3. 按平均分排序，分数越高说明与评测集分布越接近
    """
    print("=" * 70)
    print("Top-K 平均分筛选策略")
    print("=" * 70)
    
    # 加载评测集向量
    print("\n1. 加载评测集向量...")
    eval_data = []
    with open(eval_vectors_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='评测集'):
            data = json.loads(line)
            eval_data.append(data)
    
    eval_vectors = np.array([d['vector'] for d in eval_data], dtype=np.float32)
    eval_count = len(eval_vectors)
    print(f"✅ 评测集: {eval_count} 条")
    
    # 归一化评测集向量
    eval_norms = np.linalg.norm(eval_vectors, axis=1, keepdims=True)
    eval_norms = np.where(eval_norms == 0, 1e-10, eval_norms)
    eval_normalized = eval_vectors / eval_norms
    
    # 处理训练数据（分批）
    print("\n2. 计算训练数据的Top-K平均分...")
    
    with open(train_vectors_file, 'r', encoding='utf-8') as fin, \
         open(output_file + '.temp', 'w', encoding='utf-8') as fout:
        
        batch = []
        batch_ids = []
        total_processed = 0
        
        for line in tqdm(fin, desc='处理训练数据'):
            data = json.loads(line)
            batch.append(data['vector'])
            batch_ids.append(data)
            
            if len(batch) >= batch_size:
                # 处理当前批次
                scores = process_batch(batch, eval_normalized, top_k)
                
                # 保存结果
                for idx, score in enumerate(scores):
                    result = batch_ids[idx].copy()
                    result['topk_avg_score'] = float(score)
                    result['topk_matches'] = top_k
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                total_processed += len(batch)
                batch = []
                batch_ids = []
        
        # 处理最后一批
        if batch:
            scores = process_batch(batch, eval_normalized, top_k)
            for idx, score in enumerate(scores):
                result = batch_ids[idx].copy()
                result['topk_avg_score'] = float(score)
                result['topk_matches'] = top_k
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            total_processed += len(batch)
    
    print(f"✅ 处理完成: {total_processed} 条")
    
    # 排序
    print("\n3. 按Top-K平均分排序...")
    sort_by_score(output_file + '.temp', output_file)
    
    # 清理临时文件
    Path(output_file + '.temp').unlink()
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 显示统计信息
    show_statistics(output_file)


def process_batch(batch_vectors: list, eval_normalized: np.ndarray, top_k: int) -> np.ndarray:
    """处理一批训练向量"""
    # 转换为numpy数组并归一化
    train_vectors = np.array(batch_vectors, dtype=np.float32)
    train_norms = np.linalg.norm(train_vectors, axis=1, keepdims=True)
    train_norms = np.where(train_norms == 0, 1e-10, train_norms)
    train_normalized = train_vectors / train_norms
    
    # 计算相似度矩阵 [batch_size, eval_size]
    similarity_matrix = np.dot(train_normalized, eval_normalized.T)
    
    # 对每个训练样本，找出Top-K个最相似的评测样本
    # 使用 argpartition 快速找到Top-K（不完全排序）
    top_k_indices = np.argpartition(-similarity_matrix, top_k, axis=1)[:, :top_k]
    
    # 提取Top-K的相似度值
    row_indices = np.arange(similarity_matrix.shape[0])[:, np.newaxis]
    topk_scores = similarity_matrix[row_indices, top_k_indices]
    
    # 计算平均分
    avg_scores = np.mean(topk_scores, axis=1)
    
    return avg_scores


def sort_by_score(input_file: str, output_file: str):
    """按分数排序"""
    # 加载所有数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='加载数据'):
            data.append(json.loads(line))
    
    # 排序
    sorted_data = sorted(
        data,
        key=lambda x: x.get('topk_avg_score', 0),
        reverse=True
    )
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(sorted_data, desc='保存结果'):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def show_statistics(file_path: str, top_n: int = 10):
    """显示统计信息"""
    print("\n" + "=" * 70)
    print("数据质量统计")
    print("=" * 70)
    
    scores = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            scores.append(data.get('topk_avg_score', 0))
    
    scores = np.array(scores)
    
    print(f"\n总数据量: {len(scores):,} 条")
    print(f"\n分数统计:")
    print(f"  最高分: {scores.max():.4f}")
    print(f"  最低分: {scores.min():.4f}")
    print(f"  平均分: {scores.mean():.4f}")
    print(f"  中位数: {np.median(scores):.4f}")
    print(f"  标准差: {scores.std():.4f}")
    
    print(f"\n分位数:")
    for p in [90, 80, 70, 50, 30, 20, 10]:
        percentile = np.percentile(scores, p)
        count = np.sum(scores >= percentile)
        print(f"  Top {100-p}%: 分数 >= {percentile:.4f} ({count:,} 条)")
    
    print(f"\n推荐筛选策略:")
    for ratio in [0.1, 0.2, 0.3]:
        threshold_idx = int(len(scores) * ratio)
        threshold = scores[threshold_idx]
        print(f"  保留 Top {ratio*100:.0f}%: 分数 >= {threshold:.4f} ({threshold_idx:,} 条)")


def extract_top_samples(input_file: str, output_file: str, top_n: int = None, threshold: float = None):
    """提取Top-N或分数超过阈值的样本"""
    if top_n is None and threshold is None:
        raise ValueError("必须指定 top_n 或 threshold 之一")
    
    print(f"\n提取高质量样本...")
    if top_n:
        print(f"  策略: 保留 Top {top_n:,} 条")
    if threshold:
        print(f"  策略: 分数 >= {threshold:.4f}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc='提取样本'):
            if top_n and count >= top_n:
                break
            
            data = json.loads(line)
            score = data.get('topk_avg_score', 0)
            
            if threshold and score < threshold:
                continue
            
            fout.write(line)
            count += 1
    
    print(f"✅ 已提取 {count:,} 条高质量样本")
    print(f"✅ 保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="优化步骤2: Top-K平均分筛选")
    parser.add_argument('--eval_vectors', required=True, help='评测集向量文件')
    parser.add_argument('--train_vectors', required=True, help='训练集向量文件')
    parser.add_argument('--output', required=True, help='输出文件（含分数）')
    parser.add_argument('--top_k', type=int, default=5, help='Top-K值（默认5）')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小')
    
    # 筛选参数
    parser.add_argument('--extract', action='store_true', help='提取高质量样本')
    parser.add_argument('--extract_output', help='提取结果输出文件')
    parser.add_argument('--extract_top_n', type=int, help='提取Top-N条')
    parser.add_argument('--extract_threshold', type=float, help='提取分数阈值')
    
    args = parser.parse_args()
    
    # 计算Top-K平均分
    calculate_topk_average_scores(
        args.eval_vectors,
        args.train_vectors,
        args.output,
        args.top_k,
        args.batch_size
    )
    
    # 提取高质量样本
    if args.extract:
        if not args.extract_output:
            args.extract_output = args.output.replace('.jsonl', '_filtered.jsonl')
        
        extract_top_samples(
            args.output,
            args.extract_output,
            args.extract_top_n,
            args.extract_threshold
        )
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
