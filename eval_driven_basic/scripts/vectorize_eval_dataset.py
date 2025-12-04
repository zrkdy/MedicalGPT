#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量化评测集数据
支持 GLM-embedding-3 和本地 sentence-transformers 模型
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time


def get_embedding_model(model_name: str, api_key: str = None):
    """
    获取向量化模型
    支持: glm-embedding-3, sentence-transformers
    """
    if model_name == "glm-embedding-3":
        # 使用智谱AI的嵌入模型
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
            """批量获取向量"""
            embeddings = []
            for text in tqdm(texts, desc="调用API获取向量"):
                try:
                    response = client.embeddings.create(
                        model="embedding-3",
                        input=text,
                    )
                    embeddings.append(response.data[0].embedding)
                    time.sleep(0.05)  # 避免API限流
                except Exception as e:
                    print(f"获取向量失败: {e}")
                    embeddings.append([0.0] * 1024)  # 返回零向量
            return embeddings
        
        return embed_func, 1024  # GLM-embedding-3 维度为 1024
    
    else:
        # 使用本地 sentence-transformers 模型
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        
        print(f"加载本地模型: {model_name}")
        model = SentenceTransformer(model_name)
        
        def embed_func(texts: List[str]) -> List[List[float]]:
            """批量获取向量"""
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        
        return embed_func, model.get_sentence_embedding_dimension()


def load_ceval_data(file_path: str) -> List[Dict]:
    """加载 CEval 格式的评测数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    return data


def format_ceval_question(item: Dict) -> str:
    """
    格式化 CEval 问题为完整文本
    包含问题和所有选项
    """
    question = item.get('question', '')
    options = []
    for key in ['A', 'B', 'C', 'D', 'E']:
        if key in item and item[key]:
            options.append(f"{key}. {item[key]}")
    
    if options:
        return f"{question}\n" + "\n".join(options)
    return question


def vectorize_eval_dataset(
    input_dir: str,
    output_dir: str,
    model_name: str = "glm-embedding-3",
    api_key: str = None,
    batch_size: int = 100
):
    """
    向量化评测集
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取向量化模型
    embed_func, dim = get_embedding_model(model_name, api_key)
    print(f"使用模型: {model_name}, 向量维度: {dim}")
    
    # 处理所有评测集文件
    json_files = list(input_path.glob("*.jsonl")) + list(input_path.glob("*.json"))
    
    for file_path in json_files:
        print(f"\n处理文件: {file_path.name}")
        
        # 加载数据
        data = load_ceval_data(file_path)
        print(f"加载 {len(data)} 条数据")
        
        # 批量处理
        vectorized_data = []
        for i in tqdm(range(0, len(data), batch_size), desc="向量化进度"):
            batch = data[i:i+batch_size]
            
            # 格式化问题文本
            texts = [format_ceval_question(item) for item in batch]
            
            # 获取向量
            embeddings = embed_func(texts)
            
            # 保存结果
            for item, embedding in zip(batch, embeddings):
                vectorized_data.append({
                    **item,  # 保留原始字段
                    "text": format_ceval_question(item),
                    "embedding": embedding
                })
        
        # 保存向量化数据
        output_file = output_path / f"{file_path.stem}_vectorized.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in vectorized_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ 已保存: {output_file}")
        print(f"  样本数: {len(vectorized_data)}")
        print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="向量化评测集数据")
    parser.add_argument("--input_dir", type=str, required=True, help="评测集目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--model_name",
        type=str,
        default="glm-embedding-3",
        help="向量化模型 (glm-embedding-3 / paraphrase-multilingual-MiniLM-L12-v2)"
    )
    parser.add_argument("--api_key", type=str, default=None, help="API Key (GLM)")
    parser.add_argument("--batch_size", type=int, default=100, help="批处理大小")
    
    args = parser.parse_args()
    
    vectorize_eval_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        api_key=args.api_key,
        batch_size=args.batch_size
    )
    
    print("\n✅ 向量化完成！")


if __name__ == "__main__":
    main()
