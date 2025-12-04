#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 CEval 医疗评测集
"""

import os
from pathlib import Path


def download_ceval_medical():
    """下载 CEval 医疗相关评测集"""
    
    # 设置镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装 datasets: pip install datasets")
        return
    
    # 创建目录
    output_dir = Path("data/eval_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始下载 CEval 医疗评测集...")
    print("=" * 60)
    
    # 医疗相关的评测集
    medical_subjects = {
        "clinical_medicine": "临床医学",
        "basic_medicine": "基础医学",
        "physician": "医师资格",
        "pharmacology": "药理学",
        "virology": "病毒学"
    }
    
    for subject, chinese_name in medical_subjects.items():
        try:
            print(f"\n下载: {chinese_name} ({subject})")
            
            # 加载数据集
            dataset = load_dataset("ceval/ceval-exam", subject)
            
            # 保存验证集（用于评测）
            output_file = output_dir / f"{subject}.jsonl"
            
            # 合并 val 和 test 集
            data_to_save = []
            if "val" in dataset:
                data_to_save.extend(list(dataset["val"]))
                print(f"  val集: {len(dataset['val'])} 条")
            
            if "test" in dataset:
                data_to_save.extend(list(dataset["test"]))
                print(f"  test集: {len(dataset['test'])} 条")
            
            # 保存为 JSONL
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data_to_save:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"  ✓ 已保存: {output_file}")
            print(f"  总计: {len(data_to_save)} 条")
            
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ CEval 医疗评测集下载完成！")
    print(f"保存位置: {output_dir.absolute()}")
    print("\n下一步:")
    print("  运行向量化: python scripts/vectorize_eval_dataset.py --input_dir data/eval_benchmark --output_dir data/eval_vectorized")


if __name__ == "__main__":
    download_ceval_medical()
