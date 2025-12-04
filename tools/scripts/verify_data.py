#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证数据完整性
检查所有必需的数据文件是否存在且格式正确
"""

import json
from pathlib import Path
from typing import List, Dict


def check_jsonl_file(file_path: Path) -> Dict:
    """检查 JSONL 文件"""
    if not file_path.exists():
        return {
            "exists": False,
            "valid": False,
            "lines": 0,
            "size_mb": 0,
            "error": "文件不存在"
        }
    
    try:
        size_mb = file_path.stat().st_size / 1024 / 1024
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = 0
            sample_data = None
            
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    lines += 1
                    
                    # 保存第一条数据作为样例
                    if i == 0:
                        sample_data = data
                        
                except json.JSONDecodeError as e:
                    return {
                        "exists": True,
                        "valid": False,
                        "lines": lines,
                        "size_mb": size_mb,
                        "error": f"第 {i+1} 行 JSON 解析错误: {e}"
                    }
        
        return {
            "exists": True,
            "valid": True,
            "lines": lines,
            "size_mb": size_mb,
            "sample": sample_data,
            "error": None
        }
        
    except Exception as e:
        return {
            "exists": True,
            "valid": False,
            "lines": 0,
            "size_mb": 0,
            "error": f"读取错误: {e}"
        }


def verify_data():
    """验证所有必需的数据文件"""
    
    print("=" * 80)
    print("MedicalGPT 数据完整性验证")
    print("=" * 80)
    
    # 定义必需的文件
    required_files = {
        "评测集": [
            "data/eval_benchmark/clinical_medicine.jsonl",
            "data/eval_benchmark/basic_medicine.jsonl",
            "data/eval_benchmark/physician.jsonl"
        ],
        "向量化评测集": [
            "data/eval_vectorized/clinical_medicine_vectorized.jsonl",
            "data/eval_vectorized/basic_medicine_vectorized.jsonl",
            "data/eval_vectorized/physician_vectorized.jsonl"
        ],
        "向量化训练数据": [
            "data/train_vectorized/medical_vectorized.jsonl"
        ],
        "召回数据": [
            "data/recalled_data/recalled_clinical_medicine.jsonl",
            "data/recalled_data/recall_statistics.json"
        ],
        "最终训练集": [
            "data/finetune/medical_eval_driven.jsonl"
        ]
    }
    
    all_ok = True
    results = {}
    
    for category, files in required_files.items():
        print(f"\n{'─' * 80}")
        print(f"📁 {category}")
        print(f"{'─' * 80}")
        
        category_ok = True
        
        for file_path in files:
            path = Path(file_path)
            result = check_jsonl_file(path)
            results[file_path] = result
            
            if result["exists"] and result["valid"]:
                print(f"✅ {file_path}")
                print(f"   大小: {result['size_mb']:.2f} MB")
                print(f"   行数: {result['lines']:,}")
                
                # 显示样例数据结构
                if result.get("sample"):
                    sample = result["sample"]
                    keys = list(sample.keys())[:5]  # 只显示前5个键
                    print(f"   字段: {', '.join(keys)}")
                    
            elif result["exists"] and not result["valid"]:
                print(f"⚠️  {file_path}")
                print(f"   错误: {result['error']}")
                category_ok = False
                all_ok = False
            else:
                print(f"❌ {file_path}")
                print(f"   {result['error']}")
                category_ok = False
                all_ok = False
        
        if category_ok:
            print(f"✅ {category} - 所有文件正常")
        else:
            print(f"❌ {category} - 部分文件缺失或损坏")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    
    total_size = sum(r["size_mb"] for r in results.values() if r["valid"])
    total_lines = sum(r["lines"] for r in results.values() if r["valid"])
    total_files = len(results)
    valid_files = sum(1 for r in results.values() if r["valid"])
    
    print(f"总文件数: {valid_files}/{total_files}")
    print(f"总大小: {total_size:.2f} MB ({total_size / 1024:.2f} GB)")
    print(f"总行数: {total_lines:,}")
    
    # 关键文件检查
    print("\n" + "=" * 80)
    print("关键文件检查")
    print("=" * 80)
    
    key_files = [
        ("训练集", "data/finetune/medical_eval_driven.jsonl", 5000),
        ("训练数据向量", "data/train_vectorized/medical_vectorized.jsonl", 10000)
    ]
    
    for name, file_path, min_lines in key_files:
        if file_path in results:
            result = results[file_path]
            if result["valid"]:
                if result["lines"] >= min_lines:
                    print(f"✅ {name}: {result['lines']:,} 行 (≥ {min_lines:,})")
                else:
                    print(f"⚠️  {name}: {result['lines']:,} 行 (< {min_lines:,}, 可能数据不足)")
                    all_ok = False
            else:
                print(f"❌ {name}: 文件无效")
                all_ok = False
        else:
            print(f"❌ {name}: 文件不存在")
            all_ok = False
    
    # 最终结果
    print("\n" + "=" * 80)
    if all_ok:
        print("✅✅✅ 所有文件验证通过！可以开始训练。")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 提交代码到 Git: git push")
        print("  2. 传输大文件到服务器 (data/train_vectorized/)")
        print("  3. 在服务器执行训练: bash scripts/run_sft_eval_driven.sh")
        return 0
    else:
        print("❌❌❌ 部分文件缺失或损坏，请检查。")
        print("=" * 80)
        print("\n建议:")
        print("  1. 重新运行准备脚本: python scripts/local_prepare.py")
        print("  2. 检查网络连接和 API Key")
        print("  3. 查看错误日志")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(verify_data())
