#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地数据准备脚本
在本地完成所有数据准备工作，然后传输到服务器训练
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 设置 Windows 控制台 UTF-8 编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def run_command(cmd, description):
    """执行命令并显示进度"""
    print("\n" + "=" * 70)
    print(f"🔄 {description}")
    print("=" * 70)
    print(f"执行命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ 失败: {description}")
        return False
    
    print(f"\n✅ 完成: {description}")
    return True


def check_environment():
    """检查环境配置"""
    print("\n" + "=" * 70)
    print("检查环境配置")
    print("=" * 70)
    
    # 检查 API Key
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if api_key:
        print(f"✅ ZHIPUAI_API_KEY: {'*' * 20}{api_key[-4:]}")
        use_glm = True
    else:
        print("⚠️  未设置 ZHIPUAI_API_KEY，将使用本地模型")
        use_glm = False
    
    # 检查 HF 镜像
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        print(f"✅ HF_ENDPOINT: {hf_endpoint}")
    else:
        print("⚠️  未设置 HF_ENDPOINT，使用默认地址")
    
    # 检查必要的包
    try:
        import zhipuai
        print("✅ zhipuai 已安装")
    except ImportError:
        if use_glm:
            print("❌ zhipuai 未安装，请运行: pip install zhipuai")
            return False
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers 已安装")
    except ImportError:
        if not use_glm:
            print("❌ sentence-transformers 未安装，请运行: pip install sentence-transformers")
            return False
    
    return True, use_glm


def main():
    parser = argparse.ArgumentParser(description="本地数据准备脚本")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="训练数据最大样本数 (建议: 测试10000, 正式100000-500000)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="跳过下载评测集（如果已下载）"
    )
    parser.add_argument(
        "--skip_vectorize_eval",
        action="store_true",
        help="跳过向量化评测集（如果已完成）"
    )
    parser.add_argument(
        "--skip_vectorize_train",
        action="store_true",
        help="跳过向量化训练数据（如果已完成）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="向量化模型 (glm-embedding-3 或 paraphrase-multilingual-MiniLM-L12-v2)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("   MedicalGPT 本地数据准备脚本")
    print("=" * 70)
    print(f"训练数据样本数: {args.max_samples}")
    print(f"预计耗时: {args.max_samples / 10000 * 0.5:.1f} - {args.max_samples / 10000 * 1.2:.1f} 小时")
    print("=" * 70)
    
    # 检查环境
    env_ok, use_glm = check_environment()
    if not env_ok:
        print("\n❌ 环境检查失败，请安装必要的依赖")
        return 1
    
    # 确定向量化模型
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "glm-embedding-3" if use_glm else "paraphrase-multilingual-MiniLM-L12-v2"
    
    print(f"\n使用向量化模型: {model_name}")
    
    # 确认继续
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return 0
    
    # Step 1: 下载评测集
    if not args.skip_download:
        if not run_command(
            "python scripts/download_ceval.py",
            "Step 1/5: 下载 CEval 医疗评测集"
        ):
            return 1
    else:
        print("\n⏭️  跳过: 下载评测集")
    
    # Step 2: 向量化评测集
    if not args.skip_vectorize_eval:
        if not run_command(
            f"python scripts/vectorize_eval_dataset.py "
            f"--input_dir data/eval_benchmark "
            f"--output_dir data/eval_vectorized "
            f"--model_name {model_name}",
            "Step 2/5: 向量化评测集"
        ):
            return 1
    else:
        print("\n⏭️  跳过: 向量化评测集")
    
    # Step 3: 向量化训练数据（最耗时）
    if not args.skip_vectorize_train:
        print("\n" + "⚠️ " * 20)
        print("警告: 这一步可能需要 {} - {} 小时".format(
            args.max_samples / 10000 * 0.5,
            args.max_samples / 10000 * 1.2
        ))
        print("建议: 晚上启动，第二天早上查看结果")
        print("⚠️ " * 20)
        
        response = input("\n确认开始向量化训练数据? (y/n): ")
        if response.lower() != 'y':
            print("已跳过，可稍后手动执行:")
            print(f"  python scripts/vectorize_training_dataset.py --max_samples {args.max_samples}")
        else:
            if not run_command(
                f"python scripts/vectorize_training_dataset.py "
                f"--dataset_name shibing624/medical "
                f"--output_file data/train_vectorized/medical_vectorized.jsonl "
                f"--model_name {model_name} "
                f"--max_samples {args.max_samples}",
                "Step 3/5: 向量化训练数据"
            ):
                return 1
    else:
        print("\n⏭️  跳过: 向量化训练数据")
    
    # Step 4: 召回数据
    if not run_command(
        "python scripts/recall_relevant_data.py "
        "--eval_vectors data/eval_vectorized "
        "--train_vectors data/train_vectorized/medical_vectorized.jsonl "
        "--output_dir data/recalled_data "
        "--top_k 50 "
        "--similarity_threshold 0.75",
        "Step 4/5: 召回相关数据"
    ):
        return 1
    
    # Step 5: 合并数据
    if not run_command(
        "python scripts/merge_recalled_data.py "
        "--input_dir data/recalled_data "
        "--output_file data/finetune/medical_eval_driven.jsonl "
        "--format sharegpt "
        "--shuffle True",
        "Step 5/5: 合并为训练集"
    ):
        return 1
    
    # 显示文件信息
    print("\n" + "=" * 70)
    print("✅ 本地准备完成！")
    print("=" * 70)
    
    print("\n生成的文件:")
    data_files = [
        "data/eval_benchmark/",
        "data/eval_vectorized/",
        "data/train_vectorized/medical_vectorized.jsonl",
        "data/recalled_data/",
        "data/finetune/medical_eval_driven.jsonl"
    ]
    
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024 / 1024
                print(f"  ✅ {file_path} ({size:.2f} MB)")
            else:
                files = list(path.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
                print(f"  ✅ {file_path} ({len(files)} 文件, {total_size:.2f} MB)")
        else:
            print(f"  ❌ {file_path} (不存在)")
    
    print("\n" + "=" * 70)
    print("下一步:")
    print("=" * 70)
    print("1. 验证数据: python scripts/verify_data.py")
    print("2. 提交小文件到 Git:")
    print("   git add data/eval_benchmark/ data/eval_vectorized/ data/recalled_data/ data/finetune/")
    print("   git commit -m 'Add prepared training data'")
    print("   git push")
    print("\n3. 传输大文件到服务器 (data/train_vectorized/):")
    print("   方式1: 使用 WinSCP/FileZilla 图形界面上传")
    print("   方式2: scp -r data/train_vectorized/ root@server:/root/MedicalGPT/data/")
    print("   方式3: 压缩后上传 (推荐)")
    print("     本地: tar -czf train_vectorized.tar.gz data/train_vectorized/")
    print("     上传: scp train_vectorized.tar.gz root@server:/root/")
    print("     服务器: tar -xzf train_vectorized.tar.gz -C MedicalGPT/")
    print("\n4. 在服务器开始训练:")
    print("   bash scripts/run_sft_eval_driven.sh")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
