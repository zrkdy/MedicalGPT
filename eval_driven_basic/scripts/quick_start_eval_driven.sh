#!/bin/bash
# 一键启动评测驱动训练流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  评测驱动训练 - 快速启动脚本"
echo "=========================================="

# 检查环境变量
if [ -z "$ZHIPUAI_API_KEY" ]; then
    echo "⚠️  警告: 未设置 ZHIPUAI_API_KEY 环境变量"
    echo "   如果使用 GLM-embedding-3，请先设置:"
    echo "   export ZHIPUAI_API_KEY='your_api_key'"
    echo ""
    read -p "继续使用本地模型? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
else
    EMBEDDING_MODEL="glm-embedding-3"
fi

export HF_ENDPOINT=https://hf-mirror.com

# ==================== Step 1: 下载评测集 ====================
echo ""
echo "Step 1/6: 下载 CEval 医疗评测集"
echo "----------------------------------------"

if [ ! -d "data/eval_benchmark" ] || [ -z "$(ls -A data/eval_benchmark)" ]; then
    python scripts/download_ceval.py
else
    echo "✓ 评测集已存在，跳过下载"
fi

# ==================== Step 2: 向量化评测集 ====================
echo ""
echo "Step 2/6: 向量化评测集"
echo "----------------------------------------"

if [ ! -d "data/eval_vectorized" ] || [ -z "$(ls -A data/eval_vectorized)" ]; then
    python scripts/vectorize_eval_dataset.py \
        --input_dir data/eval_benchmark \
        --output_dir data/eval_vectorized \
        --model_name $EMBEDDING_MODEL \
        --batch_size 50
else
    echo "✓ 评测集向量已存在，跳过"
fi

# ==================== Step 3: 向量化训练数据 ====================
echo ""
echo "Step 3/6: 向量化训练数据 (可能需要较长时间)"
echo "----------------------------------------"

TRAIN_VECTOR_FILE="data/train_vectorized/medical_vectorized.jsonl"

if [ ! -f "$TRAIN_VECTOR_FILE" ]; then
    echo "这一步可能需要 6-12 小时..."
    echo "建议先用小数据集测试 (--max_samples 10000)"
    echo ""
    read -p "使用完整数据集 (500k样本)? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        MAX_SAMPLES=500000
    else
        MAX_SAMPLES=10000
    fi
    
    python scripts/vectorize_training_dataset.py \
        --dataset_name shibing624/medical \
        --output_file $TRAIN_VECTOR_FILE \
        --model_name $EMBEDDING_MODEL \
        --max_samples $MAX_SAMPLES \
        --batch_size 100
else
    echo "✓ 训练数据向量已存在，跳过"
fi

# ==================== Step 4: 召回相关数据 ====================
echo ""
echo "Step 4/6: 召回相关数据"
echo "----------------------------------------"

python scripts/recall_relevant_data.py \
    --eval_vectors data/eval_vectorized \
    --train_vectors $TRAIN_VECTOR_FILE \
    --output_dir data/recalled_data \
    --top_k 50 \
    --similarity_threshold 0.75 \
    --max_similarity 0.99

# ==================== Step 5: 合并数据 ====================
echo ""
echo "Step 5/6: 合并数据为训练集"
echo "----------------------------------------"

python scripts/merge_recalled_data.py \
    --input_dir data/recalled_data \
    --output_file data/finetune/medical_eval_driven.jsonl \
    --format sharegpt \
    --shuffle True \
    --deduplicate True

# ==================== Step 6: 开始训练 ====================
echo ""
echo "Step 6/6: 开始 SFT 训练"
echo "----------------------------------------"
echo ""
read -p "是否立即开始训练? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash scripts/run_sft_eval_driven.sh
else
    echo "跳过训练"
    echo ""
    echo "手动启动训练:"
    echo "  bash scripts/run_sft_eval_driven.sh"
fi

echo ""
echo "=========================================="
echo "✅ 流程完成！"
echo "=========================================="
echo ""
echo "数据文件:"
echo "  - 评测集: data/eval_benchmark/"
echo "  - 向量化评测集: data/eval_vectorized/"
echo "  - 训练数据向量: $TRAIN_VECTOR_FILE"
echo "  - 召回数据: data/recalled_data/"
echo "  - 最终训练集: data/finetune/medical_eval_driven.jsonl"
echo ""
echo "下一步:"
echo "  1. 训练: bash scripts/run_sft_eval_driven.sh"
echo "  2. 评测: python scripts/evaluate_model.py"
echo "  3. 部署: python merge_peft_adapter.py"
