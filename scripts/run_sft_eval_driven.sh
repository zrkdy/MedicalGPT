#!/bin/bash
# SFT训练脚本 - 使用评测召回数据

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# 训练参数
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
TRAIN_DATA="data/finetune/medical_eval_driven.jsonl"
OUTPUT_DIR="outputs-sft-eval-driven"
BATCH_SIZE=4
GRADIENT_ACCUM=4
NUM_EPOCHS=3
LEARNING_RATE=2e-5
LORA_RANK=64
LORA_ALPHA=128

echo "=========================================="
echo "开始 SFT 训练 (评测召回数据)"
echo "=========================================="
echo "模型: $MODEL_NAME"
echo "数据: $TRAIN_DATA"
echo "输出: $OUTPUT_DIR"
echo "=========================================="

python supervised_finetuning.py \
    --model_name_or_path $MODEL_NAME \
    --train_file_dir $TRAIN_DATA \
    --validation_file_dir $TRAIN_DATA \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --use_peft True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --target_modules all \
    --bf16 \
    --gradient_checkpointing True \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_first_step True \
    --do_train \
    --do_eval

echo ""
echo "✅ 训练完成！"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "下一步:"
echo "1. 评测模型: python scripts/evaluate_model.py --model_path $OUTPUT_DIR"
echo "2. 合并权重: python merge_peft_adapter.py --base_model $MODEL_NAME --lora_model $OUTPUT_DIR --output_dir medical-gpt-final"
