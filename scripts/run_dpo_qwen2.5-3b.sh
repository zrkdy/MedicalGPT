#!/bin/bash
# DPO训练 - Qwen2.5-3B
# 用途：直接从偏好数据优化模型，推荐使用

# 基于SFT模型训练
SFT_MODEL="outputs-sft-qwen2.5-3b/checkpoint-best"

CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
    --model_name_or_path $SFT_MODEL \
    --template_name qwen \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 5000 \
    --max_eval_samples 100 \
    --max_steps 2000 \
    --eval_steps 100 \
    --save_steps 500 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --learning_rate 5e-6 \
    --output_dir outputs-dpo-qwen2.5-3b \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --device_map auto \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache
