#!/bin/bash
# PPO训练 - Qwen2.5-3B
# 用途：使用强化学习优化模型（需要大显存）

SFT_MODEL="outputs-sft-qwen2.5-3b/checkpoint-best"
RM_MODEL="outputs-rm-qwen2.5-3b/checkpoint-best"

CUDA_VISIBLE_DEVICES=0,1 python ppo_training.py \
    --sft_model_path $SFT_MODEL \
    --reward_model_path $RM_MODEL \
    --template_name qwen \
    --torch_dtype bfloat16 \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --max_source_length 1024 \
    --response_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --do_train \
    --total_episodes 10000 \
    --output_dir outputs-ppo-qwen2.5-3b \
    --missing_eos_penalty 1.0 \
    --eval_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --cache_dir ./cache
