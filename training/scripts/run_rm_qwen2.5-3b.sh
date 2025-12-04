#!/bin/bash
# 奖励建模 - Qwen2.5-3B
# 用途：训练打分模型，用于PPO训练

# 基于SFT模型训练RM
SFT_MODEL="outputs-sft-qwen2.5-3b/checkpoint-best"

CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --model_name_or_path $SFT_MODEL \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 5000 \
    --max_eval_samples 100 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 2 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --output_dir outputs-rm-qwen2.5-3b \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 \
    --torch_dtype bfloat16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache
