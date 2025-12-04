#!/bin/bash
# 增量预训练 - Qwen2.5-3B
# 用途：在医疗领域文本上继续训练基础模型

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 pretraining.py \
    --model_name_or_path Qwen/Qwen2.5-3B \
    --train_file_dir ./data/pretrain \
    --validation_file_dir ./data/pretrain \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 100000 \
    --max_eval_samples 500 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 8 \
    --block_size 1024 \
    --group_by_length True \
    --output_dir outputs-pt-qwen2.5-3b \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache
