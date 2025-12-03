#!/bin/bash
# 监督微调 - Qwen2.5-3B - 单卡 RTX 3090 24GB 适配版
# 用途：训练医疗问答对话能力

MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

CUDA_VISIBLE_DEVICES=0 python supervised_finetuning.py \
    --model_name_or_path $MODEL_PATH \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 50 \
    --model_max_length 1024 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 100 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-sft-qwen2.5-3b \
    --overwrite_output_dir \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache
