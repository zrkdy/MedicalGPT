#!/bin/bash
# 监督微调 - Qwen2.5-3B
# 用途：训练医疗问答对话能力

# 如果进行了PT，使用PT后的模型；否则使用原始模型
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
# 如果完成了PT，取消下面的注释
# MODEL_PATH="outputs-pt-qwen2.5-3b/checkpoint-best"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
    --model_name_or_path $MODEL_PATH \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 10000 \
    --max_eval_samples 100 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 8 \
    --output_dir outputs-sft-qwen2.5-3b \
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
