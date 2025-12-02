# -*- coding: utf-8 -*-
# 导入必要的库
import os
from itertools import chain
import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)

# --------------------------------------------------------------------------------
# 核心部分 1: 数据处理函数
# 预训练(PT)的关键是将无监督文本转换为模型可以接受的输入格式(input_ids)
# 并通过滑动窗口(grouping)将文本拼接成固定长度(block_size)的序列
# --------------------------------------------------------------------------------

def process_pretrain_data(tokenizer, data_args, training_args):
    """
    数据处理核心流程:
    1. 加载数据集
    2. Tokenize (分词)
    3. Grouping (拼接与切分)
    """
    
    # 1. 加载数据集 (支持 txt, json, jsonl 等格式)
    # load_dataset 会自动处理缓存和多进程加载
    raw_datasets = load_dataset(
        data_args.extension, # 文件扩展名
        data_files=data_args.data_files, # 文件路径列表
        cache_dir=data_args.cache_dir, # 缓存目录
    )

    # 获取列名，以便在 tokenize 后删除原始文本列，节省内存
    column_names = list(raw_datasets["train"].features)
    
    # 设置文本块大小 (block_size)，通常为模型最大长度 (如 1024, 2048, 4096)
    block_size = data_args.block_size

    # ----------------------------------------------------------------------------
    # 核心函数 A: 分词函数
    # 作用: 将原始文本转换为 token id 序列
    # ----------------------------------------------------------------------------
    def tokenize_function(examples):
        # 使用 tokenizer 对文本进行编码
        # truncation=True: 超过最大长度截断 (但在 grouping 模式下通常不在这里截断，而是在 grouping 时处理)
        # padding='max_length': 填充到最大长度 (grouping 模式下通常不填充，而是拼接)
        # 这里展示的是简单的分词，不带 padding，因为后面会拼接
        return tokenizer(examples["text"])

    # ----------------------------------------------------------------------------
    # 核心函数 B: 文本拼接与切分 (Grouping)
    # 作用: 将多个短文本拼接起来，然后按 block_size 切分，最大化利用 context window
    # ----------------------------------------------------------------------------
    def group_text_function(examples):
        # 1. 将一个 batch 内的所有文本的 input_ids 拼接成一个长列表
        # examples 是一个字典，key 是 'input_ids', 'attention_mask' 等
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        
        # 2. 获取拼接后的总长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 3. 丢弃剩余部分 (如果总长度不能被 block_size 整除)
        # 也可以选择 padding 到 block_size，但预训练通常直接丢弃多余部分以保持高效
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # 4. 按照 block_size 切分
        # result 将包含多个长度为 block_size 的样本
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # 5. 设置 labels
        # 在自回归语言模型训练中 (Causal LM)，labels 通常就是 input_ids
        # 模型会自动处理 shift (预测下一个 token)
        result["labels"] = result["input_ids"].copy()
        return result

    # ----------------------------------------------------------------------------
    # 执行数据处理 pipeline
    # ----------------------------------------------------------------------------
    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        # 第一步: Tokenize
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True, # 批量处理加速
            num_proc=data_args.preprocessing_num_workers, # 多进程
            remove_columns=column_names, # 移除原始文本列
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
        # 第二步: Grouping
        lm_datasets = tokenized_datasets.map(
            group_text_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    return lm_datasets

# --------------------------------------------------------------------------------
# 核心部分 2: 模型初始化与 LoRA 配置
# --------------------------------------------------------------------------------

def get_model_and_tokenizer(model_args, script_args):
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 加载基础模型 (Base Model)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" # 自动分配 GPU
    )

    # 配置 LoRA (Low-Rank Adaptation) - 参数高效微调
    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        
        # 定义 LoRA 配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, # 任务类型: 因果语言模型
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 需要微调的层 (通常是 Attention 层的投影矩阵)
            inference_mode=False,
            r=script_args.lora_rank, # LoRA 秩 (rank)，如 8, 16, 64
            lora_alpha=script_args.lora_alpha, # 缩放系数
            lora_dropout=script_args.lora_dropout # Dropout 概率
        )
        
        # 将 Base Model 包装为 PeftModel
        model = get_peft_model(model, peft_config)
        
        # 打印可训练参数量，确认只有少量参数被激活
        model.print_trainable_parameters()
        
    return model, tokenizer

# --------------------------------------------------------------------------------
# 核心部分 3: 训练器配置与执行
# --------------------------------------------------------------------------------

def run_training(model, tokenizer, train_dataset, eval_dataset, training_args):
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # DataCollator 负责将 batch 数据整理成 tensor
        # default_data_collator 适用于简单的 input_ids/labels 字典列表
        data_collator=default_data_collator, 
    )

    # 开始训练
    logger.info("*** Train ***")
    train_result = trainer.train()
    
    # 保存模型
    trainer.save_model()
    
    # 保存训练指标 (Loss, 速度等)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
