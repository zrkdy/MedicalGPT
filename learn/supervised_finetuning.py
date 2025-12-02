# -*- coding: utf-8 -*-
import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForSeq2Seq
)
from template import get_conv_template # 项目自定义的对话模板工具

# --------------------------------------------------------------------------------
# 核心部分 1: 聊天数据预处理 (SFT的核心)
# SFT 需要将 "Instruction" + "Input" + "Output" 转换成模型训练格式
# 并且只对 "Output" 部分计算 Loss (Mask 掉 Input 部分)
# --------------------------------------------------------------------------------

def process_sft_data(tokenizer, raw_datasets, script_args, data_args, training_args):
    
    # 获取对话模板 (如 vicuna, alpaca 等)，决定了 system prompt 和分隔符
    prompt_template = get_conv_template(script_args.template_name)
    max_length = script_args.model_max_length
    
    # 定义忽略计算 Loss 的 Token ID (通常为 -100)
    IGNORE_INDEX = -100 

    # ----------------------------------------------------------------------------
    # 核心函数: 对话预处理函数
    # 作用: 解析对话列表 -> 应用模板 -> 分词 -> 构建 input_ids, attention_mask, labels
    # ----------------------------------------------------------------------------
    def preprocess_function(examples):
        input_ids_list = []
        attention_mask_list = []
        targets_list = [] # Labels
        
        roles = ["human", "gpt"] # 定义角色映射

        # 辅助生成器: 解析 dataset 中的 conversations 字段
        def get_dialog(examples):
            # ... (省略部分数据清洗与格式检查代码，核心是提取 human 和 gpt 的多轮对话)
            # 这里假设 examples['conversations'] 是标准的 [{"from": "human", "value": "xxx"}, ...] 格式
            # 并使用 prompt_template.get_dialog 生成带格式的文本列表
            # 返回格式: [User_Input_1, Assistant_Output_1, User_Input_2, Assistant_Output_2, ...]
            pass 

        # 遍历每一个对话样本
        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            # 遍历每一轮对话 (User + Assistant)
            for i in range(len(dialog) // 2):
                source_txt = dialog[2 * i]   # User Input
                target_txt = dialog[2 * i + 1] # Assistant Output

                # 分别编码 User 输入和 Assistant 输出
                source_ids = tokenizer.encode(text=source_txt, add_special_tokens=(i == 0)) # 首轮添加 BOS
                target_ids = tokenizer.encode(text=target_txt, add_special_tokens=False)

                # ... (省略长度截断逻辑，防止超出 max_length) ...

                # 构建这一轮的 input_ids
                # [User_Ids] + [Assistant_Ids] + [EOS]
                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
                
                # 构建 Labels
                # 关键点: User 部分的 Loss 不计算，设为 IGNORE_INDEX
                # 如果 script_args.train_on_inputs=False (默认):
                # Labels: [-100, -100, ...] + [Assistant_Ids] + [EOS]
                labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    # 执行 Map 操作
    with training_args.main_process_first(desc="Train dataset tokenization"):
        tokenized_dataset = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets.column_names,
        )
    
    return tokenized_dataset

# --------------------------------------------------------------------------------
# 核心部分 2: 模型初始化 (同 PT，略有不同的是可能需要处理 Flash Attention)
# --------------------------------------------------------------------------------

def get_model(model_args, script_args):
    # ... (配置 config, 如 flash_attn, rope_scaling 等) ...
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # ...
    )
    
    if script_args.use_peft:
        # LoRA 配置同 PT
        pass
        
    return model

# --------------------------------------------------------------------------------
# 核心部分 3: 训练 (使用 DataCollatorForSeq2Seq)
# --------------------------------------------------------------------------------

def run_training(model, tokenizer, train_dataset, training_args):
    # SFT 的 DataCollator 需要处理 padding
    # 因为每个样本长度不一，需要 pad 到 batch 中最长样本的长度
    # labels 也会被 pad 为 -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model()
