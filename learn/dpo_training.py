# -*- coding: utf-8 -*-
from copy import deepcopy
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------------
# 核心部分 1: DPO 数据格式
# DPO 需要的数据格式: { "prompt": ..., "chosen": ..., "rejected": ... }
# 不同于 RM 需要 tokenize 后的 input_ids，DPOTrainer 内部会处理 tokenize
# --------------------------------------------------------------------------------

def return_prompt_and_responses(examples, prompt_template):
    """
    将数据集转换为 DPO 需要的字典格式
    """
    prompts = []
    for system, history, question in zip(examples["system"], examples["history"], examples["question"]):
        # 构建 Prompt
        system_prompt = system or ""
        history_with_question = history + [[question, '']] # 添加当前问题，回答为空
        # 使用模板生成 Prompt 字符串
        prompts.append(prompt_template.get_prompt(messages=history_with_question, system_prompt=system_prompt))
        
    return {
        "prompt": prompts,
        "chosen": examples["response_chosen"],   # 偏好的回答 (字符串)
        "rejected": examples["response_rejected"], # 拒绝的回答 (字符串)
    }

# --------------------------------------------------------------------------------
# 核心部分 2: DPOTrainer 初始化
# DPO 的优势在于不需要显式的 Reward Model，也不需要 PPO 的复杂采样循环
# 它直接在 Policy Model 上进行优化
# --------------------------------------------------------------------------------

def run_dpo_training(model_name_or_path, train_dataset, training_args):
    
    # 1. 加载模型
    # DPO 仍然需要一个 Reference Model (用于计算 KL 散度约束)
    # 但通常不需要显式加载两个模型到显存 (如果是 LoRA)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    # 如果使用 LoRA，ref_model 可以设为 None (TRL 会自动处理，或者复用 Base Model)
    # 如果是全参微调，通常需要 deepcopy 一个 ref_model
    ref_model = None
    
    # 2. 配置 DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args, # DPOConfig
        train_dataset=train_dataset,
        # DPO Trainer 需要知道每列数据的名称
        beta=0.1, # KL 惩罚系数 (非常重要的超参数)
        max_prompt_length=1024,
        max_length=2048,
    )
    
    # 3. 开始训练
    # 内部逻辑:
    # - 计算 Policy 对 Chosen 和 Rejected 的 LogProb
    # - 计算 Reference 对 Chosen 和 Rejected 的 LogProb
    # - 计算 Implicit Reward = beta * (log(Policy/Ref))
    # - 计算 DPO Loss (类似 Sigmoid Loss)
    dpo_trainer.train()
    dpo_trainer.save_model()
