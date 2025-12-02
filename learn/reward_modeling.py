# -*- coding: utf-8 -*-
import torch
from typing import List, Dict, Any
from transformers import Trainer, PreTrainedTokenizerBase

# --------------------------------------------------------------------------------
# 核心部分 1: 偏好数据预处理
# RM 需要输入成对的数据: (Chosen, Rejected)
# --------------------------------------------------------------------------------

def preprocess_reward_function(examples, tokenizer, prompt_template):
    """
    将数据集转换为 Question + Answer 对。
    生成两个 Input:
    1. input_ids_chosen:  Question + Better Answer
    2. input_ids_rejected: Question + Worse Answer
    """
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    
    # 遍历 batch 中的数据
    for system, history, question, chosen, rejected in zip(
            examples["system"], examples["history"], examples["question"], 
            examples["response_chosen"], examples["response_rejected"]
    ):
        # 构建 Prompt (Question)
        # 结合 System Prompt 和 History
        chosen_messages = history + [[question, chosen]]
        chosen_prompt = prompt_template.get_prompt(messages=chosen_messages, system_prompt=system)
        
        rejected_messages = history + [[question, rejected]]
        rejected_prompt = prompt_template.get_prompt(messages=rejected_messages, system_prompt=system)

        # 分别 Tokenize
        tokenized_chosen = tokenizer(chosen_prompt)
        tokenized_rejected = tokenizer(rejected_prompt)

        # 存入列表
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        
    return new_examples

# --------------------------------------------------------------------------------
# 核心部分 2: 自定义 DataCollator
# --------------------------------------------------------------------------------

class RewardDataCollatorWithPadding:
    """
    需要自定义 Collator 来处理成对数据的 Padding
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 拆分 Chosen 和 Rejected
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append({
                "input_ids": feature["input_ids_chosen"],
                "attention_mask": feature["attention_mask_chosen"],
            })
            features_rejected.append({
                "input_ids": feature["input_ids_rejected"],
                "attention_mask": feature["attention_mask_rejected"],
            })
            
        # 分别进行 Pad (batch 内对其长度)
        batch_chosen = self.tokenizer.pad(features_chosen, return_tensors="pt")
        batch_rejected = self.tokenizer.pad(features_rejected, return_tensors="pt")
        
        # 组合回一个 Batch
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch

# --------------------------------------------------------------------------------
# 核心部分 3: 自定义 Trainer 计算 Loss
# --------------------------------------------------------------------------------

class RewardTrainer(Trainer):
    """
    继承 Trainer 并重写 compute_loss 方法
    实现 Pairwise Log Loss (InstructGPT 论文中的损失函数)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 获取 Chosen 的分数
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"]
        )[0] # model 输出通常是 logits，对于 RM 来说就是一个标量分数
        
        # 2. 获取 Rejected 的分数
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"]
        )[0]
        
        # 3. 计算 Loss
        # Loss = -log(sigmoid(Reward_Chosen - Reward_Rejected))
        # 我们希望 Chosen 的分数比 Rejected 高尽可能多
        # log_sigmoid 在差值越大时越接近 0 (Loss 越小)
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss

# --------------------------------------------------------------------------------
# 核心部分 4: 模型初始化
# --------------------------------------------------------------------------------

def get_reward_model(model_name):
    # RM 通常是一个 SequenceClassification 模型 (输出一个数值)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1, # 输出维度为 1 (分数)
    )
    return model
