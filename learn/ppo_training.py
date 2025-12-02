# -*- coding: utf-8 -*-
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

# --------------------------------------------------------------------------------
# 核心部分 1: 模型加载 (PPO 需要 4 个模型!)
# --------------------------------------------------------------------------------

def load_ppo_models(sft_model_path, reward_model_path):
    """
    PPO 算法涉及四个模型:
    1. Policy Model (Actor):  我们要训练的 SFT 模型，生成回答。
    2. Reference Model (Ref): 原始 SFT 模型 (冻结)，用于计算 KL 散度，防止 Policy 跑偏太远。
    3. Reward Model (RM):     训练好的奖励模型 (冻结)，给 Policy 生成的回答打分。
    4. Value Model (Critic):  价值模型，预测当前状态的价值 (通常基于 RM 初始化，但在 PPO 中会更新)。
    """
    
    # 1. 加载 Policy Model (Actor)
    policy = AutoModelForCausalLM.from_pretrained(sft_model_path)
    
    # 2. 加载 Reference Model
    # 如果使用 PEFT/LoRA，Ref Model 可以是 Policy 的初始状态 (不加载 Adapter)
    # 或者显式加载另一个副本
    ref_policy = AutoModelForCausalLM.from_pretrained(sft_model_path)
    
    # 3. 加载 Reward Model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, num_labels=1
    )
    
    # 4. 加载 Value Model
    # 通常使用 Reward Model 的权重初始化，但在 PPO 过程中会更新
    value_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, num_labels=1
    )
    
    return policy, ref_policy, reward_model, value_model

# --------------------------------------------------------------------------------
# 核心部分 2: PPOTrainer 初始化
# --------------------------------------------------------------------------------

def init_ppo_trainer(policy, ref_policy, reward_model, value_model, tokenizer, dataset, training_args):
    # PPOTrainer 来自 trl 库 (Transformer Reinforcement Learning)
    # 它封装了 PPO 算法的复杂性
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
    )
    return trainer

# --------------------------------------------------------------------------------
# 核心部分 3: PPO 训练循环 (Step-by-Step)
# --------------------------------------------------------------------------------

def train_ppo_step(trainer, tokenizer):
    # PPO 的训练不同于普通的 Trainer.train()，通常需要手动写循环
    # 或者是 trl 封装好的 trainer.train()
    
    # 伪代码展示一个 Step 的过程:
    
    # 1. 获取一个 Batch 的 Prompt (Query)
    # 比如: "感冒了怎么办？"
    queries = next(iterator)
    
    # 2. Policy Model 生成回答 (Response)
    # 比如: "多喝热水。"
    responses = trainer.generate(queries)
    
    # 3. Reward Model 打分 (Reward)
    # 比如: Score = 0.8
    # 注意: 这里通常会把 Query + Response 拼接后送入 RM
    scores = trainer.reward_model(queries, responses)
    
    # 4. PPO Step 更新
    # 使用 PPO 算法更新 Policy 和 Value Model
    # 输入: Query, Response, Reward
    # 内部计算:
    #   - Log Prob (当前策略生成该 Response 的概率)
    #   - Ref Log Prob (参考策略生成的概率 -> 计算 KL 散度惩罚)
    #   - Advantage (优势函数)
    #   - Loss (PPO Clip Loss + Value Loss)
    train_stats = trainer.step(queries, responses, scores)
    
    return train_stats
