# MedicalGPT 项目核心流程总结

MedicalGPT 项目旨在训练一个专注于医疗领域的语言大模型。该项目复现了完整的 ChatGPT 训练流程（PT -> SFT -> RLHF），并集成了最新的对齐技术（如 DPO, ORPO, GRPO）。

下面是该项目主要学习阶段的流程、作用及效果总结：

## 1. 增量预训练 (Continue Pre-training, PT)

*   **对应脚本**: `run_pt.sh` / `pretraining.py`
*   **输入数据**: 大规模无监督的医疗领域文本数据（如医学书籍、论文、百科、指南等），位于 `data/pretrain` 目录。
*   **核心流程**:
    *   基于通用的基础模型（如 LLaMA, Qwen, Baichuan 等）。
    *   在医疗领域语料上进行自回归（Next Token Prediction）训练。
    *   通常使用 LoRA 等参数高效微调技术（PEFT）以降低显存需求。
*   **作用**:
    *   **领域适配**: 让通用模型熟悉医疗专业术语、知识体系和语言表达习惯。
    *   **知识注入**: 将特定领域的知识“注入”到模型的参数中。
*   **效果**:
    *   模型在医疗文本上的困惑度（Perplexity）降低。
    *   模型能生成更流畅、专业的医疗文本，但此时它还不太会“聊天”或遵循复杂指令。

## 2. 有监督微调 (Supervised Fine-tuning, SFT)

*   **对应脚本**: `run_sft.sh` / `supervised_finetuning.py`
*   **输入数据**: 医疗领域的指令对话数据（Instruction-Response Pairs），位于 `data/finetune` 目录。
*   **核心流程**:
    *   在 PT 模型（或 Base 模型）的基础上进行微调。
    *   使用包含 `<Input>` (问题/指令) 和 `<Output>` (医生回答) 的数据进行训练。
    *   学习目标依然是 Next Token Prediction，但只计算 `<Output>` 部分的 Loss。
*   **作用**:
    *   **指令对齐**: 教会模型理解用户意图，学会以“问答”或“对话”的形式进行交互。
    *   **格式规范**: 让模型学会医生回答问题的语气和结构。
*   **效果**:
    *   模型变成了“医疗聊天机器人”，可以回答患者咨询。
    *   但此时模型可能会产生“幻觉”（一本正经胡说八道），或者回答不够安全、不够有用。

## 3. 强化学习人类反馈 (RLHF)

这是为了进一步提升模型回答质量，使其更符合人类偏好（有用 Helpful、诚实 Honest、无害 Harmless）的关键阶段，通常分为两步：

### 3.1 奖励建模 (Reward Modeling, RM)
*   **对应脚本**: `run_rm.sh` / `reward_modeling.py`
*   **输入数据**: 偏好数据集（Comparison Data），即针对同一个问题，有“更好”和“更差”两个回答（Chosen vs. Rejected），位于 `data/reward` 目录。
*   **作用**: 训练一个“裁判”模型（Reward Model），它能给模型的回答打分。
*   **效果**: 获得一个能模拟人类评分标准的模型，它知道什么是“好”的医疗建议。

### 3.2 强化学习 (Reinforcement Learning, PPO)
*   **对应脚本**: `run_ppo.sh` / `ppo_training.py`
*   **输入数据**: Prompt 数据（只有问题，没有标准答案）。
*   **核心流程**:
    *   **Actor (策略模型)**: SFT 后的模型，负责生成回答。
    *   **Critic (价值模型)**: 预估当前状态的价值。
    *   **Reward Model**: 给 Actor 生成的回答打分。
    *   **算法**: 使用 PPO (Proximal Policy Optimization) 算法，根据 Reward Model 的反馈更新 Actor 的参数。
*   **作用**: 鼓励模型生成 Reward Model 认为分数更高的回答。
*   **效果**: 模型生成的回答在安全性、有用性和风格上更符合人类预期，减少有害建议或胡编乱造。

## 4. 直接偏好优化 (Direct Preference Optimization, DPO)

*   **对应脚本**: `run_dpo.sh` / `dpo_training.py`
*   **输入数据**: 与 RM 阶段相同的偏好数据集（Chosen vs. Rejected）。
*   **核心流程**:
    *   不显式训练 Reward Model，也不使用 PPO。
    *   直接通过数学推导出的 Loss 函数，在偏好数据上优化策略模型。
*   **作用**: 作为 RLHF 的高效替代方案，目的是在不需要复杂强化学习过程的情况下实现对齐。
*   **效果**: 
    *   训练更稳定，资源消耗更少。
    *   效果通常与 PPO 相当甚至更好，是目前主流的对齐方法。

## 5. 其他前沿方法 (ORPO, GRPO)

*   **ORPO (Odds Ratio Preference Optimization)**:
    *   **脚本**: `run_orpo.sh`
    *   **特点**: 将 SFT 和偏好对齐合并为一步。不需要参考模型（Reference Model），减少了显存占用和训练步骤。
*   **GRPO (Group Relative Policy Optimization)**:
    *   **脚本**: `run_grpo.sh`
    *   **特点**: 主要是 DeepSeek 提出的方法，通过对同一个 Prompt 生成多个回答，利用组内相对优势来优化策略。常用于提升模型的逻辑推理能力（如数学、代码、复杂诊断推理）。

---

### 总结：该项目如何让模型“懂医疗”？

1.  **PT** 让它 **“读万卷医书”**（学知识）。
2.  **SFT** 让它 **“通过执业医师考试”**（学问诊）。
3.  **RLHF/DPO** 让它 **“成为口碑好医生”**（学医德、学规范、符合人类价值观）。
