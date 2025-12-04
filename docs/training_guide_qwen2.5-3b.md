# MedicalGPT 完整训练指南 - Qwen2.5-3B 版本

## 目录
1. [服务器配置要求](#1-服务器配置要求)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [完整训练流程](#4-完整训练流程)
5. [模型评估与部署](#5-模型评估与部署)
6. [常见问题与优化](#6-常见问题与优化)

---

## 1. 服务器配置要求

### 推荐配置（基于 Qwen2.5-3B）

| 训练阶段 | GPU要求 | 显存要求 | 内存要求 | 存储要求 |
|---------|--------|---------|---------|---------|
| **预训练 (PT)** | 2×A100/H100 (40GB) 或 4×RTX 4090 | ≥40GB | 64GB+ | 500GB+ |
| **监督微调 (SFT)** | 2×A100/H100 (40GB) 或 2×RTX 4090 | ≥24GB | 32GB+ | 200GB+ |
| **奖励建模 (RM)** | 1×A100 (40GB) 或 2×RTX 4090 | ≥24GB | 32GB+ | 100GB+ |
| **DPO训练** | 2×A100 (40GB) 或 2×RTX 4090 | ≥24GB | 32GB+ | 200GB+ |
| **PPO训练** | 2×A100 (80GB) | ≥80GB | 64GB+ | 200GB+ |

### 云服务器租赁建议

**推荐平台：**
- **AutoDL**: 性价比高，按小时计费，适合实验
- **恒源云**: 稳定性好，支持长时间训练
- **阿里云/腾讯云**: 企业级稳定性，价格较高
- **AWS/Azure**: 国际平台，资源丰富

**推荐配置：**
```
方案1（经济型）: 2×RTX 4090 (24GB)
- 价格: ~8-12元/小时
- 适合: SFT, RM, DPO训练
- 训练时间: 预计 24-48小时完成SFT

方案2（推荐型）: 2×A100 (40GB)
- 价格: ~15-20元/小时
- 适合: 所有训练阶段
- 训练时间: 预计 12-24小时完成SFT

方案3（高配型）: 2×A100 (80GB)
- 价格: ~25-35元/小时
- 适合: PPO训练 + 大规模数据
- 训练时间: 最快，支持更大batch size
```

---

## 2. 环境配置

### 2.1 克隆项目

```bash
# 连接到服务器后
cd /root
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT
```

### 2.2 创建虚拟环境

```bash
# 使用 Conda（推荐）
conda create -n medical python=3.10 -y
conda activate medical

# 或使用 venv
python3 -m venv venv_medical
source venv_medical/bin/activate
```

### 2.3 安装依赖

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 安装额外依赖（QLoRA支持）
pip install bitsandbytes
pip install flash-attn --no-build-isolation

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

### 2.4 配置 Hugging Face

```bash
# 设置 HF 镜像（国内服务器）
export HF_ENDPOINT=https://hf-mirror.com

# 登录 Hugging Face（用于下载 Qwen2.5-3B）
pip install huggingface_hub
huggingface-cli login
# 输入你的 HF Token: hf_xxxxxxxxxxxxx
```

### 2.5 目录结构

```bash
MedicalGPT/
├── data/                  # 数据目录
│   ├── pretrain/         # 预训练数据（医疗文本）
│   ├── finetune/         # SFT数据（医疗问答对）
│   └── reward/           # RM/DPO数据（偏好数据）
├── outputs-pt/           # 预训练输出
├── outputs-sft/          # SFT输出
├── outputs-rm/           # RM输出
├── outputs-dpo/          # DPO输出
├── cache/                # 模型缓存
└── logs/                 # 训练日志
```

---

## 3. 数据准备

### 3.1 预训练数据 (PT)

**格式要求：** 纯文本，每行一个医疗文档

```bash
# 创建数据目录
mkdir -p data/pretrain

# 示例数据格式（data/pretrain/medical_corpus.txt）
```

**medical_corpus.txt 示例：**
```text
高血压是一种常见的心血管疾病，主要表现为血压持续升高。治疗方法包括生活方式干预和药物治疗。常用降压药有ACEI类、ARB类、利尿剂等。
糖尿病分为1型和2型，2型糖尿病占90%以上。主要治疗药物包括二甲双胍、磺脲类、GLP-1受体激动剂等。患者需要定期监测血糖。
感冒是由病毒引起的上呼吸道感染，主要症状包括鼻塞、流涕、咳嗽、发热等。治疗以对症为主，注意休息，多饮水。
```

**获取医疗数据：**
```bash
# 1. 使用项目自带的示例数据
# 已包含在 data/pretrain 中

# 2. 下载开源医疗数据集
# Chinese Medical Dialogue Dataset
wget https://huggingface.co/datasets/Toyhom/Chinese-medical-dialogue-data/resolve/main/train.json

# 3. 准备自己的医疗数据
# 格式转换脚本（将jsonl转为txt）
python scripts/convert_to_pretrain_format.py \
    --input_file your_medical_data.jsonl \
    --output_file data/pretrain/medical_corpus.txt
```

### 3.2 监督微调数据 (SFT)

**格式要求：** JSONL格式，每行一个对话

```json
{"conversations":[{"from":"human","value":"患者：医生，我最近总是头晕，该怎么办？"},{"from":"gpt","value":"医生：头晕的原因很多，包括低血压、颈椎病、贫血等。建议您先测量血压，如果血压正常，可以做个颈椎X光检查。平时注意休息，避免突然起立。"}]}
{"conversations":[{"from":"human","value":"感冒了吃什么药？"},{"from":"gpt","value":"感冒一般是病毒感染，多数可以自愈。建议多休息、多喝水，如果症状严重可以服用对乙酰氨基酚退热，配合维生素C。如果3天后症状未缓解，请及时就医。"}]}
```

**数据准备：**
```bash
mkdir -p data/finetune

# 使用项目自带数据
cp data/finetune/medical_sft_1K_format.jsonl data/finetune/train.jsonl

# 或下载更多数据
# ChatDoctor数据集
wget https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/resolve/main/data/train-00000-of-00001.parquet

# HuatuoGPT数据集
git clone https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1
```

### 3.3 奖励建模/DPO数据 (RM/DPO)

**格式要求：** 每条包含 chosen 和 rejected 两个回答

```json
{"system":"","history":[],"question":"感冒了怎么办？","response_chosen":"建议您多休息、多喝水，保持室内通风。如果出现发热症状可以服用对乙酰氨基酚退热。如果症状持续3天以上或出现呼吸困难，请及时就医。","response_rejected":"喝热水就行了。"}
{"system":"你是一名专业医生","history":[],"question":"高血压需要一直吃药吗？","response_chosen":"高血压是一种慢性疾病，多数患者需要长期服药控制。突然停药可能导致血压反弹，增加心脑血管事件风险。建议在医生指导下调整用药，定期监测血压。","response_rejected":"血压正常了就可以停药。"}
```

**数据准备：**
```bash
mkdir -p data/reward

# 使用项目自带数据
cp data/reward/dpo_zh_500.jsonl data/reward/train.jsonl

# 或从SFT数据生成偏好数据
python scripts/generate_preference_data.py \
    --sft_data data/finetune/train.jsonl \
    --output_file data/reward/preference_data.jsonl \
    --num_samples 5000
```

---

## 4. 完整训练流程

### 4.1 阶段一：增量预训练 (PT) - 可选

**目的：** 让模型学习医疗领域的语言特征和专业知识

**预计时间：** 12-24小时（10万条数据，2×A100）

**创建训练脚本：** `scripts/run_pt_qwen2.5-3b.sh`

```bash
#!/bin/bash
# 增量预训练 - Qwen2.5-3B

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
```

**启动训练：**
```bash
chmod +x scripts/run_pt_qwen2.5-3b.sh
nohup bash scripts/run_pt_qwen2.5-3b.sh > logs/pt_training.log 2>&1 &

# 监控训练
tail -f logs/pt_training.log

# 查看TensorBoard
tensorboard --logdir outputs-pt-qwen2.5-3b --port 6006 --bind_all
```

**训练完成后：**
- 模型权重保存在：`outputs-pt-qwen2.5-3b/checkpoint-xxxx/`
- LoRA权重：`adapter_model.bin` + `adapter_config.json`

### 4.2 阶段二：监督微调 (SFT) - 必需

**目的：** 训练模型进行医疗问答对话

**预计时间：** 4-8小时（1万条数据，2×A100）

**创建训练脚本：** `scripts/run_sft_qwen2.5-3b.sh`

```bash
#!/bin/bash
# 监督微调 - Qwen2.5-3B

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
```

**启动训练：**
```bash
chmod +x scripts/run_sft_qwen2.5-3b.sh
nohup bash scripts/run_sft_qwen2.5-3b.sh > logs/sft_training.log 2>&1 &

tail -f logs/sft_training.log
```

**合并LoRA权重（可选，用于推理）：**
```bash
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_model outputs-sft-qwen2.5-3b/checkpoint-best \
    --output_dir outputs-sft-qwen2.5-3b-merged
```

### 4.3 阶段三A：奖励建模 (RM) - 用于PPO

**目的：** 训练一个打分模型，评估回答质量

**预计时间：** 2-4小时（5000条数据，1×A100）

**创建训练脚本：** `scripts/run_rm_qwen2.5-3b.sh`

```bash
#!/bin/bash
# 奖励建模 - Qwen2.5-3B

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
```

**启动训练：**
```bash
chmod +x scripts/run_rm_qwen2.5-3b.sh
nohup bash scripts/run_rm_qwen2.5-3b.sh > logs/rm_training.log 2>&1 &
```

### 4.3 阶段三B：DPO训练 (推荐) - 替代RM+PPO

**目的：** 直接从偏好数据优化模型，无需单独的RM

**预计时间：** 4-6小时（5000条数据，2×A100）

**创建训练脚本：** `scripts/run_dpo_qwen2.5-3b.sh`

```bash
#!/bin/bash
# DPO训练 - Qwen2.5-3B

# 基于SFT模型训练
SFT_MODEL="outputs-sft-qwen2.5-3b/checkpoint-best"

CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
    --model_name_or_path $SFT_MODEL \
    --template_name qwen \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 5000 \
    --max_eval_samples 100 \
    --max_steps 2000 \
    --eval_steps 100 \
    --save_steps 500 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --learning_rate 5e-6 \
    --output_dir outputs-dpo-qwen2.5-3b \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --device_map auto \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache
```

**启动训练：**
```bash
chmod +x scripts/run_dpo_qwen2.5-3b.sh
nohup bash scripts/run_dpo_qwen2.5-3b.sh > logs/dpo_training.log 2>&1 &
```

### 4.4 阶段四：PPO训练 (高级，可选)

**注意：** PPO训练非常消耗显存，需要 2×A100 (80GB)

**创建训练脚本：** `scripts/run_ppo_qwen2.5-3b.sh`

```bash
#!/bin/bash
# PPO训练 - Qwen2.5-3B（需要大显存）

SFT_MODEL="outputs-sft-qwen2.5-3b/checkpoint-best"
RM_MODEL="outputs-rm-qwen2.5-3b/checkpoint-best"

CUDA_VISIBLE_DEVICES=0,1 python ppo_training.py \
    --sft_model_path $SFT_MODEL \
    --reward_model_path $RM_MODEL \
    --template_name qwen \
    --torch_dtype bfloat16 \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --max_source_length 1024 \
    --response_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --do_train \
    --total_episodes 10000 \
    --output_dir outputs-ppo-qwen2.5-3b \
    --missing_eos_penalty 1.0 \
    --eval_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --cache_dir ./cache
```

---

## 5. 模型评估与部署

### 5.1 本地测试

**创建测试脚本：** `scripts/test_model.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的医疗模型
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_model_path, lora_model_path=None):
    """加载模型"""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if lora_model_path:
        print(f"Loading LoRA from {lora_model_path}")
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()  # 合并LoRA权重
    
    return model, tokenizer

def chat(model, tokenizer, query, history=[]):
    """对话函数"""
    # 构建prompt
    messages = history + [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    
    return response

def main():
    # 配置
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    lora_model = "outputs-dpo-qwen2.5-3b/checkpoint-best"  # 使用DPO训练后的模型
    
    print("Loading model...")
    model, tokenizer = load_model(base_model, lora_model)
    
    print("\n" + "="*50)
    print("医疗问答模型测试")
    print("="*50)
    
    # 测试问题
    test_questions = [
        "感冒了应该怎么办？",
        "高血压患者在饮食上需要注意什么？",
        "糖尿病有哪些早期症状？",
        "如何预防心血管疾病？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print(f"回答: ", end="")
        response = chat(model, tokenizer, question)
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main()
```

**运行测试：**
```bash
python scripts/test_model.py
```

### 5.2 使用 vLLM 部署（高性能推理）

```bash
# 安装 vLLM
pip install vllm

# 合并LoRA权重（如果还没合并）
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_model outputs-dpo-qwen2.5-3b/checkpoint-best \
    --output_dir outputs-dpo-qwen2.5-3b-merged

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model outputs-dpo-qwen2.5-3b-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# 测试API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "outputs-dpo-qwen2.5-3b-merged",
    "messages": [{"role": "user", "content": "感冒了怎么办？"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### 5.3 模型评估

**创建评估脚本：** `scripts/evaluate_model.py`

```python
#!/usr/bin/env python3
"""
评估模型在医疗问答任务上的表现
"""
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def evaluate(model, tokenizer, test_file, num_samples=100):
    """评估模型"""
    # 加载测试数据
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f][:num_samples]
    
    total = 0
    correct = 0
    
    for item in tqdm(test_data):
        question = item['conversations'][0]['value']
        reference = item['conversations'][1]['value']
        
        # 生成回答
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # 简单评估（实际应该用更复杂的指标）
        print(f"\nQ: {question}")
        print(f"A: {response}")
        print(f"Ref: {reference}")
        
        total += 1
    
    print(f"\nEvaluated {total} samples")

if __name__ == "__main__":
    # 配置
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    lora_model = "outputs-dpo-qwen2.5-3b/checkpoint-best"
    test_file = "data/finetune/medical_sft_1K_format.jsonl"
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_model)
    
    # 评估
    evaluate(model, tokenizer, test_file, num_samples=10)
```

---

## 6. 常见问题与优化

### 6.1 显存不足 (OOM)

**问题：** `CUDA out of memory`

**解决方案：**
```bash
# 1. 减小batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 32  # 增大累积步数

# 2. 启用梯度检查点
--gradient_checkpointing True

# 3. 使用8bit量化
--load_in_8bit True

# 4. 减小模型序列长度
--model_max_length 1024  # 从2048降到1024

# 5. 减小LoRA rank
--lora_rank 8  # 从16降到8
```

### 6.2 训练速度慢

**优化建议：**
```bash
# 1. 启用FlashAttention
--flash_attn True

# 2. 启用bf16混合精度
--bf16 True
--torch_dtype bfloat16

# 3. 增大batch size（如果显存允许）
--per_device_train_batch_size 4

# 4. 减少eval频率
--eval_steps 1000  # 从100增加到1000

# 5. 启用torch编译（PyTorch 2.0+）
export TORCH_COMPILE=1
```

### 6.3 训练不稳定/Loss不下降

**解决方案：**
```bash
# 1. 调整学习率
--learning_rate 1e-5  # 尝试更小的学习率

# 2. 增加warmup
--warmup_ratio 0.1  # 从0.05增加到0.1

# 3. 调整weight decay
--weight_decay 0.01

# 4. 检查数据质量
# 确保数据格式正确，没有异常样本

# 5. 使用更稳定的优化器
--optim adamw_torch  # 或 adafactor
```

### 6.4 数据加载慢

**优化建议：**
```bash
# 1. 增加数据处理workers
--preprocessing_num_workers 16

# 2. 使用数据缓存
# 第二次训练会自动使用缓存，除非:
--overwrite_cache  # 删除这个参数

# 3. 预处理数据
python scripts/preprocess_data.py \
    --input_dir data/finetune \
    --output_dir data/finetune_processed
```

### 6.5 监控训练进度

**使用 TensorBoard：**
```bash
# 启动TensorBoard（在本地）
tensorboard --logdir outputs-sft-qwen2.5-3b --port 6006 --bind_all

# 如果是远程服务器，需要端口转发
# 在本地电脑运行：
ssh -L 6006:localhost:6006 user@server_ip

# 然后在浏览器访问：
http://localhost:6006
```

**使用 wandb：**
```bash
# 安装wandb
pip install wandb

# 登录
wandb login

# 修改训练脚本
--report_to wandb
--run_name medical-gpt-sft-qwen2.5-3b
```

### 6.6 断点续训

```bash
# 训练中断后，从最新checkpoint继续
--resume_from_checkpoint outputs-sft-qwen2.5-3b/checkpoint-500

# 或让程序自动找最新checkpoint
--resume_from_checkpoint True
```

---

## 7. 完整训练时间表

### 假设配置：2×A100 (40GB)

| 阶段 | 数据量 | 预计时间 | 输出 |
|-----|-------|---------|------|
| **PT** | 10万条文本 | 12-18小时 | `outputs-pt-qwen2.5-3b` |
| **SFT** | 1万条对话 | 4-6小时 | `outputs-sft-qwen2.5-3b` |
| **RM** | 5千条偏好 | 2-3小时 | `outputs-rm-qwen2.5-3b` |
| **DPO** | 5千条偏好 | 4-6小时 | `outputs-dpo-qwen2.5-3b` |
| **总计** | - | **22-33小时** | - |

### 预算估算（AutoDL，2×A100 40GB）

- **单价：** ~18元/小时
- **总时长：** 30小时
- **总费用：** ~540元

---

## 8. 快速开始脚本

**创建一键训练脚本：** `scripts/train_all.sh`

```bash
#!/bin/bash
# 一键完成所有训练阶段

set -e  # 遇到错误立即退出

echo "========================================="
echo "MedicalGPT 完整训练流程"
echo "模型: Qwen2.5-3B"
echo "========================================="

# 创建必要目录
mkdir -p logs outputs cache scripts

# 阶段1: 预训练（可选，跳过则从SFT开始）
read -p "是否进行预训练? (y/n): " do_pt
if [ "$do_pt" = "y" ]; then
    echo "\n[1/4] 开始预训练..."
    nohup bash scripts/run_pt_qwen2.5-3b.sh > logs/pt.log 2>&1
    echo "预训练完成！"
fi

# 阶段2: 监督微调
echo "\n[2/4] 开始监督微调..."
nohup bash scripts/run_sft_qwen2.5-3b.sh > logs/sft.log 2>&1
echo "监督微调完成！"

# 阶段3: 选择RM+PPO 或 DPO
echo "\n选择强化学习方法:"
echo "1) DPO (推荐，更简单)"
echo "2) RM + PPO (复杂，需要大显存)"
read -p "请选择 (1/2): " rl_method

if [ "$rl_method" = "1" ]; then
    echo "\n[3/4] 开始DPO训练..."
    nohup bash scripts/run_dpo_qwen2.5-3b.sh > logs/dpo.log 2>&1
    echo "DPO训练完成！"
else
    echo "\n[3/4] 开始RM训练..."
    nohup bash scripts/run_rm_qwen2.5-3b.sh > logs/rm.log 2>&1
    echo "RM训练完成！"
    
    echo "\n[4/4] 开始PPO训练..."
    nohup bash scripts/run_ppo_qwen2.5-3b.sh > logs/ppo.log 2>&1
    echo "PPO训练完成！"
fi

echo "\n========================================="
echo "所有训练完成！"
echo "模型保存在: outputs-*-qwen2.5-3b/"
echo "========================================="
```

**使用方法：**
```bash
chmod +x scripts/train_all.sh
bash scripts/train_all.sh
```

---

## 9. 附录

### 9.1 数据格式转换工具

**创建：** `scripts/convert_data_format.py`

```python
#!/usr/bin/env python3
"""
数据格式转换工具
"""
import json
import argparse

def convert_to_sft_format(input_file, output_file):
    """转换为SFT格式"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            # 假设原格式是 {"question": "...", "answer": "..."}
            sft_data = {
                "conversations": [
                    {"from": "human", "value": data["question"]},
                    {"from": "gpt", "value": data["answer"]}
                ]
            }
            
            f_out.write(json.dumps(sft_data, ensure_ascii=False) + '\n')

def convert_to_dpo_format(input_file, output_file):
    """转换为DPO格式"""
    # 假设原格式有 question, good_answer, bad_answer
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            dpo_data = {
                "system": "",
                "history": [],
                "question": data["question"],
                "response_chosen": data["good_answer"],
                "response_rejected": data["bad_answer"]
            }
            
            f_out.write(json.dumps(dpo_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文件")
    parser.add_argument("--output", required=True, help="输出文件")
    parser.add_argument("--format", choices=["sft", "dpo"], required=True)
    
    args = parser.parse_args()
    
    if args.format == "sft":
        convert_to_sft_format(args.input, args.output)
    else:
        convert_to_dpo_format(args.input, args.output)
    
    print(f"转换完成: {args.output}")
```

### 9.2 模型合并工具

**创建：** `scripts/merge_lora.py`

```python
#!/usr/bin/env python3
"""
合并LoRA权重到基础模型
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_path, lora_model_path, output_dir):
    """合并LoRA权重"""
    print(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA from {lora_model_path}")
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="基础模型路径")
    parser.add_argument("--lora_model", required=True, help="LoRA模型路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    merge_lora(args.base_model, args.lora_model, args.output_dir)
```

### 9.3 有用的命令

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep python

# 杀死训练进程
pkill -f "pretraining.py"

# 查看磁盘空间
df -h

# 压缩模型文件
tar -czf outputs-sft-qwen2.5-3b.tar.gz outputs-sft-qwen2.5-3b/

# 下载到本地（在本地运行）
scp -r user@server:/root/MedicalGPT/outputs-sft-qwen2.5-3b ./

# 清理缓存
rm -rf cache/
rm -rf ~/.cache/huggingface/
```

---

## 10. 总结与建议

### 推荐训练路径

1. **快速验证路径**（1-2天）：
   ```
   SFT (1万条) → DPO (5千条) → 部署测试
   ```

2. **完整训练路径**（3-5天）：
   ```
   PT (10万条) → SFT (5万条) → DPO (1万条) → 部署
   ```

3. **高级路径**（需大显存）：
   ```
   PT → SFT → RM → PPO → 部署
   ```

### 关键提示

- ✅ **从SFT开始**：如果没有大量领域文本，跳过PT直接SFT
- ✅ **优先使用DPO**：比PPO更简单，效果相当
- ✅ **数据质量>数量**：1万条高质量数据优于10万条低质量数据
- ✅ **定期保存checkpoint**：避免训练中断导致重来
- ✅ **监控显存使用**：及时调整batch size避免OOM

### 获取帮助

- 项目GitHub: https://github.com/shibing624/MedicalGPT
- Issues: https://github.com/shibing624/MedicalGPT/issues
- Qwen文档: https://qwenlm.github.io/

---

**祝训练顺利！🎉**
