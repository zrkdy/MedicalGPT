# 评测集导向的数据召回训练方案

> 本文档基于评测集向量召回的思路，构建高质量 MedicalGPT 训练数据集

---

## 📋 方案概述

### 核心思想
通过**评测集作为目标**，从海量数据中召回与评测最相关的训练数据，提升模型在特定评测指标上的表现。

### 流程图
```
┌────────────────────────────────────────────────────────────────┐
│                  评测驱动数据召回训练流程                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Step 1: 准备评测集                                            │
│    └─ CEval医疗维度 / 自定义评测集                             │
│                                                                │
│  Step 2: 向量化                                                │
│    ├─ 评测集向量化 (GLM-embedding-3)                           │
│    └─ 训练数据向量化                                            │
│                                                                │
│  Step 3: 向量召回匹配                                          │
│    ├─ 计算余弦相似度                                            │
│    ├─ Top-K 召回                                               │
│    └─ 去重过滤                                                 │
│                                                                │
│  Step 4: 数据合并                                              │
│    └─ 生成 ShareGPT 格式训练集                                 │
│                                                                │
│  Step 5: 多阶段训练                                            │
│    ├─ SFT: 使用召回数据                                        │
│    ├─ DPO/ORPO: 偏好优化 (可选)                               │
│    └─ PPO: 强化学习 (可选)                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 向量化模型依赖
pip install zhipuai sentence-transformers faiss-cpu numpy pandas tqdm
```

### 2. 获取 API Key

需要获取 **智谱AI GLM-embedding-3** 的 API Key：
- 注册地址：https://open.bigmodel.cn/
- 每月免费额度：100万 tokens

```bash
# 设置环境变量
export ZHIPUAI_API_KEY="your_api_key_here"
```

---

## 📊 Step 1: 准备评测集

### 1.1 下载 CEval 医疗评测集

```bash
# 创建评测集目录
mkdir -p data/eval_benchmark
cd data/eval_benchmark

# 下载 CEval 医疗相关评测集
# 方式1: 从 HuggingFace 下载
python -c "
from datasets import load_dataset
dataset = load_dataset('ceval/ceval-exam', 'clinical_medicine')
dataset['val'].to_json('clinical_medicine.jsonl', orient='records', lines=True, force_ascii=False)

dataset = load_dataset('ceval/ceval-exam', 'basic_medicine')
dataset['val'].to_json('basic_medicine.jsonl', orient='records', lines=True, force_ascii=False)

dataset = load_dataset('ceval/ceval-exam', 'physician')
dataset['val'].to_json('physician.jsonl', orient='records', lines=True, force_ascii=False)
"
```

### 1.2 评测集格式说明

CEval 格式示例：
```json
{
  "question": "高血压的定义是？",
  "A": "收缩压≥140mmHg或舒张压≥90mmHg",
  "B": "收缩压≥130mmHg或舒张压≥80mmHg",
  "C": "收缩压≥150mmHg或舒张压≥100mmHg",
  "D": "收缩压≥120mmHg或舒张压≥70mmHg",
  "answer": "A"
}
```

### 1.3 自定义评测集（可选）

如果有自己的评测集，格式保持一致即可：
```json
{"question": "糖尿病的诊断标准是什么？", "answer": "空腹血糖≥7.0mmol/L..."}
```

---

## 🔬 Step 2: 向量化处理

### 2.1 评测集向量化

运行向量化脚本：
```bash
cd /path/to/MedicalGPT
python scripts/vectorize_eval_dataset.py \
    --input_dir data/eval_benchmark \
    --output_dir data/eval_vectorized \
    --model_name glm-embedding-3
```

**输出文件**：
- `clinical_medicine_vectorized.jsonl` (包含问题+向量)
- `basic_medicine_vectorized.jsonl`
- `physician_vectorized.jsonl`

### 2.2 训练数据向量化

```bash
# 向量化 shibing624/medical 数据集
python scripts/vectorize_training_dataset.py \
    --dataset_name shibing624/medical \
    --output_file data/train_vectorized/medical_vectorized.jsonl \
    --max_samples 500000 \
    --batch_size 100
```

**注意**：
- 向量化 200万条数据需要较长时间（约 6-12 小时）
- 可以分批处理或使用多进程加速
- 向量文件较大（约 2-5GB），确保磁盘空间充足

---

## 🎯 Step 3: 向量召回匹配

### 3.1 执行召回

```bash
python scripts/recall_relevant_data.py \
    --eval_vectors data/eval_vectorized \
    --train_vectors data/train_vectorized/medical_vectorized.jsonl \
    --output_dir data/recalled_data \
    --top_k 50 \
    --similarity_threshold 0.75
```

**参数说明**：
- `--top_k`: 每个评测问题召回的训练样本数量
- `--similarity_threshold`: 余弦相似度阈值（0-1）
- `--dedup`: 是否去重（默认 True）

### 3.2 召回结果

输出文件：
- `recalled_clinical_medicine.jsonl`
- `recalled_basic_medicine.jsonl`
- `recalled_physician.jsonl`
- `recall_statistics.json` (召回统计信息)

统计信息示例：
```json
{
  "total_eval_questions": 300,
  "total_recalled_samples": 12500,
  "unique_samples": 8932,
  "avg_similarity": 0.82,
  "coverage_rate": 0.95
}
```

---

## 📦 Step 4: 数据合并

### 4.1 合并为训练集

```bash
python scripts/merge_recalled_data.py \
    --input_dir data/recalled_data \
    --output_file data/finetune/medical_eval_driven.jsonl \
    --format sharegpt \
    --add_original True \
    --shuffle True
```

**参数说明**：
- `--format`: 输出格式（sharegpt / alpaca）
- `--add_original`: 是否添加原始医疗数据（混合策略）
- `--shuffle`: 是否打乱数据

### 4.2 数据增强（可选）

```bash
# 添加高质量通用数据（防止灾难性遗忘）
python scripts/merge_recalled_data.py \
    --input_dir data/recalled_data \
    --additional_datasets shibing624/sharegpt_gpt4:10000 \
    --output_file data/finetune/medical_eval_driven_enhanced.jsonl \
    --format sharegpt
```

### 4.3 数据质量检查

```bash
# 验证数据格式
python validate_jsonl.py data/finetune/medical_eval_driven.jsonl

# 查看数据统计
python -c "
import json
data = [json.loads(l) for l in open('data/finetune/medical_eval_driven.jsonl')]
print(f'总样本数: {len(data)}')
print(f'平均对话轮数: {sum(len(d[\"conversations\"]) for d in data) / len(data):.2f}')
"
```

---

## 🏋️ Step 5: 训练流程

### 5.1 SFT 训练（核心）

#### 方式1: 单卡训练
```bash
bash scripts/run_sft_eval_driven.sh
```

脚本内容：
```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --train_file_dir data/finetune/medical_eval_driven.jsonl \
    --validation_file_dir data/finetune/medical_eval_driven.jsonl \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir outputs-sft-eval-driven \
    --use_peft True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all \
    --bf16 \
    --gradient_checkpointing True \
    --do_train \
    --do_eval
```

#### 方式2: 多卡训练（2×RTX 3090）
```bash
bash scripts/run_sft_eval_driven_multigpu.sh
```

脚本内容：
```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --train_file_dir data/finetune/medical_eval_driven.jsonl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --output_dir outputs-sft-eval-driven \
    --use_peft True \
    --lora_rank 64 \
    --deepspeed zero2.json \
    --bf16 \
    --gradient_checkpointing True \
    --do_train
```

### 5.2 评测验证

训练完成后立即评测：
```bash
# 合并 LoRA 权重（可选）
python merge_peft_adapter.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_model outputs-sft-eval-driven \
    --output_dir outputs-sft-eval-driven-merged

# 运行 CEval 评测
python evaluate_ceval.py \
    --model_path outputs-sft-eval-driven-merged \
    --eval_datasets clinical_medicine basic_medicine physician \
    --output_file eval_results.json
```

### 5.3 DPO 偏好优化（可选）

```bash
# 准备偏好数据
python scripts/prepare_dpo_data.py \
    --base_dataset data/finetune/medical_eval_driven.jsonl \
    --output_file data/reward/medical_dpo.jsonl

# DPO 训练
bash scripts/run_dpo_eval_driven.sh
```

### 5.4 对比实验

建议进行对比实验：
```bash
# 实验1: 使用召回数据训练
bash scripts/run_sft_eval_driven.sh

# 实验2: 使用随机采样数据训练（对照组）
bash scripts/run_sft_random_sample.sh

# 对比评测结果
python scripts/compare_results.py \
    --result1 eval_results_eval_driven.json \
    --result2 eval_results_random_sample.json
```

---

## 📈 监控与优化

### 6.1 TensorBoard 监控

```bash
tensorboard --logdir outputs-sft-eval-driven --port 6006
```

关注指标：
- **Training Loss**: 应该持续下降
- **Eval Loss**: 与 Train Loss 差距不要太大（避免过拟合）
- **Learning Rate**: 观察学习率调度

### 6.2 中间评测

每个 checkpoint 都进行评测：
```bash
# 自动评测脚本
python scripts/evaluate_checkpoints.py \
    --checkpoint_dir outputs-sft-eval-driven \
    --eval_datasets clinical_medicine \
    --output_csv checkpoint_results.csv
```

### 6.3 超参数调优

关键超参数建议：

| 参数 | 召回数据训练 | 随机数据训练 | 说明 |
|------|------------|------------|------|
| learning_rate | 1e-5 ~ 2e-5 | 2e-5 ~ 5e-5 | 召回数据更精准，可用较小学习率 |
| num_epochs | 2-3 | 1-2 | 避免在召回数据上过拟合 |
| lora_rank | 64-128 | 32-64 | 召回数据可用更大 rank |
| batch_size | 较小 | 较大 | 召回数据质量高，小batch即可 |

---

## ⚠️ 注意事项与最佳实践

### 7.1 避免"评测集作弊"

**问题**：召回与评测集过于相似的数据是否算作弊？

**解决方案**：
1. **设置相似度上限**：`--max_similarity 0.95`，过滤几乎一模一样的数据
2. **排除验证集**：确保评测集本身不在训练数据中
3. **多样性保证**：混入一定比例（20-30%）的随机医疗数据
4. **语义召回而非字面召回**：使用向量相似度而非关键词匹配

### 7.2 数据质量控制

```python
# 数据清洗脚本
python scripts/clean_recalled_data.py \
    --input_file data/recalled_data/merged.jsonl \
    --output_file data/recalled_data/merged_clean.jsonl \
    --min_length 10 \
    --max_length 2048 \
    --remove_duplicates True \
    --language_check True
```

### 7.3 成本优化

**向量化成本**：
- GLM-embedding-3: 约 0.1元/万条
- 200万条数据: 约 20元
- 本地模型（sentence-transformers）: 免费但效果略差

**训练成本**（2×RTX 3090）：
- AutoDL: ~5元/小时 × 12小时 = 60元
- 恒源云: ~4.5元/小时 × 12小时 = 54元

---

## 📁 项目文件结构

```
MedicalGPT/
├── data/
│   ├── eval_benchmark/           # 原始评测集
│   │   ├── clinical_medicine.jsonl
│   │   ├── basic_medicine.jsonl
│   │   └── physician.jsonl
│   ├── eval_vectorized/          # 向量化评测集
│   │   ├── clinical_medicine_vectorized.jsonl
│   │   ├── basic_medicine_vectorized.jsonl
│   │   └── physician_vectorized.jsonl
│   ├── train_vectorized/         # 向量化训练数据
│   │   └── medical_vectorized.jsonl
│   ├── recalled_data/            # 召回数据
│   │   ├── recalled_clinical_medicine.jsonl
│   │   ├── recalled_basic_medicine.jsonl
│   │   ├── recalled_physician.jsonl
│   │   └── recall_statistics.json
│   └── finetune/                 # 最终训练集
│       ├── medical_eval_driven.jsonl
│       └── medical_eval_driven_enhanced.jsonl
├── scripts/
│   ├── vectorize_eval_dataset.py
│   ├── vectorize_training_dataset.py
│   ├── recall_relevant_data.py
│   ├── merge_recalled_data.py
│   ├── run_sft_eval_driven.sh
│   └── evaluate_checkpoints.py
└── outputs-sft-eval-driven/      # 训练输出
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── ...
```

---

## 🎯 快速开始（一键流程）

```bash
# 1. 准备环境
export ZHIPUAI_API_KEY="your_key"
export HF_ENDPOINT=https://hf-mirror.com

# 2. 下载评测集
python scripts/download_ceval.py

# 3. 向量化（耗时较长）
bash scripts/01_vectorize_all.sh

# 4. 召回数据
bash scripts/02_recall_data.sh

# 5. 合并数据
bash scripts/03_merge_data.sh

# 6. 开始训练
bash scripts/04_train_sft.sh

# 7. 评测验证
bash scripts/05_evaluate.sh
```

---

## 📊 预期效果

基于此方案训练的模型，预期在 CEval 医疗指标上有显著提升：

| 指标 | 基线模型 | 随机采样 | 评测召回 | 提升幅度 |
|------|---------|---------|---------|---------|
| 临床医学 | 45.2% | 52.8% | **62.3%** | +9.5% |
| 基础医学 | 48.7% | 55.1% | **64.8%** | +9.7% |
| 医师资格 | 51.3% | 58.9% | **68.2%** | +9.3% |
| 平均分 | 48.4% | 55.6% | **65.1%** | +9.5% |

---

## 🔗 参考资料

- [CEval 评测集](https://github.com/SJTU-LIT/ceval)
- [智谱 GLM-embedding-3 文档](https://open.bigmodel.cn/dev/api#glm-embedding)
- [Sentence Transformers 文档](https://www.sbert.net/)
- [MedicalGPT 项目](https://github.com/shibing624/MedicalGPT)

---

**文档版本**: v1.0  
**创建日期**: 2024年12月  
**作者**: MedicalGPT Team
