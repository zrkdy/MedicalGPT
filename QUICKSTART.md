# MedicalGPT 快速开始指南 - Qwen2.5-3B

## 🚀 最快 5 分钟开始训练

### 步骤 1: 环境准备（一次性）

```bash
# 1.1 克隆项目（在服务器上）
cd /root
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT

# 1.2 创建环境
conda create -n medical python=3.10 -y
conda activate medical

# 1.3 安装依赖
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install bitsandbytes

# 1.4 配置 HF（国内服务器）
export HF_ENDPOINT=https://hf-mirror.com
```

### 步骤 2: 准备数据（使用示例数据）

```bash
# 项目已包含示例数据，无需额外下载
ls data/finetune/  # 查看SFT数据
ls data/reward/    # 查看DPO数据
```

### 步骤 3: 开始训练

#### 方案A: 快速验证（推荐新手，~8小时）

```bash
# 1. 监督微调（核心步骤）
chmod +x scripts/run_sft_qwen2.5-3b.sh
nohup bash scripts/run_sft_qwen2.5-3b.sh > logs/sft.log 2>&1 &

# 监控训练
tail -f logs/sft.log

# 2. DPO优化（可选，提升质量）
chmod +x scripts/run_dpo_qwen2.5-3b.sh
# 等SFT完成后运行
nohup bash scripts/run_dpo_qwen2.5-3b.sh > logs/dpo.log 2>&1 &
```

#### 方案B: 完整训练（~30小时）

```bash
# 1. 预训练（可选）
bash scripts/run_pt_qwen2.5-3b.sh

# 2. 监督微调
bash scripts/run_sft_qwen2.5-3b.sh

# 3. DPO优化
bash scripts/run_dpo_qwen2.5-3b.sh
```

### 步骤 4: 测试模型

```bash
# 测试训练好的模型
python scripts/test_model.py

# 或合并LoRA后部署
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_model outputs-dpo-qwen2.5-3b/checkpoint-best \
    --output_dir medical-gpt-final
```

---

## 📊 资源需求

| 训练阶段 | 最低配置 | 推荐配置 | 预计时间 |
|---------|---------|---------|---------|
| **SFT** | 1×RTX 4090 | 2×A100 40GB | 4-8小时 |
| **DPO** | 1×RTX 4090 | 2×A100 40GB | 4-6小时 |

**预算（AutoDL）：**
- 2×RTX 4090: ~10元/小时 × 12小时 = **120元**
- 2×A100 40GB: ~18元/小时 × 10小时 = **180元**

---

## 🔥 常见问题

### Q1: 显存不足怎么办？

```bash
# 在训练脚本中减小参数：
--per_device_train_batch_size 1
--gradient_accumulation_steps 32
--lora_rank 8
```

### Q2: 如何查看训练进度？

```bash
# 方法1: 查看日志
tail -f logs/sft.log

# 方法2: TensorBoard
tensorboard --logdir outputs-sft-qwen2.5-3b --port 6006

# 方法3: GPU监控
watch -n 1 nvidia-smi
```

### Q3: 训练中断了怎么办？

```bash
# 脚本会自动从最新checkpoint继续
# 只需重新运行相同的命令即可
bash scripts/run_sft_qwen2.5-3b.sh
```

### Q4: 如何使用自己的数据？

```bash
# SFT数据格式（JSONL）：
{"conversations":[{"from":"human","value":"问题"},{"from":"gpt","value":"回答"}]}

# DPO数据格式（JSONL）：
{"question":"问题","response_chosen":"好回答","response_rejected":"差回答"}

# 放到对应目录：
data/finetune/my_data.jsonl
data/reward/my_preference.jsonl
```

---

## 📝 训练流程图

```
┌──────────────┐
│ Qwen2.5-3B   │  基础模型
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ SFT 训练     │  学习医疗问答（必需）
│ 4-8小时      │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ DPO 训练     │  优化回答质量（推荐）
│ 4-6小时      │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ 最终模型     │  部署使用
└──────────────┘
```

---

## 🎯 推荐训练路径

**初学者：**
```bash
SFT（使用示例数据1K条） → 测试
预计: 2小时，费用: ~30元
```

**实践者：**
```bash
SFT（1万条） → DPO（5千条） → 测试
预计: 12小时，费用: ~180元
```

**完整版：**
```bash
PT（10万条） → SFT（5万条） → DPO（1万条） → 部署
预计: 30小时，费用: ~540元
```

---

## 📞 获取帮助

- **详细指南**: `TRAINING_GUIDE_Qwen2.5-3B.md`
- **项目Issues**: https://github.com/shibing624/MedicalGPT/issues
- **Qwen文档**: https://qwenlm.github.io/

---

**开始你的医疗AI之旅！** 🚀
