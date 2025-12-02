# Scripts 目录说明

本目录包含所有训练和测试脚本。

## 📁 文件列表

### 训练脚本

| 脚本 | 用途 | 运行时间 | GPU要求 |
|------|------|---------|---------|
| `run_pt_qwen2.5-3b.sh` | 预训练（可选） | 12-18小时 | 2×A100 |
| `run_sft_qwen2.5-3b.sh` | 监督微调（必需） | 4-8小时 | 2×A100 |
| `run_rm_qwen2.5-3b.sh` | 奖励建模（PPO用） | 2-4小时 | 1×A100 |
| `run_dpo_qwen2.5-3b.sh` | DPO训练（推荐） | 4-6小时 | 2×A100 |
| `run_ppo_qwen2.5-3b.sh` | PPO训练（高级） | 8-12小时 | 2×A100 80GB |

### 工具脚本

| 脚本 | 用途 |
|------|------|
| `check_environment.py` | 环境检查工具 |
| `test_model.py` | 模型测试工具 |
| `merge_lora.py` | LoRA权重合并工具 |

---

## 🚀 快速开始

### 1. 环境检查

```bash
# 首次运行前检查环境
python scripts/check_environment.py
```

### 2. 开始训练

```bash
# 最简单的流程：SFT训练
chmod +x scripts/run_sft_qwen2.5-3b.sh
nohup bash scripts/run_sft_qwen2.5-3b.sh > logs/sft.log 2>&1 &

# 监控训练
tail -f logs/sft.log
```

### 3. 测试模型

```bash
# 测试训练好的模型
python scripts/test_model.py
```

### 4. 合并LoRA（部署用）

```bash
python scripts/merge_lora.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_model outputs-sft-qwen2.5-3b/checkpoint-best \
    --output_dir medical-gpt-merged
```

---

## 📝 训练脚本说明

### run_sft_qwen2.5-3b.sh

**功能：** 监督微调，训练医疗问答能力

**关键参数：**
```bash
--model_name_or_path Qwen/Qwen2.5-3B-Instruct  # 基础模型
--train_file_dir ./data/finetune               # 训练数据
--per_device_train_batch_size 2                # 批次大小
--num_train_epochs 3                           # 训练轮数
--learning_rate 2e-5                           # 学习率
--lora_rank 16                                 # LoRA秩
```

**调整建议：**
- 显存不足？减小 `batch_size` 和 `lora_rank`
- 数据量大？增加 `num_train_epochs`
- 想要更好效果？增大 `lora_rank` (16→32)

### run_dpo_qwen2.5-3b.sh

**功能：** 直接偏好优化，提升回答质量

**关键参数：**
```bash
--model_name_or_path outputs-sft-qwen2.5-3b/checkpoint-best  # 基于SFT模型
--train_file_dir ./data/reward                               # 偏好数据
--learning_rate 5e-6                                         # 较小学习率
--max_steps 2000                                             # 训练步数
```

**注意事项：**
- 必须先完成SFT训练
- DPO比PPO更简单，推荐使用
- 学习率要比SFT小

---

## 🛠️ 工具脚本说明

### check_environment.py

**功能：** 检查训练环境是否配置正确

**检查项目：**
- Python版本
- CUDA和GPU
- 必需的Python包
- 数据文件
- 磁盘空间
- HuggingFace访问

**使用方法：**
```bash
python scripts/check_environment.py
```

### test_model.py

**功能：** 测试训练好的模型

**支持两种模式：**
1. 批量测试：使用预设问题
2. 交互模式：实时对话

**修改测试模型：**
编辑 `test_model.py` 中的配置：
```python
lora_model = "outputs-sft-qwen2.5-3b/checkpoint-best"  # 改为你的模型路径
```

### merge_lora.py

**功能：** 合并LoRA权重，便于部署

**使用场景：**
- 部署到生产环境
- 使用vLLM等推理框架
- 分享完整模型

**注意：** 合并后模型体积会增大（约6GB）

---

## 📊 训练监控

### 方法1: 查看日志
```bash
tail -f logs/sft.log
```

### 方法2: TensorBoard
```bash
tensorboard --logdir outputs-sft-qwen2.5-3b --port 6006
# 远程服务器需要端口转发:
# ssh -L 6006:localhost:6006 user@server
```

### 方法3: GPU监控
```bash
watch -n 1 nvidia-smi
```

---

## ⚙️ 常见调整

### 显存不足 (OOM)
```bash
# 在训练脚本中修改：
--per_device_train_batch_size 1     # 减小批次
--gradient_accumulation_steps 32    # 增大累积
--lora_rank 8                       # 减小LoRA秩
```

### 训练太慢
```bash
# 增大批次（如果显存允许）
--per_device_train_batch_size 4

# 减少评估频率
--eval_steps 1000

# 启用FlashAttention
--flash_attn True
```

### 效果不好
```bash
# 增大LoRA秩
--lora_rank 32

# 增加训练轮数
--num_train_epochs 5

# 调整学习率
--learning_rate 1e-5
```

---

## 📞 获取帮助

遇到问题？
1. 检查 `TRAINING_GUIDE_Qwen2.5-3B.md` 详细指南
2. 运行 `python scripts/check_environment.py` 诊断环境
3. 查看项目 Issues: https://github.com/shibing624/MedicalGPT/issues
