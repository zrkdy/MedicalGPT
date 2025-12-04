# 快速开始：本地准备 + 服务器训练

> 3分钟速查表，快速完成本地准备和服务器训练

---

## 📍 总体流程

```
本地 Windows (6-12小时) → 传输 (30分钟) → 服务器 Linux (12-24小时)
     数据准备               Git + SCP              GPU 训练
```

---

## 💻 本地操作（Windows）

### 方式1: 一键脚本（推荐）

```powershell
# 设置 API Key（可选）
$env:ZHIPUAI_API_KEY="your_api_key"

# 执行准备脚本
.\local_prepare.ps1 -MaxSamples 100000

# 或使用 Python
python scripts/local_prepare.py --max_samples 100000
```

### 方式2: 分步执行

```powershell
# 1. 下载评测集
python scripts/download_ceval.py

# 2. 向量化评测集
python scripts/vectorize_eval_dataset.py `
    --input_dir data/eval_benchmark `
    --output_dir data/eval_vectorized

# 3. 向量化训练数据（最耗时）
python scripts/vectorize_training_dataset.py `
    --dataset_name shibing624/medical `
    --output_file data/train_vectorized/medical_vectorized.jsonl `
    --max_samples 100000

# 4. 召回数据
python scripts/recall_relevant_data.py `
    --eval_vectors data/eval_vectorized `
    --train_vectors data/train_vectorized/medical_vectorized.jsonl `
    --output_dir data/recalled_data

# 5. 合并数据
python scripts/merge_recalled_data.py `
    --input_dir data/recalled_data `
    --output_file data/finetune/medical_eval_driven.jsonl

# 6. 验证数据
python scripts/verify_data.py
```

---

## 📤 传输到服务器

### 方式1: Git + 对象存储（推荐）

```powershell
# 本地提交小文件
git add data/eval_benchmark/ data/eval_vectorized/ data/recalled_data/ data/finetune/
git add scripts/ *.md .gitignore
git commit -m "Add prepared training data and scripts"
git push

# 上传大文件到阿里云 OSS
ossutil cp -r data\train_vectorized\ oss://your-bucket/medicalgpt/
```

```bash
# 服务器拉取代码
cd /root
git clone https://github.com/yourusername/MedicalGPT.git
cd MedicalGPT

# 下载大文件
ossutil cp -r oss://your-bucket/medicalgpt/train_vectorized/ data/
```

### 方式2: Git + SCP

```powershell
# 本地提交 Git
git add . && git commit -m "Add data" && git push

# 压缩大文件
Compress-Archive -Path data\train_vectorized -DestinationPath train_vectorized.zip

# SCP 上传
scp train_vectorized.zip root@your-server:/root/
```

```bash
# 服务器解压
cd /root/MedicalGPT
unzip /root/train_vectorized.zip -d data/
```

### 方式3: WinSCP 图形界面

1. 下载 WinSCP: https://winscp.net/
2. 连接到服务器
3. 拖拽上传 `data/train_vectorized/` 目录

---

## 🚀 服务器训练（Linux）

### 1. 环境准备

```bash
# 连接服务器
ssh root@your-server-ip

# 进入目录
cd /root/MedicalGPT

# 拉取代码（如果用 Git）
git pull

# 验证数据
python scripts/verify_data.py

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 单卡训练（1×RTX 3090）

```bash
bash scripts/run_sft_eval_driven.sh
```

### 3. 多卡训练（2×RTX 3090）

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --train_file_dir data/finetune/medical_eval_driven.jsonl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --output_dir outputs-sft-eval-driven \
    --use_peft True \
    --lora_rank 64 \
    --deepspeed zero2.json \
    --bf16
```

### 4. 后台运行（推荐）

```bash
nohup bash scripts/run_sft_eval_driven.sh > train.log 2>&1 &

# 查看日志
tail -f train.log

# 查看进程
ps aux | grep python
```

---

## 📊 文件大小参考

| 文件/目录 | 大小 | 传输方式 |
|----------|------|---------|
| 评测集 | ~200KB | Git |
| 评测集向量 | ~20MB | Git |
| 训练数据向量 | **2-5GB** | OSS/SCP/WinSCP |
| 召回数据 | ~50MB | Git |
| 最终训练集 | ~50MB | Git |

---

## ⏱️ 时间估算

| 阶段 | 10万样本 | 50万样本 |
|------|---------|---------|
| 本地准备 | 2-4小时 | 6-12小时 |
| 文件传输 | 10-30分钟 | 30-60分钟 |
| 服务器训练 | 12小时 | 18-24小时 |
| **总计** | **14-16小时** | **24-36小时** |

---

## 🔧 常用命令

### 本地

```powershell
# 检查文件大小
Get-ChildItem -Recurse data\ | Measure-Object -Property Length -Sum

# 压缩文件
Compress-Archive -Path data\train_vectorized -DestinationPath train_vectorized.zip

# 计算文件哈希
Get-FileHash train_vectorized.zip -Algorithm MD5
```

### 服务器

```bash
# 查看 GPU 状态
nvidia-smi

# 查看磁盘空间
df -h

# 查看文件大小
du -sh data/*

# 监控训练进程
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir outputs-sft-eval-driven --port 6006
```

---

## ⚠️ 常见问题

### Q: 向量化太慢？
```powershell
# 使用本地模型（免费，更快）
--model_name paraphrase-multilingual-MiniLM-L12-v2

# 或减少样本数
--max_samples 10000
```

### Q: 传输中断怎么办？
```bash
# 使用 rsync（支持断点续传）
rsync -avz --progress data/ root@server:/root/MedicalGPT/data/
```

### Q: 服务器显存不够？
```bash
# 使用 4bit 量化
--load_in_4bit True \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### Q: 训练中断如何恢复？
```bash
# 从最近的 checkpoint 恢复
--resume_from_checkpoint outputs-sft-eval-driven/checkpoint-1000
```

---

## 📝 检查清单

### 本地准备完成

- [ ] 评测集已下载 `data/eval_benchmark/`
- [ ] 评测集已向量化 `data/eval_vectorized/`
- [ ] 训练数据已向量化 `data/train_vectorized/`
- [ ] 数据已召回 `data/recalled_data/`
- [ ] 训练集已生成 `data/finetune/medical_eval_driven.jsonl`
- [ ] 数据验证通过 `python scripts/verify_data.py`

### 传输到服务器

- [ ] 代码已推送到 Git
- [ ] 大文件已上传（OSS/SCP/WinSCP）
- [ ] 服务器已拉取代码 `git pull`
- [ ] 大文件已下载并解压
- [ ] 服务器数据验证通过

### 开始训练

- [ ] GPU 可用 `nvidia-smi`
- [ ] 依赖已安装 `pip install -r requirements.txt`
- [ ] 环境变量已设置
- [ ] 训练脚本已启动
- [ ] 日志正常输出

---

## 📚 相关文档

- **详细指南**: [LOCAL_PREPARE_GUIDE.md](LOCAL_PREPARE_GUIDE.md)
- **完整流程**: [EVAL_DRIVEN_TRAINING_GUIDE.md](EVAL_DRIVEN_TRAINING_GUIDE.md)
- **快速入门**: [EVAL_DRIVEN_QUICKSTART.md](EVAL_DRIVEN_QUICKSTART.md)
- **主文档**: [README_EVAL_DRIVEN.md](README_EVAL_DRIVEN.md)

---

## 🎯 一句话命令

```powershell
# 本地：一键准备
.\local_prepare.ps1

# 服务器：一键训练
bash scripts/run_sft_eval_driven.sh
```

---

**更新**: 2024年12月  
**版本**: v1.0
