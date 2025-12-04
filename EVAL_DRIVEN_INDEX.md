# 评测驱动训练 - 文档索引

> 快速找到您需要的文档和脚本

---

## 📖 文档列表

### 🚀 快速开始

| 文档 | 用途 | 推荐度 |
|------|------|--------|
| [QUICK_START_LOCAL_SERVER.md](QUICK_START_LOCAL_SERVER.md) | **3分钟速查** | ⭐⭐⭐⭐⭐ |
| [EVAL_DRIVEN_OPTIMIZATION.md](EVAL_DRIVEN_OPTIMIZATION.md) | **优化方案（新）** | ⭐⭐⭐⭐⭐ |
| [EVAL_DRIVEN_QUICKSTART.md](EVAL_DRIVEN_QUICKSTART.md) | 5分钟快速入门 | ⭐⭐⭐⭐⭐ |
| [README_EVAL_DRIVEN.md](README_EVAL_DRIVEN.md) | 方案总览 | ⭐⭐⭐⭐ |

### 📚 详细教程

| 文档 | 用途 | 适用场景 |
|------|------|---------|
| [EVAL_DRIVEN_TRAINING_GUIDE.md](EVAL_DRIVEN_TRAINING_GUIDE.md) | 完整操作指南（60页） | 深入学习 |
| [LOCAL_PREPARE_GUIDE.md](LOCAL_PREPARE_GUIDE.md) | 本地准备详细流程 | 本地+服务器分离 |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) | 训练计划汇总 | 了解训练阶段 |

---

## 🐍 脚本列表

### 核心脚本（数据准备）

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `scripts/download_ceval.py` | 下载评测集 | - | `data/eval_benchmark/` |
| `scripts/vectorize_eval_dataset.py` | 向量化评测集 | 评测集 | `data/eval_vectorized/` |
| `scripts/vectorize_training_dataset.py` | 向量化训练数据 | HF数据集 | `data/train_vectorized/` |
| `scripts/recall_relevant_data.py` | 召回相关数据 | 向量数据 | `data/recalled_data/` |
| `scripts/merge_recalled_data.py` | 合并训练集 | 召回数据 | `data/finetune/*.jsonl` |

### 辅助脚本

| 脚本 | 功能 | 说明 |
|------|------|------|
| `scripts/local_prepare.py` | 本地一键准备 | Python自动化脚本 |
| `local_prepare.ps1` | 本地一键准备 | PowerShell自动化脚本 |
| `scripts/verify_data.py` | 验证数据完整性 | 检查所有文件 |
| `scripts/evaluate_model.py` | 模型评测 | CEval评测 |

### 训练脚本

| 脚本 | 功能 | 说明 |
|------|------|------|
| `scripts/run_sft_eval_driven.sh` | SFT训练（单卡） | 使用召回数据 |
| `scripts/quick_start_eval_driven.sh` | 完整流程（Linux） | 一键自动化 |

---

## 📁 目录结构

```
MedicalGPT/
├── 📖 文档
│   ├── README_EVAL_DRIVEN.md              ⭐ 方案总览
│   ├── QUICK_START_LOCAL_SERVER.md        ⭐⭐⭐ 3分钟速查
│   ├── EVAL_DRIVEN_QUICKSTART.md          ⭐⭐ 快速入门
│   ├── EVAL_DRIVEN_TRAINING_GUIDE.md      📚 完整指南
│   ├── LOCAL_PREPARE_GUIDE.md             💻 本地准备指南
│   └── EVAL_DRIVEN_INDEX.md               📑 本文档
│
├── 🐍 脚本
│   ├── scripts/
│   │   ├── download_ceval.py              下载评测集
│   │   ├── vectorize_eval_dataset.py      向量化评测集
│   │   ├── vectorize_training_dataset.py  向量化训练数据
│   │   ├── recall_relevant_data.py        召回数据
│   │   ├── merge_recalled_data.py         合并数据
│   │   ├── local_prepare.py               本地准备（Python）
│   │   ├── verify_data.py                 验证数据
│   │   ├── evaluate_model.py              评测模型
│   │   ├── run_sft_eval_driven.sh         SFT训练
│   │   └── quick_start_eval_driven.sh     一键流程
│   └── local_prepare.ps1                  本地准备（PowerShell）
│
├── 📦 配置
│   ├── requirements_eval_driven.txt       额外依赖
│   └── .gitignore                         Git配置（已更新）
│
└── 📊 数据（生成）
    └── data/
        ├── eval_benchmark/                原始评测集
        ├── eval_vectorized/               评测集向量
        ├── train_vectorized/              训练数据向量（大）
        ├── recalled_data/                 召回数据
        └── finetune/                      最终训练集
```

---

## 🎯 使用场景导航

### 场景1: 我想快速了解这个方案

👉 阅读顺序：
1. [QUICK_START_LOCAL_SERVER.md](QUICK_START_LOCAL_SERVER.md) - 3分钟速查
2. [README_EVAL_DRIVEN.md](README_EVAL_DRIVEN.md) - 方案总览

### 场景2: 我要在本地准备数据

👉 操作流程：
1. 阅读 [LOCAL_PREPARE_GUIDE.md](LOCAL_PREPARE_GUIDE.md)
2. 运行 `.\local_prepare.ps1` 或 `python scripts/local_prepare.py`
3. 验证数据 `python scripts/verify_data.py`

### 场景3: 我要在服务器训练

👉 操作流程：
1. 传输数据（参考 [LOCAL_PREPARE_GUIDE.md](LOCAL_PREPARE_GUIDE.md)）
2. 运行 `bash scripts/run_sft_eval_driven.sh`
3. 监控训练 `tail -f train.log`

### 场景4: 我想深入了解技术细节

👉 阅读顺序：
1. [EVAL_DRIVEN_TRAINING_GUIDE.md](EVAL_DRIVEN_TRAINING_GUIDE.md) - 完整指南
2. [TRAINING_PLAN.md](TRAINING_PLAN.md) - 训练计划
3. 查看脚本源码

### 场景5: 我遇到问题需要排查

👉 排查步骤：
1. 运行 `python scripts/verify_data.py` 检查数据
2. 查看文档中的"常见问题"章节
3. 检查日志文件 `train.log`

---

## 💡 快速命令参考

### 本地操作（Windows PowerShell）

```powershell
# 一键准备
.\local_prepare.ps1 -MaxSamples 100000

# 验证数据
python scripts/verify_data.py

# 提交到 Git
git add . && git commit -m "Add data" && git push

# 压缩大文件
Compress-Archive -Path data\train_vectorized -DestinationPath train_vectorized.zip
```

### 服务器操作（Linux Bash）

```bash
# 拉取代码
git pull

# 验证数据
python scripts/verify_data.py

# 开始训练
bash scripts/run_sft_eval_driven.sh

# 后台运行
nohup bash scripts/run_sft_eval_driven.sh > train.log 2>&1 &

# 查看日志
tail -f train.log
```

---

## 📊 时间&成本估算

| 项目 | 10万样本 | 50万样本 |
|------|---------|---------|
| **本地准备** | 2-4小时 | 6-12小时 |
| **文件传输** | 10-30分钟 | 30-60分钟 |
| **服务器训练** | 12小时 | 18-24小时 |
| **向量化成本** | 免费/10元 | 免费/50元 |
| **训练成本** | 60元 | 90-120元 |
| **总计** | **~70元** | **~150元** |

---

## 🔍 关键概念

### 评测集向量召回
从海量数据中智能筛选与评测最相关的训练样本，提升模型在特定评测指标上的表现。

### 本地 vs 服务器
- **本地**: CPU密集，向量化计算（6-12小时）
- **服务器**: GPU密集，模型训练（12-24小时）

### 数据分类
- **小文件** (<100MB): Git管理，直接提交
- **大文件** (>1GB): 单独传输，OSS/SCP/WinSCP

---

## 📞 获取帮助

1. 查看文档中的"常见问题"章节
2. 运行验证脚本诊断问题
3. 查看脚本源码了解实现细节
4. 提 Issue 或讨论

---

## 🎓 学习路径

### 初级（快速上手）
1. QUICK_START_LOCAL_SERVER.md
2. 运行 local_prepare.ps1
3. 完成一次完整训练

### 中级（深入理解）
1. EVAL_DRIVEN_TRAINING_GUIDE.md
2. 理解向量召回原理
3. 调整召回参数

### 高级（自定义优化）
1. 修改脚本源码
2. 设计自己的评测集
3. 优化召回策略

---

## ✅ 推荐工作流

```
1. 阅读 QUICK_START_LOCAL_SERVER.md (5分钟)
   ↓
2. 本地运行 local_prepare.ps1 (6-12小时，可过夜)
   ↓
3. 验证数据 verify_data.py (1分钟)
   ↓
4. 提交小文件到 Git (5分钟)
   ↓
5. 传输大文件到服务器 (30分钟)
   ↓
6. 服务器启动训练 (12-24小时)
   ↓
7. 评测验证效果 (1-2小时)
```

---

**更新时间**: 2024年12月  
**维护**: MedicalGPT Team  
**版本**: v1.0
