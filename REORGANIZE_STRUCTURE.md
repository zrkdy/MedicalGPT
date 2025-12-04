# 文件重组计划

## 🎯 重组目标

将项目文件按功能模块分类，方便管理和使用：
- **原始评测驱动方案** → `eval_driven_basic/`
- **优化方案（HealthAI-2025风格）** → `eval_driven_optimized/`
- **通用训练脚本** → `training/`
- **文档** → `docs/`

---

## 📁 新的文件结构

```
MedicalGPT/
│
├── 📖 根目录（保留核心文档）
│   ├── README.md                    # 项目总览
│   ├── README_EN.md
│   ├── LICENSE
│   └── requirements.txt
│
├── 📚 docs/（所有文档集中管理）
│   ├── index.md                     # 文档索引（原EVAL_DRIVEN_INDEX.md）
│   ├── git_guide.md
│   ├── training_guide.md
│   └── ...其他文档
│
├── 🔵 eval_driven_basic/（原始评测驱动方案）
│   ├── README.md                    # 方案说明
│   ├── QUICKSTART.md                # 快速开始
│   ├── local_prepare.ps1            # 自动化脚本
│   │
│   ├── scripts/                     # 脚本
│   │   ├── download_ceval.py
│   │   ├── vectorize_eval_dataset.py
│   │   ├── vectorize_training_dataset.py
│   │   ├── recall_relevant_data.py
│   │   ├── merge_recalled_data.py
│   │   └── run_sft_eval_driven.sh
│   │
│   └── docs/                        # 文档
│       ├── QUICK_START.md
│       ├── LOCAL_PREPARE_GUIDE.md
│       ├── TRAINING_GUIDE.md
│       └── TRAINING_PLAN.md
│
├── 🟢 eval_driven_optimized/（优化方案，参考HealthAI-2025）
│   ├── README.md                    # 优化方案说明
│   ├── COMPARISON.md                # 与原始方案对比
│   ├── optimize_pipeline.ps1        # 自动化脚本
│   │
│   ├── scripts/                     # 脚本
│   │   ├── optimize_step1_format_data.py
│   │   ├── optimize_step2_topk_filter.py
│   │   ├── optimize_step3_reasoning_distill.py
│   │   └── ... (可复用basic的向量化脚本)
│   │
│   ├── docs/                        # 文档
│   │   ├── OPTIMIZATION_GUIDE.md
│   │   └── HEALTHAI_REFERENCE.md
│   │
│   └── reference/                   # 参考项目
│       └── HealthAI-2025/           # 原始参考代码
│
├── 🔧 training/（通用训练脚本）
│   ├── supervised_finetuning.py
│   ├── dpo_training.py
│   ├── ppo_training.py
│   ├── reward_modeling.py
│   ├── pretraining.py
│   └── ... (其他训练相关)
│
├── 🛠️ tools/（通用工具）
│   ├── scripts/
│   │   ├── check_environment.py
│   │   ├── verify_data.py
│   │   ├── evaluate_model.py
│   │   ├── merge_lora.py
│   │   └── test_model.py
│   └── templates/
│       └── template.py
│
├── 📊 data/（数据目录，保持不变）
│   ├── eval_benchmark/
│   ├── eval_vectorized/
│   ├── train_vectorized/
│   └── ...
│
└── 🚀 deployment/（部署相关）
    ├── gradio_demo.py
    ├── fastapi_server_demo.py
    ├── openai_api.py
    └── vllm_deployment.sh
```

---

## 🔄 文件迁移列表

### 📚 文档迁移

| 当前位置 | 新位置 |
|---------|--------|
| `EVAL_DRIVEN_INDEX.md` | `docs/index.md` |
| `GIT_GUIDE.md` | `docs/git_guide.md` |
| `TRAINING_GUIDE_Qwen2.5-3B.md` | `docs/training_guide_qwen2.5-3b.md` |
| `README_GITHUB_UPLOAD.md` | `docs/github_upload_guide.md` |

### 🔵 原始评测驱动方案

| 当前位置 | 新位置 |
|---------|--------|
| `README_EVAL_DRIVEN.md` | `eval_driven_basic/README.md` |
| `EVAL_DRIVEN_QUICKSTART.md` | `eval_driven_basic/QUICKSTART.md` |
| `QUICK_START_LOCAL_SERVER.md` | `eval_driven_basic/docs/QUICK_START.md` |
| `LOCAL_PREPARE_GUIDE.md` | `eval_driven_basic/docs/LOCAL_PREPARE_GUIDE.md` |
| `EVAL_DRIVEN_TRAINING_GUIDE.md` | `eval_driven_basic/docs/TRAINING_GUIDE.md` |
| `TRAINING_PLAN.md` | `eval_driven_basic/docs/TRAINING_PLAN.md` |
| `local_prepare.ps1` | `eval_driven_basic/local_prepare.ps1` |
| `scripts/download_ceval.py` | `eval_driven_basic/scripts/download_ceval.py` |
| `scripts/vectorize_eval_dataset.py` | `eval_driven_basic/scripts/vectorize_eval_dataset.py` |
| `scripts/vectorize_training_dataset.py` | `eval_driven_basic/scripts/vectorize_training_dataset.py` |
| `scripts/recall_relevant_data.py` | `eval_driven_basic/scripts/recall_relevant_data.py` |
| `scripts/merge_recalled_data.py` | `eval_driven_basic/scripts/merge_recalled_data.py` |
| `scripts/run_sft_eval_driven.sh` | `eval_driven_basic/scripts/run_sft_eval_driven.sh` |
| `scripts/local_prepare.py` | `eval_driven_basic/scripts/local_prepare.py` |

### 🟢 优化方案

| 当前位置 | 新位置 |
|---------|--------|
| `EVAL_DRIVEN_OPTIMIZATION.md` | `eval_driven_optimized/README.md` |
| `OPTIMIZATION_COMPARISON.md` | `eval_driven_optimized/COMPARISON.md` |
| `optimize_pipeline.ps1` | `eval_driven_optimized/optimize_pipeline.ps1` |
| `scripts/optimize_step1_format_data.py` | `eval_driven_optimized/scripts/optimize_step1_format_data.py` |
| `scripts/optimize_step2_topk_filter.py` | `eval_driven_optimized/scripts/optimize_step2_topk_filter.py` |
| `scripts/optimize_step3_reasoning_distill.py` | `eval_driven_optimized/scripts/optimize_step3_reasoning_distill.py` |
| `HealthAI-2025/` | `eval_driven_optimized/reference/HealthAI-2025/` |

### 🔧 训练脚本

| 当前位置 | 新位置 |
|---------|--------|
| `supervised_finetuning.py` | `training/supervised_finetuning.py` |
| `dpo_training.py` | `training/dpo_training.py` |
| `ppo_training.py` | `training/ppo_training.py` |
| `reward_modeling.py` | `training/reward_modeling.py` |
| `pretraining.py` | `training/pretraining.py` |
| `run_sft.sh` | `training/run_sft.sh` |
| `run_dpo.sh` | `training/run_dpo.sh` |
| ... | ... |

### 🛠️ 工具脚本

| 当前位置 | 新位置 |
|---------|--------|
| `scripts/check_environment.py` | `tools/scripts/check_environment.py` |
| `scripts/verify_data.py` | `tools/scripts/verify_data.py` |
| `scripts/evaluate_model.py` | `tools/scripts/evaluate_model.py` |
| `scripts/merge_lora.py` | `tools/scripts/merge_lora.py` |
| `scripts/test_model.py` | `tools/scripts/test_model.py` |
| `template.py` | `tools/templates/template.py` |

### 🚀 部署脚本

| 当前位置 | 新位置 |
|---------|--------|
| `gradio_demo.py` | `deployment/gradio_demo.py` |
| `fastapi_server_demo.py` | `deployment/fastapi_server_demo.py` |
| `openai_api.py` | `deployment/openai_api.py` |
| `vllm_deployment.sh` | `deployment/vllm_deployment.sh` |

---

## ⚙️ 执行重组

### 方式1: 手动重组（推荐，更安全）

```powershell
# 创建新文件夹结构
mkdir docs
mkdir eval_driven_basic\scripts, eval_driven_basic\docs
mkdir eval_driven_optimized\scripts, eval_driven_optimized\docs, eval_driven_optimized\reference
mkdir training
mkdir tools\scripts, tools\templates
mkdir deployment

# 移动文件（示例）
Move-Item EVAL_DRIVEN_INDEX.md docs\index.md
Move-Item README_EVAL_DRIVEN.md eval_driven_basic\README.md
# ... 依次移动其他文件
```

### 方式2: 使用自动化脚本

```powershell
# 运行重组脚本
.\reorganize_files.ps1
```

---

## 📝 重组后的快速开始

### 原始方案
```powershell
cd eval_driven_basic
.\local_prepare.ps1 -MaxSamples 10000
```

### 优化方案
```powershell
cd eval_driven_optimized
.\optimize_pipeline.ps1 -Mode quick
```

---

## ⚠️ 注意事项

1. **备份**: 重组前建议先提交到Git或创建备份
2. **路径更新**: 重组后需要更新脚本中的相对路径
3. **文档链接**: 需要更新文档中的文件引用路径
4. **循序渐进**: 建议分模块逐步重组，避免一次性改动过大

---

**创建时间**: 2024年12月
**版本**: v1.0
