# 本地数据准备 + 服务器训练方案

> 在本地完成数据准备工作，通过 Git + 文件传输上传到服务器训练

---

## 📋 任务划分

### ✅ 本地完成（数据准备）

| 步骤 | 任务 | 耗时 | 说明 |
|------|------|------|------|
| 1 | 下载评测集 | 5分钟 | 文件小，可用Git同步 |
| 2 | 向量化评测集 | 5-10分钟 | 文件约20MB，可用Git |
| 3 | 向量化训练数据 | 6-12小时 | **文件大（2-5GB）**，需单独传输 |
| 4 | 召回数据 | 30分钟 | 依赖步骤3 |
| 5 | 合并训练集 | 5分钟 | 最终文件约50MB |

### 🚀 服务器完成（训练）

| 步骤 | 任务 | 耗时 | 说明 |
|------|------|------|------|
| 6 | SFT训练 | 12-24小时 | 需要GPU |
| 7 | 模型评测 | 1-2小时 | 可选 |
| 8 | DPO/PPO训练 | 12-24小时 | 可选 |

---

## 🔧 本地准备流程

### 1. 环境准备

```powershell
# Windows PowerShell

# 激活虚拟环境（如果有）
# .\venv_medical\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
pip install -r requirements_eval_driven.txt

# 设置环境变量
$env:ZHIPUAI_API_KEY="5fbd8d5375e54946884bd2d796a9c12a.CzJhYueAB8LV729i"
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### 2. 执行本地准备脚本

```powershell
# 运行本地准备脚本（Windows版本）
python scripts/local_prepare.py
```

这个脚本会自动完成：
- ✅ 下载评测集
- ✅ 向量化评测集
- ✅ 向量化训练数据（可选数量）
- ✅ 召回数据
- ✅ 合并训练集

---

## 📤 数据传输方案

### 方案1: Git + 阿里云OSS/七牛云（推荐）

#### Git 同步（小文件）

```bash
# 1. 添加小文件到 Git
git add data/eval_benchmark/
git add data/eval_vectorized/
git add data/recalled_data/
git add data/finetune/medical_eval_driven.jsonl
git commit -m "Add prepared training data"
git push
```

#### 对象存储传输（大文件）

```powershell
# 使用 ossutil 上传大文件（2-5GB的向量文件）

# 安装 ossutil（Windows）
# 下载: https://help.aliyun.com/document_detail/120075.html

# 配置
ossutil config

# 上传向量文件
ossutil cp -r data\train_vectorized\ oss://your-bucket/medicalgpt/train_vectorized/

# 在服务器下载
# ossutil cp -r oss://your-bucket/medicalgpt/train_vectorized/ data/train_vectorized/
```

### 方案2: Git + SCP/SFTP

```bash
# 1. Git 同步小文件
git push

# 2. SCP 传输大文件（从本地到服务器）
scp -r data/train_vectorized/ root@your-server-ip:/root/MedicalGPT/data/

# 或使用 WinSCP（Windows图形界面工具）
# 下载: https://winscp.net/
```

### 方案3: Git + 网盘中转

```bash
# 1. 压缩大文件
tar -czf train_vectorized.tar.gz data/train_vectorized/

# 2. 上传到百度网盘/阿里云盘/坚果云

# 3. 在服务器下载并解压
wget "网盘分享链接" -O train_vectorized.tar.gz
tar -xzf train_vectorized.tar.gz -C data/
```

### 方案4: 直接在服务器向量化（备选）

如果传输困难，可以只传代码，在服务器重新向量化：

```bash
# 服务器执行
python scripts/vectorize_training_dataset.py \
    --dataset_name shibing624/medical \
    --output_file data/train_vectorized/medical_vectorized.jsonl \
    --max_samples 500000
```

---

## 📁 Git 配置建议

### .gitignore 配置

创建/更新 `.gitignore`，排除大文件：

```bash
# 大文件不提交到 Git
data/train_vectorized/*.jsonl
*.tar.gz
*.zip

# 模型权重不提交
outputs-*/
*.bin
*.safetensors

# 其他
__pycache__/
*.pyc
.DS_Store
```

### Git LFS（可选，适合中等文件）

如果想用 Git 管理 50-200MB 的文件：

```bash
# 安装 Git LFS
# Windows: https://git-lfs.github.com/

# 配置 Git LFS
git lfs install

# 追踪特定文件
git lfs track "data/finetune/*.jsonl"
git lfs track "data/recalled_data/*.jsonl"

# 提交
git add .gitattributes
git add data/finetune/
git commit -m "Add training data with LFS"
git push
```

---

## 🖥️ 服务器接收数据

### 方案A: Git克隆

```bash
# 1. SSH 连接到服务器
ssh root@your-server-ip

# 2. 克隆代码
cd /root
git clone https://github.com/yourusername/MedicalGPT.git
cd MedicalGPT

# 3. 下载大文件（如果使用 OSS）
ossutil cp -r oss://your-bucket/medicalgpt/train_vectorized/ data/train_vectorized/

# 4. 验证文件
ls -lh data/train_vectorized/
ls -lh data/finetune/
```

### 方案B: rsync 同步（增量传输）

```bash
# 从本地同步到服务器（增量，断点续传）
rsync -avz --progress \
    data/ \
    root@your-server-ip:/root/MedicalGPT/data/

# 只同步特定目录
rsync -avz --progress \
    data/train_vectorized/ \
    root@your-server-ip:/root/MedicalGPT/data/train_vectorized/
```

---

## 📝 完整操作步骤

### 阶段1: 本地准备（Windows）

```powershell
# 1. 设置环境
$env:ZHIPUAI_API_KEY="your_api_key"
$env:HF_ENDPOINT="https://hf-mirror.com"

# 2. 执行准备脚本
python scripts/local_prepare.py --max_samples 100000

# 3. 验证生成的文件
Get-ChildItem -Recurse data\ | Select-Object FullName, Length

# 4. 提交小文件到 Git
git add data/eval_benchmark/
git add data/eval_vectorized/
git add data/recalled_data/
git add data/finetune/
git commit -m "Prepare training data"
git push

# 5. 压缩大文件准备传输
Compress-Archive -Path data\train_vectorized -DestinationPath train_vectorized.zip
```

### 阶段2: 传输到服务器

**方式1: 使用 WinSCP（推荐）**
- 下载 WinSCP: https://winscp.net/
- 连接服务器，拖拽上传 `train_vectorized.zip`

**方式2: 使用命令行**
```powershell
# SCP 上传（需要 OpenSSH）
scp train_vectorized.zip root@your-server-ip:/root/
```

### 阶段3: 服务器训练（Linux）

```bash
# 1. SSH 连接
ssh root@your-server-ip

# 2. 拉取代码（如果用Git）
cd /root/MedicalGPT
git pull

# 3. 解压大文件
unzip /root/train_vectorized.zip -d data/

# 4. 验证文件完整性
python scripts/verify_data.py

# 5. 安装依赖
pip install -r requirements.txt

# 6. 开始训练
bash scripts/run_sft_eval_driven.sh
```

---

## ⚙️ 自动化脚本

### 本地准备脚本（Windows）

创建 `scripts/local_prepare.py`（已包含在前面创建的文件中）

### 快速命令（PowerShell）

```powershell
# 保存为 local_prepare.ps1
$env:ZHIPUAI_API_KEY="your_api_key"
$env:HF_ENDPOINT="https://hf-mirror.com"

Write-Host "Step 1: 下载评测集" -ForegroundColor Green
python scripts/download_ceval.py

Write-Host "Step 2: 向量化评测集" -ForegroundColor Green
python scripts/vectorize_eval_dataset.py `
    --input_dir data/eval_benchmark `
    --output_dir data/eval_vectorized `
    --model_name glm-embedding-3

Write-Host "Step 3: 向量化训练数据（可能需要6-12小时）" -ForegroundColor Yellow
python scripts/vectorize_training_dataset.py `
    --dataset_name shibing624/medical `
    --output_file data/train_vectorized/medical_vectorized.jsonl `
    --max_samples 100000

Write-Host "Step 4: 召回数据" -ForegroundColor Green
python scripts/recall_relevant_data.py `
    --eval_vectors data/eval_vectorized `
    --train_vectors data/train_vectorized/medical_vectorized.jsonl `
    --output_dir data/recalled_data

Write-Host "Step 5: 合并数据" -ForegroundColor Green
python scripts/merge_recalled_data.py `
    --input_dir data/recalled_data `
    --output_file data/finetune/medical_eval_driven.jsonl `
    --format sharegpt

Write-Host "✅ 本地准备完成！" -ForegroundColor Cyan
Write-Host "下一步: 传输数据到服务器" -ForegroundColor Yellow
```

---

## 📊 文件大小参考

| 文件/目录 | 大小 | Git同步 | 传输方式 |
|----------|------|---------|---------|
| `data/eval_benchmark/` | ~200KB | ✅ 是 | Git |
| `data/eval_vectorized/` | ~20MB | ✅ 是 | Git |
| `data/train_vectorized/` | **2-5GB** | ❌ 否 | OSS/SCP/网盘 |
| `data/recalled_data/` | ~50MB | ✅ 是 | Git |
| `data/finetune/*.jsonl` | ~50MB | ✅ 是 | Git |

**总计**:
- Git管理: ~120MB
- 单独传输: 2-5GB

---

## 🔍 数据验证脚本

创建 `scripts/verify_data.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证数据完整性"""

import json
from pathlib import Path

def verify_data():
    """验证所有必需的数据文件"""
    
    required_files = [
        "data/eval_benchmark/clinical_medicine.jsonl",
        "data/eval_vectorized/clinical_medicine_vectorized.jsonl",
        "data/train_vectorized/medical_vectorized.jsonl",
        "data/recalled_data/recalled_clinical_medicine.jsonl",
        "data/finetune/medical_eval_driven.jsonl"
    ]
    
    print("验证数据文件完整性...")
    print("=" * 60)
    
    all_ok = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024 / 1024  # MB
            
            # 验证是否为有效的JSONL
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                print(f"✅ {file_path}")
                print(f"   大小: {size:.2f} MB, 行数: {lines}")
            except Exception as e:
                print(f"⚠️  {file_path} - 文件损坏: {e}")
                all_ok = False
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_ok = False
    
    print("=" * 60)
    if all_ok:
        print("✅ 所有文件验证通过！可以开始训练。")
    else:
        print("❌ 部分文件缺失或损坏，请检查。")
    
    return all_ok

if __name__ == "__main__":
    verify_data()
```

---

## 💡 最佳实践建议

### 1. 分批准备（推荐）

```bash
# 第一批: 先用小数据测试流程
python scripts/local_prepare.py --max_samples 10000

# 验证流程正确后，再处理完整数据
python scripts/local_prepare.py --max_samples 500000
```

### 2. 断点续传

```python
# 向量化支持断点续传
python scripts/vectorize_training_dataset.py \
    --dataset_name shibing624/medical \
    --output_file data/train_vectorized/medical_vectorized.jsonl \
    --max_samples 500000 \
    --resume_from data/train_vectorized/medical_vectorized.jsonl  # 从已有文件继续
```

### 3. 压缩传输

```bash
# 压缩可减少50-70%文件大小
tar -czf data_prepared.tar.gz data/

# 服务器解压
tar -xzf data_prepared.tar.gz
```

### 4. 校验文件完整性

```bash
# 本地生成MD5
Get-FileHash -Path train_vectorized.zip -Algorithm MD5

# 服务器验证
md5sum train_vectorized.zip
```

---

## 🚨 常见问题

### Q1: 向量化在本地很慢怎么办？

A: 3种方案：
1. **过夜运行**: 睡前启动，早上完成
2. **使用本地模型**: 不需要API，速度更快
3. **减少样本数**: 先用10万条测试效果

### Q2: 网络传输太慢？

A: 优化方案：
1. **压缩后传输**: 可减少50-70%大小
2. **使用OSS**: 国内服务器上传/下载都快
3. **分块传输**: 分多个小文件传输
4. **直接在服务器准备**: 跳过传输步骤

### Q3: Git 仓库太大？

A: 使用 `.gitignore` 排除大文件：
```bash
# 只提交代码和小数据文件
data/train_vectorized/
*.tar.gz
*.zip
```

### Q4: 服务器没有外网怎么办？

A: 在本地下载好所有数据和模型：
```bash
# 本地下载模型
huggingface-cli download Qwen/Qwen2.5-3B-Instruct

# 一起打包上传
tar -czf medical_all.tar.gz MedicalGPT/ models/
```

---

## 📞 下一步

完成本地准备后：

1. ✅ 验证数据: `python scripts/verify_data.py`
2. ✅ 提交到Git: `git push`
3. ✅ 传输大文件到服务器
4. ✅ 在服务器执行训练: `bash scripts/run_sft_eval_driven.sh`

---

**更新时间**: 2024年12月  
**适用系统**: Windows本地 + Linux服务器
