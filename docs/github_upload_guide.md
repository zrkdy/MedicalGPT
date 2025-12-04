# 上传项目到 GitHub - 快速指南

## 🚀 三步上传（最简单）

### 步骤 1: 初始化（首次使用）

**双击运行：** `init_github.bat`

这个脚本会：
- ✓ 检查 Git 是否安装
- ✓ 初始化 Git 仓库
- ✓ 配置你的 GitHub 用户信息
- ✓ 提交所有文件
- ✓ 推送到 GitHub

### 步骤 2: 在 GitHub 创建仓库

1. 访问：https://github.com/new
2. 填写：
   - **Repository name**: `MedicalGPT-Qwen2.5`
   - **Description**: `Medical GPT training guide for Qwen2.5-3B`
   - 选择 **Private**（私有）或 **Public**（公开）
   - ⚠️ **不要** 勾选 "Initialize this repository with a README"
3. 点击 **Create repository**
4. 复制仓库 URL（例如：`https://github.com/你的用户名/MedicalGPT-Qwen2.5.git`）

### 步骤 3: 完成上传

回到 `init_github.bat` 窗口，粘贴仓库 URL，按回车

---

## 🔄 后续更新代码

修改代码后，**双击运行：** `upload_to_github.bat`

这个脚本会自动：
- 查看修改
- 添加所有文件
- 提交修改
- 推送到 GitHub

---

## 🖥️ 在服务器上使用

### 公开仓库：

```bash
git clone https://github.com/你的用户名/MedicalGPT-Qwen2.5.git
cd MedicalGPT-Qwen2.5
```

### 私有仓库：

需要使用 Personal Access Token：

```bash
git clone https://你的用户名:你的token@github.com/你的用户名/MedicalGPT-Qwen2.5.git
cd MedicalGPT-Qwen2.5
```

**获取 Token：**
1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 勾选 "repo" 权限
4. 生成并复制 token

---

## 📝 手动操作（命令行）

如果你更喜欢手动操作：

### 首次上传

```powershell
# 1. 初始化 Git
git init
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"

# 2. 提交文件
git add .
git commit -m "Initial commit: MedicalGPT for Qwen2.5-3B"

# 3. 添加远程仓库并推送
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

### 后续更新

```powershell
git add .
git commit -m "描述你的修改"
git push
```

---

## ⚠️ 重要提示

### 不会上传的文件（已在 .gitignore 配置）

- ✗ `outputs-*/` - 训练输出（太大）
- ✗ `cache/` - 模型缓存
- ✗ `*.log` - 日志文件
- ✗ `*.bin`, `*.pth` - 模型权重
- ✗ `logs/` - 训练日志

### 会上传的文件

- ✓ Python 脚本 (`.py`)
- ✓ Shell 脚本 (`.sh`)
- ✓ 配置文件 (`requirements.txt`)
- ✓ 文档 (`.md`)
- ✓ 示例数据（小文件）

---

## 🔧 故障排查

### 问题 1: "git 不是内部或外部命令"

**解决：** 安装 Git
- 下载：https://git-scm.com/download/win
- 安装后重启终端或重新运行脚本

### 问题 2: 推送时要求输入密码

**解决：** 使用 Personal Access Token（不是你的 GitHub 密码）
1. 获取 Token：https://github.com/settings/tokens
2. 生成时勾选 "repo" 权限
3. 推送时：
   - Username: 你的 GitHub 用户名
   - Password: 粘贴 Token

### 问题 3: "authentication failed"

**解决：** 配置 Token 到 URL

```powershell
git remote set-url origin https://你的token@github.com/你的用户名/仓库名.git
git push
```

### 问题 4: 文件太大无法推送

**解决：** 
1. 检查 `.gitignore` 是否正确配置
2. 删除已提交的大文件：

```powershell
git rm -r --cached outputs-*
git rm -r --cached cache
git commit -m "Remove large files"
git push
```

---

## 📊 工作流程图

```
┌─────────────────┐
│  本地开发       │  编写脚本和文档
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Git 提交       │  git add . && git commit
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  推送到 GitHub  │  git push
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  服务器拉取     │  git clone / git pull
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  运行训练       │  bash scripts/run_*.sh
└─────────────────┘
```

---

## 📚 相关文档

- **详细指南**: `GIT_GUIDE.md` - 完整的 Git 操作说明
- **训练指南**: `TRAINING_GUIDE_Qwen2.5-3B.md` - 完整训练流程
- **快速开始**: `QUICKSTART.md` - 5分钟快速开始

---

## 🎯 推荐流程

**首次使用（5分钟）：**
1. 双击 `init_github.bat`
2. 在 GitHub 创建仓库
3. 粘贴仓库 URL
4. 完成！

**日常更新（30秒）：**
1. 修改代码
2. 双击 `upload_to_github.bat`
3. 完成！

**服务器使用：**
```bash
git clone https://github.com/你的用户名/MedicalGPT-Qwen2.5.git
cd MedicalGPT-Qwen2.5
conda create -n medical python=3.10 -y
conda activate medical
pip install -r requirements.txt
bash scripts/run_sft_qwen2.5-3b.sh
```

---

现在就可以开始了！🚀

有问题请查看 `GIT_GUIDE.md` 获取更多帮助。
