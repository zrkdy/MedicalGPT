# 上传项目到 GitHub 指南

## 方法一：使用 GitHub Desktop（推荐新手）

### 步骤 1: 安装 GitHub Desktop
1. 下载：https://desktop.github.com/
2. 安装并登录你的 GitHub 账号

### 步骤 2: 发布项目
1. 打开 GitHub Desktop
2. 点击 `File` → `Add Local Repository`
3. 选择项目目录：`C:\Users\xiaoan\Desktop\ai\MedicalGPT`
4. 如果提示 "This directory does not appear to be a Git repository"，点击 `create a repository`
5. 填写仓库信息：
   - Name: `MedicalGPT-Qwen2.5`
   - Description: `Medical GPT training guide for Qwen2.5-3B`
   - 勾选 `Initialize this repository with a README`
6. 点击 `Publish repository`
7. 选择是否设为私有仓库（建议先设为私有）
8. 点击 `Publish`

---

## 方法二：使用命令行（推荐）

### 步骤 1: 初始化 Git 仓库

```powershell
# 进入项目目录
cd C:\Users\xiaoan\Desktop\ai\MedicalGPT

# 初始化 Git
git init

# 配置用户信息（首次使用）
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

### 步骤 2: 创建 .gitignore 文件

在项目根目录创建 `.gitignore` 文件（避免上传不必要的文件）：

```bash
# 在项目目录下执行
# 文件已自动创建，见下方内容
```

### 步骤 3: 提交代码到本地仓库

```powershell
# 添加所有文件
git add .

# 查看状态（可选）
git status

# 提交到本地仓库
git commit -m "Initial commit: MedicalGPT training guide for Qwen2.5-3B"
```

### 步骤 4: 在 GitHub 创建远程仓库

1. 访问 https://github.com/new
2. 填写信息：
   - Repository name: `MedicalGPT-Qwen2.5`
   - Description: `Medical GPT training guide for Qwen2.5-3B model`
   - 选择 `Private`（私有）或 `Public`（公开）
   - **不要** 勾选 "Initialize this repository with a README"
3. 点击 `Create repository`

### 步骤 5: 推送到 GitHub

```powershell
# 添加远程仓库（替换成你的GitHub用户名）
git remote add origin https://github.com/你的用户名/MedicalGPT-Qwen2.5.git

# 推送代码
git branch -M main
git push -u origin main
```

**如果推送失败（需要认证）：**

```powershell
# 使用 Personal Access Token (PAT)
# 1. 访问 https://github.com/settings/tokens
# 2. 点击 "Generate new token" → "Generate new token (classic)"
# 3. 勾选 "repo" 权限
# 4. 生成并复制 token

# 推送时输入：
# Username: 你的GitHub用户名
# Password: 粘贴刚才的 token（不是密码）
git push -u origin main
```

---

## 在服务器上使用

### 方法 1: 克隆仓库（推荐）

```bash
# 连接到服务器后
cd /root

# 克隆你的仓库（公开仓库）
git clone https://github.com/你的用户名/MedicalGPT-Qwen2.5.git

# 克隆私有仓库（需要认证）
git clone https://你的用户名:你的token@github.com/你的用户名/MedicalGPT-Qwen2.5.git

# 进入项目
cd MedicalGPT-Qwen2.5

# 开始使用
conda create -n medical python=3.10 -y
conda activate medical
pip install -r requirements.txt
```

### 方法 2: 下载 ZIP（简单但不推荐）

1. 在 GitHub 仓库页面点击 `Code` → `Download ZIP`
2. 上传到服务器并解压

---

## 后续更新代码

### 本地更新后推送到 GitHub

```powershell
# 在本地项目目录
cd C:\Users\xiaoan\Desktop\ai\MedicalGPT

# 查看修改
git status

# 添加修改的文件
git add .

# 提交
git commit -m "描述你的修改"

# 推送到 GitHub
git push
```

### 服务器上拉取最新代码

```bash
# 在服务器项目目录
cd /root/MedicalGPT-Qwen2.5

# 拉取最新代码
git pull
```

---

## 常用 Git 命令

```powershell
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 撤销修改（未提交）
git checkout -- 文件名

# 回退到上一个版本
git reset --hard HEAD^

# 创建分支
git branch dev
git checkout dev

# 合并分支
git checkout main
git merge dev
```

---

## 注意事项

### ⚠️ 不要上传的文件

以下文件不应该上传到 GitHub（已在 .gitignore 中配置）：

- ✗ `outputs-*/` - 训练输出（太大）
- ✗ `cache/` - 模型缓存
- ✗ `*.log` - 日志文件
- ✗ `*.pth`, `*.bin` - 模型权重文件
- ✗ `__pycache__/` - Python缓存

### ✅ 应该上传的文件

- ✓ 所有 Python 脚本 (`.py`)
- ✓ 所有 Shell 脚本 (`.sh`)
- ✓ 配置文件 (`requirements.txt`)
- ✓ 文档文件 (`.md`)
- ✓ 示例数据（小文件）

### 🔒 私有仓库 vs 公开仓库

**私有仓库（推荐）：**
- ✓ 只有你可见
- ✓ 可以包含自己的训练脚本
- ✗ 免费账户有限制

**公开仓库：**
- ✓ 所有人可见
- ✓ 可以分享给他人
- ✗ 不要包含敏感信息

---

## 快速命令脚本

创建 `upload_to_github.bat`（Windows）:

```batch
@echo off
echo 正在上传到 GitHub...
cd C:\Users\xiaoan\Desktop\ai\MedicalGPT
git add .
git commit -m "Update: %date% %time%"
git push
echo 完成！
pause
```

使用：双击 `upload_to_github.bat` 即可自动上传

---

## 故障排查

### 问题1: git 不是内部或外部命令

**解决：** 安装 Git
- 下载：https://git-scm.com/download/win
- 安装后重启终端

### 问题2: 推送失败 (403 错误)

**解决：** 使用 Personal Access Token
```powershell
# 重新设置远程仓库URL（包含token）
git remote set-url origin https://你的token@github.com/你的用户名/仓库名.git
```

### 问题3: 文件太大无法推送

**解决：** 删除大文件或使用 Git LFS
```powershell
# 找出大文件
git rev-list --objects --all | sort -k 2 > allfileshas.txt

# 删除大文件（如果已提交）
git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch 大文件路径" --prune-empty --tag-name-filter cat -- --all
```

### 问题4: 忘记添加 .gitignore，已上传大文件

**解决：**
```powershell
# 创建 .gitignore
# 从 Git 中删除但保留本地文件
git rm -r --cached outputs-*
git rm -r --cached cache
git commit -m "Remove large files"
git push
```

---

## 推荐工作流

```
本地开发 → 测试 → 提交到 Git → 推送到 GitHub
                                    ↓
                          服务器拉取 → 训练 → 保存结果
```

1. **本地**：编写脚本和文档
2. **GitHub**：版本控制和同步
3. **服务器**：运行训练

---

## 总结

**最简单的流程（使用 GitHub Desktop）：**
1. 安装 GitHub Desktop
2. Add Local Repository
3. Publish repository
4. 在服务器上 `git clone`

**推荐流程（命令行）：**
```powershell
# 本地
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main

# 服务器
git clone https://github.com/你的用户名/仓库名.git
```

现在就可以开始使用了！🚀
