@echo off
chcp 65001 >nul
echo ========================================
echo   MedicalGPT - 初始化 GitHub 仓库
echo ========================================
echo.

REM 检查是否安装了 Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [!] 未检测到 Git
    echo.
    echo 请先安装 Git:
    echo   https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo ✓ Git 已安装
echo.

REM 初始化 Git 仓库
if exist ".git" (
    echo [!] Git 仓库已经初始化
    echo.
) else (
    echo [1/5] 初始化 Git 仓库...
    git init
    echo ✓ 完成
    echo.
)

REM 配置用户信息
echo [2/5] 配置 Git 用户信息
echo.
set /p username="请输入你的 GitHub 用户名: "
set /p email="请输入你的 GitHub 邮箱: "

git config --global user.name "%username%"
git config --global user.email "%email%"

echo ✓ 用户信息配置完成
echo.

REM 添加文件
echo [3/5] 添加项目文件...
git add .
echo ✓ 完成
echo.

REM 提交
echo [4/5] 创建初始提交...
git commit -m "Initial commit: MedicalGPT training guide for Qwen2.5-3B"
echo ✓ 完成
echo.

REM 配置远程仓库
echo [5/5] 配置远程仓库
echo.
echo 请先在 GitHub 创建一个新仓库:
echo   1. 访问 https://github.com/new
echo   2. 仓库名建议: MedicalGPT-Qwen2.5
echo   3. 不要勾选 "Initialize with README"
echo   4. 创建后复制仓库 URL
echo.
set /p repo_url="请粘贴你的 GitHub 仓库 URL (例如: https://github.com/username/repo.git): "

git remote add origin %repo_url%
echo ✓ 远程仓库配置完成
echo.

REM 推送
echo 准备推送到 GitHub...
echo.
echo 如果需要认证，请使用你的 GitHub 用户名和 Personal Access Token
echo (Token 获取: https://github.com/settings/tokens)
echo.
pause

git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo [!] 推送失败
    echo.
    echo 如果提示需要认证，请:
    echo   1. 访问 https://github.com/settings/tokens
    echo   2. 生成新的 Personal Access Token
    echo   3. 勾选 "repo" 权限
    echo   4. 使用 Token 作为密码重新推送:
    echo      git push -u origin main
    echo.
) else (
    echo.
    echo ========================================
    echo   ✓ 成功上传到 GitHub!
    echo ========================================
    echo.
    echo 仓库地址: %repo_url%
    echo.
    echo 在服务器上使用:
    echo   git clone %repo_url%
    echo.
)

echo 后续更新代码请使用: upload_to_github.bat
echo 详细说明请查看: GIT_GUIDE.md
echo.
pause
