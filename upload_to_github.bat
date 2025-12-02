@echo off
chcp 65001 >nul
echo ========================================
echo   MedicalGPT - 上传到 GitHub
echo ========================================
echo.

REM 检查是否已初始化 Git
if not exist ".git" (
    echo [步骤 1/4] 初始化 Git 仓库...
    git init
    echo ✓ Git 仓库初始化完成
    echo.
) else (
    echo ✓ Git 仓库已存在
    echo.
)

REM 检查是否配置了远程仓库
git remote -v | findstr "origin" >nul 2>&1
if errorlevel 1 (
    echo [!] 尚未配置远程仓库
    echo.
    echo 请先在 GitHub 创建仓库，然后运行:
    echo   git remote add origin https://github.com/你的用户名/仓库名.git
    echo.
    echo 或参考 GIT_GUIDE.md 获取详细帮助
    pause
    exit /b 1
)

echo [步骤 2/4] 查看修改的文件...
git status
echo.

echo [步骤 3/4] 添加所有文件...
git add .
echo ✓ 文件添加完成
echo.

REM 获取当前时间
set "datetime=%date% %time%"

echo [步骤 4/4] 提交并推送到 GitHub...
git commit -m "Update: %datetime%"
if errorlevel 1 (
    echo.
    echo [!] 没有需要提交的修改
) else (
    echo ✓ 提交完成
    echo.
    echo 正在推送到 GitHub...
    git push
    if errorlevel 1 (
        echo.
        echo [!] 推送失败，可能需要先配置认证
        echo 参考 GIT_GUIDE.md 配置 Personal Access Token
    ) else (
        echo ✓ 推送成功！
    )
)

echo.
echo ========================================
echo   操作完成
echo ========================================
pause
