@echo off
chcp 65001 >nul
echo ========================================
echo   清理 Git 历史并重新上传
echo ========================================
echo.
echo 这个脚本会：
echo 1. 删除旧的 .git 文件夹
echo 2. 重新初始化 Git 仓库
echo 3. 排除数据文件后重新提交
echo 4. 强制推送到 GitHub
echo.
pause

echo [1/4] 删除旧的 .git 文件夹...
rd /s /q .git
echo ✓ 完成
echo.

echo [2/4] 重新初始化 Git...
git init
git config user.name "zrkdy"
git config user.email "1971209186@qq.com"
echo ✓ 完成
echo.

echo [3/4] 添加文件并提交（排除数据文件）...
git add .
git commit -m "Initial commit: MedicalGPT training guide for Qwen2.5-3B"
echo ✓ 完成
echo.

echo [4/4] 添加远程仓库并推送...
git remote add origin https://github.com/zrkdy/MedicalGPT.git
git branch -M main
git push -u origin main --force
echo.

if errorlevel 1 (
    echo [!] 推送失败
    echo 可能需要手动处理
) else (
    echo ✓ 成功上传到 GitHub!
)

echo.
pause
