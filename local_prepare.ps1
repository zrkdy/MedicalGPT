# MedicalGPT 本地数据准备脚本 (PowerShell)
# 用法: .\local_prepare.ps1

param(
    [int]$MaxSamples = 100000,
    [string]$ApiKey = $env:ZHIPUAI_API_KEY
)

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "   MedicalGPT 本地数据准备脚本" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

# 设置环境变量
if ($ApiKey) {
    $env:ZHIPUAI_API_KEY = $ApiKey
    Write-Host "✅ 已设置 ZHIPUAI_API_KEY" -ForegroundColor Green
} else {
    Write-Host "⚠️  未设置 ZHIPUAI_API_KEY，将使用本地模型" -ForegroundColor Yellow
}

$env:HF_ENDPOINT = "https://hf-mirror.com"
Write-Host "✅ 已设置 HF_ENDPOINT: $env:HF_ENDPOINT" -ForegroundColor Green

Write-Host "`n训练数据样本数: $MaxSamples" -ForegroundColor Cyan
$hours = [math]::Round($MaxSamples / 10000 * 1.0, 1)
Write-Host "预计耗时: $hours 小时" -ForegroundColor Cyan

# 确认继续
$response = Read-Host "`n是否继续? (y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "已取消" -ForegroundColor Yellow
    exit 0
}

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "`n✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "`n❌ 未找到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    exit 1
}

# 执行准备脚本
Write-Host "`n开始执行数据准备..." -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

try {
    python scripts/local_prepare.py --max_samples $MaxSamples
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n" + ("=" * 80) -ForegroundColor Green
        Write-Host "✅ 本地准备完成！" -ForegroundColor Green
        Write-Host ("=" * 80) -ForegroundColor Green
        
        # 显示下一步
        Write-Host "`n下一步操作:" -ForegroundColor Cyan
        Write-Host "1. 验证数据: " -NoNewline -ForegroundColor Yellow
        Write-Host "python scripts/verify_data.py" -ForegroundColor White
        
        Write-Host "`n2. 提交小文件到 Git:" -ForegroundColor Yellow
        Write-Host "   git add data/eval_benchmark/ data/eval_vectorized/ data/recalled_data/ data/finetune/" -ForegroundColor White
        Write-Host "   git commit -m 'Add prepared training data'" -ForegroundColor White
        Write-Host "   git push" -ForegroundColor White
        
        Write-Host "`n3. 传输大文件 (data/train_vectorized/) 到服务器:" -ForegroundColor Yellow
        Write-Host "   方式1: 使用 WinSCP/FileZilla" -ForegroundColor White
        Write-Host "   方式2: 使用 scp 命令" -ForegroundColor White
        Write-Host "   方式3: 压缩后上传 (推荐)" -ForegroundColor White
        
        # 询问是否压缩大文件
        Write-Host "`n" -NoNewline
        $compress = Read-Host "是否现在压缩大文件以便传输? (y/n)"
        if ($compress -eq "y" -or $compress -eq "Y") {
            Write-Host "`n压缩文件中..." -ForegroundColor Cyan
            
            if (Test-Path "data\train_vectorized\medical_vectorized.jsonl") {
                Compress-Archive -Path "data\train_vectorized" -DestinationPath "train_vectorized.zip" -Force
                $size = (Get-Item "train_vectorized.zip").Length / 1MB
                Write-Host "✅ 已创建: train_vectorized.zip ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
                Write-Host "`n使用 WinSCP 或 scp 上传此文件到服务器" -ForegroundColor Yellow
            } else {
                Write-Host "❌ 未找到训练数据向量文件" -ForegroundColor Red
            }
        }
        
    } else {
        Write-Host "`n❌ 数据准备失败，请查看错误信息" -ForegroundColor Red
        exit 1
    }
    
} catch {
    Write-Host "`n❌ 执行失败: $_" -ForegroundColor Red
    exit 1
}
