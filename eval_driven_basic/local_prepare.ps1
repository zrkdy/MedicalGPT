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

python scripts/local_prepare.py --max_samples $MaxSamples

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor Green
    Write-Host "✅ 本地准备完成！" -ForegroundColor Green
    Write-Host ("=" * 80) -ForegroundColor Green
    
    Write-Host "`n生成的文件:" -ForegroundColor Cyan
    Write-Host "  - data/eval_benchmark/" -ForegroundColor White
    Write-Host "  - data/eval_vectorized/" -ForegroundColor White
    Write-Host "  - data/train_vectorized/" -ForegroundColor White
    Write-Host "  - data/recalled_data/" -ForegroundColor White
    Write-Host "  - data/finetune/medical_eval_driven.jsonl" -ForegroundColor Green
    
    Write-Host "`n下一步: 使用生成的训练集进行模型训练" -ForegroundColor Yellow
} else {
    Write-Host "`n❌ 数据准备失败，请查看上面的错误信息" -ForegroundColor Red
    exit 1
}
