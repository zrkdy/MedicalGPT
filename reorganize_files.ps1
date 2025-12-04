# 文件重组脚本
# 将项目文件按功能模块重新组织

param(
    [switch]$DryRun = $false,  # 仅预览，不实际移动
    [switch]$Force = $false     # 强制执行，不询问
)

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "   MedicalGPT 文件重组工具" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n🔍 预览模式（不会实际移动文件）" -ForegroundColor Yellow
}

# 检查是否有未提交的更改
$gitStatus = git status --porcelain
if ($gitStatus -and -not $Force) {
    Write-Host "`n⚠️  检测到未提交的Git更改！" -ForegroundColor Yellow
    Write-Host "建议先提交或备份更改。" -ForegroundColor Yellow
    $response = Read-Host "是否继续? (输入 'yes' 继续)"
    if ($response -ne "yes") {
        Write-Host "已取消" -ForegroundColor Red
        exit 0
    }
}

# 定义移动规则
$moveRules = @{
    # 📚 文档迁移
    "docs" = @(
        @{From="EVAL_DRIVEN_INDEX.md"; To="docs/index.md"}
        @{From="GIT_GUIDE.md"; To="docs/git_guide.md"}
        @{From="TRAINING_GUIDE_Qwen2.5-3B.md"; To="docs/training_guide_qwen2.5-3b.md"}
        @{From="README_GITHUB_UPLOAD.md"; To="docs/github_upload_guide.md"}
    )
    
    # 🔵 原始评测驱动方案
    "eval_driven_basic" = @(
        # 文档
        @{From="README_EVAL_DRIVEN.md"; To="eval_driven_basic/README.md"}
        @{From="EVAL_DRIVEN_QUICKSTART.md"; To="eval_driven_basic/QUICKSTART.md"}
        @{From="QUICK_START_LOCAL_SERVER.md"; To="eval_driven_basic/docs/QUICK_START.md"}
        @{From="LOCAL_PREPARE_GUIDE.md"; To="eval_driven_basic/docs/LOCAL_PREPARE_GUIDE.md"}
        @{From="EVAL_DRIVEN_TRAINING_GUIDE.md"; To="eval_driven_basic/docs/TRAINING_GUIDE.md"}
        @{From="TRAINING_PLAN.md"; To="eval_driven_basic/docs/TRAINING_PLAN.md"}
        
        # 脚本
        @{From="local_prepare.ps1"; To="eval_driven_basic/local_prepare.ps1"}
        @{From="scripts/download_ceval.py"; To="eval_driven_basic/scripts/download_ceval.py"}
        @{From="scripts/vectorize_eval_dataset.py"; To="eval_driven_basic/scripts/vectorize_eval_dataset.py"}
        @{From="scripts/vectorize_training_dataset.py"; To="eval_driven_basic/scripts/vectorize_training_dataset.py"}
        @{From="scripts/recall_relevant_data.py"; To="eval_driven_basic/scripts/recall_relevant_data.py"}
        @{From="scripts/merge_recalled_data.py"; To="eval_driven_basic/scripts/merge_recalled_data.py"}
        @{From="scripts/run_sft_eval_driven.sh"; To="eval_driven_basic/scripts/run_sft_eval_driven.sh"}
        @{From="scripts/local_prepare.py"; To="eval_driven_basic/scripts/local_prepare.py"}
        @{From="scripts/quick_start_eval_driven.sh"; To="eval_driven_basic/scripts/quick_start_eval_driven.sh"}
    )
    
    # 🟢 优化方案
    "eval_driven_optimized" = @(
        # 文档
        @{From="EVAL_DRIVEN_OPTIMIZATION.md"; To="eval_driven_optimized/README.md"}
        @{From="OPTIMIZATION_COMPARISON.md"; To="eval_driven_optimized/COMPARISON.md"}
        
        # 脚本
        @{From="optimize_pipeline.ps1"; To="eval_driven_optimized/optimize_pipeline.ps1"}
        @{From="scripts/optimize_step1_format_data.py"; To="eval_driven_optimized/scripts/optimize_step1_format_data.py"}
        @{From="scripts/optimize_step2_topk_filter.py"; To="eval_driven_optimized/scripts/optimize_step2_topk_filter.py"}
        @{From="scripts/optimize_step3_reasoning_distill.py"; To="eval_driven_optimized/scripts/optimize_step3_reasoning_distill.py"}
        
        # 参考项目
        @{From="HealthAI-2025"; To="eval_driven_optimized/reference/HealthAI-2025"}
    )
    
    # 🔧 训练脚本
    "training" = @(
        @{From="supervised_finetuning.py"; To="training/supervised_finetuning.py"}
        @{From="supervised_finetuning_accelerate.py"; To="training/supervised_finetuning_accelerate.py"}
        @{From="dpo_training.py"; To="training/dpo_training.py"}
        @{From="ppo_training.py"; To="training/ppo_training.py"}
        @{From="grpo_training.py"; To="training/grpo_training.py"}
        @{From="orpo_training.py"; To="training/orpo_training.py"}
        @{From="reward_modeling.py"; To="training/reward_modeling.py"}
        @{From="pretraining.py"; To="training/pretraining.py"}
        @{From="run_sft.sh"; To="training/run_sft.sh"}
        @{From="run_sft_accelerate.sh"; To="training/run_sft_accelerate.sh"}
        @{From="run_full_sft.sh"; To="training/run_full_sft.sh"}
        @{From="run_dpo.sh"; To="training/run_dpo.sh"}
        @{From="run_ppo.sh"; To="training/run_ppo.sh"}
        @{From="run_grpo.sh"; To="training/run_grpo.sh"}
        @{From="run_orpo.sh"; To="training/run_orpo.sh"}
        @{From="run_rm.sh"; To="training/run_rm.sh"}
        @{From="run_pt.sh"; To="training/run_pt.sh"}
        @{From="zero1.yaml"; To="training/zero1.yaml"}
        @{From="zero2.json"; To="training/zero2.json"}
        @{From="zero2.yaml"; To="training/zero2.yaml"}
        @{From="zero3.json"; To="training/zero3.json"}
        @{From="zero3.yaml"; To="training/zero3.yaml"}
        @{From="scripts/run_sft_qwen2.5-3b.sh"; To="training/scripts/run_sft_qwen2.5-3b.sh"}
        @{From="scripts/run_dpo_qwen2.5-3b.sh"; To="training/scripts/run_dpo_qwen2.5-3b.sh"}
        @{From="scripts/run_ppo_qwen2.5-3b.sh"; To="training/scripts/run_ppo_qwen2.5-3b.sh"}
        @{From="scripts/run_rm_qwen2.5-3b.sh"; To="training/scripts/run_rm_qwen2.5-3b.sh"}
        @{From="scripts/run_pt_qwen2.5-3b.sh"; To="training/scripts/run_pt_qwen2.5-3b.sh"}
        @{From="scripts/run_sft_rtx3090.sh"; To="training/scripts/run_sft_rtx3090.sh"}
    )
    
    # 🛠️ 工具脚本
    "tools" = @(
        @{From="scripts/check_environment.py"; To="tools/scripts/check_environment.py"}
        @{From="scripts/verify_data.py"; To="tools/scripts/verify_data.py"}
        @{From="scripts/evaluate_model.py"; To="tools/scripts/evaluate_model.py"}
        @{From="scripts/merge_lora.py"; To="tools/scripts/merge_lora.py"}
        @{From="scripts/test_model.py"; To="tools/scripts/test_model.py"}
        @{From="template.py"; To="tools/templates/template.py"}
        @{From="merge_peft_adapter.py"; To="tools/merge_peft_adapter.py"}
        @{From="merge_tokenizers.py"; To="tools/merge_tokenizers.py"}
        @{From="build_domain_tokenizer.py"; To="tools/build_domain_tokenizer.py"}
        @{From="convert_dataset.py"; To="tools/convert_dataset.py"}
        @{From="validate_jsonl.py"; To="tools/validate_jsonl.py"}
        @{From="model_quant.py"; To="tools/model_quant.py"}
        @{From="eval_quantize.py"; To="tools/eval_quantize.py"}
        @{From="run_quant.sh"; To="tools/run_quant.sh"}
        @{From="run_eval_quantize.sh"; To="tools/run_eval_quantize.sh"}
    )
    
    # 🚀 部署脚本
    "deployment" = @(
        @{From="gradio_demo.py"; To="deployment/gradio_demo.py"}
        @{From="fastapi_server_demo.py"; To="deployment/fastapi_server_demo.py"}
        @{From="openai_api.py"; To="deployment/openai_api.py"}
        @{From="vllm_deployment.sh"; To="deployment/vllm_deployment.sh"}
        @{From="chatpdf.py"; To="deployment/chatpdf.py"}
        @{From="inference.py"; To="deployment/inference.py"}
        @{From="inference_multigpu_demo.py"; To="deployment/inference_multigpu_demo.py"}
    )
}

# 统计信息
$totalFiles = 0
$movedFiles = 0
$skippedFiles = 0
$errorFiles = 0

# 执行移动
foreach ($category in $moveRules.Keys) {
    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-Host "处理类别: $category" -ForegroundColor Green
    Write-Host ("=" * 80) -ForegroundColor Cyan
    
    foreach ($rule in $moveRules[$category]) {
        $totalFiles++
        $from = $rule.From
        $to = $rule.To
        
        # 检查源文件是否存在
        if (-not (Test-Path $from)) {
            Write-Host "  ⏭️  跳过: $from (文件不存在)" -ForegroundColor Yellow
            $skippedFiles++
            continue
        }
        
        # 检查目标文件是否已存在
        if (Test-Path $to) {
            Write-Host "  ⚠️  跳过: $to (目标已存在)" -ForegroundColor Yellow
            $skippedFiles++
            continue
        }
        
        if ($DryRun) {
            Write-Host "  📋 预览: $from → $to" -ForegroundColor Cyan
            $movedFiles++
        } else {
            try {
                # 创建目标目录
                $targetDir = Split-Path $to -Parent
                if ($targetDir -and -not (Test-Path $targetDir)) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                
                # 移动文件或文件夹
                if (Test-Path $from -PathType Container) {
                    # 文件夹：使用复制+删除（因为Move-Item对文件夹有时不稳定）
                    Copy-Item -Path $from -Destination $to -Recurse -Force
                    Remove-Item -Path $from -Recurse -Force
                } else {
                    # 文件：直接移动
                    Move-Item -Path $from -Destination $to -Force
                }
                
                Write-Host "  ✅ 移动: $from → $to" -ForegroundColor Green
                $movedFiles++
            } catch {
                Write-Host "  ❌ 错误: $from → $to" -ForegroundColor Red
                Write-Host "     原因: $($_.Exception.Message)" -ForegroundColor Red
                $errorFiles++
            }
        }
    }
}

# 创建新的README文件
if (-not $DryRun) {
    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-Host "创建模块README文件" -ForegroundColor Green
    Write-Host ("=" * 80) -ForegroundColor Cyan
    
    # eval_driven_basic README补充
    if (Test-Path "eval_driven_basic/README.md") {
        $content = @"

---

## 📂 文件结构

``````
eval_driven_basic/
├── README.md                  # 本文档
├── QUICKSTART.md              # 快速开始
├── local_prepare.ps1          # 自动化脚本
├── scripts/                   # 脚本目录
│   ├── download_ceval.py
│   ├── vectorize_eval_dataset.py
│   ├── vectorize_training_dataset.py
│   ├── recall_relevant_data.py
│   ├── merge_recalled_data.py
│   └── run_sft_eval_driven.sh
└── docs/                      # 详细文档
    ├── QUICK_START.md
    ├── LOCAL_PREPARE_GUIDE.md
    ├── TRAINING_GUIDE.md
    └── TRAINING_PLAN.md
``````

## 🚀 快速开始

``````powershell
# 在本目录下执行
.\local_prepare.ps1 -MaxSamples 10000
``````

详见 [QUICKSTART.md](QUICKSTART.md)
"@
        Add-Content -Path "eval_driven_basic/README.md" -Value $content
        Write-Host "  ✅ 更新: eval_driven_basic/README.md" -ForegroundColor Green
    }
    
    # eval_driven_optimized README补充
    if (Test-Path "eval_driven_optimized/README.md") {
        $content = @"

---

## 📂 文件结构

``````
eval_driven_optimized/
├── README.md                  # 本文档（优化方案说明）
├── COMPARISON.md              # 与原始方案对比
├── optimize_pipeline.ps1      # 自动化脚本
├── scripts/                   # 优化脚本
│   ├── optimize_step1_format_data.py
│   ├── optimize_step2_topk_filter.py
│   └── optimize_step3_reasoning_distill.py
└── reference/                 # 参考项目
    └── HealthAI-2025/
``````

## 🚀 快速开始

``````powershell
# 在本目录下执行
.\optimize_pipeline.ps1 -Mode quick
``````

详见 [COMPARISON.md](COMPARISON.md)
"@
        Add-Content -Path "eval_driven_optimized/README.md" -Value $content
        Write-Host "  ✅ 更新: eval_driven_optimized/README.md" -ForegroundColor Green
    }
}

# 统计报告
Write-Host "`n" + ("=" * 80) -ForegroundColor Green
Write-Host "重组完成！" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green

Write-Host "`n📊 统计信息:" -ForegroundColor Cyan
Write-Host "  总文件数: $totalFiles" -ForegroundColor White
Write-Host "  已移动: $movedFiles" -ForegroundColor Green
Write-Host "  已跳过: $skippedFiles" -ForegroundColor Yellow
Write-Host "  错误: $errorFiles" -ForegroundColor Red

if ($DryRun) {
    Write-Host "`n💡 这是预览模式，没有实际移动文件。" -ForegroundColor Yellow
    Write-Host "要实际执行，请运行: .\reorganize_files.ps1" -ForegroundColor Yellow
} else {
    Write-Host "`n✅ 文件已重新组织！" -ForegroundColor Green
    Write-Host "`n下一步:" -ForegroundColor Cyan
    Write-Host "  1. 检查文件是否正确移动" -ForegroundColor White
    Write-Host "  2. 测试各模块功能" -ForegroundColor White
    Write-Host "  3. 提交到Git: git add . && git commit -m 'Reorganize project structure'" -ForegroundColor White
    
    Write-Host "`n快速开始:" -ForegroundColor Cyan
    Write-Host "  原始方案: cd eval_driven_basic && .\local_prepare.ps1" -ForegroundColor White
    Write-Host "  优化方案: cd eval_driven_optimized && .\optimize_pipeline.ps1" -ForegroundColor White
}
