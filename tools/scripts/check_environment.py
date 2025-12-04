#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查脚本 - 在训练前运行，确保所有依赖正确安装
"""
import sys
import subprocess

def check_python_version():
    """检查Python版本"""
    print("1. 检查 Python 版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python 版本过低: {version.major}.{version.minor}")
        print(f"     需要 Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ✗ {package_name}: 未安装")
        return False

def check_cuda():
    """检查CUDA和GPU"""
    print("\n2. 检查 CUDA 和 GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA 可用")
            print(f"   ✓ CUDA 版本: {torch.version.cuda}")
            print(f"   ✓ GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"       显存: {mem:.1f} GB")
            return True
        else:
            print(f"   ✗ CUDA 不可用")
            print(f"     请检查CUDA安装和驱动")
            return False
    except Exception as e:
        print(f"   ✗ 错误: {e}")
        return False

def check_packages():
    """检查所有必需的包"""
    print("\n3. 检查 Python 包...")
    
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('peft', 'peft'),
        ('trl', 'trl'),
        ('accelerate', 'accelerate'),
        ('bitsandbytes', 'bitsandbytes'),
    ]
    
    results = []
    for pkg_name, import_name in packages:
        results.append(check_package(pkg_name, import_name))
    
    return all(results)

def check_data_files():
    """检查数据文件"""
    print("\n4. 检查数据文件...")
    import os
    
    data_dirs = {
        'data/finetune': 'SFT训练数据',
        'data/reward': 'DPO/RM训练数据',
    }
    
    all_ok = True
    for dir_path, desc in data_dirs.items():
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.json', '.jsonl'))]
            if files:
                print(f"   ✓ {desc}: {len(files)} 个文件")
            else:
                print(f"   ⚠ {desc}: 目录存在但无数据文件")
                all_ok = False
        else:
            print(f"   ✗ {desc}: 目录不存在")
            all_ok = False
    
    return all_ok

def check_disk_space():
    """检查磁盘空间"""
    print("\n5. 检查磁盘空间...")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free / (1024**3)
        print(f"   可用空间: {free_gb:.1f} GB")
        
        if free_gb < 50:
            print(f"   ⚠ 磁盘空间不足，建议至少 100GB")
            return False
        else:
            print(f"   ✓ 磁盘空间充足")
            return True
    except Exception as e:
        print(f"   ⚠ 无法检查磁盘空间: {e}")
        return True

def check_model_access():
    """检查是否能访问HuggingFace"""
    print("\n6. 检查 HuggingFace 访问...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # 尝试访问一个公开模型
        model_info = api.model_info("Qwen/Qwen2.5-0.5B")
        print(f"   ✓ HuggingFace 可访问")
        return True
    except Exception as e:
        print(f"   ⚠ HuggingFace 访问异常: {e}")
        print(f"     建议设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
        return True  # 不阻塞，只是警告

def main():
    print("="*60)
    print("MedicalGPT 环境检查工具")
    print("="*60)
    
    checks = [
        ("Python版本", check_python_version),
        ("CUDA和GPU", check_cuda),
        ("Python包", check_packages),
        ("数据文件", check_data_files),
        ("磁盘空间", check_disk_space),
        ("HuggingFace", check_model_access),
    ]
    
    results = {}
    for name, func in checks:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n检查 {name} 时出错: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)
    
    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    if all(results.values()):
        print("\n✓ 所有检查通过！可以开始训练")
        print("\n建议运行:")
        print("  bash scripts/run_sft_qwen2.5-3b.sh")
        return 0
    else:
        print("\n✗ 部分检查未通过，请根据上述提示修复问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
