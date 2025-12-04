#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的医疗模型
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_model_path, lora_model_path=None):
    """加载模型"""
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if lora_model_path:
        print(f"Loading LoRA from {lora_model_path}...")
        model = PeftModel.from_pretrained(model, lora_model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer

def chat(model, tokenizer, query, history=[]):
    """对话函数"""
    # 构建prompt
    messages = history + [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    
    return response

def main():
    # 配置 - 根据你的实际训练结果修改
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    
    # 选择要测试的模型（取消注释对应的行）
    # lora_model = "outputs-sft-qwen2.5-3b/checkpoint-best"  # 测试SFT模型
    lora_model = "outputs-dpo-qwen2.5-3b/checkpoint-best"    # 测试DPO模型
    # lora_model = None  # 测试原始基础模型
    
    print("="*60)
    print("医疗问答模型测试")
    print("="*60)
    print(f"Base Model: {base_model}")
    print(f"LoRA Model: {lora_model}")
    print("="*60)
    
    model, tokenizer = load_model(base_model, lora_model)
    
    # 测试问题
    test_questions = [
        "感冒了应该怎么办？",
        "高血压患者在饮食上需要注意什么？",
        "糖尿病有哪些早期症状？",
        "如何预防心血管疾病？",
        "咳嗽了一周还没好，需要去医院吗？",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[测试 {i}/{len(test_questions)}]")
        print(f"问题: {question}")
        print(f"回答: ", end="", flush=True)
        
        response = chat(model, tokenizer, question)
        print(response)
        print("-" * 60)
    
    # 交互式对话
    print("\n" + "="*60)
    print("进入交互模式（输入 'exit' 退出）")
    print("="*60)
    
    history = []
    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ['exit', 'quit', '退出']:
                break
            
            if not user_input:
                continue
            
            response = chat(model, tokenizer, user_input, history)
            print(f"助手: {response}")
            
            # 更新历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # 保持历史在合理长度
            if len(history) > 10:
                history = history[-10:]
        
        except KeyboardInterrupt:
            print("\n\n对话结束")
            break

if __name__ == "__main__":
    main()
