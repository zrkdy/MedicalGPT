#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化步骤3: 使用DeepSeek R1蒸馏推理过程
参考 HealthAI-2025 的推理增强策略
"""

import os
import json
import time
from openai import OpenAI
from zhipuai import ZhipuAI
from tqdm import tqdm
import argparse


class ReasoningDistiller:
    """推理过程蒸馏器"""
    
    def __init__(self, provider: str = "zhipu", api_key: str = None):
        """
        初始化蒸馏器
        
        Args:
            provider: API提供商 ("zhipu" 或 "deepseek")
            api_key: API密钥
        """
        self.provider = provider
        
        if provider == "zhipu":
            self.client = ZhipuAI(api_key=api_key)
            self.model = "glm-4-plus"
        elif provider == "deepseek":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            self.model = "deepseek-reasoner"
        else:
            raise ValueError(f"不支持的provider: {provider}")
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return '''你是一名资深全科医生，请严格按以下要求分析病例：

1. 诊断疾病（如果有多个疾病，分点列出）
2. 详细说明诊断依据（分点说明，每条依据用\\n分隔）

返回严格符合以下JSON格式：
{
    "reason": "诊断原因分点说明", 
    "diseases": "疾病名称"（如果有多种疾病，分点说明）
}

注意：
- 疾病名称使用标准医学术语
- 诊断原因需结合病例中的症状、体征、检查结果等要素
- 必须使用双引号，严格避免JSON格式错误
- diseases的推导一定要依赖于reason的逻辑

明确结构化输出要求：
- 强制分点编号（如 "1. 2. 3."）和医学逻辑顺序（主诉→现病史→体格检查→辅助检查→诊断依据）
- 要求生成文本直接引用输入中的原始术语，避免同义替换
- 强制模型覆盖输入中所有字段（主诉、现病史、体格检查等），避免遗漏
- 禁止添加输入中未提及的症状或检查建议，避免冗余内容降低精确度'''
    
    def distill_single(self, case_content: str) -> dict:
        """蒸馏单个病例"""
        try:
            if self.provider == "zhipu":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": case_content}
                    ],
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=5000
                )
                
                content = response.choices[0].message.content
                reasoning = ""  # GLM-4-Plus 没有单独的推理字段
                
            else:  # deepseek
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": case_content}
                    ],
                    max_tokens=5000
                )
                
                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, 'reasoning_content', '')
            
            # 解析JSON结果
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"reason": content, "diseases": "解析失败"}
            
            # 添加推理过程
            if reasoning:
                result['reasoning_content'] = reasoning
            
            return result
            
        except Exception as e:
            print(f"蒸馏失败: {e}")
            return None
    
    def build_batch_requests(self, input_file: str, batch_file: str, max_samples: int = None):
        """构建批量推理请求"""
        print("构建批量推理请求...")
        
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(batch_file, 'w', encoding='utf-8') as fout:
            
            count = 0
            for line in tqdm(fin, desc='构建请求'):
                if max_samples and count >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # 提取病例内容
                    if 'patient_info' in data:
                        # 格式化数据
                        info = data['patient_info']
                        content = f"""性别: {info.get('gender', '未知')}
年龄: {info.get('age', '未知')}
主诉: {info.get('chief_complaint', '')}
现病史: {info.get('history', '')}
既往史: {info.get('past_history', '')}
体格检查: {info.get('physical_exam', '')}
辅助检查: {info.get('lab_test', '')}"""
                    else:
                        # 原始文本
                        content = data.get('text', str(data))
                    
                    # 构建请求
                    request = {
                        "custom_id": f"case_{data.get('id', count)}",
                        "method": "POST",
                        "url": "/v4/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": self.get_system_prompt()},
                                {"role": "user", "content": content}
                            ],
                            "max_tokens": 5000
                        }
                    }
                    
                    fout.write(json.dumps(request, ensure_ascii=False) + '\n')
                    count += 1
                    
                except Exception as e:
                    print(f"跳过: {e}")
                    continue
        
        print(f"✅ 已生成 {count} 条请求")
        return count
    
    def submit_batch_job(self, batch_file: str) -> str:
        """提交批量任务（仅支持智谱）"""
        if self.provider != "zhipu":
            raise ValueError("批量API仅支持智谱")
        
        print("上传批量文件...")
        upload_response = self.client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        file_id = upload_response.id
        print(f"✅ 文件已上传: {file_id}")
        
        print("创建批量任务...")
        batch_job = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v4/chat/completions",
            auto_delete_input_file=True,
            metadata={"description": "推理过程蒸馏"}
        )
        batch_id = batch_job.id
        print(f"✅ 任务已创建: {batch_id}")
        
        return batch_id
    
    def wait_batch_completion(self, batch_id: str, output_file: str):
        """等待批量任务完成"""
        print("等待任务完成...")
        
        while True:
            status_info = self.client.batches.retrieve(batch_id)
            status = status_info.status
            
            completed = getattr(status_info.request_counts, 'completed', 0)
            total = getattr(status_info.request_counts, 'total', 0)
            
            print(f"状态: {status} | 进度: {completed}/{total}")
            
            if status == 'completed':
                break
            elif status == 'failed':
                print("❌ 任务失败")
                return False
            
            time.sleep(10)
        
        print("下载结果...")
        batch_info = self.client.batches.retrieve(batch_id)
        success_content = self.client.files.content(batch_info.output_file_id)
        success_content.write_to_file(output_file)
        
        print(f"✅ 结果已保存: {output_file}")
        return True
    
    def process_batch_results(self, batch_output: str, original_file: str, final_output: str):
        """处理批量结果"""
        print("处理批量结果...")
        
        # 加载原始数据
        original_data = {}
        with open(original_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_data[data.get('id', '')] = data
        
        # 处理结果
        valid_count = 0
        with open(batch_output, 'r', encoding='utf-8') as fin, \
             open(final_output, 'w', encoding='utf-8') as fout:
            
            for line in tqdm(fin, desc='处理结果'):
                try:
                    response = json.loads(line)
                    
                    if response['response']['status_code'] != 200:
                        continue
                    
                    custom_id = response['custom_id']
                    original_id = custom_id.replace('case_', '')
                    
                    content = response['response']['body']['choices'][0]['message']['content']
                    reasoning = response['response']['body']['choices'][0]['message'].get('reasoning_content', '')
                    
                    # 解析JSON
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        
                        # 添加推理过程
                        if reasoning:
                            result['reasoning_content'] = reasoning
                        
                        # 合并原始数据
                        if original_id in original_data:
                            merged = {**original_data[original_id], **result}
                        else:
                            merged = result
                        
                        fout.write(json.dumps(merged, ensure_ascii=False) + '\n')
                        valid_count += 1
                    
                except Exception as e:
                    print(f"处理错误: {e}")
                    continue
        
        print(f"✅ 有效结果: {valid_count} 条")
        return valid_count


def main():
    parser = argparse.ArgumentParser(description="优化步骤3: 推理过程蒸馏")
    parser.add_argument('--input', required=True, help='输入文件')
    parser.add_argument('--output', required=True, help='输出文件')
    parser.add_argument('--batch_input', default='batch_reasoning_input.jsonl', help='批量请求文件')
    parser.add_argument('--batch_output', default='batch_reasoning_output.jsonl', help='批量结果文件')
    parser.add_argument('--provider', choices=['zhipu', 'deepseek'], default='zhipu', help='API提供商')
    parser.add_argument('--api_key', help='API Key')
    parser.add_argument('--use_batch', action='store_true', help='使用批量API')
    parser.add_argument('--max_samples', type=int, help='最大处理数量')
    
    args = parser.parse_args()
    
    # 获取API Key
    api_key = args.api_key or os.getenv('ZHIPUAI_API_KEY' if args.provider == 'zhipu' else 'DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("请设置API Key")
    
    # 初始化蒸馏器
    distiller = ReasoningDistiller(args.provider, api_key)
    
    if args.use_batch and args.provider == 'zhipu':
        # 批量模式
        print("=" * 70)
        print("使用批量推理模式")
        print("=" * 70)
        
        distiller.build_batch_requests(args.input, args.batch_input, args.max_samples)
        batch_id = distiller.submit_batch_job(args.batch_input)
        
        if distiller.wait_batch_completion(batch_id, args.batch_output):
            distiller.process_batch_results(args.batch_output, args.input, args.output)
    else:
        # 实时模式
        print("=" * 70)
        print("使用实时API模式")
        print("=" * 70)
        
        with open(args.input, 'r', encoding='utf-8') as fin, \
             open(args.output, 'w', encoding='utf-8') as fout:
            
            count = 0
            for line in tqdm(fin, desc='蒸馏推理'):
                if args.max_samples and count >= args.max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # 提取内容
                    if 'patient_info' in data:
                        info = data['patient_info']
                        content = f"""性别: {info.get('gender', '未知')}
年龄: {info.get('age', '未知')}
主诉: {info.get('chief_complaint', '')}
现病史: {info.get('history', '')}"""
                    else:
                        content = data.get('text', str(data))[:1000]
                    
                    # 蒸馏
                    result = distiller.distill_single(content)
                    
                    if result:
                        merged = {**data, **result}
                        fout.write(json.dumps(merged, ensure_ascii=False) + '\n')
                        count += 1
                    
                except Exception as e:
                    print(f"处理失败: {e}")
                    continue
    
    print("\n" + "=" * 70)
    print("✅ 推理蒸馏完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
