#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化步骤1: 使用大模型格式化原始训练数据
参考 HealthAI-2025 的数据质量提升策略
"""

import os
import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm
import argparse

# 格式化Prompt（改编为通用医疗问答格式）
FORMAT_PROMPT = '''
你是一名专业的医疗数据处理助手。请分析以下医疗对话，提取结构化信息。

对话内容：
{content}

请按照以下步骤处理：
1. 判断这是否是真实的医疗问诊对话（而非科普、闲聊等）
2. 如果是问诊，提取以下信息：
   - 患者基本信息（性别、年龄，如未提及则标注"未知"）
   - 主诉（核心症状）
   - 现病史（症状详情、持续时间等）
   - 既往史、过敏史等（如有提及）
   - 医生的诊断或建议

3. 对信息完整度打分（0-5分）：
   - 0-2分：基本信息缺失严重
   - 3-4分：主诉和现病史基本完整
   - 5分：信息非常详细完整

严格按照以下JSON格式输出：
```json
{{
    "is_consultation": true/false,
    "patient_info": {{
        "gender": "男/女/未知",
        "age": "年龄或未知",
        "chief_complaint": "主诉",
        "history": "现病史",
        "past_history": "既往史（如无则为空字符串）",
        "diagnosis": "医生诊断或建议"
    }},
    "quality_score": 0-5,
    "reason": "评分理由"
}}
```
'''


class DataFormatter:
    def __init__(self, api_key: str, use_batch: bool = True):
        """
        初始化数据格式化器
        
        Args:
            api_key: 智谱AI API Key
            use_batch: 是否使用批量推理API（推荐）
        """
        self.client = ZhipuAI(api_key=api_key)
        self.use_batch = use_batch
    
    def format_single_record(self, record: dict) -> dict:
        """格式化单条记录（实时API）"""
        try:
            # 构建对话内容
            if 'conversations' in record:
                # ShareGPT格式
                content = "\n".join([
                    f"{msg['from']}: {msg['value']}" 
                    for msg in record['conversations']
                ])
            elif 'instruction' in record and 'output' in record:
                # Alpaca格式
                content = f"问题: {record['instruction']}\n回答: {record['output']}"
            else:
                # 尝试通用字段
                content = record.get('text', str(record))
            
            # 调用API
            response = self.client.chat.completions.create(
                model="glm-4-plus",
                messages=[
                    {"role": "system", "content": "你是一名专业的医疗数据处理助手。"},
                    {"role": "user", "content": FORMAT_PROMPT.format(content=content)}
                ],
                temperature=0.1,
                top_p=0.1
            )
            
            # 解析结果
            content = response.choices[0].message.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            result = json.loads(content[json_start:json_end])
            
            # 合并原始数据
            result['original_id'] = record.get('id', None)
            result['original_content'] = content[:200]  # 保留前200字符
            
            return result
            
        except Exception as e:
            print(f"格式化失败: {e}")
            return None
    
    def build_batch_requests(self, input_file: str, batch_file: str, max_samples: int = None):
        """构建批量推理请求文件"""
        print(f"构建批量请理请求...")
        
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(batch_file, 'w', encoding='utf-8') as fout:
            
            count = 0
            for idx, line in enumerate(tqdm(fin, desc='构建请求')):
                if max_samples and count >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # 提取内容
                    if 'conversations' in data:
                        content = "\n".join([
                            f"{msg['from']}: {msg['value']}" 
                            for msg in data['conversations']
                        ])
                    elif 'instruction' in data:
                        content = f"问题: {data['instruction']}\n回答: {data['output']}"
                    else:
                        content = data.get('text', str(data))
                    
                    # 构建请求
                    request = {
                        "custom_id": f"format_{idx}",
                        "method": "POST",
                        "url": "/v4/chat/completions",
                        "body": {
                            "model": "glm-4-plus",
                            "messages": [
                                {"role": "system", "content": "你是一名专业的医疗数据处理助手。"},
                                {"role": "user", "content": FORMAT_PROMPT.format(content=content[:2000])}
                            ],
                            "temperature": 0.1,
                            "top_p": 0.1
                        }
                    }
                    
                    fout.write(json.dumps(request, ensure_ascii=False) + '\n')
                    count += 1
                    
                except Exception as e:
                    print(f"跳过行 {idx}: {e}")
                    continue
        
        print(f"✅ 已生成 {count} 条请求")
        return count
    
    def submit_batch_job(self, batch_file: str) -> str:
        """提交批量任务"""
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
            metadata={"description": "医疗数据格式化"}
        )
        batch_id = batch_job.id
        print(f"✅ 任务已创建: {batch_id}")
        
        return batch_id
    
    def wait_batch_completion(self, batch_id: str, output_file: str):
        """等待批量任务完成并下载结果"""
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
    
    def process_batch_results(self, batch_output: str, final_output: str, min_quality: int = 3):
        """处理批量结果并过滤低质量数据"""
        print(f"处理批量结果...")
        
        valid_count = 0
        filtered_count = 0
        
        with open(batch_output, 'r', encoding='utf-8') as fin, \
             open(final_output, 'w', encoding='utf-8') as fout:
            
            for line in tqdm(fin, desc='处理结果'):
                try:
                    response = json.loads(line)
                    
                    if response['response']['status_code'] != 200:
                        continue
                    
                    content = response['response']['body']['choices'][0]['message']['content']
                    
                    # 解析JSON
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    result = json.loads(content[json_start:json_end])
                    
                    # 质量过滤
                    if (result.get('is_consultation', False) and 
                        result.get('quality_score', 0) >= min_quality):
                        
                        result['custom_id'] = response['custom_id']
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        valid_count += 1
                    else:
                        filtered_count += 1
                    
                except Exception as e:
                    print(f"解析错误: {e}")
                    continue
        
        print(f"✅ 有效数据: {valid_count} 条")
        print(f"🗑️  过滤数据: {filtered_count} 条")
        
        return valid_count


def main():
    parser = argparse.ArgumentParser(description="优化步骤1: 格式化训练数据")
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--batch_input', default='batch_format_input.jsonl', help='批量请求文件')
    parser.add_argument('--batch_output', default='batch_format_output.jsonl', help='批量结果文件')
    parser.add_argument('--max_samples', type=int, help='最大处理数量')
    parser.add_argument('--min_quality', type=int, default=3, help='最低质量分数(0-5)')
    parser.add_argument('--use_batch', type=bool, default=True, help='使用批量API')
    parser.add_argument('--api_key', default=None, help='API Key（或设置环境变量）')
    
    args = parser.parse_args()
    
    # 获取API Key
    api_key = args.api_key or os.getenv('ZHIPUAI_API_KEY')
    if not api_key:
        raise ValueError("请设置 ZHIPUAI_API_KEY 环境变量或传入 --api_key 参数")
    
    # 初始化
    formatter = DataFormatter(api_key, use_batch=args.use_batch)
    
    if args.use_batch:
        # 批量模式（推荐）
        print("=" * 70)
        print("使用批量推理模式（更稳定、更便宜）")
        print("=" * 70)
        
        # 步骤1: 构建请求
        formatter.build_batch_requests(args.input, args.batch_input, args.max_samples)
        
        # 步骤2: 提交任务
        batch_id = formatter.submit_batch_job(args.batch_input)
        
        # 步骤3: 等待完成
        if formatter.wait_batch_completion(batch_id, args.batch_output):
            # 步骤4: 处理结果
            formatter.process_batch_results(
                args.batch_output, 
                args.output,
                args.min_quality
            )
    else:
        # 实时模式
        print("=" * 70)
        print("使用实时API模式")
        print("=" * 70)
        
        with open(args.input, 'r', encoding='utf-8') as fin, \
             open(args.output, 'w', encoding='utf-8') as fout:
            
            count = 0
            for line in tqdm(fin, desc='格式化数据'):
                if args.max_samples and count >= args.max_samples:
                    break
                
                try:
                    record = json.loads(line)
                    result = formatter.format_single_record(record)
                    
                    if result and result.get('quality_score', 0) >= args.min_quality:
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                        count += 1
                    
                except Exception as e:
                    print(f"处理失败: {e}")
                    continue
    
    print("\n" + "=" * 70)
    print("✅ 数据格式化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
