'''
Author: lijixiang01 lijixiang01@baidu.com
Date: 2023-07-20 16:28:11
LastEditors: Please set LastEditors
LastEditTime: 2024-01-23 10:21:50
FilePath: /code/chatgpt_req/baidu_gpt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import openai
import os
import json
import time
import re
import random

# 允许从环境变量读取，若未设置则保留占位符（demo 兜底逻辑会在调用方处理）
openai.api_key = os.getenv("OPENAI_API_KEY", "sk---zhuyinghao---olagFJFAmsbCJ9kZjT+U1Q==")


openai.api_base = 'http://llms-se.baidu-int.com:8200'
temperature = 0.7
top_p = 1
max_tokens = 2000
model = "ernie-4.5-8k-preview"    # 切换模型 gpt-4o, gpt-4, gpt-3.5-turbo, ernie-bot,  gpt-3.5-turbo-16k,  ernie-bot-4.0, gpt-4-1106-preview, ernie-4.0-8k


def get_response(messages, temperature, top_p, max_tokens, model):

    response = openai.ChatCompletion.create(model=model,
        messages=messages,
        temperature=temperature, 
        top_p=top_p,
        max_tokens=max_tokens,
    )

    if not response.get("error"):
        # 检查回复中是否有错误信息,如果没有错误则继续下面的操作
        return response["choices"][0]["message"]["content"]
        # 从回复字典中获取第一个回复的内容并返回
    return response["error"]["message"]
        # 如果有错误信息,则返回错误信息

def get_stream_response(messages, temperature, top_p, max_tokens, model):
    # 定义一个函数get_stream_response,用于获取聊天的流式回复
    # 参数和get_response函数相同

    response = openai.ChatCompletion.create(model=model,
        messages=messages,
        temperature=temperature, 
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True
    )

    # 调用openai.ChatCompletion.create()函数来生成聊天回复
    # 设置stream参数为True,表示获取流式回复
    # 函数返回一个生成器对象（generator）

    for resp in response:
        # 遍历生成器对象中的每个回复
        yield resp["choices"][0]['delta'].get("content", "")
        # 获取每个回复的内容并返回


def res_json_process(result):

    result = result.replace("json", "")
    result = result.replace("```", "")

    result_json = json.loads(result)
    return result_json


if __name__ == "__main__":
    # 可选：仅在直接运行本文件时做一次演示调用
    prompt_gen = "你好, 你是谁"
    messages = [
        {"role": "user", "content": prompt_gen}
    ]
    try:
        result = get_response(messages, temperature, top_p, max_tokens, model)
        print(result)
    except Exception as e:
        print(f"LLM 调用失败: {e}")