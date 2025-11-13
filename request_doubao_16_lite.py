import requests

from config import get_doubao_api_key


def call_doubao_api(user_message=None, reasoning_effort="medium", max_completion_tokens=65535):
    """
    调用火山引擎豆包 API
    
    Args:
        user_message: 用户消息内容（字符串），如果提供 messages 则忽略此参数
        messages: 消息列表，格式为 [{"role": "system/user", "content": "..."}, ...]
        reasoning_effort: 推理努力程度，可选值: "low", "medium", "high"
        max_completion_tokens: 最大完成token数
    
    Returns:
        tuple: (content, reasoning_content) 
               content: 回复内容
               reasoning_content: 推理内容（如果存在）
    """
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    
    api_key = get_doubao_api_key()
    if not api_key:
        raise ValueError("请在 config.py 或环境变量中配置 DOUBAO_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    
    # 向后兼容：使用单个 user_message
    if user_message is None:
        raise ValueError("必须提供 user_message 或 messages 参数")
    formatted_messages = [
        {
            "content": [
                {
                    "text": user_message,
                    "type": "text"
                }
            ],
            "role": "user"
        }
    ]
    
    data = {
        "model": "doubao-seed-1-6-lite-251015",
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
        "messages": formatted_messages
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        # 提取 content 和 reasoning_content
        content = None
        reasoning_content = None
        
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice:
                message = choice['message']
                content = message.get('content', None)
                reasoning_content = message.get('reasoning_content', None)
        
        return content, reasoning_content
    else:
        raise Exception(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")


if __name__ == "__main__":
    # 示例用法
    user_msg = "你好，你是谁？"
    content, reasoning_content = call_doubao_api(user_msg)
    
    print("Content:")
    print(content)
    print("\nReasoning Content:")
    print(reasoning_content)

