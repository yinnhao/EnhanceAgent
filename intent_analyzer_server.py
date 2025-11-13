"""
LLM意图理解服务器
分析用户指令并确定需要调用的图像处理工具
"""
import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from request_doubao_16_lite import call_doubao_api

from config import get_mode
from prompt import INTENT_ANALYSIS_SYSTEM_PROMPT

# 创建意图理解MCP服务器
mcp = FastMCP("LLM意图理解服务器")

@mcp.tool()
def analyze_intent(user_instruction: str) -> str:
    """
    分析用户图像处理指令的意图
    
    Args:
        user_instruction (str): 用户的图像处理指令
        
    Returns:
        str: JSON格式的意图分析结果
    """
    try:
        # 设置代理，确保豆包 API 请求能够通过指定网络出口
        os.environ["https_proxy"] = "http://agent.baidu.com:8891"
        os.environ["HTTPS_PROXY"] = "http://agent.baidu.com:8891"

        # 将 system prompt 和 user instruction 合并成一个字符串
        full_prompt = f"{INTENT_ANALYSIS_SYSTEM_PROMPT}\n\n用户指令：{user_instruction}"
        print("--- 调用豆包请求 ---", file=sys.stderr)
        print(f"Full Prompt:\n{full_prompt}", file=sys.stderr)
        content, reasoning_content = call_doubao_api(user_message=full_prompt)
        print("--- 调用豆包响应 ---", file=sys.stderr)
        print(f"Content:\n{content}", file=sys.stderr)
        response = content  # 使用 content 作为响应
        if reasoning_content:
            print("--- 调用豆包推理内容 ---", file=sys.stderr)
            print(f"Reasoning Content:\n{reasoning_content}", file=sys.stderr)
        
        print("=== 调用豆包结束 ===", file=sys.stderr)
        try:
            # 对llm返回的结果进行清洗：1.移除开头和结尾的空白字符（包括空格、换行符、制表符等）2.移除开头和结尾的```json和```
            cleaned_response = response.strip() 
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            parsed_response = json.loads(cleaned_response)
            # 判断是否缺少必要字段
            if not all(key in parsed_response for key in ["action_type", "tools", "reasoning"]):
                raise ValueError("缺少必要字段")
            return json.dumps(parsed_response, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 最终兜底：无法识别
        return json.dumps({
            "action_type": "error",
            "tools": [],
            "reasoning": "无法从指令中识别出可用工具"
        }, ensure_ascii=False, indent=2)
            
    except Exception as e:
        error_response = {
            "action_type": "error",
            "tools": [],
            "reasoning": f"意图分析失败: {str(e)}"
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)

@mcp.tool()
def get_available_tools() -> str:
    """
    获取当前可用的图像处理工具列表
    
    Returns:
        str: 可用工具的JSON描述
    """
    tools_info = {
        "available_tools": [
            {
                "name": "convert_to_grayscale",
                "description": "将图像转换为灰度图像",
                "keywords": ["灰度", "黑白", "灰色", "grayscale", "gray"]
            },
            {
                "name": "rotate_clockwise_90", 
                "description": "将图像顺时针旋转90度",
                "keywords": ["旋转", "顺时针", "90度", "rotate", "clockwise"]
            },
            {
                "name": "rotate_counterclockwise_90", 
                "description": "将图像逆时针旋转90度",
                "keywords": ["旋转", "逆时针", "90度", "rotate", "counterclockwise"]
            },
            {
                "name": "get_image_info",
                "description": "获取图像基本信息（尺寸、格式等）",
                "keywords": ["信息", "尺寸", "大小", "格式", "info", "size"]
            },
            {
                "name": "colorize_ddcolor",
                "description": "使用 DDColor 为黑白图像上色",
                "keywords": ["上色", "着色", "黑白", "彩色", "colorize", "ddcolor"]
            },
            {
                "name": "derain_restormer",
                "description": "使用 Restormer 进行去雨",
                "keywords": ["去雨", "雨", "derain", "restormer"]
            },
            {
                "name": "deblur_motion_restormer",
                "description": "使用 Restormer 进行去运动模糊",
                "keywords": ["去模糊", "运动模糊", "去运动模糊", "deblur", "motion", "restormer"]
            },
            {
                "name": "denoise_scunet",
                "description": "使用 SCUNet 进行去噪",
                "keywords": ["去噪", "降噪", "denoise", "scunet"]
            },
            {
                "name": "super_resolution_bsrgan",
                "description": "使用 BSRGAN 进行超分辨率",
                "keywords": ["超分", "超分辨率", "放大", "BSRGAN", "SR", "super-resolution"]
            }
        ]
    }
    
    return json.dumps(tools_info, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("启动LLM意图理解服务器...")
    if get_mode() == "http":
        mcp.run(
            transport="http",
            host="127.0.0.1", 
            port=4202,
            path="/intent-analyzer",
            log_level="info"
        )
    elif get_mode() == "stdout":
        mcp.run()
