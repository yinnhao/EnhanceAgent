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
try:
    from request_llm import get_response
except Exception:
    get_response = None

# 创建意图理解MCP服务器
mcp = FastMCP("LLM意图理解服务器")

# LLM配置（最简demo将优先使用规则匹配，若配置完整也可调用LLM）
TEMPERATURE = 0.3
TOP_P = 1
MAX_TOKENS = 1000
MODEL = "ernie-4.5-8k-preview" 

SYSTEM_PROMPT = """你是一个图像处理意图理解专家。用户会提供一个图像处理指令，你需要分析并确定需要调用哪些图像处理工具。

当前可用的图像处理工具：
1. convert_to_grayscale - 将图像转换为灰度图像
2. rotate_clockwise_90 - 将图像顺时针旋转90度
3. get_image_info - 获取图像基本信息

请根据用户指令，返回一个JSON格式的响应，包含：
- action_type: "single" 或 "sequence" (单个操作或序列操作)
- tools: 需要调用的工具名称列表，按执行顺序排列
- reasoning: 分析推理过程

示例：
用户："请把这张图片变成灰度图"
回复：{
  "action_type": "single",
  "tools": ["convert_to_grayscale"],
  "reasoning": "用户明确要求将图片转换为灰度图，需要调用convert_to_grayscale工具"
}

用户："请先把图片转成灰色，然后顺时针旋转90度"
回复：{
  "action_type": "sequence", 
  "tools": ["convert_to_grayscale", "rotate_clockwise_90"],
  "reasoning": "用户要求先灰度化再旋转，需要按顺序执行两个操作"
}

请确保返回有效的JSON格式。如果指令不清楚或无法处理，tools字段应为空数组。
"""

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
        # 规则优先：最小可用规则解析
        text = user_instruction.strip().lower()
        tools = []
        reasoning = []

        # # 中文/英文关键词映射
        # grayscale_keywords = ["灰度", "灰色", "黑白", "grayscale", "gray"]
        # rotate_keywords = ["旋转", "顺时针", "90", "rotate", "clockwise"]

        # if any(k in user_instruction for k in grayscale_keywords):
        #     tools.append("convert_to_grayscale")
        #     reasoning.append("检测到灰度相关关键词")
        # if any(k in user_instruction for k in rotate_keywords):
        #     tools.append("rotate_clockwise_90")
        #     reasoning.append("检测到旋转相关关键词")

        # if tools:
        #     action_type = "single" if len(tools) == 1 else "sequence"
        #     return json.dumps({
        #         "action_type": action_type,
        #         "tools": tools,
        #         "reasoning": "；".join(reasoning) or "根据规则匹配得到工具序列"
        #     }, ensure_ascii=False, indent=2)

        # 若规则无匹配且具备LLM，则尝试LLM（可选）
        if get_response is not None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_instruction}
            ]
            response = get_response(
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                model=MODEL
            )
            print("llm response:", response)
            try:
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                parsed_response = json.loads(cleaned_response)
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
                "name": "get_image_info",
                "description": "获取图像基本信息（尺寸、格式等）",
                "keywords": ["信息", "尺寸", "大小", "格式", "info", "size"]
            }
        ]
    }
    
    return json.dumps(tools_info, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("启动LLM意图理解服务器...")
    mcp.run(
        transport="http",
        host="127.0.0.1", 
        port=4202,
        path="/intent-analyzer",
        log_level="info"
    )
