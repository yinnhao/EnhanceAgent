"""
图像处理多智能体系统协调器
整合LLM意图理解和图像处理工具，提供统一的处理接口
"""
import asyncio
import json
import base64
import os
from typing import Dict, Any
from fastmcp import FastMCP, Client

# 创建协调器MCP服务器
mcp = FastMCP("图像处理协调器")

# 导入统一配置
from config import (
    get_mode, is_http_mode, is_stdout_mode,
    get_intent_analyzer_source, get_image_processor_source,
    HTTP_CONFIG
)
class ImageProcessingCoordinator:
    def __init__(self):
        self.intent_client = Client(get_intent_analyzer_source())
        self.image_client = Client(get_image_processor_source())
    
    @staticmethod
    def _extract_text_from_call_result(result: Any) -> str:
        """从 fastmcp CallToolResult 或其他类型中提取字符串内容"""
        if isinstance(result, str):
            return result
        # 具有 content 属性（通常是 List[TextContent]）
        content = getattr(result, "content", None)
        if content is not None:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text") or item.get("value")
                        if txt:
                            texts.append(txt)
                    else:
                        txt = getattr(item, "text", None)
                        if txt:
                            texts.append(txt)
                return "".join(texts)
        # 兜底：转字符串
        return str(result)

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        """递归清理对象，确保可被 JSON 序列化"""
        from collections.abc import Mapping, Sequence
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): ImageProcessingCoordinator._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ImageProcessingCoordinator._sanitize_for_json(v) for v in value]
        # 其他不可序列化对象转为字符串
        return str(value)

    async def process_request(self, image_base64: str, instruction: str) -> Dict[str, Any]:
        """
        处理完整的图像处理请求
        """
        result = {"success": False}
        
        try:
            # 1. 分析用户意图
            async with self.intent_client:
                _intent_result = await self.intent_client.call_tool(
                    "analyze_intent", 
                    {"user_instruction": instruction}
                )
                intent_result = self._extract_text_from_call_result(_intent_result)
            
            # 解析意图分析结果
            try:
                intent_data = json.loads(intent_result)
            except json.JSONDecodeError:
                result["error"] = "意图分析结果格式错误"
                return result

            # 最简版本：不记录详细步骤，避免复杂对象序列化问题
            
            if intent_data.get("action_type") == "error":
                return {"success": False, "error": intent_data.get("reasoning", "意图分析失败")}
            
            tools_to_execute = intent_data.get("tools", [])
            if not tools_to_execute:
                return {"success": False, "error": "没有找到匹配的处理工具"}
            
            # 2. 按顺序执行图像处理工具
            current_image = image_base64
            
            async with self.image_client:
                for i, tool_name in enumerate(tools_to_execute):
                    try:
                        # 调用图像处理工具
                        _processed_result = await self.image_client.call_tool(
                            tool_name,
                            {"image_base64": current_image}
                        )
                        processed_result = self._extract_text_from_call_result(_processed_result)
                        
                        # 最简版本：不记录每步详情
                        
                        if isinstance(processed_result, str) and processed_result.startswith("错误："):
                            return {"success": False, "error": processed_result}
                        
                        # 更新当前图像用于下一步处理
                        current_image = processed_result
                        
                    except Exception as e:
                        result["error"] = f"执行工具 {tool_name} 时发生错误: {str(e)}"
                        return result
            
            # 3. 返回最小结果
            return {"success": True, "final_image": current_image}
            
        except Exception as e:
            return {"success": False, "error": f"协调器处理失败: {str(e)}"}

# 全局协调器实例
coordinator = ImageProcessingCoordinator()

@mcp.tool()
async def process_image_with_instruction(image_base64: str, instruction: str) -> str:
    """
    根据指令处理图像
    
    Args:
        image_base64 (str): base64编码的输入图像
        instruction (str): 用户的处理指令
        
    Returns:
        str: 处理结果的JSON字符串
    """
    try:
        result = await coordinator.process_request(image_base64, instruction)
        safe = ImageProcessingCoordinator._sanitize_for_json(result)
        return json.dumps(safe, ensure_ascii=False)
    except Exception as e:
        # 返回最小可序列化的错误字符串
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

@mcp.tool()
async def get_system_status() -> dict:
    """
    获取多智能体系统状态
    
    Returns:
        dict: 系统状态的字典
    """
    status = {
        "coordinator": "running",
        "intent_analyzer": "unknown",
        "image_processor": "unknown",
        "timestamp": asyncio.get_event_loop().time()
    }

    try:
        async with coordinator.intent_client:
            await coordinator.intent_client.call_tool("get_available_tools", {})
            status["intent_analyzer"] = "running"
    except Exception:
        status["intent_analyzer"] = "error"

    try:
        async with coordinator.image_client:
            await coordinator.image_client.call_tool("ping", {})
            status["image_processor"] = "running"
    except Exception:
        status["image_processor"] = "error"

    return status

@mcp.tool()
def load_image_from_file(file_path: str) -> str:
    """
    从文件加载图像并转换为base64格式
    
    Args:
        file_path (str): 图像文件路径
        
    Returns:
        str: base64编码的图像或错误信息
    """
    try:
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"
        
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        return image_base64
    except Exception as e:
        return f"错误：加载图像失败 - {str(e)}"

if __name__ == "__main__":
    print("启动图像处理协调器...")
    print(f"运行模式: {get_mode()}")
    if is_http_mode():
        print("确保以下服务正在运行：")
        print("- 意图分析服务器 (端口 4202)")
        print("- 图像处理服务器 (端口 4201)")
        mcp.run(
            transport="http",
            host=HTTP_CONFIG["COORDINATOR_HOST"],
            port=HTTP_CONFIG["COORDINATOR_PORT"],
            path=HTTP_CONFIG["COORDINATOR_PATH"], 
        )
    else:
        print("使用 stdout 模式，将自动启动下游服务")
        mcp.run()
