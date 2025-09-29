"""
图像处理工具服务器
提供灰度化和旋转90度的原子能力
"""
import base64
import io
import os
from PIL import Image
from fastmcp import FastMCP

# 创建图像处理MCP服务器
mcp = FastMCP("图像处理工具服务器")

def image_from_base64(base64_str: str) -> Image.Image:
    """从base64字符串创建PIL图像对象"""
    # 移除可能的data URI前缀
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """将PIL图像对象转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

@mcp.tool()
def ping() -> str:
    """
    健康检查
    """
    return "pong"

@mcp.tool()
def convert_to_grayscale(image_base64: str) -> str:
    """
    将图像转换为灰度图像
    
    Args:
        image_base64 (str): base64编码的输入图像
        
    Returns:
        str: base64编码的灰度图像
    """
    try:
        # 加载图像
        image = image_from_base64(image_base64)
        
        # 转换为灰度
        grayscale_image = image.convert('L')
        
        # 转换回base64
        result_base64 = image_to_base64(grayscale_image)
        
        return result_base64
    except Exception as e:
        return f"错误：灰度化处理失败 - {str(e)}"

@mcp.tool()
def rotate_clockwise_90(image_base64: str) -> str:
    """
    将图像顺时针旋转90度
    
    Args:
        image_base64 (str): base64编码的输入图像
        
    Returns:
        str: base64编码的旋转后图像
    """
    try:
        # 加载图像
        image = image_from_base64(image_base64)
        
        # 顺时针旋转90度 (PIL中使用负值表示顺时针)
        rotated_image = image.rotate(-90, expand=True)
        
        # 转换回base64
        result_base64 = image_to_base64(rotated_image)
        
        return result_base64
    except Exception as e:
        return f"错误：旋转处理失败 - {str(e)}"

@mcp.tool()
def get_image_info(image_base64: str) -> str:
    """
    获取图像基本信息
    
    Args:
        image_base64 (str): base64编码的输入图像
        
    Returns:
        str: 图像信息的JSON字符串
    """
    try:
        image = image_from_base64(image_base64)
        
        info = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format or "Unknown"
        }
        
        import json
        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"错误：获取图像信息失败 - {str(e)}"

if __name__ == "__main__":
    print("启动图像处理工具服务器...")
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=4201,
        path="/image-processor",
        log_level="info"
    )
