"""
命令行主脚本（不依赖 Gradio）
用法示例：
  python cli_main.py --image /path/to/input.png --instruction "先灰度化再顺时针旋转90度" --output /path/to/output.png

要求：
  - 运行前需确保以下服务已启动：
    1) image_processor_server.py  (http://127.0.0.1:4201/image-processor)
    2) intent_analyzer_server.py  (http://127.0.0.1:4202/intent-analyzer)
    3) image_processing_coordinator.py (http://127.0.0.1:4204/coordinator)
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from typing import Optional

from PIL import Image
from fastmcp import Client
from config import get_coordinator_source


def read_file_as_base64(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def base64_to_image(data: str) -> Image.Image:
    if not data:
        raise ValueError("empty base64 string")
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    binary = base64.b64decode(data)
    return Image.open(io.BytesIO(binary))


def _extract_text_from_content(value) -> Optional[str]:
    # 兼容 FastMCP v2: CallToolResult.content 可能为 TextContent 列表
    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("value")
                if txt:
                    texts.append(txt)
            else:
                txt = getattr(item, "text", None)
                if txt:
                    texts.append(txt)
        return "".join(texts) if texts else None
    return None


def to_json_data(result):
    # 将 CallToolResult / str / dict 统一转换为 dict
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    content = getattr(result, "content", None)
    if content is not None:
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            text = _extract_text_from_content(content)
            if text:
                return json.loads(text)
    raise TypeError(f"无法解析结果类型: {type(result)}")


async def _call_coordinator(image_b64: str, instruction: str):
    client = Client(get_coordinator_source())
    async with client:
        r = await client.call_tool(
            "process_image_with_instruction",
            {"image_base64": image_b64, "instruction": instruction}
        )
        return r


def process_image(image_path: str, instruction: str, output_path: Optional[str] = None) -> str:
    """
    处理单张图像。

    Args:
        image_path: 输入图片路径
        instruction: 用户指令（中文/英文均可）
        output_path: 可选，结果输出路径（含文件名）。若未提供，将在输入同目录生成 *_out.png

    Returns:
        最终结果图片文件路径
    """
    img_b64 = read_file_as_base64(image_path)

    # 调用协调器
    raw_result = asyncio.run(_call_coordinator(img_b64, instruction))
    

    data = to_json_data(raw_result)
    if not data.get("success"):
        raise RuntimeError(f"处理失败: {data.get('error', '未知错误')}")

    out_b64 = data.get("final_image")
    if not out_b64:
        raise RuntimeError("未返回结果图像")

    out_img = base64_to_image(out_b64)

    # 输出路径
    if not output_path:
        root, _ = os.path.splitext(os.path.abspath(image_path))
        output_path = root + "_out.png"

    # 保存
    out_img.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="图像编辑多智能体 - 命令行主脚本")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--instruction", required=True, help="用户处理指令")
    parser.add_argument("--output", default=None, help="输出图片路径（可选）")
    args = parser.parse_args()

    try:
        result_path = process_image(args.image, args.instruction, args.output)
        print(result_path)
    except Exception as e:
        print(f"处理异常: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


