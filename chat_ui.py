"""
对话式可视化界面（Gradio）
功能：多轮对话 + 上传/更换图像 + 指令处理 -> 返回结果图像

启动前确保以下服务已运行：
1) python demo/image_processor_server.py         (4201)
2) python demo/intent_analyzer_server.py        (4202)
3) python demo/image_processing_coordinator.py  (4204)

运行：
  python demo/chat_ui.py
"""
import asyncio
import base64
import io
import json
from typing import Optional, Tuple, List
import tempfile
import os

import gradio as gr
from PIL import Image
from fastmcp import Client
from config import get_coordinator_source

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
try:
    with open(ICON_PATH, "rb") as _icon_file:
        ICON_BASE64 = base64.b64encode(_icon_file.read()).decode("utf-8")
except Exception:
    ICON_BASE64 = ""

def image_to_base64_str(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_str_to_image(data: str) -> Image.Image:
    if not data:
        raise ValueError("empty base64 string")
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    binary = base64.b64decode(data)
    return Image.open(io.BytesIO(binary))


def _extract_text_from_content(value) -> Optional[str]:
    # 兼容 FastMCP v2: CallToolResult.content 为 TextContent 列表
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


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def process_turn(message: str, chat_history: List, current_image: Optional[Image.Image]) -> Tuple[List, Optional[Image.Image]]:
    chat_history = list(chat_history or [])
    if not current_image:
        reply = "请先在左侧上传一张图片，再发送指令。"
        chat_history = chat_history + [{"sender": "assistant", "type": "text", "text": reply}]
        return chat_history, None

    try:
        input_b64 = image_to_base64_str(current_image)

        async def _call():
            client = Client(get_coordinator_source())
            async with client:
                r = await client.call_tool(
                    "process_image_with_instruction",
                    {"image_base64": input_b64, "instruction": message}
                )
                return r

        result = run_async(_call())
        data = to_json_data(result)

        if not data.get("success"):
            chat_history = chat_history + [{
                "sender": "assistant",
                "type": "text",
                "text": f"处理失败: {data.get('error', '未知错误')}"
            }]
            return chat_history, None

        out_b64 = data.get("final_image")
        if not out_b64:
            chat_history = chat_history + [{
                "sender": "assistant",
                "type": "text",
                "text": "未返回结果图像"
            }]
            return chat_history, None

        out_img = base64_str_to_image(out_b64)
        # 返回更新后的历史与结果图像（由上层决定如何展示）
        return chat_history, out_img
    except Exception as e:
        chat_history = chat_history + [{
            "sender": "assistant",
            "type": "text",
            "text": f"处理异常: {e}"
        }]
        return chat_history, None


with gr.Blocks(title="图像处理多智能体 - 对话式") as demo:
    gr.HTML(
        f"""
        <div style="display:flex;align-items:center;gap:16px;padding:12px 16px;">
            <img src="data:image/png;base64,{ICON_BASE64}" alt="Enhance Agent 图标" style="height:72px;width:72px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.12);flex-shrink:0;"/>
            <div style="display:flex;flex-direction:column;gap:6px;">
                <div style="font-size:20px;font-weight:600;color:#111827;">Enhance Agent: 通过自然语言实现图像质量增强</div>
                <ul style="margin:0;padding-left:20px;color:#374151;font-size:14px;line-height:1.6;">
                    <li>右侧上传或更换图片，左侧进行多轮对话指令</li>
                    <li>示例："请把这张图片转换为灰度图"、"先上色再超分"、"去掉雨点"</li>
                </ul>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            # 自定义 HTML 聊天区，完全可控：文本 + 图片
            chat_html = gr.HTML(value="", label="Chat")
            with gr.Row():
                msg = gr.Textbox(show_label=False, placeholder="输入指令；可在右侧选择或拖入图片后再发送")
                send = gr.Button("发送", variant="primary")
            clear = gr.Button("清空对话")
        with gr.Column(scale=4):
            # 将图片入口挪到输入区右侧，仍保留一个可见的图像选择器
            current_img = gr.Image(type="pil", label="图片（可拖入/点击上传）", height=260)
            # gr.Markdown("图片上传后会插入到聊天区作为一条图片消息，并作为后续处理的输入。")

    # 状态：当前图像 + 聊天历史（自定义结构，供 HTML 渲染）
    state_image = gr.State(value=None)
    state_history = gr.State(value=[])
    # 跳过下一次由助手更新 current_img 触发的 on_image_change（避免把助手图片当成用户图片再追加一次）
    state_skip_next_img_event = gr.State(value=False)

    def _render_html(history: List[dict]) -> str:
        # 生成简洁的聊天 HTML（不引入外部资源）
        style = """
        <style>
        .chat{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;display:flex;flex-direction:column}
        .bubble{max-width:70%;padding:10px 12px;margin:8px;border-radius:10px;white-space:pre-wrap;word-break:break-word}
        .user{background:#f0f7ff;border:1px solid #c7e2ff;align-self:flex-end}
        .assistant{background:#f7f7f8;border:1px solid #e5e7eb;align-self:flex-start}
        .bubble img{max-width:100%;border-radius:6px;border:1px solid #e5e7eb}
        </style>
        """
        parts = ["<div class='chat'>"]
        for m in history or []:
            who = m.get("sender","user")
            cls = "user" if who=="user" else "assistant"
            if m.get("type") in ("image", "image_b64"):
                b64 = m.get("b64")
                if not b64 and m.get("path"):
                    try:
                        with open(m.get("path"), "rb") as f:
                            import base64 as _b
                            b64 = _b.b64encode(f.read()).decode()
                    except Exception:
                        b64 = None
                if b64:
                    parts.append(f"<div class='bubble {cls}'><img src='data:image/png;base64,{b64}' alt='img'/></div>")
                else:
                    parts.append(f"<div class='bubble {cls}'>[图片无法显示]</div>")
            else:
                from html import escape
                parts.append(f"<div class='bubble {cls}'>"+escape(m.get("text",""))+"</div>")
        parts.append("</div>")
        return style+"\n"+"\n".join(parts)

    def _append_user_image(pil_img: Image.Image, hist: List[dict]) -> List[dict]:
        if pil_img is None:
            return hist or []
        b64 = image_to_base64_str(pil_img)
        return (hist or []) + [{"sender":"user","type":"image_b64","b64":b64}]

    def on_image_change(img, hist, skip):
        # 更新当前图像输入，并把图像作为一条消息显示在聊天区
        if skip:
            # 助手刚刚更新了 current_img，这次变更不追加用户图片
            return img, _render_html(hist), hist, False
        if img is not None:
            hist = _append_user_image(img, hist)
        return img, _render_html(hist), hist, False

    def on_submit(user_text, hist, img):
        # 文本加入历史
        hist = hist or []
        if user_text and user_text.strip():
            hist = hist + [{"sender":"user","type":"text","text":user_text.strip()}]

        # 选择输入图像
        state_img = img if img is not None else state_image.value
        updated_hist, new_img = process_turn(user_text or "", hist, state_img)

        # 结果图像加入历史
        if new_img is not None:
            b64 = image_to_base64_str(new_img)
            updated_hist = updated_hist + [{"sender":"assistant","type":"image_b64","b64":b64}]

        html = _render_html(updated_hist)
        # 如果有新图片输出，设置跳过标记，避免 current_img 更新触发 on_image_change 再追加一条“用户图片”
        skip_flag = True if new_img is not None else False
        return "", (new_img if new_img is not None else state_img), html, updated_hist, skip_flag

    send.click(on_submit, inputs=[msg, state_history, current_img], outputs=[msg, current_img, chat_html, state_history, state_skip_next_img_event])
    msg.submit(on_submit, inputs=[msg, state_history, current_img], outputs=[msg, current_img, chat_html, state_history, state_skip_next_img_event])

    current_img.change(on_image_change, inputs=[current_img, state_history, state_skip_next_img_event], outputs=[state_image, chat_html, state_history, state_skip_next_img_event])
    clear.click(lambda: ("", None, []), outputs=[chat_html, state_image, state_history])

if __name__ == "__main__":
    # 自动选择可用端口，避免端口占用错误
    # demo.launch(server_name="127.0.0.1", server_port=None, inbrowser=True)
    demo.launch(server_name="0.0.0.0", server_port=8080, inbrowser=False)


