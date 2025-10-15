"""
图像处理工具服务器
提供灰度化和旋转90度的原子能力
"""
import base64
import io
import os
from PIL import Image
from fastmcp import FastMCP
from config import get_mode

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

# --------------- 扩展：Restormer 去雨 / 去运动模糊（调用官方 demo） ---------------

def _run_restormer_task(task: str, image_base64: str) -> str:
    try:
        import tempfile
        import subprocess
        import shutil
        import uuid
        from glob import glob
        # 准备临时目录
        work_dir = tempfile.mkdtemp(prefix="restormer_")
        input_dir = os.path.join(work_dir, "input")
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # 保存输入图像
        img = image_from_base64(image_base64).convert('RGB')
        in_name = f"{uuid.uuid4().hex}.png"
        in_path = os.path.join(input_dir, in_name)
        img.save(in_path)

        # 切换到 Restormer 目录调用 demo.py
        restormer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "Restormer"))
        cmd = [
            "python",
            "demo.py",
            "--task", task,
            "--input_dir", in_path,  # 传入单个文件路径，符合 demo.py 单图用法
            "--result_dir", output_dir,
        ]
        subprocess.check_call(cmd, cwd=restormer_root)

        # 严格按 demo.py 输出规则：result_dir/task/<basename>.png
        task_out_dir = os.path.join(output_dir, task)
        base = os.path.splitext(in_name)[0]
        out_path = os.path.join(task_out_dir, base + ".png")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"未找到预期输出文件: {out_path}")

        with open(out_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        # 清理
        shutil.rmtree(work_dir, ignore_errors=True)
        return b64
    except subprocess.CalledProcessError as e:
        return f"错误：Restormer 执行失败 - {e}"
    except Exception as e:
        return f"错误：Restormer 处理失败 - {str(e)}"


@mcp.tool()
def derain_restormer(image_base64: str) -> str:
    """
    使用 Restormer 进行去雨（Deraining）。

    Args:
        image_base64: base64 输入图像
    Returns:
        base64 输出图像
    """
    return _run_restormer_task("Deraining", image_base64)


@mcp.tool()
def deblur_motion_restormer(image_base64: str) -> str:
    """
    使用 Restormer 进行去运动模糊（Motion_Deblurring）。

    Args:
        image_base64: base64 输入图像
    Returns:
        base64 输出图像
    """
    return _run_restormer_task("Motion_Deblurring", image_base64)

# --------------- 扩展：黑白图像上色（DDColor） ---------------

@mcp.tool()
def colorize_ddcolor(
    image_base64: str,
    model_path: str = "DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt",
    input_size: int = 512,
    model_size: str = "large"
) -> str:
    """
    使用 DDColor 为黑白图像上色。

    Args:
        image_base64: base64 输入图像
        model_path: 模型权重路径（默认与仓库中的 infer.py 一致）
        input_size: 模型输入尺寸（默认 512）
        model_size: 模型规模（"tiny" 或 "large"，默认 large）

    Returns:
        base64 输出图像
    """
    try:
        import sys
        import numpy as np
        import cv2
        import torch  # 仅用于确认环境；推理在 DDColor 内部完成

        # 将项目的 DDColor 目录加入路径
        ddcolor_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "DDColor"))
        if ddcolor_root not in sys.path:
            sys.path.insert(0, ddcolor_root)

        # 延迟导入以避免全局依赖
        from infer import ImageColorizationPipeline

        # PIL -> OpenCV BGR
        pil_img = image_from_base64(image_base64).convert('RGB')
        img_rgb = np.array(pil_img)  # RGB uint8
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        pipeline = ImageColorizationPipeline(
            model_path=model_path,
            input_size=input_size,
            model_size=model_size
        )
        out_bgr = pipeline.process(img_bgr)

        # OpenCV BGR -> PIL
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        out_img = Image.fromarray(out_rgb)
        return image_to_base64(out_img)
    except Exception as e:
        return f"错误：DDColor 上色失败 - {str(e)}"

# --------------- 扩展：去噪与超分（最小改动，依赖 KAIR） ---------------

@mcp.tool()
def denoise_scunet(image_base64: str, model_name: str = "scunet_color_real_psnr", model_zoo: str = "") -> str:
    """
    使用 KAIR 的 SCUNet 模型进行去噪。

    Args:
        image_base64: base64 输入图像（任意模式，会转换为 RGB）
        model_name: 模型名（默认 scunet_color_real_psnr），对应 {model_zoo}/{model_name}.pth
        model_zoo: 模型目录，默认使用 KAIR/model_zoo
    Returns:
        base64 输出图像
    """
    try:
        import sys
        import numpy as np
        import torch
        # 动态引入 KAIR
        kair_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "KAIR"))
        if kair_root not in sys.path:
            sys.path.insert(0, kair_root)
        from models.network_scunet import SCUNet as net

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载图像 -> Tensor
        img = image_from_base64(image_base64).convert('RGB')
        arr = np.array(img)  # HWC, uint8
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().div(255.0).to(device)

        # 加载模型
        model = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
        weights_root = model_zoo or os.path.join(kair_root, "model_zoo")
        weight_path = os.path.join(weights_root, f"{model_name}.pth")
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model = model.to(device)

        with torch.no_grad():
            y = model(x)
        y = y.clamp(0,1).cpu().squeeze(0)
        out = (y.permute(1,2,0).numpy() * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image as _Image
        return image_to_base64(_Image.fromarray(out))
    except Exception as e:
        return f"错误：SCUNet 去噪失败 - {str(e)}"


@mcp.tool()
def super_resolution_bsrgan(image_base64: str, model_name: str = "BSRGAN", model_zoo: str = "", scale: int = 4) -> str:
    """
    使用 KAIR 的 BSRGAN 模型进行超分辨率。

    Args:
        image_base64: base64 输入图像（任意模式，会转换为 RGB）
        model_name: 模型名（默认 BSRGAN，若使用 x2 模型可设为 BSRGANx2）
        model_zoo: 模型目录，默认使用 KAIR/model_zoo
        scale: 放大倍数（2 或 4），需与权重匹配
    Returns:
        base64 输出图像
    """
    try:
        import sys
        import numpy as np
        import torch
        # 动态引入 KAIR
        kair_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "KAIR"))
        if kair_root not in sys.path:
            sys.path.insert(0, kair_root)
        from models.network_rrdbnet import RRDBNet as net

        # 确定放大倍数
        sf = 2 if ("x2" in model_name.lower() or scale == 2) else 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载图像 -> Tensor
        img = image_from_base64(image_base64).convert('RGB')
        arr = np.array(img)
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().div(255.0).to(device)

        # 加载模型
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
        weights_root = model_zoo or os.path.join(kair_root, "model_zoo")
        weight_path = os.path.join(weights_root, f"{model_name}.pth")
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model = model.to(device)

        with torch.no_grad():
            y = model(x)
        y = y.clamp(0,1).cpu().squeeze(0)
        out = (y.permute(1,2,0).numpy() * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image as _Image
        return image_to_base64(_Image.fromarray(out))
    except Exception as e:
        return f"错误：BSRGAN 超分失败 - {str(e)}"

if __name__ == "__main__":
    print("启动图像处理工具服务器...")
    if get_mode() == "http":
        mcp.run(
            transport="http",
            host="127.0.0.1",
            port=4201,
            path="/image-processor",
            log_level="info"
        )
    elif get_mode() == "stdout":    
        mcp.run()
