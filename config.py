"""
配置文件 - 统一管理系统运行模式和相关配置
"""
import os

# 运行模式配置 - 可通过环境变量覆盖
# 支持的模式: "http", "stdout" 
MODE = os.getenv("IMAGE_EDIT_MODE", "stdout")

# HTTP 模式配置
HTTP_CONFIG = {
    "INTENT_ANALYZER_URL": "http://127.0.0.1:4202/intent-analyzer",
    "IMAGE_PROCESSOR_URL": "http://127.0.0.1:4201/image-processor", 
    "COORDINATOR_URL": "http://127.0.0.1:4204/coordinator",
    "COORDINATOR_HOST": "127.0.0.1",
    "COORDINATOR_PORT": 4204,
    "COORDINATOR_PATH": "/coordinator"
}

# STDOUT 模式配置
STDOUT_CONFIG = {
    "INTENT_ANALYZER_SCRIPT": "./intent_analyzer_server.py",
    "IMAGE_PROCESSOR_SCRIPT": "./image_processor_server.py",
    "COORDINATOR_SCRIPT": "./image_processing_coordinator.py"
}

# 模型预加载开关（默认关闭）。设置 IMAGE_EDIT_PRELOAD_MODELS=true 可开启。
PRELOAD_MODELS = os.getenv("IMAGE_EDIT_PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")

# 预加载所用的默认模型配置（可按需修改或通过环境变量覆盖）
DEFAULT_MODEL_CONFIG = {
    # KAIR - SCUNet 去噪
    "scunet_model_name": os.getenv("SCUNET_MODEL_NAME", "scunet_color_real_psnr"),
    # KAIR - BSRGAN 超分
    "bsrgan_model_name": os.getenv("BSRGAN_MODEL_NAME", "BSRGAN"),
    # DDColor 上色
    "ddcolor_model_path": os.getenv("DDCOLOR_MODEL_PATH", "DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"),
    "ddcolor_input_size": int(os.getenv("DDCOLOR_INPUT_SIZE", "512")),
    "ddcolor_model_size": os.getenv("DDCOLOR_MODEL_SIZE", "large"),
}

# 豆包 API 密钥配置
DOUBAO_API_KEY = "64785b58-cc0c-4013-967c-e4c762f5f5ae"

def get_mode():
    """获取当前运行模式"""
    return MODE

def is_http_mode():
    """检查是否为 HTTP 模式"""
    return MODE == "http"

def is_stdout_mode():
    """检查是否为 STDOUT 模式"""
    return MODE == "stdout"

def get_intent_analyzer_source():
    """根据模式获取意图分析器源"""
    if is_http_mode():
        return HTTP_CONFIG["INTENT_ANALYZER_URL"]
    else:
        return STDOUT_CONFIG["INTENT_ANALYZER_SCRIPT"]

def get_image_processor_source():
    """根据模式获取图像处理器源"""
    if is_http_mode():
        return HTTP_CONFIG["IMAGE_PROCESSOR_URL"]
    else:
        return STDOUT_CONFIG["IMAGE_PROCESSOR_SCRIPT"]

def get_coordinator_source():
    """根据模式获取协调器源"""
    if is_http_mode():
        return HTTP_CONFIG["COORDINATOR_URL"]
    else:
        return STDOUT_CONFIG["COORDINATOR_SCRIPT"]

def get_preload_models() -> bool:
    """是否在服务启动时预加载模型"""
    return PRELOAD_MODELS

def get_default_model_config() -> dict:
    """获取默认的模型配置（用于预加载）"""
    return DEFAULT_MODEL_CONFIG.copy()

def get_doubao_api_key() -> str:
    """获取豆包 API 密钥"""
    return DOUBAO_API_KEY
