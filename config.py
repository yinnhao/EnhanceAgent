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
