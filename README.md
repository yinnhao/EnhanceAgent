# 图像编辑多智能体系统

基于 FastMCP 实现的图像处理智能体系统，支持通过自然语言指令进行图像灰度化和旋转操作。

## 环境要求

```bash
# 需要安装 fastmcp 和 Pillow
pip install fastmcp Pillow
# openai 的版本需要是 0.28.0（如果使用 LLM 功能）
pip install openai==0.28.0
```

## 运行模式配置

系统支持两种运行模式，可通过配置统一切换：

### 查看和切换模式

```bash
# 查看当前模式
python set_mode.py

# 切换到 STDOUT 模式（推荐，自动启动服务）
python set_mode.py stdout

# 切换到 HTTP 模式（需要手动启动服务）
python set_mode.py http
```

### 1. STDOUT 模式（推荐）

无需手动启动服务，系统自动管理：

```bash
# 直接使用 CLI
python cli_main.py --image ./test_img/test.png --instruction "先灰度化再顺时针旋转90度" --output ./output/test_out.png
```

### 2. HTTP 模式

需要手动启动各个服务：

```bash
# 1. 启动服务（分别在三个终端）
python image_processor_server.py
python intent_analyzer_server.py
python image_processing_coordinator.py

# 2. 使用 CLI
python cli_main.py --image ./test_img/test.png --instruction "先灰度化再顺时针旋转90度" --output ./output/test_out.png
```

## Gradio UI（可选）

```bash
python chat_ui.py
```

## 环境变量配置

也可以通过环境变量临时切换模式：

```bash
# 临时使用 HTTP 模式
IMAGE_EDIT_MODE=http python cli_main.py --image ./test_img/test.png --instruction "先灰度化再顺时针旋转90度"

# 临时使用 STDOUT 模式  
IMAGE_EDIT_MODE=stdout python cli_main.py --image ./test_img/test.png --instruction "先灰度化再顺时针旋转90度"
```

## 项目架构

### 核心模块
- **`config.py`** - 统一配置管理（运行模式、服务端点等）
- **`prompt.py`** - 统一 Prompt 管理（LLM 提示词、错误消息、UI 文本等）
- **`cli_main.py`** - 命令行主入口
- **`chat_ui.py`** - Gradio 可视化界面

### 服务器模块
- **`image_processor_server.py`** - 图像处理工具服务器
- **`intent_analyzer_server.py`** - 意图分析服务器
- **`image_processing_coordinator.py`** - 协调器服务器

### 统一管理优势
- **配置集中化**：通过 `config.py` 统一管理所有配置
- **Prompt 集中化**：通过 `prompt.py` 统一管理所有提示词和消息
- **易于维护**：修改配置或 Prompt 只需在一个地方
- **标准化**：统一的错误处理和消息格式