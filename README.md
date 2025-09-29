# conda环境
```
# 如果需要本地运行kontext，参考kontext/build_env.sh配置环境
# 需要安装fastmcp
# openai=0.28.0
conda activate kontext

```

# 启动mcp服务

分别在三个终端启动以下服务

```
python image_processor_server.py

python intent_analyzer_server.py

python image_processing_coordinator.py
```

# 启动gradio

```shell
python chat_ui.py
```