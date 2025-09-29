# conda环境
```
# 需要安装fastmcp
# openai的版本需要是0.28.0
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


# 非gradio请求方式
sh run.sh

```
python cli_main.py --image ./test_img/test.png --instruction "先灰度化再顺时针旋转90度" --output ./output/test_out.png
```