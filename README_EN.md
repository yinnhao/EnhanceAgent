<div align="right">
  <strong>English</strong> | <a href="README.md">中文</a>
</div>

<div align="center">



<div>
  <img src="assets/icon.png" alt="Enhance Agent Logo" width="120" height="120">
</div>

# Enhance Agent: Image Quality Enhancement via Natural Language
</div>



## 📖 Project Overview

**Enhance Agent** is an intelligent image processing system capable of performing image editing through natural language instructions. The system adopts a multi-agent collaborative architecture that understands user commands, breaks tasks down, calls the appropriate tools, and automatically executes the required image processing steps.

### Key Features
- Supports natural language instructions
- Handles compound instructions
- Enables multi-stage enhancement

### Example Results

| Input Image | Prompt | Output Image |
| --- | --- | --- |
| <img src="testset/gray.png" alt="Example Input 1" width="200"/> | Remove rain, then colorize, then apply super-resolution | <img src="assets/output1.png" alt="Example Output 1" width="200"/> |
| <img src="testset/noise.png" alt="Example Input 2" width="200"/> | Remove noise first, then apply super-resolution | <img src="assets/output2.png" alt="Example Output 2" width="200"/> |

### UI Interface

<div align="center">
  <img src="assets/ui2.png" alt="Gradio UI" width="800">
</div>

## 🚀 Quick Start

### Environment Setup
```bash
conda create -n enhanceAgent python=3.10
conda activate enhanceAgent
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
cd KAIR
pip install -r requirement.txt
pip install basicsr natsort gdown fastmcp Pillow gradio
```

### Model Download

```bash
sh download_model.sh
```

### Configure LLM Keys

The project uses Volcano Engine's `doubao-seed-1-6-lite-251015` model for intent analysis and task coordination. Before running, please set your own `DOUBAO_API_KEY` in `config.py` (Volcano Engine provides a free quota of 500k tokens).

```python
DOUBAO_API_KEY = "****"
```


### Run

#### 1. Gradio UI Mode
```bash
python chat_ui.py
```

#### 2. CLI Mode
```bash
python cli_main.py --image ./testset/grayscale.png --instruction "Colorize first, then apply super-resolution" --output ./output/test_out.png
```

### Supported Instruction Examples

- ✅ "Convert the image to grayscale"
- ✅ "Convert to grayscale first, then rotate 90 degrees"
- ✅ "Colorize this image"
- ✅ "Improve image resolution"
- ✅ "Remove image noise"
- ✅ "Remove rain and then apply super-resolution"
- ✅ "Remove the raindrops"
- ✅ "Make it clearer"
- ✅ "First denoise, then deblur, then apply super-resolution, then colorize"

  And many other combinations.
  
  More capabilities are on the way.
---

