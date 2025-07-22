# SIALFCVD

A deep learning-based computer vision project for color vision deficiency


## 📋 Requirements

- Python 3.9+
- 8GB+ GPU memory 

## 🛠️ Installation

```bash
# Clone the project
git clone https://github.com/your-username/SIALFCVD.git
cd SIALFCVD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download model's weight
cd checkpoints
wget https://huggingface.co/Chen-Zhencheng/Alpha_BLIP_CVD/resolve/main/pytorch_model.pt
```

### Method 2: Using conda

```bash
# Create conda environment
conda create -n sialfcvd python=3.11
conda activate sialfcvd

# Install PyTorch (according to your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
SIALFCVD/
├── models/                 # Model-related code
│   ├── Alpha_BLIP/        # BLIP model implementation
│   ├── YOLO_World/        # YOLO World object detection
│   └── ...
├── utils/                 # Utility functions
├── configs/               # Configuration files
├── data/                  # Data directory
├── checkpoints/           # Model weights
├── results/               # Result outputs
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── main.py               # Main program entry
```

## 🚦 Quick Start

### Basic Usage

```bash
# Activate environment
conda activate sialfcvd  # or source venv/bin/activate

# Run main program
python main.py

# Start Gradio interface (if supported)
python app.py
```

### Object Detection Example

```python
from models.yolo_world import YOLOWorld

# Initialize model
model = YOLOWorld('path/to/weights.pth')

# Perform inference
results = model.predict('path/to/image.jpg')

# Visualize results
model.visualize(results, save_path='output.jpg')
```

### Multi-modal Task Example

```python
from models.alpha_blip import AlphaBLIP

# Load multi-modal model
model = AlphaBLIP()

# Image caption generation
caption = model.generate_caption('path/to/image.jpg')
print(f"Image caption: {caption}")

# Visual question answering
answer = model.visual_qa('path/to/image.jpg', "What's in this image?")
print(f"Answer: {answer}")
```

## 📊 Supported Models

| Model Type | Model Name | Description | Status |
|------------|------------|-------------|--------|
| Object Detection | YOLO World | Open-vocabulary object detection | ✅ |
| Object Detection | MMDetection | General object detection framework | ✅ |
| Pose Estimation | MMPose | Human pose estimation | ✅ |
| Multi-modal | BLIP | Image-text understanding | ✅ |
| Multi-modal | LLaVA | Large vision-language model | ✅ |
| Classification | TimM | Image classification model library | ✅ |

## 🔧 Configuration

Main configuration files are located in the `configs/` directory:

- `model_config.yaml`: Model parameter configuration
- `data_config.yaml`: Data path configuration
- `train_config.yaml`: Training parameter configuration

### Custom Configuration Example

```yaml
# configs/model_config.yaml
model:
  name: "yolo_world"
  weights: "checkpoints/yolo_world.pth"
  device: "cuda"
  confidence_threshold: 0.5
  nms_threshold: 0.4

data:
  input_size: [640, 640]
  classes: ["person", "car", "bicycle"]
```

## 📈 Performance Metrics

| Model | Dataset | mAP | FPS | GPU Memory |
|-------|---------|-----|-----|------------|
| YOLO World | COCO | 52.3 | 45 | 4.2GB |
| MMDet | COCO | 48.1 | 38 | 3.8GB |

## 🛡️ Model Deployment

### ONNX Export

```bash
python tools/export_onnx.py --model yolo_world --weights checkpoints/yolo_world.pth
```

### TensorRT Optimization

```bash
python tools/tensorrt_convert.py --onnx model.onnx --output model.trt
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_models.py -v

# Performance testing
python benchmark/benchmark_models.py
```

## 📝 Development

### Adding New Models

1. Create new model files in the `models/` directory
2. Implement standard model interfaces
3. Add corresponding configuration files
4. Write unit tests

### Code Standards

```bash
# Install development dependencies
pip install black flake8 isort

# Code formatting
black .
isort .

# Code checking
flake8 .
```

## 🤝 Contributing

Pull Requests and Issues are welcome!

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [OpenMMLab](https://github.com/open-mmlab) - Excellent computer vision toolkits
- [YOLO World](https://github.com/AILab-CVC/YOLO-World) - Open-vocabulary object detection
- [Hugging Face](https://huggingface.co/) - Multi-modal model support
- [Gradio](https://gradio.app/) - Easy-to-use interface framework

## 📞 Contact

- Author: chenzc
- Email: your.email@example.com
- Project Link: https://github.com/your-username/SIALFCVD

## 🔄 Changelog

### v1.0.0 (2025-01-XX)
- Initial release
- Support for basic object detection and multi-modal functionality
- Integrated Gradio interface

### v0.9.0 (2025-01-XX)
- Added YOLO World support
- Optimized model loading speed
- Fixed known issues

---

⭐ If this project helps you, please give it a star!
