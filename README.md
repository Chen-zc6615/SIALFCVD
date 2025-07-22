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
python -m venv myenv
source myvenv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Download model's weight
cd checkpoints
wget https://huggingface.co/Chen-Zhencheng/Alpha_BLIP_CVD/resolve/main/pytorch_model.pt
```



## 🏗️ Project Structure

```
SIALFCVD/
├── models/                
│   └── Alpha_BLIP/                       
├── configs/
├── demo/             
├── datasets/                  
├── checkpoints/                        
├── requirements.txt       
└── README.md                         
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments


- [YOLO World](https://github.com/AILab-CVC/YOLO-World) - Open-vocabulary object detection
- [Hugging Face](https://huggingface.co/) - Multi-modal model support


## ✉️ Contact

- Author: Chenzc
- Email: threechen6615@gmail.com

## 🔄 Changelog

### v1.0.0 (2025-07-20)
- Initial release

