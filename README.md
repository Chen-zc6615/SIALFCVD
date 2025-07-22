# SIALFCVD

A deep learning-based computer vision project for color vision deficiency


## 📋 Requirements

- Python 3.11
- 16GB+ GPU memory 

## 🛠️ Installation

```bash
# Clone the project
git clone https://github.com/Chen-zc6615/SIALFCVD.git
cd SIALFCVD

# Create virtual environment
python -m venv myenv
source myenv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Download model's weight
cd checkpoints
wget https://huggingface.co/Chen-Zhencheng/Alpha_BLIP_CVD/resolve/main/alpha_blip_cvd.pt
```



## 📁 Project Structure

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
## 🎮 Demo
```
cd demo
python demo/demo1.py --image_path demo/test.jpg --cvd_type 0 --cvd_levels 60
# simulate with machado method
#cvd_type = {0: "Protanopia",  1: "Deuteranopia",  2: "Tritanopia"}
#cvd_levels = {0, 10, 20, 30, ... 100}


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

