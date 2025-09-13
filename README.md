# 🌿 Plant Disease Classification

This project implements a deep learning model to classify plant diseases using the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).  
It is optimized for **GPU training** with **PyTorch** and **CUDA Toolkit 12.6** for faster performance.  

## 🛠️ Setup Instructions

### 1. Install Python
Download and install **Python 3.12.6** (or a compatible version) from [python.org](https://www.python.org/).

### 2. Clone the Repository
```bash
git clone https://github.com/Durjoy01/Plant-Disease-Classification-with-GPU-Support-MLP-.git
cd Plant-Disease-Classification-with-GPU-Support-MLP-
```

### 3. Download the Dataset
Download from Kaggle: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  

Unzip and arrange directories like this:
```
project445/
├── train/
├── valid/
├── test/
├── code.ipynb
├── hudai.py
├── gpu_monitor/
└── README.md
```

### 4. Create Virtual Environment
```bash
python -m venv venv
# Activate venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 5. Install VSCode Jupyter Extension
- Download VSCode from [code.visualstudio.com](https://code.visualstudio.com/)  
- Open VSCode  
- Go to **Extensions** → Search **Jupyter** → Install  

### 6. GPU Setup
- Ensure you have an **NVIDIA GPU** → check with:
  ```bash
  nvidia-smi
  ```
- Update your GPU driver via the [NVIDIA GeForce Experience App](https://www.nvidia.com/en-us/geforce/geforce-experience/download/)  
  (install latest **Game Ready / Studio Driver**).

### 7. Install CUDA Toolkit 12.6
Download from NVIDIA’s official site:  
[CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads)

### 8. Install Dependencies
Install base packages:
```bash
pip install -r requirements.txt
```

Create a file named **requirements.txt** with the following content:
```txt
numpy
matplotlib
scikit-learn
pandas
opencv-python
jupyter
torchvision
```

### 9. Install PyTorch with CUDA Support
Visit [PyTorch.org](https://pytorch.org/get-started/locally/) to get the correct install command for your system.  

Example for CUDA 12.6:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 🚀 Running the Project
Run in Jupyter Notebook:
1. Open VSCode  
2. Launch `code.ipynb`  
3. Run all cells  

Or run the script directly:
```bash
python hudai.py
```

## 📊 GPU Monitoring
Monitor GPU usage during training:
```bash
watch -n 1 nvidia-smi
```

Or use the Python-based monitoring script:
```bash
python gpu_monitor/monitor.py
```

## 📌 Notes
- Always **activate your virtual environment** before running the project.  
- Ensure dataset is structured inside `train/`, `valid/`, and `test/`.  
- Training time may vary depending on dataset size and GPU capability.  

## 👥 Authors
- Soleman Hossain  
- Durjoy Barua  
- Nafis Mahmud
- Shahriar Ratul  

🔗 GitHub Repo: [Plant Disease Classification with GPU Support](https://github.com/Durjoy01/Plant-Disease-Classification-with-GPU-Support-MLP-)  
🔗 Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
🔗 Python: [Download Python](https://www.python.org/)  
🔗 PyTorch: [Get Started with PyTorch](https://pytorch.org/get-started/locally/)  
🔗 CUDA Toolkit: [CUDA 12.6 Download](https://developer.nvidia.com/cuda-downloads)  
🔗 VSCode: [Download VSCode](https://code.visualstudio.com/)  
🔗 NVIDIA Drivers: [NVIDIA GeForce Drivers](https://www.nvidia.com/download/index.aspx)  
S