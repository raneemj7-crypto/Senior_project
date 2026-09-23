# KAIST Multispectral Pedestrian Dataset Visualization Tool

This repository provides a minimal toolkit to visualize the KAIST Multispectral Pedestrian Dataset. You can overlay annotations on both Visible (RGB) and Thermal (LWIR) images and export them as a side-by-side video.

## 🛠 Installation

First, ensure you have Python installed. Then, install the required dependencies using `pip`:

```bash
$ pip install -r requirements.txt
```

Note: Required packages include opencv-python, matplotlib, and tqdm.

## 🚀 How to Use
- Jupyter Notebook
If you want to explore the dataset interactively, use the provided notebook:

Open `demo.ipynb` in Jupyter or Google Colab.

Follow the steps to visualize individual pairs of Visible and Thermal images with class-specific bounding boxes.


## 📂 Dataset Structure
To ensure the scripts run correctly, please maintain the following directory structure:


```Plaintext
project/
├── annotations-xml-new-sanitized/
│   └── set04/
│       └── V001/           # XML files (e.g., I00000.xml)
├── images/
│   └── set04/
│       └── V001/
│           ├── visible/    # RGB images (I00000.jpg)
│           └── lwir/       # Thermal images (I00000.jpg)
├── train-preview-01.txt    # List of frames to process
├── demo.ipynb
└── demo.py
```
