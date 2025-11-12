# Halal Logo & Barcode Detection System

**A Computer Vision Application for Automated Halal Certification & Barcode Recognition**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Academic Context](#academic-context)
3. [Features](#features)
4. [System Architecture](#system-architecture)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [Technical Specifications](#technical-specifications)
9. [Dataset Information](#dataset-information)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [License](#license)

---

## 🎯 Project Overview

This repository implements an integrated **computer vision system** for detecting and validating halal logos and product barcodes from images. The system leverages YOLOv8 (You Only Look Once) for real-time object detection and pyzbar for barcode decoding, providing an end-to-end solution for food product verification.

**Key Capabilities:**

- ✅ Detect halal certification logos in product packaging
- ✅ Identify and decode barcodes (EAN, UPC, QR codes, etc.)
- ✅ Dual-model inference pipeline (halal detection + barcode detection)
- ✅ Interactive Streamlit web interface for easy testing and demonstration
- ✅ Cross-platform support (Linux, macOS, Windows)

---

## 🏫 Academic Context

**Course:** Computer Vision & Computer Pattern Recognition (CCP)  
**Institution:** Bahria University
**Academic Year:** 2024-2025

This project demonstrates:

- **Deep Learning Fundamentals**: YOLOv8 architecture for object detection
- **Computer Vision Techniques**: Image preprocessing, annotation, multi-model inference
- **Machine Learning Pipeline**: Dataset preparation, model training, evaluation, and deployment
- **Software Engineering**: Clean code practices, modular design, documentation, version control

---

## ✨ Features

### 1. Halal Logo Detection

- Real-time detection of halal certification symbols on product packaging
- Confidence scores for each detected logo
- Bounding box visualization with annotations

### 2. Barcode Detection & Decoding

- **Dual-layer approach:**
  - **pyzbar library**: Direct barcode decoding from images (EAN-13, UPC-A, QR codes, etc.)
  - **YOLOv8 model**: Barcode region localization and detection
- Barcode value extraction and display
- Support for multiple barcode formats

### 3. Interactive User Interface

- **Streamlit-powered** web application
- Upload images or capture via webcam
- Adjustable confidence thresholds (sidebar sliders)
- Real-time annotation overlays
- Downloadable results as PNG

### 4. Visualization & Feedback

- Color-coded detection badges (success/warning states)
- Detailed confidence scores and class labels
- Annotated output images with bounding boxes
- User-friendly visual feedback

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web Interface                │
│              (deploy/my.py - Main Application)          │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
   │  Image  │        │  Image  │        │  Image  │
   │ Upload  │        │ Webcam  │        │ Process |
   │         │        │         │        │         |
   └────┬────┘        └────┬────┘        └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼──────────┐ ┌────▼──────────┐ ┌────▼──────────┐
   │ YOLOv8 Model  │ │ YOLOv8 Model  │ │   pyzbar      │
   │  (Halal)      │ │  (Barcode)    │ │  (Decode)     │
   └────┬──────────┘ └────┬──────────┘ └────┬──────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  Annotation & Visualization       │
        │  (PIL ImageDraw, YOLO plot)       │
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  Streamlit Display & Download     │
        │  (PNG export, interactive UI)     │
        └───────────────────────────────────┘
```

---

## 📦 Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on Python 3.12)
- **pip** or **mamba/conda** package manager
- **Git** for version control
- **libzbar** native library (for barcode decoding)
- 2GB+ RAM recommended for model inference

### Step 1: Clone Repository

```bash
git clone https://github.com/Asad-10x/halal_food_classifier.git
cd halal_food_classifier
```

### Step 2: Create Virtual Environment (Recommended)

#### Using venv

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using mamba/conda

```bash
mamba create -n halal-cv python=3.12
mamba activate halal-cv
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

- `streamlit` – Web UI framework
- `ultralytics` – YOLOv8 object detection
- `pyzbar` – Barcode decoding
- `Pillow` – Image processing
- `opencv-python` – Computer vision utilities
- `numpy`, `pandas`, `scikit-learn` – Data processing & ML

### Step 4: Install Native Library for Barcode Decoding

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y libzbar0
```

#### macOS

```bash
brew install zbar
```

#### Windows

Download `libzbar-64.dll` from [pyzbar releases](https://github.com/NaturalHistoryMuseum/pyzbar/releases) and place it in the project directory or system PATH.

#### Alternative (Conda)

```bash
mamba install -c conda-forge zbar pyzbar
```

### Step 5: Download & Extract Dataset (Optional)

The training dataset is included as a zip file. To extract:

```bash
cd data
unzip -q halal_logo.v5i.yolov8.zip
cd ..
```

**Dataset structure:**

```
data/halal_logo_dataset/
├── data.yaml          # Dataset metadata (classes, paths)
├── train/
│   ├── images/        # Training images
│   └── labels/        # YOLO format annotations
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # YOLO format annotations
└── test/
    ├── images/        # Test images
    └── labels/        # YOLO format annotations
```

---

## 🚀 Usage Guide

### Running the Streamlit Application

Navigate to the `deploy` directory and launch the app:

```bash
cd deploy
streamlit run my.py
```

The application will start on `http://localhost:8501` (default Streamlit port).

**Note:** Use `streamlit run` instead of `python my.py` to avoid Streamlit context warnings.

### Interactive Workflow

1. **Upload or Capture Image**

   - Use the "Upload Logo Image" widget to select a JPG, JPEG, or PNG file
   - Or use "Or Take a Photo" to capture via webcam

2. **Set Confidence Thresholds** (Optional)

   - Adjust sliders in the sidebar:
     - **Halal Confidence Threshold** (default: 0.5)
     - **Barcode Confidence Threshold** (default: 0.4)
   - Lower threshold = more detections (higher false positives)
   - Higher threshold = stricter detection (may miss objects)

3. **View Results**

   - Input image is displayed
   - Halal logo detections appear with bounding boxes and confidence scores
   - Barcode region is highlighted if detected
   - Decoded barcode value is shown (if successfully decoded)

4. **Download Results**
   - Click the "⬇️ Download Annotated Result" button to save the annotated image as PNG

### Example Usage Scenarios

#### Scenario 1: Verify Halal Certification

```
1. Upload a product image containing a halal logo
2. Model detects logo and displays confidence score
3. If confidence > threshold → "HALAL CERTIFIED" badge appears
4. Download annotated image for documentation
```

#### Scenario 2: Extract Product Barcode

```
1. Upload product packaging image with visible barcode
2. App displays:
   - Barcode location (from YOLOv8 detection)
   - Decoded barcode value (from pyzbar)
   - Barcode type (EAN-13, UPC-A, QR code, etc.)
3. Use decoded value for product database lookup
```

---

## 📂 Project Structure

```
halal_food_classifier/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── deploy/
│   └── my.py                          # Main Streamlit application
│
├── src/
│   ├── cv_model.ipynb                # Jupyter notebook for model training/experimentation
│   ├── kernel_build.py                # Kernel setup utilities
│   └── utils/
│       └── virt_env.py                # Virtual environment helper scripts
│
├── data/
│   ├── halal_logo.v5i.yolov8.zip    # Compressed dataset (YOLO format)
│   ├── Deoply.zip                    # Deployment-related files
│   └── halal_logo_dataset/            # Extracted dataset (after unzipping)
│       ├── data.yaml
│       ├── train/
│       ├── valid/
│       └── test/
│
└── tmp/                               # Temporary files directory

```

---

## 🔧 Technical Specifications

### Model Architecture

**YOLOv8 (Ultralytics)**

- **Architecture:** Convolutional Neural Network (CNN) with anchor-free detection heads
- **Training Framework:** PyTorch
- **Input Size:** 640×640 pixels (automatically resized)
- **Output:** Bounding boxes with class labels and confidence scores
- **Models Used:**
  - `halal_logo_detector.pt` – Detects halal certification logos
  - `barcode_detector.pt` – Localizes barcode regions

### Barcode Decoding Pipeline

**pyzbar Integration**

- Uses ZBar library for efficient barcode scanning
- Supports formats:
  - EAN-13, EAN-8 (European Article Number)
  - UPC-A, UPC-E (Universal Product Code)
  - QR Code
  - Code 128, Code 39
  - And 25+ other formats

### Image Processing

- **Preprocessing:** RGB conversion, PIL Image objects
- **Annotation:** PIL ImageDraw for bounding boxes and text overlays
- **Fonts:** Arial.ttf (with fallback to system default)
- **Output:** PNG format (lossless compression)

### Performance Considerations

- **Inference Time:** ~100-300ms per image (GPU: ~50-100ms)
- **Memory Usage:** ~1.5-2GB for model + inference
- **Supported Resolutions:** 480×640 to 1920×1080 pixels

---

## 📊 Dataset Information

### Halal Logo Dataset (Roboflow)

- **Source:** Roboflow (YOLOv8 format)
- **Total Images:** Varies (check `data.yaml`)
- **Classes:** Halal certification logos (e.g., standard halal symbol, crescent & star, etc.)
- **Train/Valid/Test Split:** 70% / 15% / 15% (approx.)
- **Annotations:** YOLO format (normalized bounding box coordinates)

### Dataset YAML Structure

```yaml
path: /path/to/halal_logo_dataset
train: train/images
val: valid/images
test: test/images

nc: 1 # Number of classes
names:
  0: "halal" # Class name
```

### Using Your Own Dataset

To train with a custom dataset:

1. Prepare images and YOLO format annotations
2. Create a `data.yaml` file with paths and class names
3. Update `src/cv_model.ipynb` with your dataset path
4. Train using YOLOv8: `yolo detect train data=custom_data.yaml`

---

## 🐛 Troubleshooting

### Issue: Import Errors for pyzbar/libzbar

**Problem:** `ModuleNotFoundError: No module named 'pyzbar'` or `OSError: libzbar not found`

**Solution:**

```bash
# Install pyzbar
pip install pyzbar

# Install native zbar library
# Linux:
sudo apt-get install libzbar0

# macOS:
brew install zbar

# Windows: Download DLL from https://github.com/NaturalHistoryMuseum/pyzbar/releases
```

---

### Issue: Streamlit "Missing ScriptRunContext" Warnings

**Problem:** Repeated warnings about `missing ScriptRunContext` when running with `python my.py`

**Solution:** Always use the Streamlit command to launch the app:

```bash
streamlit run deploy/my.py
```

---

### Issue: Model Files Not Found

**Problem:** `FileNotFoundError: halal_logo_detector.pt not found`

**Solution:**

1. Ensure model `.pt` files are in the `deploy/` directory
2. Download pre-trained YOLOv8 models from [Ultralytics](https://github.com/ultralytics/ultralytics)
3. Or train your own using `src/cv_model.ipynb`

---

### Issue: Poor Detection Accuracy

**Problem:** Halal logos or barcodes not being detected

**Troubleshooting:**

- Lower confidence threshold in sidebar (trade-off: more false positives)
- Ensure image is well-lit and clear
- Check image resolution (minimum 480×480 recommended)
- Verify model `.pt` files are trained on similar data

---

### Issue: Barcode Not Decoding

**Problem:** Barcode detected but not decoded (pyzbar failure)

**Solutions:**

1. Ensure barcode is clearly visible and not rotated
2. Increase image contrast/brightness
3. Verify libzbar is installed: `python -c "from pyzbar import zbar_library; print('OK')"`
4. Check barcode format is supported by ZBar

---

### Issue: Out of Memory Errors

**Problem:** `RuntimeError: CUDA out of memory` or `MemoryError`

**Solutions:**

- Close other applications
- Reduce image resolution
- Use CPU inference (slower): Set environment variable `export CUDA_VISIBLE_DEVICES=""`
- Use a smaller model variant (if available)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Make changes** and **commit:** `git commit -m "Description of changes"`
4. **Push to branch:** `git push origin feature/your-feature`
5. **Submit a Pull Request** with detailed description

### Code Style

- Follow PEP 8 Python conventions
- Use type hints where possible
- Include docstrings for functions
- Keep functions focused and modular

---

## 📝 License

This project is provided for academic and educational purposes. Please check with your institution for specific licensing requirements.

---

## 📧 Contact & Support

For questions or issues:

- **Repository:** [https://github.com/Asad-10x/halal_food_classifier](https://github.com/Asad-10x/halal_food_classifier)
- **Branch:** `dev` (development), `main` (stable)
- **Issues:** Use GitHub Issues for bug reports and feature requests

---

## 🙏 Acknowledgments

- **YOLOv8 Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Barcode Detection:** [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar)
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Dataset:** Roboflow Halal Logo Dataset

---

**Last Updated:** November 2024  
**Status:** Active Development  
**Python Version:** 3.8+
