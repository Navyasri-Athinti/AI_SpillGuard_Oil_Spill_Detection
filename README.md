# AI_SpillGuard_Oil_Spill_Detection

# 🛢️ AI-Based Oil Spill Detection System

An end-to-end deep learning project for detecting and segmenting oil spills from satellite imagery using a U-Net architecture, with real-time deployment via a Streamlit web application.

---

## 📌 Project Overview
Oil spills pose serious environmental threats to marine ecosystems. Manual monitoring is slow and error-prone.  
This project proposes an **AI-driven oil spill detection system** using **semantic segmentation** to automatically identify oil spill regions in satellite images.

The system covers:
- Dataset analysis
- Model training
- Evaluation
- Visualization
- Web-based deployment

---

## 🗂️ Dataset Overview
- Satellite images of ocean regions
- Corresponding binary segmentation masks
- Images resized to **256×256**
- Masks indicate oil spill regions (white) vs background (black)

**Key challenges:**
- Class imbalance
- Varying lighting and textures
- Irregular spill boundaries

---

## 🔍 Methodology
1. Image preprocessing and normalization
2. Feature extraction using encoder–decoder CNN (U-Net)
3. Pixel-wise segmentation
4. Post-processing using thresholding and morphology
5. Visualization and reporting

---

## 📊 Exploratory Data Analysis (EDA)
- Visual inspection of sample images and masks
- Pixel distribution analysis
- Understanding oil spill shapes and coverage

EDA helped guide preprocessing decisions and threshold selection.

---

## 🧹 Data Preprocessing
- Image resizing to 256×256
- Normalization to [0,1]
- Mask binarization
- Dataset splitting (train / validation / test)

---

## 🧠 Model Architecture
- U-Net architecture
- Encoder for feature extraction
- Decoder for spatial reconstruction
- Skip connections for fine-grained segmentation

Loss & metrics:
- Binary Cross-Entropy + Dice Loss
- Dice Coefficient
- Intersection over Union (IoU)

---

## 🏋️ Training & Evaluation
- Optimizer: Adam
- Batch size: dataset-dependent
- Epochs: tuned experimentally

### Evaluation Metrics:
| Metric | Value |
|------|------|
| Accuracy | ~95% |
| Dice Score | ~0.79 |
| IoU | ~0.81 |
| Precision | ~0.91 |
| Recall | ~0.96 |

---

## 📈 Results & Visualization
- Side-by-side comparison:
  - Original image
  - Ground truth mask
  - Predicted mask
- Overlay visualization
- Probability heatmaps for uncertainty regions

The model generalizes well but may under-segment edges due to domain shift.

---

## 🌐 Streamlit Web Application
Features:
- Upload satellite image
- Real-time oil spill prediction
- Oil spill percentage estimation
- Overlay visualization
- Probability heatmap
- PDF report generation
- Prediction history (session-based)
- Animated ocean-themed UI

---

## 🧪 Deployment
The application is deployed locally using Streamlit.

### Run locally:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
