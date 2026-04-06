# 🧠 NeuroDetect AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Multimodal Machine Learning Framework for Early Parkinson's Disease and RBD Screening**

*Accepted / Submitted to IEEE — Final Year Project 2026*

[Features](#-features) · [Demo](#-demo) · [Setup](#-setup) · [Architecture](#-architecture) · [Results](#-results) · [Paper](#-paper)

</div>

---

## 📌 Overview

NeuroDetect AI is a production-ready, multimodal machine learning system for early Parkinson's Disease (PD) detection. It uniquely integrates **three independent biomarker modalities** — speech acoustics, gait dynamics, and handwriting analysis — including detection of **REM Sleep Behavior Disorder (RBD)** as a prodromal stage, which is absent in most prior multimodal approaches.

> ⚠️ **Disclaimer:** This system is intended for research purposes only and does not constitute a clinical diagnosis. Always consult a qualified neurologist for medical evaluation.

---

## ✨ Features

- 🎤 **Speech Analysis** — 24 acoustic features (jitter, shimmer, HNR, MFCC) via soft voting ensemble
- 🚶 **Gait Analysis** — 24 spatiotemporal features from PhysioNet FSR signals
- ✍️ **Handwriting Analysis** — EfficientNetB0 transfer learning on spiral/wave drawings
- 🔗 **Weighted Late Fusion** — ROC-AUC weighted combination of all three modalities
- 🎯 **3-Class Classification** — HC / RBD (prodromal) / PD (unlike binary-only prior work)
- 🎙️ **Live Microphone Recording** — Record speech directly in browser
- ✏️ **Live Drawing Pad** — Draw spirals/waves directly in browser
- 📄 **Automated PDF Reports** — Clinical-grade report generation
- 🔍 **Feature Importance Panel** — SHAP-style explainability per modality
- 🌙 **Dark Mode** — Full light/dark theme support
- ⚡ **Real-time Inference** — Sub-300ms end-to-end latency

---

## 🎬 Demo

> Try the demo mode in the dashboard — no files needed.

| Dashboard | Results | Dark Mode |
|-----------|---------|-----------|
| ![Dashboard](.github/screenshots/dashboard.png) | ![Results](.github/screenshots/results.png) | ![Dark](.github/screenshots/dark.png) |

---

## 📁 Project Structure

```
NeuroDetect-AI/
│
├── backend/
│   └── main.py                  # FastAPI server — all prediction endpoints
│
├── models/
│   ├── train_speech.py          # Speech ensemble training
│   ├── train_gait.py            # Gait ensemble training
│   ├── train_handwriting.py     # EfficientNetB0 training
│   ├── fusion.py                # Weighted late fusion logic
│   └── saved/                   # Trained model files (see note below)
│       ├── speech_model.pkl
│       ├── gait_model.pkl
│       └── handwriting_model_best.pth
│
├── preprocessing/
│   ├── preprocess_speech.py     # Czech UDPR feature extraction
│   ├── preprocess_gait.py       # PhysioNet FSR processing
│   └── preprocess_handwriting.py
│
├── data/
│   ├── speech/processed/        # Scaled CSV + feature names
│   ├── gait/processed/          # Scaled CSV + feature names
│   └── handwriting/processed/   # Augmented image splits
│
├── results/
│   ├── figures/                 # ROC curves, confusion matrices
│   └── tables/                  # Performance metric tables
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app + all pages
│   │   ├── App.css
│   │   └── components/
│   │       ├── Sidebar.jsx
│   │       ├── StatsBar.jsx
│   │       ├── UploadSection.jsx
│   │       ├── RiskGauge.jsx
│   │       ├── ResultsPanel.jsx
│   │       ├── Recommendation.jsx
│   │       ├── ExplainabilityPanel.jsx
│   │       └── PDFReport.jsx
│   ├── package.json
│   └── vite.config.js
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/NeuroDetect-AI.git
cd NeuroDetect-AI
```

### 2. Backend setup
```bash
pip install -r requirements.txt
```

### 3. Train the models (or use pre-trained)
> Skip this if you have the saved model files.
```bash
# Preprocess data first
python preprocessing/preprocess_speech.py
python preprocessing/preprocess_gait.py
python preprocessing/preprocess_handwriting.py

# Train models
python models/train_speech.py
python models/train_gait.py
python models/train_handwriting.py
```

### 4. Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Frontend setup
```bash
cd frontend
npm install
npm run dev
```

### 6. Open in browser
```
http://localhost:5173
```

---

## 🏗️ Architecture

### Model Summary

| Modality | Architecture | Dataset | CV ROC-AUC | CV Accuracy |
|----------|-------------|---------|------------|-------------|
| Speech | Soft Voting Ensemble (RF + XGBoost + SVM + LR) | Czech UDPR (n=130) | **0.9802 ± 0.012** | 55.4 ± 3.2% |
| Gait | Soft Voting Ensemble (RF + XGBoost + SVM + LR) | PhysioNet (n=165) | **0.7940 ± 0.024** | 73.3 ± 2.8% |
| Handwriting | EfficientNetB0 (Transfer Learning) | Kaggle Spiral/Wave (n=204) | **0.8844 ± 0.031** | 81.7 ± 4.1% |

### Fusion Strategy

Weighted Late Fusion with weights proportional to ROC-AUC:

```
P_fusion = 0.45 · P_speech + 0.25 · P_gait + 0.30 · P_handwriting

Risk Score R = 0·P(HC) + 50·P(RBD) + 100·P(PD)
```

| Risk Score | Classification |
|------------|---------------|
| R < 35 | Low Risk (Healthy) |
| 35 ≤ R < 60 | Moderate Risk (At Risk / RBD) |
| R ≥ 60 | High Risk (Parkinson's Disease) |

---

## 📊 Results

### Fusion Performance (Validation Set, n=25)

| Strategy | Accuracy | Notes |
|----------|----------|-------|
| **Weighted Fusion (ours)** | **87.2%** | p=0.032 vs equal-weight |
| Equal-weight averaging | 82.4% | — |
| Max-voting | 79.6% | — |

### RBD Detection
- Sensitivity: **93.3%** (14/15 correctly identified)
- Specificity: **88.9%** (16/18 HC correctly classified)
- PPV: **87.5%**

### System Performance
- Average end-to-end latency: **237ms**
- Throughput: **4.2 analyses/second**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Backend health check |
| `GET` | `/model/info` | Loaded model information |
| `POST` | `/predict/speech` | Speech-only prediction |
| `POST` | `/predict/gait` | Gait-only prediction |
| `POST` | `/predict/handwriting` | Handwriting-only prediction |
| `POST` | `/predict/fusion` | Full multimodal fusion prediction |

---

## 📄 Paper

This project is associated with the following paper:

> **NeuroDetect AI: A Multimodal Framework for Parkinson's Disease and RBD Screening**  
> Mihir Vallala, Diptanu Debnath  
> Department of Computing Technologies, SRM University, Chennai, India  
> *Submitted to IEEE — 2026*

If you use this work, please cite:
```bibtex
@inproceedings{vallala2026neurodetect,
  title     = {NeuroDetect AI: A Multimodal Framework for Parkinson's Disease and RBD Screening},
  author    = {Vallala, Mihir and Debnath, Diptanu},
  booktitle = {Proceedings of the IEEE},
  year      = {2026},
  note      = {SRM University, Chennai, India}
}
```

---

## 📦 Key Dependencies

```
fastapi==0.109.0
uvicorn==0.27.0
scikit-learn==1.4.0
xgboost==2.0.3
torch==2.1.0
torchvision==0.16.0
librosa==0.10.1
praat-parselmouth==0.4.3
pandas==2.1.4
numpy==1.26.3
Pillow==10.2.0
joblib==1.3.2
```

---

## ⚠️ Note on Model Files

The trained model files (`models/saved/`) are not included in this repository due to file size constraints. To use the system:

1. Download the datasets (links in the paper)
2. Run the preprocessing scripts
3. Run the training scripts

Or contact the authors for access to pre-trained weights.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ at SRM University · Final Year Project 2026
</div>
