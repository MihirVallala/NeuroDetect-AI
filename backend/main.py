

import os
import sys
import io
import json
import time
import logging
import tempfile
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from scipy import stats
from PIL import Image
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


#  AUDIO FEATURE EXTRACTION

try:
    import librosa
    import parselmouth
    AUDIO_EXTRACTION_AVAILABLE = True
    logger_audio = logging.getLogger('audio_extraction')
    logger_audio.info("✅ Audio extraction libraries loaded (librosa, parselmouth)")
except ImportError:
    AUDIO_EXTRACTION_AVAILABLE = False
    logger_audio = logging.getLogger('audio_extraction')
    logger_audio.warning("⚠️ Audio extraction NOT available. Install: pip install librosa praat-parselmouth")


def extract_speech_features_from_audio(audio_path: str) -> np.ndarray:
    """
    Extract 24 acoustic features from .wav audio file
    
    Features:
    - Jitter (3): local, RAP, PPQ5
    - Shimmer (5): local, APQ3, APQ5, APQ11, DDA
    - HNR (1): Harmonics-to-Noise Ratio
    - F0 (2): mean, std
    - MFCC (13): mean values
    
    Returns: np.ndarray of shape (24,)
    """
    if not AUDIO_EXTRACTION_AVAILABLE:
        raise RuntimeError("Audio extraction not available. Install librosa and praat-parselmouth.")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=48000)
        sound = parselmouth.Sound(audio_path)
        
        # Pitch analysis
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan  # unvoiced frames → NaN
        f0_mean = float(np.nanmean(pitch_values))
        f0_std  = float(np.nanstd(pitch_values))
        
        if np.isnan(f0_mean) or f0_mean == 0:
            f0_mean = 120.0
        if np.isnan(f0_std) or f0_std == 0:
            f0_std = 10.0
        
        # Jitter
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        try:
            jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_local, jitter_rap, jitter_ppq5 = 0.005, 0.003, 0.004
        
        # Shimmer
        try:
            shimmer_local = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq11 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = parselmouth.praat.call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda = 0.03, 0.025, 0.028, 0.032, 0.075
        
        # HNR
        harmonicity = sound.to_harmonicity()
        hnr = harmonicity.values[harmonicity.values != -200].mean()
        if np.isnan(hnr):
            hnr = 15.0
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1)
        
        # Construct feature array (24 features)
        features = np.array([
            jitter_local, jitter_rap, jitter_ppq5,
            shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda,
            hnr, f0_mean, f0_std
        ] + mfcc_mean.tolist(), dtype=float)
        
        return features
        
    except Exception as e:
        raise ValueError(f"Failed to extract features from audio: {str(e)}")


# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

PATHS = {
    'speech_model':   BASE_DIR / 'models/saved/speech_model.pkl',
    'gait_model':     BASE_DIR / 'models/saved/gait_model.pkl',
    'hw_model':       BASE_DIR / 'models/saved/handwriting_model_best.pth',
    'speech_scaler':  BASE_DIR / 'data/speech/processed/speech_scaler.pkl',
    'gait_scaler':    BASE_DIR / 'data/gait/processed/gait_scaler.pkl',
    'speech_features':BASE_DIR / 'data/speech/processed/feature_names.txt',
    'gait_features':  BASE_DIR / 'data/gait/processed/gait_feature_names.txt',
}

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  HANDWRITING CNN (must match train_handwriting.py)
# ─────────────────────────────────────────────
class HandwritingCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────
#  MODEL STORE (loaded once at startup)
# ─────────────────────────────────────────────
class ModelStore:
    """Holds all loaded models in memory"""
    speech_model  = None
    gait_model    = None
    hw_model      = None
    speech_scaler = None
    gait_scaler   = None
    device        = None
    speech_feature_names = []
    gait_feature_names   = []


store = ModelStore()


def load_models():
    """Load all models at startup"""
    logger.info("Loading all models...")

    store.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Speech model
    store.speech_model = joblib.load(PATHS['speech_model'])
    logger.info("✅ Speech model loaded")

    # Gait model
    store.gait_model = joblib.load(PATHS['gait_model'])
    logger.info("✅ Gait model loaded")

    # Handwriting CNN
    store.hw_model = HandwritingCNN(num_classes=2)
    store.hw_model.load_state_dict(
        torch.load(PATHS['hw_model'], map_location=store.device)
    )
    store.hw_model.eval()
    store.hw_model = store.hw_model.to(store.device)
    logger.info(f"✅ Handwriting model loaded (device: {store.device})")

    # Scalers
    store.speech_scaler = joblib.load(PATHS['speech_scaler'])
    store.gait_scaler   = joblib.load(PATHS['gait_scaler'])
    logger.info("✅ Scalers loaded")

    # Feature names
    with open(PATHS['speech_features']) as f:
        store.speech_feature_names = [
            line.split('. ', 1)[1].strip()
            for line in f if '. ' in line
        ]

    with open(PATHS['gait_features']) as f:
        store.gait_feature_names = [
            line.split('. ', 1)[1].strip()
            for line in f if '. ' in line
        ]

    logger.info(
        f"✅ All models ready | "
        f"Speech features: {len(store.speech_feature_names)} | "
        f"Gait features: {len(store.gait_feature_names)} | "
        f"Audio extraction: {'ENABLED' if AUDIO_EXTRACTION_AVAILABLE else 'DISABLED'}"
    )


# ─────────────────────────────────────────────
#  APP LIFESPAN
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    load_models()
    logger.info("🚀 API ready at http://localhost:8000")
    yield
    logger.info("Shutting down...")


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Parkinson's Multimodal Detection API",
    description=(
        "Detects Parkinson's Disease and RBD using "
        "Speech, Gait, and Handwriting analysis. "
        "NOW SUPPORTS: .wav audio files for speech!"
    ),
    version="1.0.1",
    lifespan=lifespan
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  RESPONSE MODELS
# ─────────────────────────────────────────────
class SpeechPrediction(BaseModel):
    prediction:   int
    label:        str
    confidence:   float
    probabilities: dict
    risk_score:   float
    processing_time_ms: float
    input_type:   str = "csv"  # NEW: 'csv' or 'audio'


class GaitPrediction(BaseModel):
    prediction:   int
    label:        str
    confidence:   float
    probabilities: dict
    risk_score:   float
    top_features: dict
    processing_time_ms: float


class HandwritingPrediction(BaseModel):
    prediction:   int
    label:        str
    confidence:   float
    probabilities: dict
    risk_score:   float
    processing_time_ms: float


class FusionPrediction(BaseModel):
    final_prediction: int
    final_label:      str
    risk_score:       float
    risk_level:       str
    confidence:       float
    modality_results: dict
    fusion_weights:   dict
    recommendation:   str
    processing_time_ms: float


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_risk_level(risk_score: float) -> str:
    if risk_score < 35:
        return "Low Risk"
    elif risk_score < 60:
        return "Moderate Risk"
    else:
        return "High Risk"


def get_recommendation(risk_score: float, label: str) -> str:
    if risk_score < 35:
        return (
            "No significant indicators of Parkinson's Disease detected. "
            "Continue regular health monitoring."
        )
    elif risk_score < 60:
        return (
            "Some indicators of early-stage or at-risk condition detected. "
            "Recommend consultation with a neurologist for further evaluation."
        )
    else:
        return (
            "Strong indicators of Parkinson's Disease detected. "
            "Immediate neurological consultation strongly recommended. "
            "This is a screening tool only and not a clinical diagnosis."
        )


def extract_gait_features_from_file(content: bytes) -> np.ndarray:
    """
    Extract 24 gait features from uploaded .txt file

    Same logic as preprocess_gait.py extract_features()
    """
    try:
        data = np.loadtxt(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse gait file: {str(e)}"
        )

    if data.ndim != 2 or data.shape[1] < 19:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 19 columns, got {data.shape[1] if data.ndim==2 else 'invalid'}"
        )

    if data.shape[1] > 19:
        data = data[:, :19]

    left_total  = data[:, 17]
    right_total = data[:, 18]

    features = {}

    # Left foot
    features['left_mean'] = left_total.mean()
    features['left_std']  = left_total.std()
    features['left_max']  = left_total.max()
    features['left_cv']   = left_total.std() / (left_total.mean() + 1e-8)

    # Right foot
    features['right_mean'] = right_total.mean()
    features['right_std']  = right_total.std()
    features['right_max']  = right_total.max()
    features['right_cv']   = right_total.std() / (right_total.mean() + 1e-8)

    # Symmetry
    total = left_total.mean() + right_total.mean() + 1e-8
    features['lr_asymmetry'] = abs(left_total.mean() - right_total.mean()) / total
    features['lr_ratio']     = left_total.mean() / (right_total.mean() + 1e-8)

    # Steps
    threshold     = 50
    left_contact  = (left_total > threshold).astype(int)
    right_contact = (right_total > threshold).astype(int)
    left_steps    = int(np.sum(np.diff(left_contact) > 0))
    right_steps   = int(np.sum(np.diff(right_contact) > 0))

    features['left_steps']     = left_steps
    features['right_steps']    = right_steps
    features['step_asymmetry'] = (
        abs(left_steps - right_steps) /
        (left_steps + right_steps + 1e-8)
    )

    # Stride timing
    left_onsets = np.where(np.diff(left_contact) > 0)[0]
    if len(left_onsets) > 2:
        stride_times = np.diff(left_onsets) / 100.0
        stride_times = stride_times[
            (stride_times > 0.3) & (stride_times < 3.0)
        ]
        if len(stride_times) > 1:
            features['stride_mean'] = stride_times.mean()
            features['stride_std']  = stride_times.std()
            features['stride_cv']   = (
                stride_times.std() / (stride_times.mean() + 1e-8)
            )
        else:
            features['stride_mean'] = 0
            features['stride_std']  = 0
            features['stride_cv']   = 0
    else:
        features['stride_mean'] = 0
        features['stride_std']  = 0
        features['stride_cv']   = 0

    # Total force
    total_force = left_total + right_total
    features['total_force_mean'] = total_force.mean()
    features['total_force_std']  = total_force.std()
    features['total_force_cv']   = (
        total_force.std() / (total_force.mean() + 1e-8)
    )

    # Distribution shape
    features['left_skew']      = float(stats.skew(left_total))
    features['left_kurtosis']  = float(stats.kurtosis(left_total))
    features['right_skew']     = float(stats.skew(right_total))
    features['right_kurtosis'] = float(stats.kurtosis(right_total))

    # Duration
    features['duration'] = float(data[-1, 0] - data[0, 0])

    return np.array(list(features.values()), dtype=float)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image for EfficientNet"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img).unsqueeze(0).to(store.device)


def compute_fusion_risk(
    speech_probs: np.ndarray,
    gait_probs:   np.ndarray,
    hw_probs:     np.ndarray
) -> tuple:
    """
    Compute fused risk score from all 3 modalities

    Returns: (risk_score, final_class, final_label)
    """
    WEIGHTS = {
        'speech':      0.45,
        'gait':        0.25,
        'handwriting': 0.30
    }

    # Convert each modality to PD risk (0-1)
    speech_risk = speech_probs[1] * 0.5 + speech_probs[2] * 1.0
    gait_risk   = gait_probs[1]
    hw_risk     = hw_probs[1]

    # Weighted fusion
    risk = (
        speech_risk * WEIGHTS['speech'] +
        gait_risk   * WEIGHTS['gait'] +
        hw_risk     * WEIGHTS['handwriting']
    )

    risk_score = float(np.clip(risk * 100, 0, 100))

    # Determine class
    speech_class = int(np.argmax(speech_probs))
    pd_votes = sum([
        speech_class == 2,
        gait_risk > 0.5,
        hw_risk > 0.5
    ])

    if pd_votes >= 2 or risk_score >= 60:
        final_class = 2
        final_label = "Parkinson's Disease"
    elif pd_votes == 1 or speech_class == 1 or risk_score >= 35:
        final_class = 1
        final_label = "At Risk (RBD / Early Stage)"
    else:
        final_class = 0
        final_label = "Healthy"

    return risk_score, final_class, final_label


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Check if API and all models are loaded"""
    models_loaded = all([
        store.speech_model is not None,
        store.gait_model   is not None,
        store.hw_model     is not None,
    ])

    return {
        "status":        "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "device":        str(store.device),
        "version":       "1.0.1",
        "audio_extraction_available": AUDIO_EXTRACTION_AVAILABLE
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata and performance metrics"""
    return {
        "models": {
            "speech": {
                "architecture": "Soft Voting Ensemble (RF + XGBoost + SVM + LR)",
                "task":         "3-class: HC / RBD / PD",
                "dataset":      "Czech UDPR",
                "subjects":     130,
                "features":     24,
                "cv_roc_auc":   0.9802,
                "cv_accuracy":  0.5538,
                "accepts":      "CSV (24 features) or WAV audio"
            },
            "gait": {
                "architecture": "Soft Voting Ensemble (RF + XGBoost + SVM + LR)",
                "task":         "Binary: HC vs PD",
                "dataset":      "PhysioNet Gait",
                "subjects":     165,
                "features":     24,
                "cv_roc_auc":   0.7940,
                "cv_accuracy":  0.7333
            },
            "handwriting": {
                "architecture": "EfficientNetB0 (Transfer Learning)",
                "task":         "Binary: HC vs PD",
                "dataset":      "Kaggle Spiral/Wave",
                "images":       204,
                "test_roc_auc": 0.8844,
                "test_accuracy":0.8167
            }
        },
        "fusion": {
            "strategy": "Weighted Late Fusion",
            "weights": {
                "speech":      0.45,
                "gait":        0.25,
                "handwriting": 0.30
            }
        },
        "audio_extraction": {
            "available": AUDIO_EXTRACTION_AVAILABLE,
            "features_extracted": 24 if AUDIO_EXTRACTION_AVAILABLE else 0,
            "libraries": "librosa + praat-parselmouth" if AUDIO_EXTRACTION_AVAILABLE else "Not installed"
        }
    }


@app.post("/predict/speech", response_model=SpeechPrediction)
async def predict_speech(file: UploadFile = File(...)):
    """
    Predict from speech features CSV OR WAV audio
    
    NEW: Now accepts .wav audio files! Features extracted automatically.
    
    Upload:
    - CSV file with one row of 24 speech features, OR
    - WAV audio file (3-5 seconds of sustained vowel "Ahhhh")
    """
    start = time.time()
    
    filename = file.filename.lower()
    input_type = "unknown"

    # ══════════════════════════════════════════════════════════
    # OPTION 1: WAV AUDIO FILE (NEW!)
    # ══════════════════════════════════════════════════════════
    if filename.endswith('.wav'):
        if not AUDIO_EXTRACTION_AVAILABLE:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Audio extraction not available on this server. "
                    "Please upload CSV with pre-extracted features, or "
                    "ask administrator to install: pip install librosa praat-parselmouth"
                )
            )
        
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract 24 features from audio
            logger.info(f"Extracting features from audio: {filename}")
            features = extract_speech_features_from_audio(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if features.shape[0] != 24:
                raise ValueError(f"Expected 24 features, extracted {features.shape[0]}")
            
            X = features.reshape(1, -1)
            input_type = "audio"
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Audio feature extraction failed: {str(e)}"
            )
    
    # ══════════════════════════════════════════════════════════
    # OPTION 2: CSV FILE (ORIGINAL)
    # ══════════════════════════════════════════════════════════
    elif filename.endswith('.csv'):
        content = await file.read()
        
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse CSV: {str(e)}"
            )

        # Drop metadata columns if present
        drop_cols = ['participant_code', 'label', 'label_name']
        feature_df = df.drop(
            columns=[c for c in drop_cols if c in df.columns]
        )

        if feature_df.shape[1] != 24:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 24 features, got {feature_df.shape[1]}"
            )

        X = feature_df.values[:1]  # Use first row
        input_type = "csv"
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Upload .csv (features) or .wav (audio)"
        )

    # ══════════════════════════════════════════════════════════
    # PREDICT
    # ══════════════════════════════════════════════════════════
    try:
        probs = store.speech_model.predict_proba(X)[0]
        pred  = int(np.argmax(probs))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    label_map = {0: 'Healthy (HC)', 1: 'At Risk (RBD)', 2: "Parkinson's (PD)"}
    risk      = float(probs[1] * 0.5 + probs[2] * 1.0) * 100

    elapsed = (time.time() - start) * 1000

    return SpeechPrediction(
        prediction   = pred,
        label        = label_map[pred],
        confidence   = float(probs[pred]),
        probabilities= {
            'HC':  round(float(probs[0]), 4),
            'RBD': round(float(probs[1]), 4),
            'PD':  round(float(probs[2]), 4)
        },
        risk_score          = round(risk, 1),
        processing_time_ms  = round(elapsed, 2),
        input_type          = input_type
    )


@app.post("/predict/gait", response_model=GaitPrediction)
async def predict_gait(file: UploadFile = File(...)):
    """
    Predict from gait recording .txt file

    Upload a standard PhysioNet gait file (19 columns, tab-separated).
    The file should be a normal walking trial (not dual-task).
    """
    start = time.time()

    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .txt gait file"
        )

    content = await file.read()

    # Extract features
    raw_features = extract_gait_features_from_file(content)

    # Scale features
    try:
        X_scaled = store.gait_scaler.transform(
            raw_features.reshape(1, -1)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature scaling failed: {str(e)}"
        )

    # Predict
    try:
        probs = store.gait_model.predict_proba(X_scaled)[0]
        pred  = int(np.argmax(probs))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    label_map = {0: 'Healthy (HC)', 1: "Parkinson's (PD)"}
    risk      = float(probs[1]) * 100

    # Top 3 most important features for this prediction
    feat_names = store.gait_feature_names
    top_features = {}
    if len(feat_names) == len(raw_features):
        feat_dict = dict(zip(feat_names, raw_features.tolist()))
        key_feats = ['stride_cv', 'lr_asymmetry', 'step_asymmetry',
                     'stride_std', 'left_cv', 'right_cv']
        for k in key_feats:
            if k in feat_dict:
                top_features[k] = round(feat_dict[k], 4)

    elapsed = (time.time() - start) * 1000

    return GaitPrediction(
        prediction   = pred,
        label        = label_map[pred],
        confidence   = float(probs[pred]),
        probabilities= {
            'HC': round(float(probs[0]), 4),
            'PD': round(float(probs[1]), 4)
        },
        risk_score         = round(risk, 1),
        top_features       = top_features,
        processing_time_ms = round(elapsed, 2)
    )


@app.post("/predict/handwriting", response_model=HandwritingPrediction)
async def predict_handwriting(file: UploadFile = File(...)):
    """
    Predict from handwriting image

    Upload a PNG/JPG image of a spiral or wave drawing.
    Image will be resized to 224x224 automatically.
    """
    start = time.time()

    allowed = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Please upload an image file ({', '.join(allowed)})"
        )

    content = await file.read()

    # Preprocess image
    img_tensor = preprocess_image(content)

    # Predict
    try:
        with torch.no_grad():
            output = store.hw_model(img_tensor)
            probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred   = int(np.argmax(probs))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    label_map = {0: 'Healthy (HC)', 1: "Parkinson's (PD)"}
    risk      = float(probs[1]) * 100

    elapsed = (time.time() - start) * 1000

    return HandwritingPrediction(
        prediction   = pred,
        label        = label_map[pred],
        confidence   = float(probs[pred]),
        probabilities= {
            'HC': round(float(probs[0]), 4),
            'PD': round(float(probs[1]), 4)
        },
        risk_score         = round(risk, 1),
        processing_time_ms = round(elapsed, 2)
    )


@app.post("/predict/fusion", response_model=FusionPrediction)
async def predict_fusion(
    speech_file:      Optional[UploadFile] = File(None),
    gait_file:        Optional[UploadFile] = File(None),
    handwriting_file: Optional[UploadFile] = File(None),
):
    """
    Full multimodal fusion prediction

    Upload any combination of:
    - speech_file:      CSV with 24 features OR WAV audio (NEW!)
    - gait_file:        .txt gait recording file
    - handwriting_file: PNG/JPG spiral or wave image

    At least ONE file must be provided.
    Missing modalities use neutral probabilities (0.5).
    """
    start = time.time()

    # Validate at least one file
    if all(f is None for f in [speech_file, gait_file, handwriting_file]):
        raise HTTPException(
            status_code=400,
            detail="At least one file must be provided"
        )

    modality_results = {}
    WEIGHTS = {'speech': 0.45, 'gait': 0.25, 'handwriting': 0.30}

    # ── Speech ────────────────────────────────────────────────
    if speech_file is not None:
        content = await speech_file.read()
        filename = speech_file.filename.lower()
        
        # Handle WAV audio
        if filename.endswith('.wav'):
            if not AUDIO_EXTRACTION_AVAILABLE:
                raise HTTPException(
                    status_code=400,
                    detail="Audio extraction not available. Upload CSV or install libraries."
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                features = extract_speech_features_from_audio(tmp_path)
                os.unlink(tmp_path)
                X = features.reshape(1, -1)
                speech_probs = store.speech_model.predict_proba(X)[0]
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise HTTPException(status_code=400, detail=f"Speech audio error: {str(e)}")
        
        # Handle CSV
        else:
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                drop_cols = ['participant_code', 'label', 'label_name']
                feature_df = df.drop(
                    columns=[c for c in drop_cols if c in df.columns]
                )
                X = feature_df.values[:1]
                speech_probs = store.speech_model.predict_proba(X)[0]
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speech file error: {str(e)}"
                )

        modality_results['speech'] = {
            'prediction': int(np.argmax(speech_probs)),
            'label': ['HC', 'RBD', 'PD'][int(np.argmax(speech_probs))],
            'confidence': round(float(speech_probs.max()), 4),
            'probabilities': {
                'HC':  round(float(speech_probs[0]), 4),
                'RBD': round(float(speech_probs[1]), 4),
                'PD':  round(float(speech_probs[2]), 4)
            },
            'weight': WEIGHTS['speech']
        }
    else:
        # Neutral: equal probability for all 3 classes
        speech_probs = np.array([0.333, 0.333, 0.334])
        modality_results['speech'] = {
            'prediction': None,
            'label': 'Not provided',
            'confidence': None,
            'probabilities': None,
            'weight': WEIGHTS['speech']
        }

    # ── Gait ──────────────────────────────────────────────────
    if gait_file is not None:
        content = await gait_file.read()
        try:
            raw_features = extract_gait_features_from_file(content)
            X_scaled     = store.gait_scaler.transform(
                raw_features.reshape(1, -1)
            )
            gait_probs = store.gait_model.predict_proba(X_scaled)[0]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Gait file error: {str(e)}"
            )

        modality_results['gait'] = {
            'prediction': int(np.argmax(gait_probs)),
            'label': ['HC', 'PD'][int(np.argmax(gait_probs))],
            'confidence': round(float(gait_probs.max()), 4),
            'probabilities': {
                'HC': round(float(gait_probs[0]), 4),
                'PD': round(float(gait_probs[1]), 4)
            },
            'weight': WEIGHTS['gait']
        }
    else:
        # Neutral: 50/50
        gait_probs = np.array([0.5, 0.5])
        modality_results['gait'] = {
            'prediction': None,
            'label': 'Not provided',
            'confidence': None,
            'probabilities': None,
            'weight': WEIGHTS['gait']
        }

    # ── Handwriting ───────────────────────────────────────────
    if handwriting_file is not None:
        content = await handwriting_file.read()
        try:
            img_tensor = preprocess_image(content)
            with torch.no_grad():
                output    = store.hw_model(img_tensor)
                hw_probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Handwriting file error: {str(e)}"
            )

        modality_results['handwriting'] = {
            'prediction': int(np.argmax(hw_probs)),
            'label': ['HC', 'PD'][int(np.argmax(hw_probs))],
            'confidence': round(float(hw_probs.max()), 4),
            'probabilities': {
                'HC': round(float(hw_probs[0]), 4),
                'PD': round(float(hw_probs[1]), 4)
            },
            'weight': WEIGHTS['handwriting']
        }
    else:
        # Neutral: 50/50
        hw_probs = np.array([0.5, 0.5])
        modality_results['handwriting'] = {
            'prediction': None,
            'label': 'Not provided',
            'confidence': None,
            'probabilities': None,
            'weight': WEIGHTS['handwriting']
        }

    # ── Fusion ────────────────────────────────────────────────
    risk_score, final_class, final_label = compute_fusion_risk(
        speech_probs, gait_probs, hw_probs
    )

    # Overall confidence: average of provided modalities
    provided_confs = [
        v['confidence']
        for v in modality_results.values()
        if v['confidence'] is not None
    ]
    confidence = float(np.mean(provided_confs)) if provided_confs else 0.5

    elapsed = (time.time() - start) * 1000

    return FusionPrediction(
        final_prediction   = final_class,
        final_label        = final_label,
        risk_score         = round(risk_score, 1),
        risk_level         = get_risk_level(risk_score),
        confidence         = round(confidence, 4),
        modality_results   = modality_results,
        fusion_weights     = WEIGHTS,
        recommendation     = get_recommendation(risk_score, final_label),
        processing_time_ms = round(elapsed, 2)
    )


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )    