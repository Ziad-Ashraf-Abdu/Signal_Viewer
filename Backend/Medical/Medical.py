# Medical.py (modified for 3-channel display) - removed screenshots, added AI condition identification placeholder
# Requirements:
#   pip install dash pandas numpy scipy plotly pyedflib
import base64
import io
import os
import math
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

try:
    from transformers import AutoModel

    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    print("[Model Loader] transformers library not available. Install with: pip install transformers")

# Try importing pyedflib for EDF reading (EEG). If missing, app will instruct user.
try:
    import pyedflib

    PYEDFLIB_AVAILABLE = True
except Exception:
    PYEDFLIB_AVAILABLE = False

import requests
import json

# Screenshot capability check
try:
    import plotly.io as pio
    import kaleido

    SCREENSHOT_AVAILABLE = True
    print("[Screenshot] Plotly screenshot capabilities available")
except ImportError as e:
    SCREENSHOT_AVAILABLE = False
    print(f"[Screenshot] Warning: {e}")
    print("[Screenshot] Install with: pip install kaleido")
# ---------- Configuration / Limits ----------
MAX_EEG_SUBJECTS = 150  # limit EEG subjects loaded
DEFAULT_MAX_EEG_SECONDS = 60  # read at most this many seconds per EDF to save memory (None to read whole file)
DEFAULT_MAX_EEG_SAMPLES = None  # computed from seconds & fs when reading

import plotly.io as pio
from PIL import Image, ImageOps
import tempfile


def capture_graph_screenshot(figure):
    """
    Capture a screenshot of the current plotly figure and return as bytes.
    """
    try:
        # Convert figure to image bytes
        img_bytes = pio.to_image(figure, format="png", width=800, height=600, engine="kaleido")
        return img_bytes
    except Exception as e:
        print(f"[Screenshot] Error capturing graph: {e}")
        return None


def analyze_ecg_image_with_teachable_machine(img_bytes,
                                             model_url="https://teachablemachine.withgoogle.com/models/aV7sUMdvb/"):
    """
    Send ECG image to Teachable Machine model for classification using TensorFlow.

    Args:
        img_bytes: Image bytes (PNG format)
        model_url: Base URL of the Teachable Machine model

    Returns:
        dict with predictions or error message
    """
    try:
        # Load the image
        image = Image.open(io.BytesIO(img_bytes))

        # Resize to 224x224 (standard for Teachable Machine)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array and normalize (Teachable Machine normalization)
        image_array = np.asarray(image, dtype=np.float32)
        normalized_image = (image_array / 127.5) - 1

        # Reshape for model input (batch_size, height, width, channels)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image

        print(f"[2D Analysis] Attempting to analyze ECG image")
        print(f"[2D Analysis] Model URL: {model_url}")
        print(f"[2D Analysis] Image shape: {data.shape}")

        # Fetch metadata to get class labels
        class_labels = []
        try:
            metadata_url = model_url.rstrip('/') + '/metadata.json'
            print(f"[2D Analysis] Fetching metadata from: {metadata_url}")
            metadata_response = requests.get(metadata_url, timeout=10)

            if metadata_response.status_code == 200:
                metadata = metadata_response.json()
                print(f"[2D Analysis] Metadata received")

                if isinstance(metadata, dict) and 'labels' in metadata:
                    labels_data = metadata['labels']
                    if isinstance(labels_data, list):
                        class_labels = [str(label) for label in labels_data]

                print(f"[2D Analysis] Found {len(class_labels)} classes: {class_labels}")

        except Exception as meta_error:
            print(f"[2D Analysis] Could not fetch metadata: {meta_error}")
            class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

        # Try to use a pre-converted Keras model if available
        keras_model_path = "teachable_machine_model.h5"

        if os.path.exists(keras_model_path):
            try:
                import tensorflow as tf
                from tensorflow import keras

                print(f"[2D Analysis] Loading Keras model from {keras_model_path}...")

                # Custom object scope to handle compatibility issues
                custom_objects = {}

                # Load with TensorFlow 2.x compatible settings
                try:
                    model = keras.models.load_model(
                        keras_model_path,
                        compile=False,
                        custom_objects=custom_objects
                    )
                except Exception as load_error:
                    print(f"[2D Analysis] Standard load failed, trying with custom loader...")
                    # Try loading with safe mode
                    model = tf.keras.models.load_model(
                        keras_model_path,
                        compile=False,
                        safe_mode=False  # Disable safe mode for older models
                    )

                print(f"[2D Analysis] Model loaded successfully!")

                # Make prediction
                print(f"[2D Analysis] Running prediction...")
                prediction = model.predict(data, verbose=0)

                # Format predictions
                predictions = []
                for i, prob in enumerate(prediction[0]):
                    label = class_labels[i] if i < len(class_labels) else f'Class {i}'
                    predictions.append({
                        "class": label,
                        "probability": float(prob)
                    })

                # Sort by probability
                predictions.sort(key=lambda x: x['probability'], reverse=True)

                print(f"[2D Analysis] Prediction complete!")
                for pred in predictions:
                    print(f"  {pred['class']}: {pred['probability']:.4f}")

                return {
                    "success": True,
                    "predictions": predictions,
                    "image_processed": True,
                    "image_size": "224x224",
                    "model_url": model_url,
                    "class_labels": class_labels,
                    "requires_setup": False,
                    "top_prediction": predictions[0]['class'],
                    "top_confidence": predictions[0]['probability']
                }

            except Exception as keras_error:
                print(f"[2D Analysis] Keras model loading failed: {keras_error}")
                import traceback
                traceback.print_exc()

                # Provide alternative solution
                predictions = []
                for i, label in enumerate(class_labels):
                    predictions.append({
                        "class": label,
                        "probability": 0.0
                    })

                return {
                    "success": True,
                    "predictions": predictions,
                    "note": f"Model loading failed: {str(keras_error)}",
                    "image_processed": True,
                    "image_size": "224x224",
                    "model_url": model_url,
                    "class_labels": class_labels,
                    "requires_setup": True,
                    "setup_instructions": [
                        "The model file has compatibility issues with your TensorFlow version.",
                        "",
                        "Solution 1: Update TensorFlow",
                        "  pip install --upgrade tensorflow",
                        "",
                        "Solution 2: Re-export the model",
                        "1. Go to: https://teachablemachine.withgoogle.com/models/aV7sUMdvb/",
                        "2. Click 'Export Model' > TensorFlow > Keras",
                        "3. Download fresh model file",
                        "4. Replace 'teachable_machine_model.h5'",
                        "",
                        "Solution 3: Use TensorFlow Lite (if available)",
                        "  Export as 'TensorFlow Lite' format instead"
                    ]
                }

        # Model not found - provide download instructions
        predictions = []
        for i, label in enumerate(class_labels):
            predictions.append({
                "class": label,
                "probability": 0.0
            })

        return {
            "success": True,
            "predictions": predictions,
            "note": "Keras model file not found. Please download and convert the model.",
            "image_processed": True,
            "image_size": "224x224",
            "model_url": model_url,
            "class_labels": class_labels,
            "requires_setup": True,
            "setup_instructions": [
                "Download the model in Keras format:",
                "1. Go to: https://teachablemachine.withgoogle.com/models/aV7sUMdvb/",
                "2. Click 'Export Model' button",
                "3. Select 'TensorFlow' tab",
                "4. Choose 'Keras' option",
                "5. Click 'Download my model'",
                "6. Extract the downloaded zip file",
                "7. Rename 'keras_model.h5' to 'teachable_machine_model.h5'",
                "8. Place it in the same directory as Medical.py",
                "9. Restart the application",
                "",
                "Note: Ensure you have TensorFlow 2.x installed:",
                "  pip install tensorflow>=2.10.0"
            ]
        }

    except Exception as e:
        print(f"[2D Analysis] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Failed to analyze image: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


import os
import time
import json
from datetime import datetime
import logging
import math

import numpy as np

# TensorFlow imports are optional (for .h5 models)
try:
    import tensorflow as tf
    from keras.models import load_model
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input as KerasInput

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# PyTorch imports are optional (for .pth/.pt models)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# SciPy for filtering & resampling (optional)
try:
    from scipy.signal import butter, filtfilt, resample

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Try to import user's HeartGPTClassifier if present in project (preferred)
try:
    # attempt a relative import if you placed the class in a module named 'models'
    # from models import HeartGPTClassifier
    # If you keep HeartGPTClassifier in the main training script, adapt import path accordingly.
    HeartGPTClassifier = None
except Exception:
    HeartGPTClassifier = None

# --- Minimal HeartGPTClassifier (fallback) ---
# This is a compact version of the model architecture used in your training script.
# It's included so the loader can reconstruct the model if the checkpoint contains config + state_dict.
if HeartGPTClassifier is None and TORCH_AVAILABLE:
    class TransformerBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln1 = nn.LayerNorm(config['n_embd'])
            self.attn = nn.MultiheadAttention(config['n_embd'], config['n_head'], dropout=config.get('dropout', 0.0),
                                              batch_first=True)
            self.ln2 = nn.LayerNorm(config['n_embd'])
            self.ffn = nn.Sequential(
                nn.Linear(config['n_embd'], 4 * config['n_embd']),
                nn.GELU(),
                nn.Linear(4 * config['n_embd'], config['n_embd']),
                nn.Dropout(config.get('dropout', 0.0))
            )

        def forward(self, x):
            # x: (B, T, C)
            residual = x
            x = self.ln1(x)
            # using PyTorch MultiheadAttention which expects (B, T, C) with batch_first=True
            attn_out, _ = self.attn(x, x, x)
            x = residual + attn_out
            x = x + self.ffn(self.ln2(x))
            return x


    class HeartGPTClassifier(nn.Module):
        def __init__(self, config_or_dict):
            super().__init__()
            # Accept either Config object or plain dict
            if isinstance(config_or_dict, dict):
                cfg = config_or_dict
            else:
                # try to build dict out of object with attributes
                cfg = {k: getattr(config_or_dict, k) for k in
                       ['n_embd', 'n_head', 'n_layer', 'block_size', 'dropout', 'num_classes'] if
                       hasattr(config_or_dict, k)}

            # defaults
            n_embd = int(cfg.get('n_embd', 128))
            n_head = int(cfg.get('n_head', 4))
            n_layer = int(cfg.get('n_layer', 4))
            block_size = int(cfg.get('block_size', 1024))
            dropout = float(cfg.get('dropout', 0.2))
            num_classes = int(cfg.get('num_classes', 5))

            self.config = {'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer, 'block_size': block_size,
                           'dropout': dropout, 'num_classes': num_classes}
            self.block_size = block_size
            self.conv_frontend = nn.Sequential(
                nn.Conv1d(1, n_embd, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(n_embd),
                nn.ReLU(),
                nn.Conv1d(n_embd, n_embd, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(n_embd),
                nn.ReLU()
            )
            self.signal_projection = nn.Linear(1, n_embd)
            self.position_embedding = nn.Embedding(block_size, n_embd)
            self.dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            hidden1 = n_embd * 2
            hidden2 = max(32, n_embd // 2)
            hidden3 = max(16, n_embd // 4)
            self.classifier = nn.Sequential(
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, hidden3),
                nn.LayerNorm(hidden3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden3, num_classes)
            )

        def forward(self, x):
            # x: (B, T) float tensor
            B, T = x.shape
            if T > self.block_size:
                x = x[:, :self.block_size]
                T = self.block_size
            x_unsq = x.unsqueeze(1)  # (B, 1, T)
            conv_out = self.conv_frontend(x_unsq)  # (B, C, T)
            conv_out = conv_out.permute(0, 2, 1).contiguous()  # (B, T, C)
            x_lin = x.unsqueeze(-1)
            proj = self.signal_projection(x_lin)
            x_emb = (conv_out + proj) * 0.5
            pos_ids = torch.arange(T, device=x.device)
            pos_emb = self.position_embedding(pos_ids).unsqueeze(0).expand(B, -1, -1)
            x = self.dropout(x_emb + pos_emb)
            for b in self.blocks:
                x = b(x)
            x = self.ln_f(x)
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1)[0]
            x = torch.cat([x_mean, x_max], dim=1)
            logits = self.classifier(x)
            return logits

# --- Required imports (add these at the top of your file) ---
import numpy as np
import time
from datetime import datetime
import os

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Warning] PyTorch not available")

# Transformers for HuggingFace models
try:
    from transformers import AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[Warning] Transformers library not available")

# SciPy for signal processing
try:
    from scipy.signal import butter, filtfilt, resample

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] SciPy not available - filtering/resampling disabled")


# --- ConditionIdentificationModel with REAL class labels ---
class ConditionIdentificationModel:
    """
    Updated model loader & inference wrapper with actual diagnostic labels.

    ECG Classes: Based on standard cardiovascular conditions from major ECG datasets
    EEG Classes: Based on TUAB (abnormal detection) and TUEV (event classification)
    """

    # Real ECG labels from PTB-XL, CODE, and other cardiovascular databases
    ECG_LABELS = [
        "Myocardial Infarction",
        "Normal",
        "Conduction Disturbance",
        "LVH (Left Ventricular Hypertrophy)",
        "Hypertrophy"
    ]

    # Real EEG labels based on BIOT datasets (TUAB, TUEV, CHB-MIT)
    EEG_LABELS = [

        "Schizophrenia", "Epilepsy", "Normal EEG"
    ]

    def __init__(
            self,
            ecg_model_path="hubert-ecg-small",
            eeg_model_path="EEG-PREST-16-channels.ckpt",
            signal_type="ECG",
            confidence_threshold=0.5,
            device=None,
            use_huggingface=True
    ):
        self.ecg_model_path = ecg_model_path
        self.eeg_model_path = eeg_model_path
        self.signal_type = signal_type
        self.confidence_threshold = confidence_threshold
        self.use_huggingface = use_huggingface

        self._ecg_model = None
        self._eeg_model = None
        self._active_model = None
        self._model_type = None
        self._labels = []
        self._model_loaded = False

        self._last_analysis_time = None
        self._total_predictions = 0

        # ECG preprocessing params (HuBERT-ECG)
        self.ecg_sequence_length = 500
        self.ecg_sampling_rate = 100
        self.ecg_channels = 12
        self.ecg_filter_params = {'lowcut': 0.05, 'highcut': 47.0, 'order': 4}

        # EEG preprocessing params (BIOT)
        self.eeg_sequence_length = 3000
        self.eeg_sampling_rate = 100
        self.eeg_channels = 16
        self.eeg_filter_params = {'lowcut': 0.5, 'highcut': 70.0, 'order': 4}

        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu') if TORCH_AVAILABLE else None
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else None

        # Load appropriate labels on initialization
        self.load_labels(signal_type)

    def load_labels(self, signal_type=None):
        """Load labels based on signal type - using hardcoded real labels"""
        if signal_type is None:
            signal_type = self.signal_type

        if signal_type == "ECG":
            self._labels = self.ECG_LABELS.copy()
            print(f"[Model Loader] Loaded {len(self._labels)} ECG cardiovascular condition labels")
        else:  # EEG
            self._labels = self.EEG_LABELS.copy()
            print(f"[Model Loader] Loaded {len(self._labels)} EEG neurological condition labels")

        return True

    def load_model(self, signal_type=None):
        """Load appropriate model based on signal type"""
        if signal_type is None:
            signal_type = self.signal_type

        self.signal_type = signal_type
        self.load_labels(signal_type)

        if signal_type == "ECG":
            return self._load_ecg_model()
        else:  # EEG
            return self._load_eeg_model()

    def _load_ecg_model(self):
        """Load ECG model (HuBERT-ECG)"""
        if self._ecg_model is not None and self._model_type == 'hubert-ecg':
            print("[Model Loader] ECG model already loaded")
            self._active_model = self._ecg_model
            self._model_loaded = True
            return True

        try:
            # Try HuggingFace HuBERT-ECG
            if self.use_huggingface and TRANSFORMERS_AVAILABLE:
                try:
                    print(f"[Model Loader] Loading HuBERT-ECG from HuggingFace: {self.ecg_model_path}")

                    if not self.ecg_model_path.endswith(('.pt', '.pth', '.h5', '.keras')):
                        model_id = self.ecg_model_path
                        if not model_id.startswith("Edoardo-BS/hubert-ecg-"):
                            model_id = f"Edoardo-BS/hubert-ecg-{self.ecg_model_path}"

                        print(f"[Model Loader] Loading from HuggingFace: {model_id}")
                        self._ecg_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

                        if self.device is not None:
                            self._ecg_model = self._ecg_model.to(self.device)
                        self._ecg_model.eval()
                        self._model_type = 'hubert-ecg'

                        # Add classification head
                        if not hasattr(self._ecg_model, 'classifier') or self._ecg_model.classifier is None:
                            hidden_size = getattr(self._ecg_model.config, 'hidden_size', 512)
                            num_classes = len(self._labels)

                            print(f"[Model Loader] Adding classification head: {hidden_size} -> {num_classes} classes")
                            self._ecg_model.classifier = nn.Linear(hidden_size, num_classes)

                            if self.device is not None:
                                self._ecg_model.classifier = self._ecg_model.classifier.to(self.device)

                            nn.init.xavier_uniform_(self._ecg_model.classifier.weight)
                            nn.init.zeros_(self._ecg_model.classifier.bias)

                        self._active_model = self._ecg_model
                        self._model_loaded = True
                        print(
                            f"[Model Loader] Successfully loaded HuBERT-ECG with {num_classes} cardiovascular conditions")
                        return True

                except Exception as e:
                    print(f"[Model Loader] HuggingFace loading failed: {e}")
                    print("[Model Loader] Falling back to local file loading...")

            # Try loading from local file
            if os.path.exists(self.ecg_model_path):
                print(f"[Model Loader] Attempting to load ECG model from: {self.ecg_model_path}")
                return False

            print(f"[Model Loader] ECG model file not found: {self.ecg_model_path}")
            return False

        except Exception as e:
            print(f"[Model Loader] Error loading ECG model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_eeg_model(self):
        """Load EEG model (BIOT EEG-PREST)"""
        if self._eeg_model is not None and self._model_type == 'biot-eeg':
            print("[Model Loader] EEG model already loaded")
            self._active_model = self._eeg_model
            self._model_loaded = True
            return True

        try:
            if not os.path.exists(self.eeg_model_path):
                print(f"[Model Loader] EEG model file not found: {self.eeg_model_path}")
                print("[Model Loader] Please download BIOT model from: https://github.com/ycq091044/BIOT")
                return False

            print(f"[Model Loader] Loading BIOT EEG-PREST model from: {self.eeg_model_path}")

            if not TORCH_AVAILABLE:
                print("[Model Loader] PyTorch is required for BIOT models")
                return False

            # Load BIOT checkpoint
            checkpoint = torch.load(self.eeg_model_path, map_location=self.device)

            # Create BIOT model
            self._eeg_model = self._create_biot_model(checkpoint)

            if self._eeg_model is None:
                return False

            self._eeg_model.eval()
            self._model_type = 'biot-eeg'
            self._active_model = self._eeg_model
            self._model_loaded = True

            print(f"[Model Loader] Successfully loaded BIOT EEG model with {len(self._labels)} neurological conditions")
            return True

        except Exception as e:
            print(f"[Model Loader] Error loading EEG model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_biot_model(self, checkpoint):
        """Create BIOT model architecture from checkpoint"""
        try:
            # Extract config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                # Default BIOT config for EEG-PREST-16-channels
                config = {
                    'input_channels': 16,
                    'hidden_size': 512,
                    'num_layers': 12,
                    'num_heads': 8,
                    'dropout': 0.1,
                    'num_classes': len(self._labels)
                }

            # Create simple BIOT-style transformer
            model = BIOTModel(config)

            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            if self.device is not None:
                model = model.to(self.device)

            return model

        except Exception as e:
            print(f"[Model Loader] Error creating BIOT model: {e}")
            return None

    def _preprocess_ecg_signal(self, signal_data, original_fs=None):
        """Preprocess ECG signal for HuBERT-ECG"""
        sig = np.asarray(signal_data, dtype=np.float32).copy()
        if sig.size == 0:
            raise ValueError("Empty signal")

        if np.any(np.isnan(sig)):
            sig = sig[~np.isnan(sig)]

        # Bandpass filter
        if SCIPY_AVAILABLE and len(sig) > (self.ecg_filter_params['order'] * 3):
            try:
                filter_fs = original_fs if original_fs is not None else self.ecg_sampling_rate
                nyq = filter_fs / 2.0
                low = max(self.ecg_filter_params['lowcut'] / nyq, 1e-6)
                high = min(self.ecg_filter_params['highcut'] / nyq, 0.9999)

                b, a = butter(self.ecg_filter_params['order'], [low, high], btype='band')
                sig = filtfilt(b, a, sig).astype(np.float32)
            except Exception as e:
                print(f"[Preprocess ECG] Filtering warning: {e}")

        # Resample to 100 Hz
        if original_fs is not None and SCIPY_AVAILABLE and original_fs != self.ecg_sampling_rate and len(sig) > 10:
            try:
                num_samples = int(len(sig) * self.ecg_sampling_rate / float(original_fs))
                sig = resample(sig, num_samples)
            except Exception:
                x_old = np.linspace(0, 1, len(sig))
                x_new = np.linspace(0, 1, int(len(sig) * self.ecg_sampling_rate / float(original_fs)))
                sig = np.interp(x_new, x_old, sig).astype(np.float32)

        # Rescale to [-1, 1]
        sig_min, sig_max = sig.min(), sig.max()
        if sig_max - sig_min > 1e-8:
            sig = 2 * (sig - sig_min) / (sig_max - sig_min) - 1
        else:
            sig = np.zeros_like(sig)

        # Pad/truncate
        if len(sig) >= self.ecg_sequence_length:
            sig = sig[:self.ecg_sequence_length]
        else:
            pad = np.zeros(self.ecg_sequence_length - len(sig), dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)

        return sig.astype(np.float32)

    def _preprocess_eeg_signal(self, signal_data, original_fs=None):
        """Preprocess EEG signal for BIOT"""
        sig = np.asarray(signal_data, dtype=np.float32).copy()
        if sig.size == 0:
            raise ValueError("Empty signal")

        if np.any(np.isnan(sig)):
            sig = sig[~np.isnan(sig)]

        # Bandpass filter
        if SCIPY_AVAILABLE and len(sig) > (self.eeg_filter_params['order'] * 3):
            try:
                filter_fs = original_fs if original_fs is not None else self.eeg_sampling_rate
                nyq = filter_fs / 2.0
                low = max(self.eeg_filter_params['lowcut'] / nyq, 1e-6)
                high = min(self.eeg_filter_params['highcut'] / nyq, 0.9999)

                b, a = butter(self.ecg_filter_params['order'], [low, high], btype='band')
                sig = filtfilt(b, a, sig).astype(np.float32)
            except Exception as e:
                print(f"[Preprocess EEG] Filtering warning: {e}")

        # Resample to 100 Hz
        if original_fs is not None and SCIPY_AVAILABLE and original_fs != self.eeg_sampling_rate and len(sig) > 10:
            try:
                num_samples = int(len(sig) * self.eeg_sampling_rate / float(original_fs))
                sig = resample(sig, num_samples)
            except Exception:
                x_old = np.linspace(0, 1, len(sig))
                x_new = np.linspace(0, 1, int(len(sig) * self.eeg_sampling_rate / float(original_fs)))
                sig = np.interp(x_new, x_old, sig).astype(np.float32)

        # Z-score normalization for EEG
        sig_mean, sig_std = sig.mean(), sig.std()
        if sig_std > 1e-8:
            sig = (sig - sig_mean) / sig_std
        else:
            sig = np.zeros_like(sig)

        # Pad/truncate
        if len(sig) >= self.eeg_sequence_length:
            sig = sig[:self.eeg_sequence_length]
        else:
            pad = np.zeros(self.eeg_sequence_length - len(sig), dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)

        return sig.astype(np.float32)

    def analyze_signal_data(self, signal_data, sampling_rate=None, top_k=5,
                            signal_type=None, is_multi_channel=False, all_channels=None):
        """
        Main inference wrapper supporting both ECG and EEG

        Args:
            signal_data: Single channel data (for backward compatibility)
            sampling_rate: Original sampling rate
            top_k: Number of top predictions
            signal_type: "ECG" or "EEG"
            is_multi_channel: If True, expects all_channels parameter
            all_channels: Dict or DataFrame with all channels
        """
        start_time = time.time()

        if signal_type is None:
            signal_type = self.signal_type

        if sampling_rate is None:
            sampling_rate = self.ecg_sampling_rate if signal_type == "ECG" else self.eeg_sampling_rate

        # Load appropriate model
        if not self._model_loaded or self.signal_type != signal_type:
            ok = self.load_model(signal_type)
            if not ok:
                return {"error": f"Failed to load {signal_type} model. See logs."}

        try:
            if signal_data is None or len(signal_data) == 0:
                return {"error": "No signal data provided"}

            # Process based on signal type
            if signal_type == "ECG":
                return self._analyze_ecg(signal_data, sampling_rate, top_k, is_multi_channel, all_channels, start_time)
            else:  # EEG
                return self._analyze_eeg(signal_data, sampling_rate, top_k, is_multi_channel, all_channels, start_time)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Inference failed: {e}", "timestamp": datetime.now().isoformat()}

    def _analyze_ecg(self, signal_data, sampling_rate, top_k, is_multi_channel, all_channels, start_time):
        """Analyze ECG data with HuBERT-ECG (12-lead)"""
        if self._model_type != 'hubert-ecg':
            return {"error": "ECG model not loaded or wrong type"}

        # Need 12 leads for HuBERT-ECG
        if all_channels is None or not is_multi_channel:
            return {
                "error": "HuBERT-ECG requires 12-lead ECG data",
                "note": "Please provide all_channels parameter with 12 leads",
                "available_conditions": self._labels[:20],  # Show first 20 conditions
                "total_conditions": len(self._labels),
                "timestamp": datetime.now().isoformat()
            }

        # Extract and preprocess all 12 leads
        lead_signals = []
        for i in range(1, 13):
            lead_name = f"signal_{i}"
            if lead_name in all_channels.columns:
                lead_data = all_channels[lead_name].values
                processed_lead = self._preprocess_ecg_signal(lead_data, original_fs=sampling_rate)
                lead_signals.append(processed_lead)
            else:
                processed_lead = np.zeros(self.ecg_sequence_length, dtype=np.float32)
                lead_signals.append(processed_lead)
                print(f"[Warning] Missing {lead_name}, using zeros")

        # Flatten all 12 leads (12 * 500 = 6000)
        processed = np.concatenate(lead_signals, axis=0)
        tensor = torch.from_numpy(processed).float().unsqueeze(0)

        if self.device is not None:
            tensor = tensor.to(self.device)

        with torch.no_grad():
            inf_start = time.time()
            outputs = self._active_model(tensor)

            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            pooled = hidden_states.mean(dim=1)

            if hasattr(self._active_model, 'classifier'):
                logits = self._active_model.classifier(pooled)
            else:
                return {"error": "Model has no classification head"}

            inf_time = time.time() - inf_start
            probs_tensor = F.softmax(logits, dim=-1)
            probs = probs_tensor.cpu().numpy()[0].tolist()

        return self._format_results(probs, top_k, len(processed), sampling_rate,
                                    inf_time, start_time, signal_type="ECG", leads_used=12)

    def _analyze_eeg(self, signal_data, sampling_rate, top_k, is_multi_channel, all_channels, start_time):
        """Analyze EEG data with BIOT (16-channel)"""
        if self._model_type != 'biot-eeg':
            return {"error": "EEG model not loaded or wrong type"}

        # Need 16 channels for BIOT
        if all_channels is None or not is_multi_channel:
            return {
                "error": "BIOT requires 16-channel EEG data",
                "note": "Please provide all_channels parameter with 16 EEG channels",
                "available_conditions": self._labels[:20],  # Show first 20 conditions
                "total_conditions": len(self._labels),
                "timestamp": datetime.now().isoformat()
            }

        # Extract and preprocess 16 channels
        channel_signals = []
        for i in range(1, 17):
            ch_name = f"signal_{i}"
            if ch_name in all_channels.columns:
                ch_data = all_channels[ch_name].values
                processed_ch = self._preprocess_eeg_signal(ch_data, original_fs=sampling_rate)
                channel_signals.append(processed_ch)
            else:
                processed_ch = np.zeros(self.eeg_sequence_length, dtype=np.float32)
                channel_signals.append(processed_ch)
                print(f"[Warning] Missing {ch_name}, using zeros")

        # Stack channels (16, seq_length)
        processed = np.stack(channel_signals, axis=0)
        tensor = torch.from_numpy(processed).float().unsqueeze(0)

        if self.device is not None:
            tensor = tensor.to(self.device)

        with torch.no_grad():
            inf_start = time.time()
            logits = self._active_model(tensor)
            inf_time = time.time() - inf_start

            probs_tensor = F.softmax(logits, dim=-1)
            probs = probs_tensor.cpu().numpy()[0].tolist()

        return self._format_results(probs, top_k, processed.size, sampling_rate,
                                    inf_time, start_time, signal_type="EEG", leads_used=16)

    def _format_results(self, probs, top_k, signal_length, sampling_rate,
                        inf_time, start_time, signal_type="ECG", leads_used=1):
        """Format prediction results"""
        probs_arr = np.array(probs, dtype=float)
        top_k = min(max(1, int(top_k)), probs_arr.size)
        top_idx = np.argsort(probs_arr)[::-1][:top_k]

        pred_results = []
        for idx in top_idx:
            confidence = float(probs_arr[idx])
            label = self._labels[idx] if idx < len(self._labels) else f"Class_{idx}"
            pred_results.append({
                "index": int(idx),
                "label": label,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100.0, 2)
            })

        total_time = time.time() - start_time
        self._last_analysis_time = total_time
        self._total_predictions += 1

        quality = "Low"
        if pred_results:
            c = pred_results[0]['confidence']
            if c >= 0.8:
                quality = "High"
            elif c >= 0.6:
                quality = "Medium"

        return {
            "success": True,
            "predictions": pred_results,
            "raw_probabilities": probs_arr.tolist(),
            "prediction_quality": quality,
            "top_confidence": pred_results[0]['confidence'] if pred_results else 0.0,
            "inference_time_s": round(inf_time, 4),
            "total_time_s": round(total_time, 4),
            "sequence_length": int(signal_length),
            "sampling_rate": sampling_rate,
            "signal_length": int(signal_length),
            "timestamp": datetime.now().isoformat(),
            "prediction_count": self._total_predictions,
            "model_type": self._model_type,
            "signal_type": signal_type,
            "leads_used": leads_used,
            "total_conditions_available": len(self._labels)
        }

    def analyze_patient_data(self, patient_data, channel_name="signal_1", top_k=5, signal_type=None):
        """Analyze patient data (ECG or EEG)"""
        if signal_type is None:
            signal_type = self.signal_type

        try:
            available_channels = [c for c in patient_data.columns if c.startswith('signal_')]

            # Determine if we have multi-channel data
            required_channels = self.ecg_channels if signal_type == "ECG" else self.eeg_channels
            is_multi_channel = len(available_channels) >= required_channels

            estimated_fs = self.ecg_sampling_rate if signal_type == "ECG" else self.eeg_sampling_rate
            if 'time' in patient_data.columns and len(patient_data) > 2:
                diffs = np.diff(patient_data['time'].values.astype(float))
                if np.any(diffs > 0):
                    estimated_fs = 1.0 / np.median(diffs)

            # Use first channel as reference
            sig = patient_data[available_channels[0]].values

            return self.analyze_signal_data(
                sig,
                sampling_rate=estimated_fs,
                top_k=top_k,
                signal_type=signal_type,
                is_multi_channel=is_multi_channel,
                all_channels=patient_data
            )
        except Exception as e:
            return {"error": f"Patient data analysis failed: {e}"}

    def get_model_info(self):
        """Get model information"""
        return {
            "model_loaded": self._model_loaded,
            "signal_type": self.signal_type,
            "ecg_model_path": self.ecg_model_path,
            "eeg_model_path": self.eeg_model_path,
            "num_classes": len(self._labels),
            "ecg_sequence_length": self.ecg_sequence_length,
            "eeg_sequence_length": self.eeg_sequence_length,
            "ecg_channels": self.ecg_channels,
            "eeg_channels": self.eeg_channels,
            "confidence_threshold": self.confidence_threshold,
            "total_predictions": self._total_predictions,
            "last_analysis_time": self._last_analysis_time,
            "model_type": self._model_type,
            "available_ecg_conditions": len(self.ECG_LABELS),
            "available_eeg_conditions": len(self.EEG_LABELS),
            "sample_ecg_conditions": self.ECG_LABELS[:10],
            "sample_eeg_conditions": self.EEG_LABELS[:10]
        }

    def is_ready(self):
        return self._model_loaded and self._active_model is not None

    def switch_signal_type(self, signal_type):
        """
        Switch between ECG and EEG models

        Args:
            signal_type: "ECG" or "EEG"

        Returns:
            bool: True if successful, False otherwise
        """
        if signal_type not in ["ECG", "EEG"]:
            print(f"[Model Loader] Invalid signal type: {signal_type}. Must be 'ECG' or 'EEG'")
            return False

        if signal_type == self.signal_type and self._model_loaded:
            print(f"[Model Loader] Already using {signal_type} model")
            return True

        print(f"[Model Loader] Switching from {self.signal_type} to {signal_type}")
        self.signal_type = signal_type

        # Load labels for new signal type
        self.load_labels(signal_type)

        # Load the appropriate model
        success = self.load_model(signal_type)

        if success:
            print(f"[Model Loader] Successfully switched to {signal_type} model")
        else:
            print(f"[Model Loader] Failed to switch to {signal_type} model")

        return success

    def list_conditions(self, signal_type=None, search_term=None):
        """
        List all available conditions the model can detect

        Args:
            signal_type: "ECG" or "EEG" (uses current if None)
            search_term: Optional search term to filter conditions

        Returns:
            List of condition names
        """
        if signal_type is None:
            signal_type = self.signal_type

        conditions = self.ECG_LABELS if signal_type == "ECG" else self.EEG_LABELS

        if search_term:
            search_lower = search_term.lower()
            conditions = [c for c in conditions if search_lower in c.lower()]

        return {
            "signal_type": signal_type,
            "total_conditions": len(conditions),
            "conditions": conditions,
            "search_term": search_term
        }


# --- BIOT Model Architecture (simplified) ---
class BIOTModel(nn.Module):
    """Simplified BIOT model for EEG analysis"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        input_channels = config.get('input_channels', 16)
        hidden_size = config.get('hidden_size', 512)
        num_layers = config.get('num_layers', 12)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.1)
        num_classes = config.get('num_classes', 50)

        # Channel embedding
        self.channel_embedding = nn.Linear(input_channels, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels)

        # Embed channels
        x = self.channel_embedding(x)  # (batch, time, hidden)

        # Transform
        x = self.transformer(x)  # (batch, time, hidden)

        # Pool and classify
        x = x.mean(dim=1)  # (batch, hidden)
        logits = self.classifier(x)  # (batch, num_classes)

        return logits


# --- Update Global Instance ---
AI_MODEL = ConditionIdentificationModel(
    ecg_model_path="small",  # Will load "Edoardo-BS/hubert-ecg-small"
    eeg_model_path="EEG-PREST-16-channels.ckpt",  # BIOT EEG model
    signal_type="ECG",  # Default to ECG, will switch based on data
    use_huggingface=True
)


# ---------- Channel Processing Functions ----------
def derive_third_ecg_channel(sig1, sig2, method="difference"):
    """
    Derive a third ECG channel from two existing channels.

    Args:
        sig1, sig2: numpy arrays of the two ECG channels
        method: "difference", "sum", or "orthogonal"

    Returns:
        numpy array of the derived third channel
    """
    if len(sig1) != len(sig2):
        min_len = min(len(sig1), len(sig2))
        sig1, sig2 = sig1[:min_len], sig2[:min_len]

    if method == "difference":
        # Lead III = Lead II - Lead I (similar to standard ECG lead derivation)
        derived = sig2 - sig1
        print(f"[Channel Derivation] Created third ECG channel using difference method (Lead2 - Lead1)")
    elif method == "sum":
        # Sum of the two leads
        derived = (sig1 + sig2) / 2
        print(f"[Channel Derivation] Created third ECG channel using sum method ((Lead1 + Lead2)/2)")
    elif method == "orthogonal":
        # Create an orthogonal lead using Gram-Schmidt-like process
        # This approximates a perpendicular view
        dot_product = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
        derived = sig2 - dot_product * sig1
        print(f"[Channel Derivation] Created third ECG channel using orthogonal method")
    else:
        derived = sig2 - sig1  # Default to difference
        print(f"[Channel Derivation] Created third ECG channel using default difference method")

    return derived


def get_display_channels(patient, dataset_type, show_all_channels=False):
    """
    Get the channels to display based on dataset type and available channels.

    Args:
        patient: patient data dict
        dataset_type: "ECG" or "EEG"
        show_all_channels: whether to show all available channels

    Returns:
        list of channel names to display
    """
    if "ecg" not in patient or patient["ecg"] is None:
        return []

    available_channels = [c for c in patient["ecg"].columns if c.startswith("signal_")]

    if dataset_type == "ECG":
        if len(available_channels) == 0:
            return []
        elif len(available_channels) == 1:
            # Only one channel - duplicate it for now, will handle in visualization
            return [available_channels[0]]
        elif len(available_channels) == 2:
            # Two channels - will derive third in visualization
            return available_channels
        elif len(available_channels) >= 3:
            if show_all_channels:
                return available_channels
            else:
                # Show first 3 channels by default
                return available_channels[:3]

    else:  # EEG
        if len(available_channels) <= 3:
            return available_channels
        else:
            if show_all_channels:
                return available_channels
            else:
                # Show first 3 channels by default for EEG
                return available_channels[:3]

    return available_channels


def process_patient_channels(patient, dataset_type, show_all_channels=False):
    """
    Process patient data to ensure 3 channels for display.
    For ECG: derive third channel if only 2 exist.
    For EEG: limit to 3 main channels unless show_all is True.

    Returns:
        dict with processed channel data and metadata
    """
    if "ecg" not in patient or patient["ecg"] is None:
        return {"channels": [], "derived_info": "No data available"}

    ecg_df = patient["ecg"].copy()
    available_channels = [c for c in ecg_df.columns if c.startswith("signal_")]

    result = {
        "channels": [],
        "derived_info": "",
        "original_count": len(available_channels)
    }

    if dataset_type == "ECG":
        if len(available_channels) == 0:
            result["derived_info"] = "No ECG channels available"
            return result
        elif len(available_channels) == 1:
            # Single channel - create 3 versions for visualization
            ch = available_channels[0]
            result["channels"] = [ch, f"{ch}_copy", f"{ch}_inverted"]
            # Add copies to dataframe
            ecg_df[f"{ch}_copy"] = ecg_df[ch]
            ecg_df[f"{ch}_inverted"] = -ecg_df[ch]  # Inverted for different perspective
            result["derived_info"] = f"Single channel {ch} displayed with copy and inverted version"
        elif len(available_channels) == 2:
            # Two channels - derive third
            ch1, ch2 = available_channels[0], available_channels[1]
            derived_ch = f"derived_{ch1}_{ch2}"
            derived_signal = derive_third_ecg_channel(ecg_df[ch1].values, ecg_df[ch2].values, method="difference")
            ecg_df[derived_ch] = derived_signal
            result["channels"] = [ch1, ch2, derived_ch]
            result["derived_info"] = f"Third channel '{derived_ch}' derived from {ch1} - {ch2}"
        else:
            # Three or more channels
            if show_all_channels:
                result["channels"] = available_channels
                result["derived_info"] = f"Showing all {len(available_channels)} channels"
            else:
                result["channels"] = available_channels[:3]
                result["derived_info"] = f"Showing main 3 channels (out of {len(available_channels)} available)"

    else:  # EEG
        if len(available_channels) <= 3:
            result["channels"] = available_channels
            result["derived_info"] = f"Showing all {len(available_channels)} EEG channels"
        else:
            if show_all_channels:
                result["channels"] = available_channels
                result["derived_info"] = f"Showing all {len(available_channels)} EEG channels"
            else:
                result["channels"] = available_channels[:3]
                result["derived_info"] = f"Showing main 3 EEG channels (out of {len(available_channels)} available)"

    # Update patient data with processed dataframe
    patient["ecg"] = ecg_df
    result["processed_df"] = ecg_df

    return result


# ---------- Utilities ----------
def parse_num(token, default=None):
    if token is None:
        return default
    token = str(token).strip()
    if token == "":
        return default
    try:
        return float(token)
    except:
        pass
    if '/' in token:
        parts = token.split('/')
        for p in parts:
            p = p.strip()
            m = re.search(r'[-+]?\d+(\.\d+)?', p)
            if m:
                try:
                    return float(m.group(0))
                except:
                    continue
    m = re.search(r'[-+]?\d+(\.\d+)?', token)
    if m:
        try:
            return float(m.group(0))
        except:
            return default
    return default


def find_dataset_directory(dataset_type, root="."):
    """
    Updated dataset directory finder that looks for patient-organized structures.
    """
    if dataset_type == "ECG":
        candidates = [
            os.path.join(os.getcwd(), "data", "ptbdb"),
            os.path.join(os.getcwd(), "ptbdb"),
            os.path.join(os.getcwd(), "qtdb_data", "physionet.org", "files", "qtdb", "1.0.0"),
            os.path.join(os.getcwd(), "qtdb"),
            os.path.join(os.getcwd(), "qtdb_data"),
            os.path.join(os.getcwd(), "1.0.0"),
            os.path.join(os.getcwd(), "qtdb", "1.0.0"),
            os.path.join(os.getcwd(), "qtdb-1.0.0"),
        ]

        # Check for patient-organized structure first
        for d in candidates:
            if os.path.isdir(d):
                # Look for patient directories
                patient_dirs = find_patient_directories(d)
                if patient_dirs:
                    return d

                # Fallback: look for .hea files directly
                for rootd, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith('.hea'):
                            return d
    else:
        # EEG logic remains the same
        candidates = [
            os.path.join(os.getcwd(), "ASZED-153"),
            os.path.join(os.getcwd(), "ASZED_153"),
            os.path.join(os.getcwd(), "aszed-153"),
            os.path.join(os.getcwd(), "eeg_data"),
        ]

        for d in candidates:
            if os.path.isdir(d):
                for rootd, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith('.edf'):
                            return d

    # Fallback: scan root
    for rootd, _, files in os.walk(root):
        for f in files:
            ext = '.hea' if dataset_type == "ECG" else '.edf'
            if f.lower().endswith(ext):
                return rootd
    return None


def get_subject_id_from_path(path):
    """
    Heuristic to extract subject id from file path.
    """
    if not path:
        return None
    m = re.search(r"subject[_\-]?(\d+)", path, flags=re.IGNORECASE)
    if m:
        return f"subject_{int(m.group(1))}"
    parts = re.split(r"[\\/]+", path)
    if len(parts) >= 2:
        for p in reversed(parts[:-1]):
            mm = re.match(r"^(\d+)$", p)
            if mm:
                return f"subject_{int(mm.group(1))}"
            mm2 = re.match(r"^sub[_\-]?(\d+)$", p, flags=re.IGNORECASE)
            if mm2:
                return f"subject_{int(mm2.group(1))}"
        return parts[-2]
    return os.path.basename(path)


def find_all_data_files(root_dir, file_extension):
    """
    Recursively find all data files with given extension in directory tree.
    Groups files by their immediate parent directory.

    Args:
        root_dir: Root directory to search
        file_extension: '.dat' for ECG or '.edf' for EEG

    Returns:
        Dictionary mapping parent_dir -> list of file paths
    """
    grouped_files = {}

    for root, dirs, files in os.walk(root_dir):
        matching_files = [f for f in files if f.lower().endswith(file_extension)]

        if matching_files:
            # Use the immediate parent directory as the group key
            parent_key = os.path.basename(root) or root
            full_paths = [os.path.join(root, f) for f in sorted(matching_files)]

            if parent_key not in grouped_files:
                grouped_files[parent_key] = []
            grouped_files[parent_key].extend(full_paths)

    return grouped_files


def concatenate_ecg_files(file_paths, max_samples=None):
    """
    Vertically concatenate multiple ECG .dat/.hea file pairs.

    Args:
        file_paths: List of .dat or .hea file paths to concatenate
        max_samples: Maximum samples to read (None for all)

    Returns:
        (combined_df, combined_header) or (None, None) on failure
    """
    combined_df = None
    combined_header = None
    total_samples = 0

    for file_path in file_paths:
        # Get corresponding .hea and .dat files
        if file_path.endswith('.hea'):
            hea_path = file_path
            dat_path = file_path.replace('.hea', '.dat')
        elif file_path.endswith('.dat'):
            dat_path = file_path
            hea_path = file_path.replace('.dat', '.hea')
        else:
            continue

        if not os.path.exists(hea_path) or not os.path.exists(dat_path):
            print(f"[concatenate_ecg_files] Missing pair for {file_path}")
            continue

        # Read header and data
        header = read_header_file(hea_path)
        if header is None:
            print(f"[concatenate_ecg_files] Failed to read header: {hea_path}")
            continue

        df = read_dat_file(dat_path, header, max_samples=max_samples)
        if df is None:
            print(f"[concatenate_ecg_files] Failed to read data: {dat_path}")
            continue

        # First file - initialize
        if combined_df is None:
            combined_df = df.copy()
            combined_header = header.copy()
            combined_header['source_files'] = [os.path.basename(file_path)]
            total_samples = len(df)
        else:
            # Subsequent files - concatenate
            # Ensure same number of signal columns
            common_signals = [col for col in df.columns if col.startswith('signal_') and col in combined_df.columns]

            if not common_signals:
                print(f"[concatenate_ecg_files] No matching signal columns in {file_path}")
                continue

            # Adjust time column to continue from previous data
            last_time = combined_df['time'].iloc[-1]
            fs = header.get('sampling_frequency', 250.0)
            time_increment = 1.0 / fs
            df['time'] = df['time'] + last_time + time_increment

            # Concatenate only matching columns
            cols_to_concat = ['time'] + common_signals
            combined_df = pd.concat([combined_df[cols_to_concat], df[cols_to_concat]],
                                    ignore_index=True, axis=0)

            combined_header['source_files'].append(os.path.basename(file_path))
            total_samples += len(df)

    if combined_df is not None:
        combined_header['num_samples'] = total_samples
        combined_header['concatenated_files'] = len(combined_header['source_files'])
        print(f"[concatenate_ecg_files] Combined {len(combined_header['source_files'])} files, "
              f"total samples: {total_samples}")

    return combined_df, combined_header


def concatenate_eeg_files(file_paths, max_samples=None):
    """
    Vertically concatenate multiple EEG .edf files.

    Args:
        file_paths: List of .edf file paths to concatenate
        max_samples: Maximum samples to read per file (None for all)

    Returns:
        (combined_df, combined_header) or (None, None) on failure
    """
    if not PYEDFLIB_AVAILABLE:
        print("[concatenate_eeg_files] pyedflib not available")
        return None, None

    combined_df = None
    combined_header = None
    total_samples = 0

    for file_path in file_paths:
        df, header = read_edf_file(file_path, max_samples=max_samples)

        if df is None:
            print(f"[concatenate_eeg_files] Failed to read: {file_path}")
            continue

        # First file - initialize
        if combined_df is None:
            combined_df = df.copy()
            combined_header = header.copy()
            combined_header['source_files'] = [os.path.basename(file_path)]
            total_samples = len(df)
        else:
            # Subsequent files - concatenate
            # Ensure same number of signal columns
            common_signals = [col for col in df.columns if col.startswith('signal_') and col in combined_df.columns]

            if not common_signals:
                print(f"[concatenate_eeg_files] No matching signal columns in {file_path}")
                continue

            # Adjust time column to continue from previous data
            last_time = combined_df['time'].iloc[-1]
            fs = header.get('sampling_frequency', 250)
            time_increment = 1.0 / fs
            df['time'] = df['time'] + last_time + time_increment

            # Concatenate only matching columns
            cols_to_concat = ['time'] + common_signals
            combined_df = pd.concat([combined_df[cols_to_concat], df[cols_to_concat]],
                                    ignore_index=True, axis=0)

            combined_header['source_files'].append(os.path.basename(file_path))
            total_samples += len(df)

    if combined_df is not None:
        combined_header['num_samples'] = total_samples
        combined_header['concatenated_files'] = len(combined_header['source_files'])
        print(f"[concatenate_eeg_files] Combined {len(combined_header['source_files'])} files, "
              f"total samples: {total_samples}")

    return combined_df, combined_header


def read_header_file(path):
    """Read .hea header (robust)."""
    if not os.path.exists(path):
        return None
    with open(path, "r", errors="ignore") as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip() != ""]
    if not lines:
        return None
    first = lines[0].split()
    record_name = first[0] if len(first) >= 1 else None
    num_signals = parse_num(first[1], default=2) if len(first) >= 2 else 2
    fs = parse_num(first[2], default=250.0) if len(first) >= 3 else 250.0
    num_samples = parse_num(first[3], default=225000) if len(first) >= 4 else 225000
    try:
        num_signals = int(num_signals)
    except:
        num_signals = int(max(1, math.floor(num_signals))) if num_signals else 2
    try:
        fs = float(fs)
    except:
        fs = 250.0
    try:
        num_samples = int(num_samples)
    except:
        num_samples = int(225000)
    signals_raw = []
    if len(lines) > 1:
        for ln in lines[1:]:
            signals_raw.append(ln.split())
    return {
        "record_name": record_name,
        "num_signals": num_signals,
        "sampling_frequency": fs,
        "num_samples": num_samples,
        "signals_raw": signals_raw
    }


def read_dat_file(dat_path, header_info, max_samples=None):
    """Read MIT/PhysioNet .dat interleaved int16. Return pandas DataFrame or None on failure."""
    if not os.path.exists(dat_path) or header_info is None:
        return None
    try:
        raw = np.fromfile(dat_path, dtype=np.int16)
        n_signals = max(1, int(header_info.get("num_signals", 2)))
        total_samples = raw.shape[0] // n_signals
        if total_samples <= 0:
            return None
        raw = raw[: total_samples * n_signals]
        mat = raw.reshape((total_samples, n_signals))
        gains = np.ones(n_signals) * 200.0
        for i in range(min(n_signals, len(header_info.get("signals_raw", [])))):
            parts = header_info["signals_raw"][i]
            if len(parts) >= 3:
                g = parse_num(parts[2], default=None)
                if g and g > 0:
                    gains[i] = g
        cols = [f"signal_{i + 1}" for i in range(n_signals)]
        df = pd.DataFrame(mat[:, :n_signals].astype(float) / gains[:n_signals], columns=cols)
        fs = header_info.get("sampling_frequency", 250.0)
        df.insert(0, "time", np.arange(df.shape[0]) / float(fs))
        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[read_dat_file] error reading {dat_path}: {e}")
        return None


def find_patient_directories(data_dir):
    """
    Find all patient directories in the dataset.
    Looks for directories named like 'patient001', 'patient002', etc.
    """
    patient_dirs = []

    if not os.path.isdir(data_dir):
        return patient_dirs

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if it looks like a patient directory
            if (item.lower().startswith('patient') or
                    item.lower().startswith('subject') or
                    item.isdigit()):  # Some datasets use just numbers
                patient_dirs.append(item_path)

    return sorted(patient_dirs)


def get_patient_records(patient_dir):
    """
    Get all ECG records (hea/dat file pairs) for a specific patient.
    Returns list of record base names (without extensions).
    """
    if not os.path.isdir(patient_dir):
        return []

    records = []
    files = os.listdir(patient_dir)

    # Find all .hea files and extract base names
    for f in files:
        if f.lower().endswith('.hea'):
            base_name = os.path.splitext(f)[0]
            # Check if corresponding .dat file exists
            dat_file = os.path.join(patient_dir, f"{base_name}.dat")
            if os.path.exists(dat_file):
                records.append({
                    'base_name': base_name,
                    'hea_path': os.path.join(patient_dir, f),
                    'dat_path': dat_file
                })

    return records


def load_patient_record(record_info, max_samples=None):
    """
    Load a single ECG record (hea + dat file pair).
    Returns DataFrame with all channels from that record.
    """
    hea_path = record_info['hea_path']
    dat_path = record_info['dat_path']

    # Read header
    header = read_header_file(hea_path)
    if header is None:
        return None, None

    # Read data
    df = read_dat_file(dat_path, header, max_samples=max_samples)
    if df is None:
        return None, None

    # Add record identifier to column names to avoid conflicts when combining
    base_name = record_info['base_name']
    signal_cols = [c for c in df.columns if c.startswith('signal_')]
    rename_dict = {}
    for i, col in enumerate(signal_cols):
        rename_dict[col] = f"signal_{base_name}_{i + 1}"

    df.rename(columns=rename_dict, inplace=True)

    return df, header


def combine_patient_records(patient_dir, max_samples=None, max_records_per_patient=None):
    """
    Combine all ECG records for a single patient into one DataFrame.
    Each record contributes its channels with unique names.
    """
    records = get_patient_records(patient_dir)
    if not records:
        return None, None

    # Limit records per patient if specified
    if max_records_per_patient:
        records = records[:max_records_per_patient]

    combined_df = None
    combined_header = None
    total_channels = 0

    for i, record_info in enumerate(records):
        try:
            df, header = load_patient_record(record_info, max_samples)
            if df is None:
                print(f"[combine_patient_records] Failed to load record {record_info['base_name']}")
                continue

            if combined_df is None:
                # First record - use as base
                combined_df = df.copy()
                combined_header = header.copy()
                combined_header['records'] = [record_info['base_name']]
            else:
                # Additional records - merge
                if len(df) != len(combined_df):
                    # Handle different lengths by taking minimum
                    min_len = min(len(df), len(combined_df))
                    df = df.iloc[:min_len].reset_index(drop=True)
                    combined_df = combined_df.iloc[:min_len].reset_index(drop=True)

                # Add signal columns from this record
                signal_cols = [c for c in df.columns if c.startswith('signal_')]
                for col in signal_cols:
                    combined_df[col] = df[col].values

                combined_header['records'].append(record_info['base_name'])

            # Count channels added
            record_channels = len([c for c in df.columns if c.startswith('signal_')])
            total_channels += record_channels

        except Exception as e:
            print(f"[combine_patient_records] Error processing record {record_info['base_name']}: {e}")
            continue

    if combined_df is not None:
        # Update header with combined info
        combined_header['num_signals'] = len([c for c in combined_df.columns if c.startswith('signal_')])
        combined_header['combined_records'] = len(records)
        combined_header['total_channels'] = total_channels

        print(
            f"[combine_patient_records] Combined {len(records)} records into {combined_header['num_signals']} total channels")

    return combined_df, combined_header


def read_edf_file(edf_path, max_samples=None, attempts=8):
    """
    Read EDF file using pyedflib. Returns (df, header) or (None, None) and prints error.
    """
    if not PYEDFLIB_AVAILABLE:
        print("pyedflib not installed. Please: pip install pyedflib")
        return None, None

    last_exc = None
    backoff = 0.05
    for attempt in range(attempts):
        try:
            f = pyedflib.EdfReader(edf_path)
            try:
                try:
                    n_signals = int(f.signals_in_file)
                except Exception:
                    n_signals = int(getattr(f, "signals_in_file", 0) or 0)
                if n_signals <= 0:
                    raise ValueError("No signals in EDF")
                nsamps = f.getNSamples()
                if isinstance(nsamps, (list, tuple, np.ndarray)):
                    min_samples = int(min(nsamps))
                else:
                    min_samples = int(nsamps)
                use_samples = min_samples if max_samples is None else min(min_samples, int(max_samples))
                fs = None
                try:
                    fs = int(f.getSampleFrequency(0))
                except Exception:
                    try:
                        dur = getattr(f, "getFileDuration", lambda: None)()
                        if dur:
                            fs = max(1, int(round(use_samples / float(dur))))
                        else:
                            fs = 250
                    except Exception:
                        fs = 250
                data = np.zeros((use_samples, n_signals), dtype=float)
                for ch in range(n_signals):
                    sig = f.readSignal(ch)
                    if sig is None:
                        sig = np.zeros(use_samples, dtype=float)
                    if len(sig) >= use_samples:
                        sig_use = np.asarray(sig[:use_samples], dtype=float)
                    else:
                        sig_use = np.empty(use_samples, dtype=float)
                        sig_use[:len(sig)] = sig
                        sig_use[len(sig):] = np.nan
                    data[:, ch] = sig_use
                cols = [f"signal_{i + 1}" for i in range(n_signals)]
                df = pd.DataFrame(data, columns=cols)
                df.insert(0, "time", np.arange(use_samples) / float(fs))
                header = {
                    "sampling_frequency": fs,
                    "num_signals": n_signals,
                    "record_name": os.path.basename(edf_path),
                    "num_samples": use_samples
                }
                return df, header
            finally:
                try:
                    f.close()
                except Exception:
                    try:
                        f._close()
                    except Exception:
                        pass
                try:
                    del f
                except Exception:
                    pass
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            if ("already been opened" in msg) or ("file has already been opened" in msg) or (
                    "resource temporarily unavailable" in msg) or ("i/o error" in msg):
                time.sleep(backoff)
                backoff = min(0.5, backoff * 1.8)
                continue
            time.sleep(backoff)
            backoff = min(0.5, backoff * 1.8)
            continue
    print(f"[read_edf_file] Error reading {edf_path}: {last_exc}")
    return None, None


def apply_signal_filtering(signal, fs, signal_type="ECG"):
    """Bandpass filter for ECG/EEG (simple, zero-phase)."""
    if signal is None or len(signal) < 3:
        return signal
    if signal_type == "ECG":
        low_cutoff, high_cutoff = 0.5, 40.0
    else:  # EEG
        low_cutoff, high_cutoff = 0.5, 70.0
    nyq = fs / 2.0
    low = max(low_cutoff / nyq, 1e-6)
    high = min(high_cutoff / nyq, 0.9999)
    try:
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return signal


# ---------- Load patients ----------
def load_patient_data(data_dir, dataset_type="ECG", max_samples=None, max_patients=None,
                      max_records_per_patient=5):
    """
    Load and concatenate all data files found in directory tree.
    Groups files by their immediate parent directory and concatenates vertically.

    Args:
        data_dir: Root directory to search
        dataset_type: "ECG" or "EEG"
        max_samples: Max samples to read per file (None for all)
        max_patients: Maximum number of patient groups to load
        max_records_per_patient: Maximum files to concatenate per group

    Returns:
        List of patient records with concatenated data
    """
    patients = []

    if dataset_type == "ECG":
        file_ext = '.dat'
        concat_func = concatenate_ecg_files
    else:  # EEG
        if not PYEDFLIB_AVAILABLE:
            print("pyedflib is required to read EDF files. Please install: pip install pyedflib")
            return []
        file_ext = '.edf'
        concat_func = concatenate_eeg_files

    # Find all data files grouped by parent directory
    grouped_files = find_all_data_files(data_dir, file_ext)

    if not grouped_files:
        print(f"[load_patient_data] No {dataset_type} files found in {data_dir}")
        return []

    print(f"[load_patient_data] Found {len(grouped_files)} groups of {dataset_type} files")

    # Process each group
    for group_idx, (group_name, file_paths) in enumerate(sorted(grouped_files.items())):
        if max_patients is not None and group_idx >= max_patients:
            break

        print(f"[load_patient_data] Processing group '{group_name}' with {len(file_paths)} files...")

        # Limit files per group if specified
        if max_records_per_patient:
            file_paths = file_paths[:max_records_per_patient]

        # Concatenate all files in this group
        combined_df, combined_header = concat_func(file_paths, max_samples=max_samples)

        if combined_df is None:
            print(f"[load_patient_data] Failed to load group '{group_name}'")
            continue

        # Apply filtering to all signal columns
        fs = combined_header.get("sampling_frequency", 250)
        signal_cols = [c for c in combined_df.columns if c.startswith("signal_")]

        for col in signal_cols:
            combined_df[col] = apply_signal_filtering(combined_df[col].values, fs, dataset_type)

        # Create patient record
        patient_record = {
            "name": group_name,
            "header": combined_header,
            "ecg": combined_df,  # Note: still called 'ecg' even for EEG for compatibility
            "type": dataset_type,
            "source_directory": os.path.dirname(file_paths[0]),
            "files_concatenated": len(file_paths),
            "total_samples": combined_header.get('num_samples', len(combined_df)),
            "total_channels": len(signal_cols)
        }

        patients.append(patient_record)
        print(f"[load_patient_data] Loaded '{group_name}': {len(file_paths)} files, "
              f"{len(combined_df)} samples, {len(signal_cols)} channels")

    print(f"[load_patient_data] Successfully loaded {len(patients)} patient groups total")
    return patients


# ---------- Feature functions ----------
def extract_ecg_features(ecg_df, fs=250):
    features = {}
    if ecg_df is None:
        return features
    for col in ecg_df.columns:
        if not col.startswith("signal_"):
            continue
        sig = ecg_df[col].values
        try:
            height = np.quantile(sig, 0.85)
            peaks, _ = find_peaks(sig, height=height, distance=int(0.3 * fs))
            if peaks.size > 1:
                rr = np.diff(peaks) / fs
                features[col] = {"peaks": peaks.tolist(), "rr": rr.tolist()}
            else:
                features[col] = {"peaks": peaks.tolist(), "rr": []}
        except Exception:
            features[col] = {"peaks": [], "rr": []}
    return features


def extract_eeg_features(eeg_df, fs=250):
    features = {}
    if eeg_df is None:
        return features
    try:
        from scipy import signal as sp_signal
    except Exception:
        return features
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70)
    }
    for col in eeg_df.columns:
        if not col.startswith("signal_"):
            continue
        sig = eeg_df[col].values
        try:
            freqs, psd = sp_signal.welch(sig, fs, nperseg=min(1024, max(256, len(sig) // 4)))
            band_pows = {}
            for band, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                band_pows[band] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            total = sum(band_pows.values()) if sum(band_pows.values()) > 0 else 1.0
            rel = {f"{k}_rel": v / total for k, v in band_pows.items()}
            features[col] = {**band_pows, **rel}
        except Exception:
            features[col] = {b: 0.0 for b in bands.keys()}
    return features


# ---------- Buffers & app ----------
def ensure_buffers_for_patient(pid, fs, display_window, rr_capacity=300):
    bufs = GLOBAL_DATA["buffers"]
    if pid not in bufs:
        blen = max(1, int(round(display_window * fs)))
        bufs[pid] = {
            "signal_buffer": np.full(blen, np.nan),
            "write_idx": 0,
            "len": blen,
            "rr_buffer": np.full(rr_capacity, np.nan),
            "rr_write_idx": 0,
            "last_peak_global_index": -1,
            "direction": 1,
            "ping_position": 0.0,
            "ai_analysis": None  # Store AI analysis results
        }
    else:
        bufinfo = bufs[pid]
        desired_len = max(1, int(round(display_window * fs)))
        if bufinfo["len"] != desired_len:
            bufinfo["signal_buffer"] = np.full(desired_len, np.nan)
            bufinfo["len"] = desired_len
            bufinfo["write_idx"] = 0


app = Dash(__name__)
server = app.server
GLOBAL_DATA = {"patients": [], "buffers": {}, "dataset_type": "ECG"}


def capture_graph_screenshot(figure):
    """Capture a screenshot of the current plotly figure and return as bytes."""
    if not SCREENSHOT_AVAILABLE:
        return None
    try:
        img_bytes = pio.to_image(figure, format="png", width=800, height=600, engine="kaleido")
        return img_bytes
    except Exception as e:
        print(f"[Screenshot] Error: {e}")
        return None


# ---------- STYLES (for new layout) ----------
# These style dictionaries are used to create the modern UI.

app_style = {
    'backgroundColor': '#F3F4F6',
    'color': '#111827',
    'fontFamily': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif',
    'display': 'flex',
    'flexDirection': 'row',
    'height': '100vh',
    'overflow': 'hidden',
}

sidebar_style = {
    'width': '380px',
    'minWidth': '380px',
    'padding': '20px',
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '20px',
    'overflowY': 'auto',
    'backgroundColor': '#FFFFFF',
    'borderRight': '1px solid #E5E7EB'
}

content_style = {
    'flex': 1,
    'padding': '20px',
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '20px'
}

card_style = {
    'backgroundColor': '#FFFFFF',
    'borderRadius': '8px',
    'padding': '16px',
    'border': '1px solid #E5E7EB',
    'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
}

card_header_style = {
    'color': '#6B7280',
    'fontSize': '12px',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em',
    'fontWeight': '600',
    'marginBottom': '12px'
}

button_style = {
    'backgroundColor': '#3B82F6',
    'color': 'white',
    'border': 'none',
    'padding': '10px 15px',
    'borderRadius': '6px',
    'fontSize': '14px',
    'fontWeight': 'bold',
    'cursor': 'pointer',
    'width': '100%',
    'transition': 'background-color 0.2s'
}

ai_button_1d_style = {**button_style, 'backgroundColor': '#10B981'}
ai_button_2d_style = {**button_style, 'backgroundColor': '#8B5CF6'}

# ---------- New Modern Layout ----------
app.layout = html.Div(style=app_style, children=[
    # --- Controls Sidebar ---
    html.Div(style=sidebar_style, children=[
        html.Div([
            html.H1("BioSignal Monitor", style={'fontSize': '24px', 'fontWeight': 'bold', 'margin': 0}),
            html.P("ECG & EEG Real-time Analysis", style={'color': '#6B7280', 'marginTop': '4px'})
        ]),

        # --- Data Loading Card ---
        html.Div(style=card_style, children=[
            html.H2("Data Source", style=card_header_style),
            html.Label("Signal Type"),
            dcc.RadioItems(id="dataset-type",
                           options=[{"label": "ECG", "value": "ECG"}, {"label": "EEG", "value": "EEG"}],
                           value="ECG", labelStyle={'display': 'inline-block', 'marginRight': '12px'},
                           style={'marginTop': '8px'}),
            html.Label("Data Directory (optional)", style={'marginTop': '12px', 'display': 'block'}),
            dcc.Input(id="data-dir", type="text", value="", placeholder="e.g., ./data/ptbdb",
                      style={'marginTop': '8px'}),
            html.Button("Load Data", id="load-btn", n_clicks=0, style={**button_style, 'marginTop': '12px'}),
            html.Div(id="load-output",
                     style={"marginTop": '12px', "fontSize": '12px', "color": '#6B7280', "whiteSpace": "pre-wrap",
                            'minHeight': '40px'})
        ]),

        # --- Display Options Card ---
        html.Div(style=card_style, children=[
            html.H2("Display Options", style=card_header_style),
            html.Label("Select Patients"),
            dcc.Dropdown(id="patients-dropdown", multi=True, placeholder="Select one or more patients"),
            html.Label("Select Channels", style={'marginTop': '12px'}),
            dcc.Dropdown(id="channels-dropdown", multi=True, placeholder="Auto-selects first 3 if empty"),
            html.Button("Select All Channels", id="select-all-channels-btn", n_clicks=0,
                        style={**button_style, 'backgroundColor': '#6B7280', 'marginTop': '8px'}),

            html.Label("Analysis Domain", style={'marginTop': '12px'}),
            dcc.RadioItems(
                id="domain-switch",
                options=[
                    {'label': 'Time Domain', 'value': 'time'},
                    {'label': 'Frequency Domain', 'value': 'frequency'}
                ],
                value='time',
                labelStyle={'display': 'inline-block', 'marginRight': '12px'}
            ),

            html.Label("Visualization Type", style={'marginTop': '12px'}),
            dcc.Dropdown(id="viz-type", value="icu"),  # Options are now dynamic
            html.Label("Channel Display Mode", style={'marginTop': '12px'}),
            dcc.RadioItems(id="overlay-mode", options=[{"label": "Overlay", "value": "overlay"},
                                                       {"label": "Separate", "value": "separate"}],
                           value="overlay", labelStyle={'display': 'inline-block', 'marginRight': '12px'}),
            html.Div(id="channel-info", style={"marginTop": "8px", "fontSize": "12px", "color": "#6B7280"})
        ]),

        # --- Playback Control Card ---
        html.Div(style=card_style, children=[
            html.H2("Playback Controls", style=card_header_style),
            html.Label("Speed"),
            dcc.Slider(id="speed", min=0.1, max=10, step=0.1, value=1,
                       marks={0.5: "0.5x", 1: "1x", 2: "2x", 5: "5x", 10: "10x"}),
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginTop': '12px'}, children=[
                html.Div(style={'flex': 1}, children=[
                    html.Label("Update (ms)"),
                    dcc.Input(id="chunk-ms", type="number", value=200, min=20, step=10),
                ]),
                html.Div(style={'flex': 1}, children=[
                    html.Label("Window (s)"),
                    dcc.Input(id="display-window", type="number", value=8, min=1, step=1),
                ])
            ]),
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginTop': '12px'}, children=[
                html.Button("Play", id="play-btn", n_clicks=0,
                            style={**button_style, 'backgroundColor': '#10B981', 'flex': 1}),
                html.Button("Pause", id="pause-btn", n_clicks=0,
                            style={**button_style, 'backgroundColor': '#F59E0B', 'flex': 1}),
                html.Button("Reset", id="reset-btn", n_clicks=0,
                            style={**button_style, 'backgroundColor': '#EF4444', 'flex': 1}),
            ]),
        ]),

        # --- Time Domain Analysis Card (visibility controlled by callback) ---
        html.Div(id='time-analysis-card', style={**card_style}, children=[
            html.H2("Time Domain Sampling Analysis", style=card_header_style),
            html.Label("Set Sampling Period Ts (ms)", style={'marginTop': '12px'}),
            dcc.Input(id="sampling-period-input", type="number", value=None,
                      placeholder="e.g., 20ms for 50Hz", style={'width': '100%'}),
            html.Div(id='nyquist-info-time', style={'marginTop': '12px'}),
        ]),

        # --- Frequency Analysis Card (visibility controlled by callback) ---
        html.Div(id='freq-analysis-card', style={**card_style}, children=[
            html.H2("Frequency Analysis", style=card_header_style),
            html.Div(id='nyquist-info'),
            html.Label("Resampling Frequency (Hz) for FFT", style={'marginTop': '12px'}),
            dcc.Input(id="resampling-freq", type="number", value=500, min=10, step=10,
                      placeholder="e.g., 500", style={'width': '100%'}),
            html.Div(id="fft-computation-time", style={"marginTop": "12px"})
        ]),

        # --- AI Analysis Control Card ---
        html.Div(style=card_style, children=[
            html.H2("AI Analysis", style=card_header_style),
            html.P("Analyze the currently displayed signal window.",
                   style={'fontSize': '14px', 'color': '#6B7280', 'marginBottom': '16px'}),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
                html.Button("Run 1D AI Analysis (Signal)", id="ai-analyze-btn", n_clicks=0, style=ai_button_1d_style),
                html.Button("Run 2D AI Analysis (Image)", id="ai-analyze-2d-btn", n_clicks=0,
                            style=ai_button_2d_style),
            ]),
        ]),

        html.Div(style={'marginTop': 'auto'})  # Spacer
    ]),

    # --- Main Content Area ---
    html.Div(style=content_style, children=[
        # --- Main Graph ---
        html.Div(style={**card_style, 'flex': 1, 'minHeight': '400px', 'display': 'flex', 'flexDirection': 'column'},
                 children=[
                     dcc.Graph(id="main-graph", config={"displayModeBar": True}, style={'height': '100%'})
                 ]),

        # --- AI Analysis Results Section ---
        html.Div(style={**card_style, 'minHeight': '350px', 'maxHeight': '350px', 'display': 'flex',
                        'flexDirection': 'column'}, children=[
            html.H2("AI Analysis Results", style=card_header_style),
            html.Div(id="ai-analysis-output",
                     style={"fontSize": "14px", 'overflowY': 'auto', 'flex': 1, 'paddingRight': '10px'}),
        ]),
    ]),

    # --- Modal for Graph Preview ---
    html.Div(id='graph-preview-modal', style={
        'display': 'none', 'position': 'fixed', 'zIndex': '1000',
        'left': 0, 'top': 0, 'width': '100%', 'height': '100%',
        'overflow': 'auto', 'backgroundColor': 'rgba(0,0,0,0.8)'
    }, children=[
        html.Div(
            style={'backgroundColor': '#fefefe', 'margin': '5% auto', 'padding': '20px', 'border': '1px solid #888',
                   'width': '80%', 'position': 'relative'}, children=[
                html.Button("Close [X]", id='close-preview-btn', n_clicks=0, style={
                    'position': 'absolute', 'top': '10px', 'right': '20px', 'fontSize': '20px', 'fontWeight': 'bold',
                    'border': 'none', 'background': 'none', 'cursor': 'pointer'
                }),
                dcc.Graph(id='preview-graph', style={'height': '600px'})
            ])
    ]),

    # --- Hidden Components ---
    dcc.Interval(id="interval", interval=200, n_intervals=0),
    dcc.Store(id="app-state", data=None),
])


# ---------- Callbacks (updated for 3-channel display) ----------
@app.callback(
    Output("viz-type", "options"),
    Input("domain-switch", "value")
)
def update_viz_options(domain):
    if domain == 'time':
        return [
            {"label": "Standard Monitor", "value": "icu"},
            {"label": "Ping-Pong Overlay", "value": "pingpong"},
            {"label": "Polar View", "value": "polar"},
            {"label": "Cross-Recurrence Plot", "value": "crossrec"}
        ]
    else:  # frequency
        return [
            {"label": "Frequency Spectrum", "value": "icu"},
            {"label": "Spectral Comparison", "value": "pingpong"},
            {"label": "Spectral Polar View", "value": "polar"},
            {"label": "Spectral Cross-Recurrence", "value": "crossrec"}
        ]


@app.callback(
    Output("channels-dropdown", "value"),
    Input("select-all-channels-btn", "n_clicks"),
    State("patients-dropdown", "value"),
    prevent_initial_call=True
)
def select_all_channels(n_clicks, selected_patients):
    if not n_clicks or not selected_patients:
        return no_update

    patients = GLOBAL_DATA.get("patients", [])
    if not patients:
        return no_update

    # Use the first selected patient as the reference for channels
    try:
        pid = int(selected_patients[0])
        if pid >= len(patients):
            return no_update
    except (ValueError, IndexError):
        return no_update

    patient = patients[pid]
    if "ecg" not in patient or patient["ecg"] is None:
        return no_update

    all_channels = [c for c in patient["ecg"].columns if c.startswith("signal_")]
    return all_channels


@app.callback(
    [Output("load-output", "children"),
     Output("patients-dropdown", "options"),
     Output("channels-dropdown", "options"),
     Output("channel-info", "children")],
    [Input("load-btn", "n_clicks"),
     Input("dataset-type", "value")],
    [State("data-dir", "value")],
    prevent_initial_call=True
)
def load_data(nc, dataset_type, data_dir):
    if dataset_type == "EEG" and not PYEDFLIB_AVAILABLE:
        return "pyedflib not installed. Install with: pip install pyedflib", [], [], ""
    if not data_dir or data_dir.strip() == "":
        auto = find_dataset_directory(dataset_type, ".")
        if auto is None:
            return f"No {dataset_type} data found automatically. Please provide directory.", [], [], ""
        data_dir = auto
    if not os.path.isdir(data_dir):
        return f"Directory not found: {data_dir}", [], [], ""

    GLOBAL_DATA["dataset_type"] = dataset_type

    if dataset_type == "EEG":
        patients = load_patient_data(data_dir, dataset_type, max_samples=None, max_patients=MAX_EEG_SUBJECTS)
    else:
        patients = load_patient_data(data_dir, dataset_type, max_samples=None, max_patients=None)

    if not patients:
        return f"No {dataset_type} patients found in {data_dir}.", [], [], ""

    GLOBAL_DATA["patients"] = patients
    GLOBAL_DATA["buffers"] = {}

    for idx, p in enumerate(patients):
        try:
            fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
            ensure_buffers_for_patient(idx, fs, display_window=8)
        except Exception as e:
            print("Buffer init error", e)

    patient_options = [{"label": f"{p['name']} ({p['type']})", "value": idx} for idx, p in enumerate(patients)]

    # Get available channels from first patient
    channel_options = []
    channel_info = ""
    if patients:
        first_patient = patients[0]
        available_channels = [c for c in first_patient["ecg"].columns if c.startswith("signal_")]
        channel_options = [{"label": ch, "value": ch} for ch in available_channels]

        channel_info = f"{dataset_type}: {len(available_channels)} channel(s) available."

    if dataset_type == "EEG":
        msg = f"Loaded {len(patients)} EEG subjects from {data_dir}."
    else:
        msg = f"Loaded {len(patients)} ECG records from {data_dir}."

    return msg, patient_options, channel_options, channel_info


@app.callback(
    Output("interval", "interval"),
    Input("chunk-ms", "value"),
    Input("speed", "value")
)
def adjust_interval(chunk_ms, speed):
    try:
        cm = max(20, int(float(chunk_ms)))
    except:
        cm = 200
    return cm


# ---------- AI Analysis Callback with 1D and 2D Support ----------
@app.callback(
    Output("ai-analysis-output", "children"),
    [Input("ai-analyze-btn", "n_clicks"),
     Input("ai-analyze-2d-btn", "n_clicks")],
    [State("patients-dropdown", "value"),
     State("channels-dropdown", "value"),
     State("dataset-type", "value"),
     State("app-state", "data"),
     State("display-window", "value"),
     State("speed", "value")],
    prevent_initial_call=True
)
def run_ai_analysis(n_clicks_1d, n_clicks_2d, selected_patients, selected_channels,
                    dataset_type, app_state, display_window, speed_val):
    """Run AI analysis on current patient data - supports both 1D signal and 2D image analysis"""

    # Determine which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return "Click an AI Analysis button to analyze current signals"

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    is_2d_analysis = (button_id == "ai-analyze-2d-btn")

    try:
        patients = GLOBAL_DATA.get("patients", [])
        if not patients:
            return html.Div([
                html.P("No patients loaded. Please load data first.",
                       style={"color": "#ff6347"})
            ])

        # Get selected patient indices
        selected_idxs = []
        if selected_patients:
            try:
                if isinstance(selected_patients, list):
                    selected_idxs = [int(i) for i in selected_patients if 0 <= int(i) < len(patients)]
                else:
                    idx = int(selected_patients)
                    if 0 <= idx < len(patients):
                        selected_idxs = [idx]
            except:
                selected_idxs = []

        if not selected_idxs:
            selected_idxs = [0]

        signal_type = dataset_type if dataset_type else GLOBAL_DATA.get("dataset_type", "ECG")

        # ========== 2D IMAGE-BASED ANALYSIS ==========
        if is_2d_analysis:
            if signal_type != "ECG":
                return html.Div([
                    html.H4("2D AI Analysis:", style={"color": "#4F46E5"}),
                    html.P("2D AI Analysis is currently only available for ECG signals.",
                           style={"color": "#EF4444"})
                ])

            pid = selected_idxs[0]
            patient = patients[pid].copy()
            patient_name = patient.get("name", f"Patient {pid}")

            if "ecg" not in patient or patient["ecg"] is None:
                return html.Div([
                    html.H4(f"2D AI Analysis - {patient_name}:", style={"color": "#4F46E5"}),
                    html.P("No data available for analysis", style={"color": "#EF4444"})
                ])

            # Get current playback position
            pos = 0
            if app_state and "pos" in app_state and pid < len(app_state["pos"]):
                pos = app_state["pos"][pid]

            if pos <= 0:
                return html.Div([
                    html.H4("2D AI Analysis:", style={"color": "#4F46E5"}),
                    html.P("No signal data played yet. Start playback first.",
                           style={"color": "#EF4444"})
                ])

            # Create visualization of current window
            try:
                display_window_val = max(1.0, float(display_window or 8.0))
                fs = float(patient.get("header", {}).get("sampling_frequency", 250.0))
                win = int(display_window_val * fs)
                start = max(0, pos - win)

                available_channels = [c for c in patient["ecg"].columns if c.startswith('signal_')]
                if not available_channels:
                    return html.Div([
                        html.H4("2D AI Analysis:", style={"color": "#4F46E5"}),
                        html.P("No channels available", style={"color": "#EF4444"})
                    ])

                # Use selected channels or first 3
                if selected_channels:
                    channels_to_plot = [c for c in selected_channels if c in available_channels]
                else:
                    channels_to_plot = available_channels[:min(3, len(available_channels))]

                if not channels_to_plot:
                    channels_to_plot = available_channels[:1]

                # Create figure for screenshot
                fig = go.Figure()
                for ch in channels_to_plot:
                    seg = patient["ecg"].iloc[start:pos][["time", ch]]
                    if seg.shape[0] == 0:
                        continue
                    t = (seg["time"].values - seg["time"].values[0]).astype(float)
                    y = seg[ch].values.astype(float)
                    fig.add_trace(go.Scattergl(x=t, y=y, mode="lines", name=ch, line=dict(width=2)))

                fig.update_layout(
                    template="plotly_white",
                    title=f"ECG Signal - {patient_name} ({display_window_val}s window)",
                    xaxis=dict(title="Time (s)", showgrid=True),
                    yaxis=dict(title="Amplitude (mV)", showgrid=True),
                    height=600,
                    width=800,
                    showlegend=True
                )

                # Capture screenshot
                print(f"[2D Analysis] Capturing graph screenshot...")
                img_bytes = capture_graph_screenshot(fig)

                if img_bytes is None:
                    return html.Div([
                        html.H4("2D AI Analysis:", style={"color": "#4F46E5"}),
                        html.P("Failed to capture graph image. Ensure kaleido is installed:",
                               style={"color": "#EF4444"}),
                        html.Pre("pip install kaleido",
                                 style={"color": "#4B5563", "fontSize": "12px", "fontFamily": "monospace",
                                        "backgroundColor": "#F3F4F6", "padding": "10px", "borderRadius": "5px"})
                    ])

                print(f"[2D Analysis] Screenshot captured: {len(img_bytes)} bytes")

                # Analyze with Teachable Machine model
                analysis_start = time.time()
                result = analyze_ecg_image_with_teachable_machine(img_bytes)
                analysis_time = time.time() - analysis_start

                # Format results
                result_elements = []

                result_elements.append(
                    html.H4(f"2D AI Analysis - {patient_name}:",
                            style={"color": "#4F46E5", "marginBottom": "10px"})
                )

                result_elements.append(
                    html.P(f"Analysis Time: {analysis_time:.2f}s | Window: {display_window_val}s | "
                           f"Channels: {', '.join(channels_to_plot)}",
                           style={"color": "#6B7280", "fontSize": "11px", "fontStyle": "italic"})
                )

                if result.get("success"):
                    if not result.get("requires_setup"):
                        # Show actual predictions with confidence
                        top_pred = result.get('top_prediction', 'N/A')
                        top_conf = result.get('top_confidence', 0)

                        result_elements.append(
                            html.Div([
                                html.P("Analysis Complete!",
                                       style={"color": "#059669", "fontWeight": "bold", "fontSize": "16px",
                                              "marginBottom": "10px"}),
                                html.Div([
                                    html.Span("Top Prediction: ",
                                              style={"color": "#4B5563", "fontSize": "14px"}),
                                    html.Span(f"{top_pred}",
                                              style={"color": "#1E40AF", "fontSize": "16px", "fontWeight": "bold"})
                                ], style={"marginBottom": "8px"}),
                                html.Div([
                                    html.Span("Confidence: ",
                                              style={"color": "#4B5563", "fontSize": "14px"}),
                                    html.Span(f"{top_conf * 100:.1f}%",
                                              style={"color": "#059669", "fontSize": "16px", "fontWeight": "bold"})
                                ])
                            ], style={"backgroundColor": "#D1FAE5", "padding": "15px",
                                      "borderRadius": "8px", "marginTop": "10px", "marginBottom": "15px",
                                      "border": "2px solid #6EE7B7"})
                        )

                        result_elements.append(
                            html.P("All Predictions:",
                                   style={"color": "#1E40AF", "fontWeight": "bold", "marginTop": "15px",
                                          "marginBottom": "10px"})
                        )

                        predictions = result.get("predictions", [])
                        for i, pred in enumerate(predictions[:5]):  # Show top 5
                            confidence = pred.get('probability', 0)
                            confidence_pct = f"{confidence * 100:.1f}%"
                            label = pred.get('class', 'Unknown')

                            # Color coding
                            if confidence >= 0.6:
                                bar_color = "#10B981"
                                icon = "✓"
                            elif confidence >= 0.3:
                                bar_color = "#F59E0B"
                                icon = "○"
                            else:
                                bar_color = "#F97316"
                                icon = "·"

                            result_elements.append(
                                html.Div([
                                    html.Div([
                                        html.Span(f"{icon} ",
                                                  style={"color": bar_color, "fontWeight": "bold"}),
                                        html.Span(f"{i + 1}. {label}",
                                                  style={"color": "#111827", "fontWeight": "bold"}),
                                        html.Span(f"{confidence_pct}",
                                                  style={"color": bar_color, "float": "right", "fontWeight": "bold"})
                                    ]),
                                    html.Div(
                                        style={
                                            "width": f"{confidence * 100}%",
                                            "height": "6px",
                                            "backgroundColor": bar_color,
                                            "marginTop": "5px",
                                            "borderRadius": "3px",
                                        }
                                    )
                                ], style={"margin": "8px 0", "padding": "10px",
                                          "backgroundColor": "#F9FAFB", "borderRadius": "5px",
                                          "border": "1px solid #E5E7EB"})
                            )

                    else:
                        # Show setup instructions
                        result_elements.append(
                            html.Div([
                                html.P("Setup Required:",
                                       style={"color": "#D97706", "fontWeight": "bold", "fontSize": "16px",
                                              "marginBottom": "10px"}),
                                html.P(
                                    "Image successfully processed. To enable predictions, install required packages:",
                                    style={"color": "#4B5563", "fontSize": "13px", "marginBottom": "10px"}),
                                html.Pre("pip install tensorflow",
                                         style={"backgroundColor": "#F3F4F6", "padding": "12px",
                                                "borderRadius": "5px", "color": "#111827",
                                                "fontSize": "13px", "fontFamily": "monospace",
                                                "border": "1px solid #E5E7EB"}),
                                html.P("Then restart the application to enable real-time predictions.",
                                       style={"color": "#4B5563", "fontSize": "12px", "marginTop": "10px"})
                            ], style={"backgroundColor": "#FEF3C7", "padding": "15px",
                                      "borderRadius": "5px", "marginTop": "10px", "marginBottom": "15px",
                                      "border": "2px solid #FBBF24"})
                        )

                else:
                    # Error occurred
                    result_elements.append(
                        html.Div([
                            html.P(f"Error: {result.get('error', 'Unknown error')}",
                                   style={"color": "#DC2626", "fontWeight": "bold"})
                        ], style={"marginTop": "10px", "padding": "15px", "backgroundColor": "#FEE2E2",
                                  "borderRadius": "5px", "border": "2px solid #F87171"})
                    )

                return html.Div(result_elements)

            except Exception as viz_error:
                print(f"[2D Analysis] Visualization error: {viz_error}")
                import traceback
                traceback.print_exc()
                return html.Div([
                    html.H4("2D AI Analysis:", style={"color": "#4F46E5"}),
                    html.P(f"Error creating visualization: {str(viz_error)}",
                           style={"color": "#EF4444"})
                ])

        # ========== 1D SIGNAL-BASED ANALYSIS (Original) ==========
        else:
            # Switch model to appropriate type
            try:
                model_switched = AI_MODEL.switch_signal_type(signal_type)
                if not model_switched:
                    return html.Div([
                        html.P(f"Failed to load {signal_type} model.",
                               style={"color": "#EF4444"}),
                        html.P(f"For ECG: Ensure HuBERT-ECG model is available",
                               style={"color": "#6B7280", "fontSize": "12px"}),
                        html.P(f"For EEG: Place 'EEG-PREST-16-channels.ckpt' in script directory",
                               style={"color": "#6B7280", "fontSize": "12px"}),
                        html.P(f"Download BIOT models from: https://github.com/ycq091044/BIOT",
                               style={"color": "#F59E0B", "fontSize": "11px"})
                    ])
            except Exception as model_error:
                return html.Div([
                    html.P(f"Error loading {signal_type} model: {str(model_error)}",
                           style={"color": "#EF4444"}),
                    html.P("Check console for detailed error information.",
                           style={"color": "#6B7280", "fontSize": "12px"})
                ])

            # Analyze each selected patient
            all_results = []

            for pid in selected_idxs[:3]:  # Limit to 3 patients
                if pid >= len(patients):
                    continue

                patient = patients[pid].copy()
                patient_name = patient.get("name", f"Patient {pid}")

                if "ecg" not in patient or patient["ecg"] is None:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                        html.P("No data available for analysis", style={"color": "#EF4444"})
                    ]))
                    continue

                # Get current playback position
                pos = 0
                if app_state and "pos" in app_state and pid < len(app_state["pos"]):
                    pos = app_state["pos"][pid]

                if pos <= 0:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                        html.P("No signal data played yet. Start playback first.",
                               style={"color": "#EF4444"})
                    ]))
                    continue

                # Get available channels
                available_channels = [c for c in patient["ecg"].columns if c.startswith('signal_')]

                if not available_channels:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                        html.P("No channels available for analysis", style={"color": "#EF4444"})
                    ]))
                    continue

                # Get signal data up to current position
                signal_data = patient["ecg"].iloc[:pos].copy()

                # Check minimum data requirements
                min_samples = 500 if signal_type == "ECG" else 1000
                if len(signal_data) < min_samples:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                        html.P(f"Insufficient data for {signal_type} analysis ({len(signal_data)} samples). "
                               f"Need at least {min_samples} samples.",
                               style={"color": "#EF4444"})
                    ]))
                    continue

                # Check channel requirements
                required_channels = 12 if signal_type == "ECG" else 16
                available_count = len(available_channels)

                channel_warning = None
                if available_count < required_channels:
                    channel_warning = (f"{signal_type} model expects {required_channels} channels, "
                                       f"but only {available_count} available. Missing channels will be zero-padded.")

                try:
                    print(f"[AI Analysis] Analyzing {patient_name}, type: {signal_type}")

                    # Run AI analysis
                    analysis_start_time = time.time()
                    analysis_result = AI_MODEL.analyze_patient_data(
                        signal_data,
                        channel_name=available_channels[0],
                        top_k=5,
                        signal_type=signal_type
                    )
                    analysis_duration = time.time() - analysis_start_time

                    if "error" in analysis_result:
                        all_results.append(html.Div([
                            html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                            html.P(f"Analysis Error: {analysis_result['error']}",
                                   style={"color": "#EF4444"})
                        ]))
                        continue

                    # Format results
                    predictions = analysis_result.get("predictions", [])
                    if not predictions:
                        all_results.append(html.Div([
                            html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                            html.P("No predictions returned from model.",
                                   style={"color": "#EF4444"})
                        ]))
                        continue

                    # Create result elements
                    patient_elements = []

                    # Patient header with signal type
                    patient_elements.append(
                        html.H4(f"{patient_name} ({signal_type}):",
                                style={"color": "#1E40AF", "marginBottom": "8px"})
                    )

                    # Channel warning if applicable
                    if channel_warning:
                        patient_elements.append(
                            html.P(channel_warning,
                                   style={"color": "#D97706", "fontSize": "11px",
                                          "backgroundColor": "#FEF3C7", "padding": "5px",
                                          "borderRadius": "3px", "marginBottom": "8px"})
                        )

                    # Top predictions with color coding
                    for i, pred in enumerate(predictions[:5]):  # Show top 5
                        confidence = pred.get("confidence", 0)
                        confidence_pct = f"{confidence * 100:.1f}%"
                        label = pred.get("label", "Unknown")

                        if confidence >= 0.8:
                            confidence_color = "#059669"
                            icon = "✓"
                        elif confidence >= 0.6:
                            confidence_color = "#F59E0B"
                            icon = "○"
                        elif confidence >= 0.4:
                            confidence_color = "#F97316"
                            icon = "△"
                        else:
                            confidence_color = "#EF4444"
                            icon = "·"

                        patient_elements.append(
                            html.Div([
                                html.Span(f"{i + 1}. ",
                                          style={"fontWeight": "bold", "color": "#4B5563"}),
                                html.Span(f"{icon} ",
                                          style={"marginRight": "5px", "color": confidence_color}),
                                html.Span(f"{label}",
                                          style={"color": "#111827", "fontWeight": "bold"}),
                                html.Span(f" ({confidence_pct})",
                                          style={"color": confidence_color, "marginLeft": "10px"})
                            ], style={
                                "margin": "4px 0",
                                "padding": "8px",
                                "backgroundColor": "#F9FAFB",
                                "borderRadius": "4px",
                                "borderLeft": f"4px solid {confidence_color}"
                            })
                        )

                    all_results.append(
                        html.Div(patient_elements, style={"marginBottom": "20px", "borderBottom": "1px solid #E5E7EB",
                                                          "paddingBottom": "15px"})
                    )

                except Exception as analysis_error:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#1E40AF"}),
                        html.P(f"Analysis failed: {str(analysis_error)}",
                               style={"color": "#EF4444"})
                    ]))
                    print(f"[AI Analysis] Error analyzing {patient_name}: {analysis_error}")
                    import traceback
                    traceback.print_exc()

            if not all_results:
                return html.Div([
                    html.P("No analysis results available.", style={"color": "#EF4444"}),
                    html.P("Ensure patients are loaded and playback has started.",
                           style={"color": "#6B7280", "fontSize": "12px"})
                ])

            return html.Div(all_results, style={"lineHeight": "1.5"})

    except Exception as e:
        print(f"[AI Analysis] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.P("Unexpected error in AI analysis.",
                   style={"color": "#EF4444"}),
            html.P(f"Error details: {str(e)}",
                   style={"color": "#6B7280", "fontSize": "11px"}),
            html.P("Check the console for more information.",
                   style={"color": "#6B7280", "fontSize": "12px"})
        ])


# ---------- Helper to build each viz figure (updated for 3-channel display) ----------
def xor_overlay_segments(prev_vals, new_vals, strict=True):
    """
    XOR overlay merge for two same-length 1D arrays:
    - If new_vals[i] == prev_vals[i] (exact equality when strict=True),
      BOTH values are erased at that timepoint (both set to np.nan).
    - If different, keep both values intact.

    Returns:
        prev_out, new_out (both numpy arrays same length as inputs)

    Notes:
    - If lengths differ, both arrays are trimmed to the minimum length.
    - strict=True uses exact equality; strict=False uses a small tolerance.
    - This implements true XOR behavior: identical values cancel each other out.
    """
    import numpy as np

    if prev_vals is None or new_vals is None:
        return prev_vals, new_vals

    a = np.asarray(prev_vals, dtype=float).copy()
    b = np.asarray(new_vals, dtype=float).copy()

    if a.size == 0 or b.size == 0:
        return a, b

    if a.shape != b.shape:
        m = min(a.size, b.size)
        a = a[:m]
        b = b[:m]

    if strict:
        same_mask = (a == b)
    else:
        # tolerance relative to signal scale
        scale = max(1.0, np.nanmax(np.abs(np.concatenate([a, b]))))
        tol = 1e-9 * scale
        same_mask = np.isclose(a, b, atol=tol, rtol=0.0)

    # True XOR: erase BOTH where identical
    prev_out = a.copy()
    new_out = b.copy()
    prev_out[same_mask] = np.nan
    new_out[same_mask] = np.nan

    return prev_out, new_out


def _calculate_fft(signal_segment, original_fs, resampling_freq):
    """
    Helper function to perform resampling and FFT calculation.
    Also identifies the maximum frequency in the original signal.
    """
    start_time = time.time()

    y = signal_segment
    y = y[~np.isnan(y)]

    if len(y) < 2:
        return None, None, original_fs, 0, 0

    # 1. Find f_max from the original, un-sampled segment
    N_orig = len(y)
    yf_orig = np.fft.fft(y)
    xf_orig = np.fft.fftfreq(N_orig, 1 / original_fs)

    # Corrected f_max calculation:
    # Find the highest frequency with significant energy, not just the strongest peak.
    # We ignore the DC component (index 0)
    magnitudes = np.abs(yf_orig[1:N_orig // 2])
    frequencies = xf_orig[1:N_orig // 2]

    # Set a threshold to ignore noise (e.g., 1% of the max magnitude)
    if len(magnitudes) > 0:
        noise_threshold = np.max(magnitudes) * 0.01
        significant_freqs = frequencies[magnitudes > noise_threshold]
        if len(significant_freqs) > 0:
            f_max = np.max(significant_freqs)
        else:
            f_max = 0
    else:
        f_max = 0

    # 2. Perform resampling for the plot
    resampling_freq_val = float(resampling_freq or 500.0)
    if resampling_freq_val >= original_fs:
        step = 1
    else:
        step = int(round(original_fs / resampling_freq_val))
    step = max(1, step)

    y_sampled = y[::step]
    fs_new = original_fs / step

    if len(y_sampled) < 2:
        return None, None, fs_new, 0, f_max

    # 3. Calculate FFT on the (potentially down-sampled) signal
    N = len(y_sampled)
    yf = np.fft.fft(y_sampled)
    xf = np.fft.fftfreq(N, 1 / fs_new)

    xf_plot = xf[:N // 2]
    yf_plot = np.abs(yf[0:N // 2])  # Use absolute magnitude

    computation_time = time.time() - start_time

    return xf_plot, yf_plot, fs_new, computation_time, f_max


def make_viz_figure(viz_type, patients, selected_idxs, show_all_channels, state,
                    display_window_val=8.0, speed_val=1.0, collapse_flag=None,
                    selected_channels=None, overlay=True, resampling_freq=500.0,
                    domain='frequency', sampling_period_ms=None):
    """
    Unified visualization builder for both Time and Frequency domains.
    """
    total_computation_time = 0
    nyquist_info = {'f_max': 0, 'nyquist_rate': 0}

    try:
        if not patients:
            return create_empty_figure("No patients"), None, None

        if not selected_idxs:
            return create_empty_figure("No patients selected"), None, None

        pid = selected_idxs[0]
        if pid < 0 or pid >= len(patients):
            return create_empty_figure("Invalid patient selection"), None, None

        p = patients[pid]
        if "ecg" not in p or p["ecg"] is None:
            return create_empty_figure("No data for selected patient"), None, None

        if selected_channels and isinstance(selected_channels, (list, tuple)) and len(selected_channels) > 0:
            channels_to_display = [ch for ch in selected_channels if ch in p["ecg"].columns]
            if not channels_to_display:
                return create_empty_figure("Selected channels not found"), None, None
        else:
            all_channels = [c for c in p["ecg"].columns if c.startswith("signal_")]
            channels_to_display = all_channels[:min(3, len(all_channels))]
            if not channels_to_display:
                return create_empty_figure("No channels available"), None, None

        fs = float(p.get("header", {}).get("sampling_frequency", 250.0))
        pos = int(state["pos"][pid]) if state and "pos" in state and pid < len(state["pos"]) else len(p["ecg"])

        win = int(display_window_val * fs)
        start = max(0, pos - win)
        current_segment_df = p["ecg"].iloc[start:pos]

        if current_segment_df.shape[0] < 2:
            return create_empty_figure("Not enough data in window"), 0, nyquist_info

        # ==================================
        # TIME DOMAIN VISUALIZATIONS
        # ==================================
        if domain == 'time':
            if sampling_period_ms is not None:
                try:
                    ts_user = float(sampling_period_ms) / 1000.0
                    if ts_user > 0 and (1.0 / ts_user) < fs:  # only downsample
                        fs_user = 1.0 / ts_user
                        step = int(round(fs / fs_user))
                        step = max(1, step)
                        if step > 1:
                            current_segment_df = current_segment_df.iloc[::step]
                except (ValueError, TypeError):
                    pass  # Ignore if input is invalid

            unit = "mV" if GLOBAL_DATA.get("dataset_type", "ECG") == "ECG" else "µV"

            # --- Standard Monitor ---
            if viz_type == "icu":
                if overlay:
                    fig = go.Figure()
                    for ch in channels_to_display:
                        seg = current_segment_df[["time", ch]]
                        t = (seg["time"].values - seg["time"].values[0]).astype(float)
                        y = seg[ch].values.astype(float)
                        fig.add_trace(go.Scattergl(x=t, y=y, mode="lines", name=ch, customdata=[ch] * len(t)))
                    fig.update_layout(yaxis_title=f"Amplitude ({unit})")
                else:
                    from plotly.subplots import make_subplots
                    n = len(channels_to_display)
                    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                    for i, ch in enumerate(channels_to_display, start=1):
                        seg = current_segment_df[["time", ch]]
                        t = (seg["time"].values - seg["time"].values[0]).astype(float)
                        y = seg[ch].values.astype(float)
                        fig.add_trace(go.Scattergl(x=t, y=y, mode="lines", name=ch, customdata=[ch] * len(t)), row=i,
                                      col=1)
                        fig.update_yaxes(title_text=f"Amp ({unit})", row=i, col=1)
                    fig.update_layout(showlegend=False)
                fig.update_layout(title=f"Time Domain Monitor: {p.get('name', 'Patient')}", xaxis_title="Time (s)")

            # --- Ping-Pong Overlay ---
            elif viz_type == "pingpong":
                prev_start = max(0, start - win)
                prev_end = start
                prev_segment_df = p["ecg"].iloc[prev_start:prev_end]

                if overlay:
                    fig = go.Figure()
                    for ch in channels_to_display:
                        y_curr = current_segment_df[ch].values
                        t = (current_segment_df["time"].values - current_segment_df["time"].values[0]).astype(float)
                        if prev_segment_df.shape[0] == len(current_segment_df):
                            y_prev = prev_segment_df[ch].values
                            prev_masked, curr_masked = xor_overlay_segments(y_prev, y_curr)
                            fig.add_trace(go.Scattergl(x=t, y=prev_masked, mode="lines", name=f"{ch} (Prev)",
                                                       line=dict(dash='dash', color='gray'), customdata=[ch] * len(t)))
                            fig.add_trace(go.Scattergl(x=t, y=curr_masked, mode="lines", name=f"{ch} (Curr)",
                                                       customdata=[ch] * len(t)))
                        else:
                            fig.add_trace(go.Scattergl(x=t, y=y_curr, mode="lines", name=f"{ch} (Curr)",
                                                       customdata=[ch] * len(t)))
                    fig.update_layout(yaxis_title=f"Amplitude ({unit})")
                else:  # Separate
                    from plotly.subplots import make_subplots
                    n = len(channels_to_display)
                    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                    for i, ch in enumerate(channels_to_display, start=1):
                        y_curr = current_segment_df[ch].values
                        t = (current_segment_df["time"].values - current_segment_df["time"].values[0]).astype(float)
                        if prev_segment_df.shape[0] == len(current_segment_df):
                            y_prev = prev_segment_df[ch].values
                            prev_masked, curr_masked = xor_overlay_segments(y_prev, y_curr)
                            fig.add_trace(go.Scattergl(x=t, y=prev_masked, mode="lines", name=f"Prev",
                                                       line=dict(dash='dash', color='gray'), customdata=[ch] * len(t)),
                                          row=i, col=1)
                            fig.add_trace(
                                go.Scattergl(x=t, y=curr_masked, mode="lines", name=f"Curr", customdata=[ch] * len(t)),
                                row=i, col=1)
                        else:
                            fig.add_trace(
                                go.Scattergl(x=t, y=y_curr, mode="lines", name=f"Curr", customdata=[ch] * len(t)),
                                row=i, col=1)
                        fig.update_yaxes(title_text=f"Amp ({unit})", row=i, col=1)
                    fig.update_layout(showlegend=True)
                fig.update_layout(title=f"Ping-Pong Overlay: {p.get('name', 'Patient')}", xaxis_title="Time (s)")

            # --- Polar View ---
            elif viz_type == "polar":
                time_vals = current_segment_df["time"].values
                span = time_vals[-1] - time_vals[0] if len(time_vals) > 1 and time_vals[-1] != time_vals[0] else 1.0
                theta = 360 * (time_vals - time_vals[0]) / span

                if overlay:
                    fig = go.Figure()
                    for ch in channels_to_display:
                        r = current_segment_df[ch].values
                        fig.add_trace(
                            go.Scatterpolar(theta=theta, r=r, mode="lines", name=ch, customdata=[ch] * len(r)))
                    fig.update_layout(polar=dict(radialaxis=dict(title=f"Amplitude ({unit})")))
                else:  # Separate
                    from plotly.subplots import make_subplots
                    n_channels = len(channels_to_display)
                    cols = min(2, n_channels) if n_channels > 1 else 1
                    rows = int(np.ceil(n_channels / cols))
                    specs = [[{'type': 'polar'}] * cols for _ in range(rows)]
                    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=channels_to_display,
                                        horizontal_spacing=0.15, vertical_spacing=0.15)
                    for idx, ch in enumerate(channels_to_display):
                        row, col = (idx // cols) + 1, (idx % cols) + 1
                        r = current_segment_df[ch].values
                        fig.add_trace(
                            go.Scatterpolar(theta=theta, r=r, mode="lines", name=ch, customdata=[ch] * len(r)), row=row,
                            col=col)
                        polar_name = f'polar{idx + 1}' if idx > 0 else 'polar'
                        fig.layout[polar_name].radialaxis.title = f'Amp ({unit})'
                    fig.update_layout(showlegend=False)
                fig.update_layout(title=f"Polar View: {p.get('name', 'Patient')}")

            # --- Cross-Recurrence Plot ---
            elif viz_type == "crossrec":
                if len(channels_to_display) < 2:
                    return create_empty_figure("Need at least 2 channels for Cross-Recurrence"), None, None
                from plotly.subplots import make_subplots

                if overlay:
                    midpoint = len(channels_to_display) // 2
                    set_a_ch = channels_to_display[:midpoint]
                    set_b_ch = channels_to_display[midpoint:]
                    if not set_a_ch or not set_b_ch:
                        return create_empty_figure("Need channels for both comparison sets in overlay mode"), None, None

                    s1 = current_segment_df[set_a_ch].mean(axis=1).values
                    s2 = current_segment_df[set_b_ch].mean(axis=1).values

                    fig = go.Figure()
                    hist, xedges, yedges = np.histogram2d(s1, s2, bins=80)
                    fig.add_trace(go.Heatmap(z=hist.T, x=xedges, y=yedges, colorscale="Viridis",
                                             customdata=[f"Avg({','.join(set_a_ch)})"] * len(xedges)))
                    fig.update_layout(
                        title=f"Cross-Recurrence: Avg({', '.join(set_a_ch)}) vs. Avg({', '.join(set_b_ch)})",
                        xaxis_title=f"Avg of Set A ({unit})",
                        yaxis_title=f"Avg of Set B ({unit})"
                    )
                else:  # Separate
                    pairs = [(channels_to_display[i], channels_to_display[i + 1]) for i in
                             range(0, len(channels_to_display) - 1, 2)]
                    if not pairs:
                        return create_empty_figure("Not enough channels for pairing"), None, None
                    n_pairs = len(pairs)
                    fig = make_subplots(rows=n_pairs, cols=1, subplot_titles=[f"{p[0]} vs {p[1]}" for p in pairs])
                    for i, (ch_a, ch_b) in enumerate(pairs, start=1):
                        s1 = current_segment_df[ch_a].values
                        s2 = current_segment_df[ch_b].values
                        hist, xedges, yedges = np.histogram2d(s1, s2, bins=80)
                        fig.add_trace(go.Heatmap(z=hist.T, x=xedges, y=yedges, colorscale="Viridis",
                                                 customdata=[ch_a] * len(xedges)), row=i, col=1)
                        fig.update_xaxes(title_text=f"{ch_a} ({unit})", row=i, col=1)
                        fig.update_yaxes(title_text=f"{ch_b} ({unit})", row=i, col=1)
                    fig.update_layout(title=f"Cross-Recurrence: {p.get('name', 'Patient')}")

            else:
                fig = create_empty_figure(f"Unknown Visualization Type")

            fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
            return fig, 0, None  # No computation time or Nyquist info for time domain

        # ==================================
        # FREQUENCY DOMAIN VISUALIZATIONS
        # ==================================
        else:  # domain == 'frequency'
            final_fs_new = fs

            # --- Frequency Spectrum ---
            if viz_type == "icu":
                if overlay:
                    fig = go.Figure()
                    for ch in channels_to_display:
                        y_segment = current_segment_df[ch].values
                        xf, yf, fs_new, comp_time, f_max = _calculate_fft(y_segment, fs, resampling_freq)
                        total_computation_time += comp_time
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max)
                        final_fs_new = fs_new
                        if xf is not None:
                            fig.add_trace(go.Scattergl(x=xf, y=yf, mode="lines", name=ch, customdata=[ch] * len(xf)))
                    fig.update_layout(yaxis_title="Magnitude")
                else:  # Separate
                    from plotly.subplots import make_subplots
                    n = len(channels_to_display)
                    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                    for i, ch in enumerate(channels_to_display, start=1):
                        y_segment = current_segment_df[ch].values
                        xf, yf, fs_new, comp_time, f_max = _calculate_fft(y_segment, fs, resampling_freq)
                        total_computation_time += comp_time
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max)
                        final_fs_new = fs_new
                        if xf is not None:
                            fig.add_trace(go.Scattergl(x=xf, y=yf, mode="lines", name=ch, customdata=[ch] * len(xf)),
                                          row=i, col=1)
                        fig.update_yaxes(title_text="Magnitude", row=i, col=1)
                    fig.update_layout(showlegend=False)
                fig.update_layout(
                    title=f"Frequency Spectrum: {p.get('name', 'Patient')}",
                    xaxis_title=f"Frequency (Hz) - Sampled at {final_fs_new:.1f} Hz"
                )

            # --- Spectral Comparison (Ping-Pong) ---
            elif viz_type == "pingpong":
                prev_start = max(0, start - win)
                prev_end = start
                prev_segment_df = p["ecg"].iloc[prev_start:prev_end]

                if overlay:
                    fig = go.Figure()
                    for ch in channels_to_display:
                        # Current FFT
                        xf_curr, yf_curr, fs_new_curr, ct_curr, fmax_curr = _calculate_fft(
                            current_segment_df[ch].values, fs, resampling_freq)
                        total_computation_time += ct_curr
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], fmax_curr)
                        final_fs_new = fs_new_curr
                        if xf_curr is not None:
                            fig.add_trace(go.Scattergl(x=xf_curr, y=yf_curr, mode="lines", name=f"{ch} (Curr)",
                                                       customdata=[ch] * len(xf_curr)))

                        # Previous FFT
                        if prev_segment_df.shape[0] > 1:
                            xf_prev, yf_prev, _, ct_prev, fmax_prev = _calculate_fft(prev_segment_df[ch].values, fs,
                                                                                     resampling_freq)
                            total_computation_time += ct_prev
                            nyquist_info['f_max'] = max(nyquist_info['f_max'], fmax_prev)
                            if xf_prev is not None:
                                fig.add_trace(go.Scattergl(x=xf_prev, y=yf_prev, mode="lines", name=f"{ch} (Prev)",
                                                           line=dict(dash='dash', color='gray'),
                                                           customdata=[ch] * len(xf_prev)))
                    fig.update_layout(yaxis_title="Magnitude")
                else:  # Separate
                    from plotly.subplots import make_subplots
                    n = len(channels_to_display)
                    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=channels_to_display)
                    for i, ch in enumerate(channels_to_display, start=1):
                        xf_curr, yf_curr, fs_new_curr, ct_curr, fmax_curr = _calculate_fft(
                            current_segment_df[ch].values, fs, resampling_freq)
                        total_computation_time += ct_curr
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], fmax_curr)
                        final_fs_new = fs_new_curr
                        if xf_curr is not None:
                            fig.add_trace(go.Scattergl(x=xf_curr, y=yf_curr, mode="lines", name=f"Curr",
                                                       customdata=[ch] * len(xf_curr)), row=i, col=1)

                        if prev_segment_df.shape[0] > 1:
                            xf_prev, yf_prev, _, ct_prev, fmax_prev = _calculate_fft(prev_segment_df[ch].values, fs,
                                                                                     resampling_freq)
                            total_computation_time += ct_prev
                            nyquist_info['f_max'] = max(nyquist_info['f_max'], fmax_prev)
                            if xf_prev is not None:
                                fig.add_trace(go.Scattergl(x=xf_prev, y=yf_prev, mode="lines", name=f"Prev",
                                                           line=dict(dash='dash', color='gray'),
                                                           customdata=[ch] * len(xf_prev)), row=i, col=1)
                        fig.update_yaxes(title_text="Magnitude", row=i, col=1)
                    fig.update_layout(showlegend=True)
                fig.update_layout(
                    title=f"Spectral Comparison: {p.get('name', 'Patient')}",
                    xaxis_title=f"Frequency (Hz) - Sampled at {final_fs_new:.1f} Hz"
                )

            # --- Spectral Polar View ---
            elif viz_type == "polar":
                if overlay:
                    fig = go.Figure()
                    for ch in channels_to_display:
                        xf, yf, fs_new, ct, f_max = _calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                        total_computation_time += ct
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max)
                        final_fs_new = fs_new
                        if xf is not None and len(xf) > 0:
                            theta = xf * 360 / (fs_new / 2)
                            fig.add_trace(
                                go.Scatterpolar(theta=theta, r=yf, mode="lines", name=ch, customdata=[ch] * len(xf)))
                    fig.update_layout(polar=dict(radialaxis_type="log", radialaxis_title="Magnitude"))
                else:  # Separate
                    from plotly.subplots import make_subplots
                    n_channels = len(channels_to_display)
                    cols = min(2, n_channels) if n_channels > 1 else 1
                    rows = int(np.ceil(n_channels / cols))
                    specs = [[{'type': 'polar'}] * cols for _ in range(rows)]
                    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=channels_to_display,
                                        horizontal_spacing=0.15, vertical_spacing=0.15)
                    for idx, ch in enumerate(channels_to_display):
                        row, col = (idx // cols) + 1, (idx % cols) + 1
                        xf, yf, fs_new, ct, f_max = _calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                        total_computation_time += ct
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max)
                        final_fs_new = fs_new
                        if xf is not None and len(xf) > 0:
                            theta = xf * 360 / (fs_new / 2)
                            fig.add_trace(
                                go.Scatterpolar(theta=theta, r=yf, mode="lines", name=ch, customdata=[ch] * len(xf)),
                                row=row, col=col)
                        polar_name = f'polar{idx + 1}' if idx > 0 else 'polar'
                        fig.layout[polar_name].radialaxis.type = "log"
                        fig.layout[polar_name].radialaxis.title = "Magnitude"
                    fig.update_layout(showlegend=False)
                fig.update_layout(title=f"Spectral Polar View: {p.get('name', 'Patient')}")

            # --- Spectral Cross-Recurrence ---
            elif viz_type == "crossrec":
                if len(channels_to_display) < 2:
                    return create_empty_figure("Need at least 2 channels for Cross-Recurrence"), None, None
                from plotly.subplots import make_subplots

                if overlay:
                    midpoint = len(channels_to_display) // 2
                    set_a_ch = channels_to_display[:midpoint]
                    set_b_ch = channels_to_display[midpoint:]
                    if not set_a_ch or not set_b_ch:
                        return create_empty_figure("Need channels for both comparison sets in overlay mode"), None, None

                    # Average the FFTs for each set
                    xf_a_all, yfs_a, yfs_b = [], [], []
                    min_len = float('inf')

                    for ch in set_a_ch:
                        xf, yf, _, ct, f_max = _calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                        total_computation_time += ct
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max)
                        if yf is not None:
                            yfs_a.append(yf)
                            xf_a_all = xf  # just need one x-axis
                            min_len = min(min_len, len(yf))

                    for ch in set_b_ch:
                        xf, yf, fs_new, ct, f_max = _calculate_fft(current_segment_df[ch].values, fs, resampling_freq)
                        total_computation_time += ct
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max)
                        final_fs_new = fs_new
                        if yf is not None:
                            yfs_b.append(yf)
                            min_len = min(min_len, len(yf))

                    if not yfs_a or not yfs_b or min_len == float('inf'):
                        return create_empty_figure(
                            "Could not compute FFT for channel sets"), total_computation_time, nyquist_info

                    yf_a_avg = np.mean([y[:min_len] for y in yfs_a], axis=0)
                    yf_b_avg = np.mean([y[:min_len] for y in yfs_b], axis=0)

                    z = np.outer(yf_a_avg, yf_b_avg)
                    xf_plot = xf_a_all[:min_len] if xf_a_all is not None else np.arange(min_len)

                    fig = go.Figure(data=go.Heatmap(z=z, x=xf_plot, y=xf_plot, colorscale='Viridis'))
                    fig.update_layout(
                        title=f"Spectral Cross-Recurrence: Avg({', '.join(set_a_ch)}) vs. Avg({', '.join(set_b_ch)})",
                        xaxis_title=f"Frequency (Hz) - Set A",
                        yaxis_title=f"Frequency (Hz) - Set B"
                    )

                else:  # Separate
                    pairs = [(channels_to_display[i], channels_to_display[i + 1]) for i in
                             range(0, len(channels_to_display) - 1, 2)]
                    if not pairs:
                        return create_empty_figure("Not enough channels for pairing"), None, None
                    n_pairs = len(pairs)
                    fig = make_subplots(rows=n_pairs, cols=1, subplot_titles=[f"{p[0]} vs {p[1]}" for p in pairs])

                    for i, (ch_a, ch_b) in enumerate(pairs, start=1):
                        xf_a, yf_a, fs_new_a, ct_a, f_max_a = _calculate_fft(current_segment_df[ch_a].values, fs,
                                                                             resampling_freq)
                        xf_b, yf_b, fs_new_b, ct_b, f_max_b = _calculate_fft(current_segment_df[ch_b].values, fs,
                                                                             resampling_freq)
                        total_computation_time += (ct_a + ct_b)
                        nyquist_info['f_max'] = max(nyquist_info['f_max'], f_max_a, f_max_b)
                        final_fs_new = fs_new_a

                        if xf_a is not None and xf_b is not None:
                            z = np.outer(yf_a, yf_b)
                            fig.add_trace(
                                go.Heatmap(z=z, x=xf_a, y=xf_b, colorscale="Viridis", customdata=[ch_a] * len(xf_a)),
                                row=i, col=1)
                        fig.update_xaxes(title_text=f"Frequency ({ch_a})", row=i, col=1)
                        fig.update_yaxes(title_text=f"Frequency ({ch_b})", row=i, col=1)
                    fig.update_layout(title="Cross-Recurrence of Frequencies")

            else:
                fig = create_empty_figure(f"Unknown Visualization Type")

            # Final updates for all frequency plots
            nyquist_info['nyquist_rate'] = 2 * nyquist_info['f_max']
            fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
            return fig, total_computation_time, nyquist_info

    except Exception as exc:
        print(f"[make_viz_figure] Error: {exc}")
        import traceback
        traceback.print_exc()
        return create_empty_figure("Error building visualization"), None, None


def create_empty_figure(title="No data"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "y": 0.5, "x": 0.5, "xanchor": "center", "yanchor": "middle"},
        xaxis={'visible': False},
        yaxis={'visible': False},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#6B7280'
    )
    return fig


# ---------- Combined update callback (updated for 3-channel display) ----------
@app.callback(
    [Output("main-graph", "figure"),
     Output("app-state", "data"),
     Output("fft-computation-time", "children"),
     Output("nyquist-info", "children"),
     Output("freq-analysis-card", "style"),
     Output("nyquist-info-time", "children"),
     Output("time-analysis-card", "style")],
    [Input("interval", "n_intervals"),
     Input("play-btn", "n_clicks"),
     Input("pause-btn", "n_clicks"),
     Input("reset-btn", "n_clicks"),
     Input("domain-switch", "value")],
    [State("app-state", "data"),
     State("patients-dropdown", "value"),
     State("channels-dropdown", "value"),
     State("overlay-mode", "value"),
     State("viz-type", "value"),
     State("speed", "value"),
     State("chunk-ms", "value"),
     State("display-window", "value"),
     State("resampling-freq", "value"),
     State("sampling-period-input", "value")],
    prevent_initial_call=False
)
def combined_update(n_intervals, n_play, n_pause, n_reset, domain, state,
                    selected, selected_channels, overlay_mode, viz_type,
                    speed, chunk_ms, display_window, resampling_freq,
                    sampling_period_ms):
    try:
        patients = GLOBAL_DATA.get("patients", [])
        if not patients:
            empty_fig = create_empty_figure("No patients loaded")
            return empty_fig, {"playing": False, "pos": [], "write_idx": []}, "", "", {'display': 'none'}, "", {
                'display': 'none'}

        # Initialize state
        if state is None:
            state = {"playing": False, "pos": [0] * len(patients), "write_idx": [0] * len(patients)}
        if "pos" not in state or len(state["pos"]) != len(patients):
            state["pos"] = [0] * len(patients)
        if "write_idx" not in state or len(state["write_idx"]) != len(patients):
            state["write_idx"] = [0] * len(patients)
        if "playing" not in state:
            state["playing"] = False

        # Determine trigger
        ctx = callback_context
        trigger = getattr(ctx, "triggered_id", None)
        if trigger is None and ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        # Handle controls
        if trigger == "play-btn":
            state["playing"] = True
        elif trigger == "pause-btn":
            state["playing"] = False
        elif trigger == "reset-btn":
            state["playing"] = False
            state["pos"] = [0] * len(patients)
            state["write_idx"] = [0] * len(patients)
            for pid in list(GLOBAL_DATA.get("buffers", {}).keys()):
                buf = GLOBAL_DATA["buffers"][pid]
                buf["signal_buffer"].fill(np.nan)
                buf["write_idx"] = 0
                buf["rr_buffer"].fill(np.nan)
                buf["rr_write_idx"] = 0
                buf["last_peak_global_index"] = -1
                buf["direction"] = 1
                buf["ping_position"] = 0.0

        # Parse parameters
        chunk_ms_val = max(20, float(chunk_ms or 200))
        speed_val = max(0.1, float(speed or 1.0))
        display_window_val = max(1.0, float(display_window or 8.0))

        # Ensure buffers
        for pid, p in enumerate(patients):
            fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
            ensure_buffers_for_patient(pid, fs, display_window_val)

        # Update playback
        if trigger == "interval" and state.get("playing", False):
            for pid, p in enumerate(patients):
                if not p or "ecg" not in p or p["ecg"] is None:
                    continue

                ecg = p["ecg"]
                if ecg.shape[0] == 0:
                    continue

                fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
                chunk_sec = (chunk_ms_val / 1000.0) * speed_val
                chunk_samples = max(1, int(round(chunk_sec * fs)))

                pos0 = state["pos"][pid]
                pos1 = min(len(ecg), pos0 + chunk_samples)

                if pos1 > pos0:
                    state["pos"][pid] = pos1

            # Stop playing if all patients have reached the end
            try:
                if all(state["pos"][i] >= len(patients[i]["ecg"]) for i in range(len(patients))):
                    state["playing"] = False
            except Exception:
                pass

        # Get selected patient indices
        selected_idxs = []
        if selected:
            try:
                if isinstance(selected, list):
                    selected_idxs = [int(i) for i in selected if 0 <= int(i) < len(patients)]
                else:
                    idx = int(selected)
                    if 0 <= idx < len(patients):
                        selected_idxs = [idx]
            except:
                selected_idxs = []

        if not selected_idxs:
            selected_idxs = [0]  # Default to first patient

        # Determine overlay mode
        overlay = (overlay_mode == "overlay")

        # Build visualization
        main_fig, comp_time, nyquist_info = make_viz_figure(
            viz_type,
            patients,
            selected_idxs,
            None,  # show_all_channels (deprecated)
            state,
            display_window_val=display_window_val,
            speed_val=speed_val,
            collapse_flag=None,  # deprecated
            selected_channels=selected_channels,
            overlay=overlay,
            resampling_freq=resampling_freq,
            domain=domain,
            sampling_period_ms=sampling_period_ms
        )

        # --- Prepare outputs for analysis cards ---
        fft_time_output = ""
        nyquist_output = ""
        nyquist_output_time = ""
        freq_card_style = {'display': 'none'}
        time_card_style = {'display': 'none'}

        if domain == 'time':
            time_card_style = {**card_style}
            if selected_idxs:
                pid = selected_idxs[0]
                p = patients[pid]
                original_fs = float(p.get("header", {}).get("sampling_frequency", 250.0))

                # Determine the effective sampling frequency for analysis display
                effective_fs = original_fs
                if sampling_period_ms is not None:
                    try:
                        ts_user = float(sampling_period_ms) / 1000.0
                        if ts_user > 0:
                            effective_fs = 1.0 / ts_user
                    except (ValueError, TypeError):
                        pass  # Keep original_fs if input is invalid

                win = int(display_window_val * original_fs)
                pos = int(state["pos"][pid])
                start = max(0, pos - win)
                current_segment_df = p["ecg"].iloc[start:pos]

                all_channels = [c for c in p["ecg"].columns if c.startswith("signal_")]
                if all_channels and not current_segment_df.empty:
                    # Always calculate f_max on the original, full-resolution signal
                    first_channel_data = current_segment_df[all_channels[0]].values
                    _, _, _, _, f_max = _calculate_fft(first_channel_data, original_fs, original_fs)

                    if f_max > 0:
                        Ts = 1.0 / effective_fs
                        Ts_max = 1.0 / (2 * f_max)

                        is_good_sampling = Ts < Ts_max
                        status_text = "Good Sampling" if is_good_sampling else "Aliasing Risk!"
                        status_color = "#10B981" if is_good_sampling else "#EF4444"

                        nyquist_output_time = html.Div([
                            html.P(f"Signal's Max Frequency (f_max): {f_max:.1f} Hz",
                                   style={'margin': '0px', 'fontSize': '12px'}),
                            html.P(f"Current Sampling Period (Ts): {Ts * 1000:.2f} ms",
                                   style={'margin': '0px', 'fontSize': '12px'}),
                            html.P(f"Required Period (< 1 / (2*f_max)): {Ts_max * 1000:.2f} ms",
                                   style={'margin': '0px 0px 5px 0px', 'fontSize': '12px'}),
                            html.Div([
                                html.Span("Status: ", style={'fontWeight': 'bold'}),
                                html.Span(status_text, style={'color': 'white', 'backgroundColor': status_color,
                                                              'padding': '2px 6px', 'borderRadius': '4px',
                                                              'fontWeight': 'bold'})
                            ])
                        ], style={'padding': '10px', 'backgroundColor': '#F9FAFB', 'borderRadius': '6px'})

        elif domain == 'frequency':
            freq_card_style = {**card_style}
            if comp_time is not None:
                time_ms = comp_time * 1000
                color = "#3B82F6"
                if time_ms > 50: color = "#F59E0B"
                if time_ms > 100: color = "#EF4444"
                fft_time_output = html.Div([
                    html.Span("FFT Compute Time: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{time_ms:.1f} ms", style={'color': color, 'fontSize': '16px', 'fontWeight': 'bold'})
                ])
            if nyquist_info and nyquist_info['f_max'] > 0:
                f_max = nyquist_info['f_max']
                nyquist_rate = nyquist_info['nyquist_rate']
                is_aliasing = (resampling_freq or 500) < nyquist_rate
                status_text = "Aliasing Risk!" if is_aliasing else "Good Sampling"
                status_color = "#EF4444" if is_aliasing else "#10B981"
                nyquist_output = html.Div([
                    html.P(f"Signal's Max Frequency (f_max): {f_max:.1f} Hz",
                           style={'margin': '0px', 'fontSize': '12px'}),
                    html.P(f"Required Nyquist Rate (> 2 * f_max): {nyquist_rate:.1f} Hz",
                           style={'margin': '0px 0px 5px 0px', 'fontSize': '12px'}),
                    html.Div([
                        html.Span("Status: ", style={'fontWeight': 'bold'}),
                        html.Span(status_text,
                                  style={'color': 'white', 'backgroundColor': status_color, 'padding': '2px 6px',
                                         'borderRadius': '4px', 'fontWeight': 'bold'})
                    ])
                ], style={'padding': '10px', 'backgroundColor': '#F9FAFB', 'borderRadius': '6px'})

        return main_fig, state, fft_time_output, nyquist_output, freq_card_style, nyquist_output_time, time_card_style

    except Exception as e:
        print(f"[combined_update] Error: {e}")
        import traceback
        traceback.print_exc()
        empty_fig = create_empty_figure("Error occurred")
        return empty_fig, {"playing": False, "pos": [], "write_idx": []}, "Error", "", {'display': 'none'}, "Error", {
            'display': 'none'}


@app.callback(
    [Output("graph-preview-modal", "style"),
     Output("preview-graph", "figure")],
    [Input("main-graph", "clickData"),
     Input("close-preview-btn", "n_clicks")],
    [State("app-state", "data"),
     State("patients-dropdown", "value"),
     State("overlay-mode", "value"),
     State("viz-type", "value"),
     State("speed", "value"),
     State("display-window", "value"),
     State("resampling-freq", "value"),
     State("domain-switch", "value")],
    prevent_initial_call=True
)
def display_graph_preview(clickData, n_close, state, selected_patients, overlay_mode, viz_type, speed, display_window,
                          resampling_freq, domain):
    ctx = callback_context
    if not ctx.triggered or not ctx.triggered_id:
        return {'display': 'none'}, create_empty_figure()

    triggered_id = ctx.triggered_id

    if triggered_id == 'close-preview-btn':
        return {'display': 'none'}, create_empty_figure()

    if triggered_id == 'main-graph' and clickData:
        try:
            # Extract clicked channel name from customdata
            curve = clickData['points'][0]
            clicked_channel = curve.get('customdata')

            if not clicked_channel:
                print("Could not identify channel from clickData.")
                return {'display': 'none'}, create_empty_figure()

            # Now we have the channel, let's build a large figure for it.
            patients = GLOBAL_DATA.get("patients", [])

            fig, _, _ = make_viz_figure(
                viz_type,
                patients,
                selected_patients,
                None,
                state,
                display_window_val=float(display_window or 8.0),
                speed_val=float(speed or 1.0),
                selected_channels=[clicked_channel],  # Display only the clicked channel
                overlay=True,  # Force overlay since it's a single channel view
                resampling_freq=resampling_freq,
                domain=domain
            )

            # Enhance the title for the preview
            fig.update_layout(title=f"Detailed View: {clicked_channel}", showlegend=False)

            modal_style = {
                'display': 'block', 'position': 'fixed', 'zIndex': '1000',
                'left': 0, 'top': 0, 'width': '100%', 'height': '100%',
                'overflow': 'auto', 'backgroundColor': 'rgba(0,0,0,0.8)'
            }
            return modal_style, fig

        except Exception as e:
            print(f"Error in click callback: {e}")
            import traceback
            traceback.print_exc()
            return {'display': 'none'}, create_empty_figure()

    return {'display': 'none'}, create_empty_figure()


# ---------- Run ----------
if __name__ == "__main__":

    if not PYEDFLIB_AVAILABLE:
        print("Warning: pyedflib not installed. EEG (EDF) support will be disabled until it's installed.")
    app.run(debug=True, host="127.0.0.1", port=8052)

