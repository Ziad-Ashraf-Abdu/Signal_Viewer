# ai_model_manager.py
import os
import time
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageOps
import io
import base64

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt, resample
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class AIModelManager:
    def __init__(self):
        self.ecg_model = None
        self.eeg_model = None
        self.active_model = None
        self.model_loaded = False
        self.signal_type = "ECG"
        
        # ECG labels based on cardiovascular conditions
        self.ECG_LABELS = [
            "Myocardial Infarction",
            "Normal", 
            "Conduction Disturbance",
            "LVH (Left Ventricular Hypertrophy)",
            "Hypertrophy"
        ]
        
        # EEG labels based on neurological conditions
        self.EEG_LABELS = [
            "Schizophrenia", 
            "Epilepsy", 
            "Normal EEG"
        ]
        
    def analyze_ecg_image_with_teachable_machine(self, img_bytes,
                                                model_url="https://teachablemachine.withgoogle.com/models/aV7sUMdvb/"):
        """
        Send ECG image to Teachable Machine model for classification
        """
        try:
            image = Image.open(io.BytesIO(img_bytes))
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_array = np.asarray(image, dtype=np.float32)
            normalized_image = (image_array / 127.5) - 1

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image

            print(f"[2D Analysis] Attempting to analyze ECG image")
            print(f"[2D Analysis] Model URL: {model_url}")

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

            keras_model_path = "teachable_machine_model.h5"

            if os.path.exists(keras_model_path):
                try:
                    import tensorflow as tf
                    from tensorflow import keras

                    print(f"[2D Analysis] Loading Keras model from {keras_model_path}...")
                    
                    custom_objects = {}
                    try:
                        model = keras.models.load_model(
                            keras_model_path,
                            compile=False,
                            custom_objects=custom_objects
                        )
                    except Exception as load_error:
                        print(f"[2D Analysis] Standard load failed, trying with custom loader...")
                        model = tf.keras.models.load_model(
                            keras_model_path,
                            compile=False,
                            safe_mode=False
                        )

                    print(f"[2D Analysis] Model loaded successfully!")
                    print(f"[2D Analysis] Running prediction...")
                    prediction = model.predict(data, verbose=0)

                    predictions = []
                    for i, prob in enumerate(prediction[0]):
                        label = class_labels[i] if i < len(class_labels) else f'Class {i}'
                        predictions.append({
                            "class": label,
                            "probability": float(prob)
                        })

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

    def switch_signal_type(self, signal_type):
        """
        Switch between ECG and EEG models
        """
        if signal_type not in ["ECG", "EEG"]:
            print(f"[Model Loader] Invalid signal type: {signal_type}. Must be 'ECG' or 'EEG'")
            return False

        if signal_type == self.signal_type and self.model_loaded:
            print(f"[Model Loader] Already using {signal_type} model")
            return True

        print(f"[Model Loader] Switching from {self.signal_type} to {signal_type}")
        self.signal_type = signal_type
        return True

    def get_available_conditions(self, signal_type=None):
        """
        Get available conditions for the specified signal type
        """
        if signal_type is None:
            signal_type = self.signal_type
            
        if signal_type == "ECG":
            return self.ECG_LABELS
        else:
            return self.EEG_LABELS

    def analyze_patient_data_simple(self, patient_data, signal_type="ECG"):
        """
        Simple analysis for demonstration purposes
        In a real implementation, this would use actual trained models
        """
        if signal_type == "ECG":
            conditions = self.ECG_LABELS
        else:
            conditions = self.EEG_LABELS
            
        # Simulate analysis with random probabilities
        np.random.seed(hash(str(patient_data.shape)) % 10000)
        probs = np.random.dirichlet(np.ones(len(conditions)))
        probs = probs / probs.sum()
        
        predictions = []
        for i, (condition, prob) in enumerate(zip(conditions, probs)):
            predictions.append({
                "index": i,
                "label": condition,
                "confidence": float(prob),
                "confidence_percent": round(float(prob) * 100, 2)
            })
            
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "success": True,
            "predictions": predictions[:5],  # Top 5
            "signal_type": signal_type,
            "timestamp": datetime.now().isoformat(),
            "note": "This is a simulated analysis. Install proper models for real diagnosis."
        }