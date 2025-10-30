"""
Utility functions specific to Human application
"""
import numpy as np
from shared.audio_processing import audio_processor
from shared.model_utils import model_manager
from models import MODEL_LOADED, GENDER_MODEL_LOADED

def analyze_audio(y, sr):
    """Calculate basic audio features including FFT magnitude spectrum"""
    return audio_processor.analyze_audio_basic(y, sr)

def apply_model_reconstruction(y_input, sr_input):
    """Apply AI model for reconstruction or fall back to standard resampling"""
    if not MODEL_LOADED:
        print("Reconstruction model not available. Performing standard upsampling.")
        return y_input

    try:
        reconstructed_audio = model_manager.apply_audio_reconstruction(y_input, sr_input)
        return reconstructed_audio
    except Exception as e:
        print(f"❌ Error during model reconstruction: {e}")
        return y_input

def create_audio_data(audio, sr):
    """Make audio playable in browser"""
    return audio_processor.make_playable_wav(audio, sr)

def predict_gender(y, sr):
    """Predict gender from audio using Hugging Face model"""
    if not GENDER_MODEL_LOADED:
        return "Gender detection model not available."
    
    try:
        results = model_manager.predict_with_transformers('gender_detection', y, sr)
        
        if results and 'error' not in results[0]:
            result = results[0]
            predicted_label = result.get('label', 'Unknown').capitalize()
            return f"Predicted Gender: {predicted_label}"
        else:
            return "Could not predict gender from audio."
    except Exception as e:
        print(f"❌ Error during gender prediction: {e}")
        return "Could not predict gender from audio."