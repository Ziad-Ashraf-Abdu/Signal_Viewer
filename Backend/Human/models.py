# models.py
import warnings
import numpy as np
import librosa
import torch

warnings.filterwarnings('ignore')

# --- Model Loading ---

ANTI_ALIASING_MODEL = None
MODEL_LOADED = False
try:
    from keras.models import load_model
    ANTI_ALIASING_MODEL = load_model('./Anti-Aliasing.keras')
    MODEL_LOADED = True
    print("✅ AI model for audio reconstruction loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load reconstruction model. App will run in fallback mode. Error: {e}")

GENDER_MODEL_LOADED = False
gender_feature_extractor = None
gender_model = None
try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    GENDER_MODEL_NAME = "prithivMLmods/Common-Voice-Gender-Detection"
    gender_feature_extractor = AutoFeatureExtractor.from_pretrained(GENDER_MODEL_NAME)
    gender_model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL_NAME)
    GENDER_MODEL_LOADED = True
    print("✅ Gender detection model loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load gender detection model. Feature will be disabled. Error: {e}")


# --- Model Functions ---

def apply_model_reconstruction(y_input, sr_input):
    target_sr = 16000
    if sr_input != target_sr:
        y_input = librosa.resample(y_input, orig_sr=sr_input, target_sr=target_sr)
        sr_input = target_sr

    if not MODEL_LOADED:
        return y_input

    try:
        model_len = 48000
        reconstructed_chunks = []
        for i in range(0, len(y_input), model_len):
            chunk = y_input[i:i + model_len]
            len_chunk = len(chunk)
            if len_chunk < model_len:
                chunk = np.pad(chunk, (0, model_len - len_chunk), 'constant')
            model_input = chunk[np.newaxis, ..., np.newaxis]
            pred_chunk = np.squeeze(ANTI_ALIASING_MODEL.predict(model_input, verbose=0))
            if len_chunk < model_len:
                pred_chunk = pred_chunk[:len_chunk]
            reconstructed_chunks.append(pred_chunk)
        return np.concatenate(reconstructed_chunks) if reconstructed_chunks else np.array([])
    except Exception as e:
        print(f"❌ Error during model reconstruction: {e}")
        return y_input


def predict_gender(y, sr):
    if not GENDER_MODEL_LOADED:
        return "Gender detection model not available."
    try:
        if sr != 16000:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            y_resampled = y
        inputs = gender_feature_extractor(y_resampled, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = gender_model(inputs.input_values).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        predicted_label = gender_model.config.id2label[pred_id].capitalize()
        return f"Predicted Gender: {predicted_label}"
    except Exception as e:
        print(f"❌ Error during gender prediction: {e}")
        return "Could not predict gender from audio."