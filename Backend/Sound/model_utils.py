# --- SEARCHABLE COMMENT: Imports ---
import os
import threading
import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

# --- SEARCHABLE COMMENT: Import Config ---
# Import configuration constants from config.py
import config

# ============================================
# --- SEARCHABLE COMMENT: Global Model Variables ---
# ============================================
_processor, _model, _device = None, None, "cpu"
_model_lock = threading.Lock()

# ============================================
# --- SEARCHABLE COMMENT: Model Setup ---
# ============================================

# --- SEARCHABLE COMMENT: Ensure Model Function ---
def ensure_model():
    """
    Loads the Hugging Face model and processor into global variables if they haven't been loaded yet.
    Uses a lock to ensure thread safety during loading.

    Returns:
        tuple: (processor, model, device, error_message)
    """
    global _processor, _model, _device
    if _model is not None: return _processor, _model, _device, None

    with _model_lock:
        if _model is not None: return _processor, _model, _device, None
        try:
            print(f"Loading model: {config.MODEL_ID}")
            auth_token = config.HF_TOKEN if config.HF_TOKEN else None
            # Use trust_remote_code=True if required by the model
            proc = AutoProcessor.from_pretrained(config.MODEL_ID, token=auth_token, trust_remote_code=True)
            mod = AutoModelForAudioClassification.from_pretrained(config.MODEL_ID, token=auth_token, trust_remote_code=True)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mod.to(dev)
            mod.eval()
            _processor, _model, _device = proc, mod, dev
            print(f"Model loaded successfully on {dev}")
            return _processor, _model, _device, None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None, f"Failed to load model: {e}"

# ============================================
# --- SEARCHABLE COMMENT: Prediction Helper ---
# ============================================
def predict_with_local_model(processor, model, device, audio, sr, chunk_s=2):
    """
    Runs inference using the loaded Hugging Face model on the provided audio data.
    Handles necessary preprocessing like mono conversion and resampling to the model's required rate (16kHz).
    Processes audio in chunks to handle potentially long files.

    Args:
        processor: The Hugging Face processor.
        model: The Hugging Face model.
        device: The device (CPU or CUDA) the model is on.
        audio (np.ndarray): Input audio signal.
        sr (int): Sampling rate of the input audio.
        chunk_s (int): Duration of audio chunks in seconds for processing.

    Returns:
        list: A list of dictionaries, each containing prediction results ('label', 'score') for a chunk,
              or an error message.
    """
    # --- SEARCHABLE COMMENT: Prediction Audio Preprocessing ---
    if not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
        max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0
        if max_abs > 1.5:
            print(f"Warning: Input audio max abs value is {max_abs}. Assuming int16 scale and normalizing.")
            if max_abs > 0 : audio /= 32767.0

    if audio.ndim > 1:
        print("Audio has multiple channels, converting to mono.")
        audio = np.mean(audio, axis=1)

    results = []
    target_sr_model = 16000

    # --- SEARCHABLE COMMENT: Prediction Resampling (to Model Rate) ---
    if sr != target_sr_model:
        print(f"Resampling audio from {sr} Hz to model's required {target_sr_model} Hz for prediction.")
        try:
            audio_resampled_for_model = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr_model, res_type='soxr_hq')
            current_sr_for_model = target_sr_model
        except Exception as e:
            print(f"Error during resampling for model prediction: {e}")
            return [{"error": f"Resampling failed: {e}"}]
    else:
        audio_resampled_for_model = audio
        current_sr_for_model = sr

    # --- SEARCHABLE COMMENT: Prediction Chunking ---
    chunk_len = int(chunk_s * current_sr_for_model)
    if len(audio_resampled_for_model) < current_sr_for_model:
        print(f"Audio too short for prediction ({len(audio_resampled_for_model)} samples < {current_sr_for_model}).")
        return [{"label": "N/A", "score": 0.0, "message": "Audio too short"}]

    num_chunks_processed = 0
    for i in range(0, len(audio_resampled_for_model), chunk_len):
        chunk = audio_resampled_for_model[i:i + chunk_len]
        # --- SEARCHABLE COMMENT: Prediction Chunk Handling ---
        min_chunk_len = current_sr_for_model * 0.5
        if len(chunk) < min_chunk_len:
            print(f"Skipping final chunk of length {len(chunk)} < {min_chunk_len:.0f} samples.")
            continue

        num_chunks_processed += 1
        try:
            # --- SEARCHABLE COMMENT: Model Inference ---
            inputs = processor(chunk, sampling_rate=current_sr_for_model, return_tensors="pt", padding="longest")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            # --- SEARCHABLE COMMENT: Model Postprocessing ---
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            label_id = int(torch.argmax(probs))
            results.append({
                "chunk": i // chunk_len,
                "label": model.config.id2label[label_id],
                "score": float(probs[label_id]),
            })
        except Exception as e:
            # --- SEARCHABLE COMMENT: Prediction Error Handling ---
            print(f"Error during model inference on chunk {i // chunk_len}: {e}")
            results.append({"error": f"Inference failed on chunk: {e}"})

    if num_chunks_processed == 0 and len(audio_resampled_for_model) >= min_chunk_len :
        print("Warning: No chunks were processed despite audio being long enough.")
        return [{"label": "N/A", "score": 0.0, "message": "Chunk processing error"}]
    elif num_chunks_processed == 0:
        print(f"Audio too short for prediction ({len(audio_resampled_for_model)} samples).")
        return [{"label": "N/A", "score": 0.0, "message": "Audio too short"}]

    return results
