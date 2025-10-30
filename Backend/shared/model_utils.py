"""
Unified model management for all AI/ML models across applications
Handles loading, caching, and inference for various model types
"""
import os
import threading
import numpy as np
import torch
import tensorflow as tf
import librosa
from transformers import AutoProcessor, AutoModelForAudioClassification

from .config import MODEL_CONFIG, AUDIO_CONFIG, get_model_config


class ModelManager:
    """
    Centralized model management for:
    - Hugging Face Transformers models (Sound, Human apps)
    - Keras/TensorFlow models (Human, Doppler apps) 
    - Custom PyTorch models
    """
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.devices = {}
        self._lock = threading.Lock()
        
        # Auto-detect available device
        self.default_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 ModelManager initialized on device: {self.default_device}")
    
    # ==================== TRANSFORMERS MODELS ====================
    
    def load_transformers_model(self, model_name, model_key=None, auth_token=None, trust_remote_code=True):
        """
        Load Hugging Face Transformers model - used by Sound and Human apps
        
        Args:
            model_name: Hugging Face model identifier
            model_key: unique key for caching (defaults to model_name)
            auth_token: Hugging Face API token for private models
            trust_remote_code: whether to trust custom model code
            
        Returns:
            tuple: (processor, model, device, error_message)
        """
        if model_key is None:
            model_key = model_name
        
        with self._lock:
            # Check if already loaded
            if model_key in self.models:
                processor = self.processors.get(model_key)
                model_info = self.models[model_key]
                return processor, model_info[0], model_info[1], None
            
            try:
                print(f"🔄 Loading Transformers model: {model_name}")
                
                # Load processor and model
                processor = AutoProcessor.from_pretrained(
                    model_name, 
                    token=auth_token,
                    trust_remote_code=trust_remote_code
                )
                
                model = AutoModelForAudioClassification.from_pretrained(
                    model_name,
                    token=auth_token,
                    trust_remote_code=trust_remote_code
                )
                
                # Move to appropriate device
                device = self.default_device
                model.to(device)
                model.eval()
                
                # Cache for future use
                self.processors[model_key] = processor
                self.models[model_key] = (model, device)
                self.devices[model_key] = device
                
                print(f"✅ Successfully loaded {model_name} on {device}")
                return processor, model, device, None
                
            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {str(e)}"
                print(f"❌ {error_msg}")
                return None, None, None, error_msg
    
    def predict_with_transformers(self, model_key, audio_data, sr, chunk_duration=None):
        """
        Run inference with loaded Transformers model
        
        Args:
            model_key: key of loaded model
            audio_data: input audio signal
            sr: sample rate
            chunk_duration: duration of audio chunks in seconds
            
        Returns:
            list: prediction results
        """
        if model_key not in self.models:
            return [{"error": f"Model '{model_key}' not loaded"}]
        
        processor, model_info = self.processors.get(model_key), self.models.get(model_key)
        if processor is None or model_info is None:
            return [{"error": f"Model '{model_key}' not properly initialized"}]
        
        model, device = model_info
        chunk_duration = chunk_duration or AUDIO_CONFIG["chunk_duration"]
        
        try:
            # Get model configuration
            config = get_model_config(model_key)
            target_sr = config.get("target_sr", 16000)
            
            # Resample if needed
            if sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            results = []
            chunk_len = int(chunk_duration * sr)
            
            # Process in chunks
            for i in range(0, len(audio_data), chunk_len):
                chunk = audio_data[i:i + chunk_len]
                
                # Skip chunks that are too short
                if len(chunk) < sr * 0.5:  # Minimum 0.5 seconds
                    continue
                
                try:
                    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding="longest")
                    
                    with torch.no_grad():
                        logits = model(inputs.input_values.to(device)).logits
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                    label_id = int(torch.argmax(probs))
                    
                    results.append({
                        "chunk": i // chunk_len,
                        "label": model.config.id2label[label_id],
                        "score": float(probs[label_id]),
                    })
                    
                except Exception as e:
                    results.append({"error": f"Inference failed on chunk {i // chunk_len}: {str(e)}"})
            
            return results if results else [{"error": "No chunks processed successfully"}]
            
        except Exception as e:
            return [{"error": f"Prediction failed: {str(e)}"}]
    
    # ==================== KERAS/TENSORFLOW MODELS ====================
    
    def load_keras_model(self, model_path, model_key=None):
        """
        Load Keras/TensorFlow model - used by Human and Doppler apps
        
        Args:
            model_path: path to .h5 or .keras model file
            model_key: unique key for caching
            
        Returns:
            tuple: (model, error_message)
        """
        if model_key is None:
            model_key = os.path.basename(model_path)
        
        with self._lock:
            # Check if already loaded
            if model_key in self.models:
                return self.models[model_key], None
            
            if not os.path.exists(model_path):
                error_msg = f"Model file not found: {model_path}"
                return None, error_msg
            
            try:
                print(f"🔄 Loading Keras model: {model_path}")
                
                # Load model
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Cache for future use
                self.models[model_key] = model
                
                print(f"✅ Successfully loaded Keras model: {model_key}")
                return model, None
                
            except Exception as e:
                error_msg = f"Failed to load Keras model {model_path}: {str(e)}"
                print(f"❌ {error_msg}")
                return None, error_msg
    
    def predict_with_keras(self, model_key, input_data, preprocess_fn=None):
        """
        Run inference with loaded Keras model
        
        Args:
            model_key: key of loaded model
            input_data: model input data
            preprocess_fn: optional preprocessing function
            
        Returns:
            numpy array: model predictions
        """
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        
        try:
            # Apply preprocessing if provided
            if preprocess_fn:
                input_data = preprocess_fn(input_data)
            
            # Run prediction
            predictions = model.predict(input_data, verbose=0)
            return predictions
            
        except Exception as e:
            print(f"❌ Keras prediction failed: {str(e)}")
            return None
    
    # ==================== AUDIO RECONSTRUCTION (Human App) ====================
    
    def apply_audio_reconstruction(self, audio_data, sr, model_key='audio_reconstruction'):
        """
        Apply audio reconstruction/upsampling model - used by Human app
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            model_key: key of reconstruction model
            
        Returns:
            numpy array: reconstructed audio
        """
        model = self.models.get(model_key)
        if model is None:
            print("⚠️ Reconstruction model not available, returning original audio")
            return audio_data
        
        try:
            config = get_model_config(model_key)
            target_sr = config.get("target_sr", 16000)
            
            # Resample to target rate if needed
            if sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            model_len = 48000  # Model's expected input length
            reconstructed_chunks = []
            
            # Process in chunks compatible with model
            for i in range(0, len(audio_data), model_len):
                chunk = audio_data[i:i + model_len]
                len_chunk = len(chunk)
                
                # Pad if necessary
                if len_chunk < model_len:
                    chunk = np.pad(chunk, (0, model_len - len_chunk), 'constant')
                
                # Prepare for model (add batch and channel dimensions)
                model_input = chunk[np.newaxis, ..., np.newaxis]
                
                # Get prediction
                pred_chunk = np.squeeze(model.predict(model_input, verbose=0))
                
                # Remove padding if added
                if len_chunk < model_len:
                    pred_chunk = pred_chunk[:len_chunk]
                
                reconstructed_chunks.append(pred_chunk)
            
            # Combine all chunks
            if reconstructed_chunks:
                return np.concatenate(reconstructed_chunks)
            else:
                return np.array([])
                
        except Exception as e:
            print(f"❌ Audio reconstruction failed: {str(e)}")
            return audio_data  # Fallback to original
    
    # ==================== VELOCITY PREDICTION (Doppler App) ====================
    
    def predict_velocity(self, audio_data, sr, model_key='velocity_prediction'):
        """
        Predict velocity from audio features - used by Doppler app
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            model_key: key of velocity prediction model
            
        Returns:
            float: predicted velocity in m/s, or None if prediction fails
        """
        model = self.models.get(model_key)
        if model is None:
            return None
        
        try:
            config = get_model_config(model_key)
            target_sr = config.get("target_sr", 3000)
            
            # Ensure audio is float32
            audio_data = np.asarray(audio_data, dtype=np.float32)
            
            # Resample if needed
            if sr != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            
            # Standardize length
            target_length = target_sr * 2  # 2 seconds
            if len(audio_data) > target_length:
                # Center crop
                start = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start:start + target_length]
            else:
                # Pad with zeros
                audio_data = librosa.util.fix_length(data=audio_data, size=target_length)
            
            # Extract MFCC features (same parameters as training)
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=target_sr, 
                n_mfcc=30, 
                n_fft=2048, 
                hop_length=512
            )
            mfccs_mean = np.mean(mfccs, axis=1)  # shape: (30,)
            
            # Predict
            features = mfccs_mean.reshape(1, -1)
            prediction = model.predict(features, verbose=0)
            
            return float(prediction[0][0])
            
        except Exception as e:
            print(f"❌ Velocity prediction failed: {str(e)}")
            return None
    
    # ==================== MODEL MANAGEMENT ====================
    
    def unload_model(self, model_key):
        """Unload a specific model to free memory"""
        with self._lock:
            if model_key in self.models:
                del self.models[model_key]
            if model_key in self.processors:
                del self.processors[model_key]
            if model_key in self.devices:
                del self.devices[model_key]
            print(f"🗑️ Unloaded model: {model_key}")
    
    def get_loaded_models(self):
        """Get list of currently loaded models"""
        return list(self.models.keys())
    
    def clear_all_models(self):
        """Clear all loaded models"""
        with self._lock:
            self.models.clear()
            self.processors.clear()
            self.devices.clear()
            print("🧹 Cleared all loaded models")


# Global instance for convenience
model_manager = ModelManager()

# Convenience functions for common operations
def ensure_model_loaded(model_type, **kwargs):
    """Ensure a specific model type is loaded"""
    config = get_model_config(model_type)
    
    if model_type in ['drone_detection', 'gender_detection']:
        return model_manager.load_transformers_model(
            config['model_id'], 
            model_type,
            **kwargs
        )
    elif model_type in ['velocity_prediction', 'audio_reconstruction']:
        return model_manager.load_keras_model(
            config['model_path'],
            model_type
        )
    else:
        return None, None, None, f"Unknown model type: {model_type}"