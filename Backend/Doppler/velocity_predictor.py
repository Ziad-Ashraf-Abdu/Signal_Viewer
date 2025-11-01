# velocity_predictor.py
import os
import numpy as np
import tensorflow as tf
import librosa

class VelocityPredictor:
    def __init__(self, model_path='velocity_regressor_dense.h5'):
        self.model = None
        self.load_model(model_path)

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                print("✅ Velocity prediction model loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading velocity model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found at '{path}'. Prediction disabled.")
            self.model = None

    def predict(self, audio_data, sr):
        if self.model is None:
            return None
        try:
            target_sr = 3000
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if sr != target_sr:
                audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=target_sr)
            target_length = target_sr * 2
            if len(audio_data) > target_length:
                start = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start:start + target_length]
            else:
                audio_data = librosa.util.fix_length(data=audio_data, size=target_length)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sr, n_mfcc=30, n_fft=2048, hop_length=512)
            mfccs_mean = np.mean(mfccs, axis=1)
            features = mfccs_mean.reshape(1, -1)
            prediction = self.model.predict(features, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"Error during velocity prediction: {e}")
            return None