"""
Utility functions specific to Doppler application
"""
import numpy as np
import librosa
from pydub import AudioSegment
from shared.audio_processing import audio_processor
from shared.analysis_utils import analysis_utils
from shared.model_utils import model_manager

# Import from models using absolute import
from models import VELOCITY_MODEL_LOADED

def load_audio_from_bytes(wav_bytes, filename):
    """Load audio from bytes with fallback support"""
    try:
        # Try librosa first
        data_float, sr = librosa.load(wav_bytes, sr=None, mono=True)
        return data_float, sr
    except Exception:
        # Fallback to pydub
        try:
            wav_bytes.seek(0)
            audio_segment = AudioSegment.from_file(wav_bytes, format="wav")
            sr = audio_segment.frame_rate
            audio_segment = audio_segment.set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples())
            
            if audio_segment.sample_width == 2:
                data_float = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                data_float = samples.astype(np.float32) / 2147483648.0
            else:
                data_float = samples.astype(np.float32) / 128.0
                
            return data_float, sr
        except Exception as e:
            raise Exception(f"Could not load audio file '{filename}': {str(e)}")

def detect_dominant_frequency(audio_data, sr):
    """Detect dominant frequency in audio"""
    return audio_processor.detect_dominant_frequency(audio_data, sr)

def predict_velocity(audio_data, sr):
    """Predict velocity from audio features"""
    if not VELOCITY_MODEL_LOADED:
        return None
    
    try:
        return model_manager.predict_velocity(audio_data, sr)
    except Exception as e:
        print(f"❌ Error during velocity prediction: {e}")
        return None

def create_audio_spectrum_plot(audio_data, sr, dominant_freq):
    """Create frequency spectrum plot"""
    xf, magnitude = audio_processor.compute_fft_analysis(audio_data, sr)
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xf, y=magnitude, mode='lines', name='Spectrum', 
        line=dict(color='#667eea')
    ))
    fig.add_vline(
        x=dominant_freq,
        line=dict(color='#ef4444', dash='dash', width=3),
        annotation_text=f"Highest Sig.: {dominant_freq:.1f} Hz",
        annotation_position="top right",
        annotation_font=dict(size=14, color='#ef4444', family='Arial Black')
    )
    fig.update_layout(
        title="Uploaded Audio Frequency Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        xaxis_range=[0, min(2000, sr // 2)]
    )
    return fig