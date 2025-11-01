# audio_processor.py
import numpy as np
import librosa
from scipy.io import wavfile
import io
import base64

def analyze_audio(y, sr):
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    magnitude = np.abs(fft)
    if np.max(magnitude) > 0:
        magnitude /= np.max(magnitude)
    return {
        'waveform': y.tolist(),
        'sample_rate': sr,
        'duration': len(y) / sr,
        'frequencies': freqs.tolist(),
        'magnitude': magnitude.tolist(),
        'nyquist_freq': sr / 2
    }



def create_audio_data(audio, sr):
    audio_array = np.array(audio)
    if len(audio_array) == 0:
        return ""
    playback_sr = 3000
    if sr < 3000:
        duration_sec = len(audio_array) / sr
        new_len = int(duration_sec * playback_sr)
        if new_len == 0:
            return ""
        indices = np.round(np.linspace(0, len(audio_array) - 1, new_len)).astype(int)
        final_audio = audio_array[indices]
        final_sr = playback_sr
    else:
        final_audio = audio_array
        final_sr = sr
    final_audio = np.clip(final_audio, -1.0, 1.0)
    wav_data = (final_audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, final_sr, wav_data)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"  # ✅ Correct format with "data:"