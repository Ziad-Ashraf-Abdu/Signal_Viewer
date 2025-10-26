import numpy as np
import base64
import io
from scipy.fft import fft, fftfreq
from scipy.io import wavfile

class AudioProcessor:
    def make_playable_wav(self, audio, sr):
        if len(audio) == 0:
            return ""
        playback_sr = 3000
        if sr >= 3000:
            final_audio = audio
            final_sr = sr
        else:
            duration_sec = len(audio) / sr
            new_len = int(duration_sec * playback_sr)
            if new_len == 0:
                return ""
            indices = np.round(np.linspace(0, len(audio) - 1, new_len)).astype(int)
            final_audio = audio[indices]
            final_sr = playback_sr
        final_audio = np.clip(final_audio, -1.0, 1.0)
        wav_data = (final_audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, final_sr, wav_data)
        wav_bytes = buf.getvalue()
        b64 = base64.b64encode(wav_bytes).decode()
        return f"data:audio/wav;base64,{b64}"

    def detect_dominant_frequency(self, data, sr):
        N = len(data)
        yf = fft(data)
        xf = fftfreq(N, 1 / sr)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])
        threshold = 0.1 * np.max(magnitude)
        significant_indices = np.where(magnitude >= threshold)[0]
        if len(significant_indices) > 0:
            highest_significant_idx = significant_indices[-1]
            return float(xf[highest_significant_idx])
        else:
            return 500.0

    def downsample_without_anti_aliasing(self, data, orig_sr, target_fs):
        if target_fs >= orig_sr:
            return data, orig_sr
        ratio = orig_sr / target_fs
        new_len = int(len(data) / ratio)
        indices = (np.arange(new_len) * ratio).astype(int)
        indices = indices[indices < len(data)]
        aliased = data[indices]
        return aliased, target_fs