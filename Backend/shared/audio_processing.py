"""
Unified audio processing utilities for all applications
Handles file I/O, format conversion, resampling, and analysis
"""
import base64
import io
import numpy as np
import warnings
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import resample, resample_poly

from .config import AUDIO_CONFIG, PHYSICS_CONSTANTS

warnings.filterwarnings('ignore', category=UserWarning)


class AudioProcessor:
    """
    Comprehensive audio processing class used by SAR, Sound, Human, and Doppler apps
    """
    
    def __init__(self):
        self.max_duration = AUDIO_CONFIG["max_duration"]
        self.playback_sr = AUDIO_CONFIG["playback_sr"]
        self.speed_of_sound = PHYSICS_CONSTANTS["speed_of_sound"]
    
    # ==================== FILE I/O & FORMAT CONVERSION ====================
    
    def load_audio_from_base64(self, contents, filename=None):
        """
        Load audio from base64 upload - supports WAV files only
        
        Args:
            contents: Base64 encoded audio content
            filename: Optional filename for error messages
            
        Returns:
            tuple: (audio_data, sample_rate, error_message)
        """
        if contents is None:
            return None, None, "No content provided"
        
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Load WAV file using scipy
            audio_data, sr = self._load_wav_file(decoded)
            
            if audio_data is None:
                error_msg = f"Could not load audio file '{filename or 'unknown'}'. Please use WAV format."
                return None, None, error_msg
            
            # Convert to mono if needed
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Limit duration
            if len(audio_data) > sr * self.max_duration:
                audio_data = audio_data[:sr * self.max_duration]
            
            return audio_data, sr, None
            
        except Exception as e:
            error_msg = f"Error loading audio file: {str(e)}"
            return None, None, error_msg
    
    def _load_wav_file(self, decoded_bytes):
        """Load WAV file using scipy - supports standard WAV files"""
        try:
            buffer = io.BytesIO(decoded_bytes)
            sr, audio_data = wavfile.read(buffer)
            
            # Convert to float32 and normalize based on data type
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            return audio_data, sr
            
        except Exception as e:
            print(f"WAV loading failed: {e}")
            return None, None
    
    def make_playable_wav(self, audio, sr):
        """
        Create browser-playable audio data URL
        
        Args:
            audio: numpy array of audio data
            sr: sample rate
            
        Returns:
            str: data URL for HTML audio element
        """
        if len(audio) == 0:
            return ""
        
        # Ensure audio is properly formatted
        audio = self.normalize_audio(audio)
        
        # Handle very low sample rates for browser compatibility
        if sr < self.playback_sr:
            final_audio, final_sr = self._upsample_for_playback(audio, sr)
        else:
            final_audio, final_sr = audio, sr
        
        # Convert to WAV format (16-bit PCM)
        wav_data = np.clip(final_audio * 32767, -32767, 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, final_sr, wav_data)
        wav_bytes = buf.getvalue()
        
        # Create data URL
        b64 = base64.b64encode(wav_bytes).decode()
        return f"data:audio/wav;base64,{b64}"
    
    def _upsample_for_playback(self, audio, sr):
        """Upsample audio for browser compatibility"""
        duration_sec = len(audio) / sr
        new_len = int(duration_sec * self.playback_sr)
        
        if new_len == 0:
            return np.array([]), self.playback_sr
        
        # Use scipy's resample for better quality
        try:
            final_audio = resample(audio, new_len)
        except:
            # Fallback to simple interpolation
            final_audio = np.interp(
                np.linspace(0, len(audio)-1, new_len),
                np.arange(len(audio)),
                audio
            )
        
        return final_audio, self.playback_sr
    
    # ==================== AUDIO PROCESSING & ANALYSIS ====================
    
    def compute_fft_analysis(self, audio_data, sr):
        """
        Perform FFT analysis
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            
        Returns:
            tuple: (frequencies, magnitude_spectrum)
        """
        N = len(audio_data)
        yf = fft(audio_data)
        xf = fftfreq(N, 1 / sr)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])
        return xf, magnitude
    
    def detect_dominant_frequency(self, audio_data, sr, threshold_ratio=0.1):
        """
        Find dominant frequency in audio
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            threshold_ratio: relative threshold for significant frequencies
            
        Returns:
            float: dominant frequency in Hz
        """
        xf, magnitude = self.compute_fft_analysis(audio_data, sr)
        
        if len(magnitude) == 0:
            return 500.0  # Fallback
        
        threshold = threshold_ratio * np.max(magnitude)
        significant_indices = np.where(magnitude >= threshold)[0]
        
        if len(significant_indices) > 0:
            return float(xf[significant_indices[-1]])
        
        return 500.0
    
    def calculate_nyquist_info(self, audio_data, sr):
        """
        Calculate Nyquist rate and frequency information
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            
        Returns:
            dict: Nyquist analysis results
        """
        xf, magnitude = self.compute_fft_analysis(audio_data, sr)
        
        if len(magnitude) > 1:
            energy = magnitude[1:] ** 2
            cumulative_energy = np.cumsum(energy)
            total_energy = cumulative_energy[-1] if cumulative_energy.size > 0 else 0
            
            if total_energy > 1e-9:
                freq_99_percentile_idx = np.where(cumulative_energy >= total_energy * 0.99)[0]
                if len(freq_99_percentile_idx) > 0:
                    max_freq = xf[freq_99_percentile_idx[0] + 1]
                else:
                    max_freq = xf[-1] if len(xf) > 0 else sr / 4
            else:
                max_freq = xf[-1] if len(xf) > 0 else sr / 4
        else:
            max_freq = sr / 4
        
        max_freq = max(500, max_freq)
        nyquist_rate = PHYSICS_CONSTANTS["nyquist_multiplier"] * max_freq
        
        return {
            'max_freq': max_freq,
            'nyquist_rate': nyquist_rate,
            'original_sr': sr
        }
    
    # ==================== RESAMPLING & EFFECTS ====================
    
    def downsample_with_aliasing(self, audio_data, orig_sr, target_sr):
        """
        Manual downsampling without anti-aliasing
        
        Args:
            audio_data: input audio signal
            orig_sr: original sample rate
            target_sr: target sample rate
            
        Returns:
            tuple: (downsampled_audio, new_sample_rate)
        """
        if target_sr >= orig_sr:
            return audio_data, orig_sr
        
        ratio = orig_sr / target_sr
        new_len = int(len(audio_data) / ratio)
        indices = (np.arange(new_len) * ratio).astype(int)
        indices = indices[indices < len(audio_data)]
        aliased = audio_data[indices]
        return aliased, target_sr
    
    def resample_audio(self, audio_data, orig_sr, target_sr, method='poly'):
        """
        High-quality resampling
        
        Args:
            audio_data: input audio signal
            orig_sr: original sample rate
            target_sr: target sample rate
            method: resampling method ('poly', 'fft')
            
        Returns:
            numpy array: resampled audio
        """
        if orig_sr == target_sr:
            return audio_data
        
        if method == 'poly':
            # Polynomial resampling (good quality)
            return resample_poly(audio_data, target_sr, orig_sr)
        else:
            # FFT-based resampling (high quality)
            num_samples = int(len(audio_data) * target_sr / orig_sr)
            return resample(audio_data, num_samples)
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def normalize_audio(self, audio_data):
        """
        Normalize audio data to float32 in range [-1, 1]
        """
        audio_data = np.asarray(audio_data)
        
        if not np.issubdtype(audio_data.dtype, np.floating):
            audio_data = audio_data.astype(np.float32)
            if np.issubdtype(audio_data.dtype, np.integer):
                max_val = np.iinfo(audio_data.dtype).max
            else:
                max_val = 32767.0
            
            if max_val != 0:
                audio_data = audio_data / max_val
        
        # Ensure range [-1, 1]
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            audio_data = audio_data / max_abs
        
        return audio_data
    
    def get_audio_duration(self, audio_data, sr):
        """Get duration of audio in seconds"""
        return len(audio_data) / sr if sr > 0 else 0
    
    def analyze_audio_basic(self, audio_data, sr):
        """
        Comprehensive audio analysis
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            
        Returns:
            dict: comprehensive audio analysis
        """
        xf, magnitude = self.compute_fft_analysis(audio_data, sr)
        
        return {
            'waveform': audio_data.tolist(),
            'sample_rate': sr,
            'duration': self.get_audio_duration(audio_data, sr),
            'frequencies': xf.tolist(),
            'magnitude': magnitude.tolist(),
            'nyquist_freq': sr / 2,
            'rms': float(np.sqrt(np.mean(audio_data**2))),
            'dynamic_range': float(np.max(np.abs(audio_data)))
        }


# Global instance for convenience
audio_processor = AudioProcessor()