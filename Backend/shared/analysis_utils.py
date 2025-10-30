"""
Analysis utilities for image and signal processing
Used by SAR, Sound, and Human applications
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import fft, fftfreq

from .config import PHYSICS_CONSTANTS


class AnalysisUtils:
    """
    Unified analysis utilities for image statistics, signal processing, and feature extraction
    """
    
    # ==================== IMAGE ANALYSIS (SAR App) ====================
    
    @staticmethod
    def compute_image_stats_and_histogram(pil_img, bins=50):
        """
        Calculate image statistics and intensity histogram - used by SAR app
        
        Args:
            pil_img: PIL Image object
            bins: number of histogram bins
            
        Returns:
            tuple: (statistics_dict, histogram_dataframe)
        """
        # Convert to grayscale for intensity analysis
        gray = pil_img.convert('L')
        intensities = np.array(gray).flatten()
        
        # Compute basic statistics
        stats = {
            'mean': round(float(np.mean(intensities)), 2),
            'median': round(float(np.median(intensities)), 2),
            'stdDev': round(float(np.std(intensities)), 2),
            'min': round(float(np.min(intensities)), 2),
            'max': round(float(np.max(intensities)), 2),
            'pixels': int(intensities.size),
            'width': pil_img.width,
            'height': pil_img.height,
            'dynamic_range': round(float(np.max(intensities) - np.min(intensities)), 2),
        }
        
        # Compute histogram
        hist_counts, bin_edges = np.histogram(intensities, bins=bins, range=(0, 255))
        histogram = pd.DataFrame({
            'intensity': 0.5 * (bin_edges[:-1] + bin_edges[1:]),  # Bin centers
            'count': hist_counts
        })
        
        return stats, histogram
    
    @staticmethod
    def apply_image_threshold(pil_img, threshold_percent):
        """
        Apply threshold filter to image - used by SAR app
        
        Args:
            pil_img: PIL Image object
            threshold_percent: threshold percentage (0-100)
            
        Returns:
            PIL.Image: thresholded image
        """
        # Calculate threshold value
        thr_value = (threshold_percent / 100.0) * 255.0
        
        # Convert to grayscale for mask creation
        gray = pil_img.convert('L')
        mask = np.array(gray) < thr_value
        
        # Apply mask to original image
        rgba_arr = np.array(pil_img.convert('RGBA'))
        rgba_arr[mask, :3] = 0  # Set RGB to black, keep alpha
        
        return Image.fromarray(rgba_arr)
    
    # ==================== SIGNAL ANALYSIS (Sound & Human Apps) ====================
    
    @staticmethod
    def compute_signal_to_noise_ratio(audio_data, noise_floor_percentile=10):
        """
        Estimate signal-to-noise ratio
        
        Args:
            audio_data: input signal
            noise_floor_percentile: percentile to estimate noise floor
            
        Returns:
            float: estimated SNR in dB
        """
        signal_power = np.mean(audio_data ** 2)
        noise_floor = np.percentile(np.abs(audio_data), noise_floor_percentile)
        noise_power = noise_floor ** 2
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    @staticmethod
    def detect_peaks(audio_data, sr, min_height_ratio=0.1, min_distance_hz=50):
        """
        Detect spectral peaks in audio signal
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            min_height_ratio: minimum peak height relative to maximum
            min_distance_hz: minimum distance between peaks in Hz
            
        Returns:
            tuple: (peak_frequencies, peak_magnitudes)
        """
        # Compute FFT
        N = len(audio_data)
        yf = fft(audio_data)
        xf = fftfreq(N, 1 / sr)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])
        
        # Find peaks
        min_height = min_height_ratio * np.max(magnitude)
        min_distance_samples = int(min_distance_hz / (sr / N))
        
        peak_indices, peak_properties = find_peaks(
            magnitude, 
            height=min_height, 
            distance=min_distance_samples
        )
        
        peak_frequencies = xf[peak_indices]
        peak_magnitudes = peak_properties['peak_heights']
        
        return peak_frequencies, peak_magnitudes
    
    @staticmethod
    def compute_spectral_centroid(audio_data, sr):
        """
        Compute spectral centroid (brightness measure)
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            
        Returns:
            float: spectral centroid in Hz
        """
        xf, magnitude = AnalysisUtils._compute_fft(audio_data, sr)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        centroid = np.sum(xf * magnitude) / np.sum(magnitude)
        return float(centroid)
    
    @staticmethod
    def compute_spectral_rolloff(audio_data, sr, rolloff_percent=0.85):
        """
        Compute spectral rolloff frequency
        
        Args:
            audio_data: input audio signal
            sr: sample rate
            rolloff_percent: rolloff percentage (0-1)
            
        Returns:
            float: rolloff frequency in Hz
        """
        xf, magnitude = AnalysisUtils._compute_fft(audio_data, sr)
        
        if len(magnitude) == 0:
            return 0.0
        
        cumulative_magnitude = np.cumsum(magnitude)
        total_magnitude = cumulative_magnitude[-1]
        
        if total_magnitude == 0:
            return 0.0
        
        target_magnitude = rolloff_percent * total_magnitude
        rolloff_index = np.where(cumulative_magnitude >= target_magnitude)[0]
        
        if len(rolloff_index) > 0:
            return float(xf[rolloff_index[0]])
        
        return float(xf[-1])
    
    # ==================== DOPPLER & VELOCITY ANALYSIS ====================
    
    @staticmethod
    def compute_doppler_shift(f_emit, relative_velocity, speed_of_sound=None):
        """
        Compute Doppler frequency shift
        
        Args:
            f_emit: emitted frequency (Hz)
            relative_velocity: relative velocity between source and observer (m/s)
            speed_of_sound: speed of sound (m/s), defaults to 343
            
        Returns:
            float: perceived frequency (Hz)
        """
        if speed_of_sound is None:
            speed_of_sound = PHYSICS_CONSTANTS["speed_of_sound"]
        
        if abs(speed_of_sound) < 1e-6:
            return f_emit
        
        f_perceived = f_emit * (speed_of_sound + relative_velocity) / speed_of_sound
        return max(20, min(20000, f_perceived))  # Clamp to audible range
    
    @staticmethod
    def estimate_velocity_from_doppler(f_emit, f_perceived, speed_of_sound=None):
        """
        Estimate relative velocity from Doppler shift
        
        Args:
            f_emit: emitted frequency (Hz)
            f_perceived: perceived frequency (Hz)
            speed_of_sound: speed of sound (m/s)
            
        Returns:
            float: relative velocity (m/s)
        """
        if speed_of_sound is None:
            speed_of_sound = PHYSICS_CONSTANTS["speed_of_sound"]
        
        if f_emit <= 0:
            return 0.0
        
        relative_velocity = speed_of_sound * (f_perceived / f_emit - 1)
        return float(relative_velocity)
    
    # ==================== UTILITY METHODS ====================
    
    @staticmethod
    def _compute_fft(audio_data, sr):
        """Internal FFT computation"""
        N = len(audio_data)
        yf = fft(audio_data)
        xf = fftfreq(N, 1 / sr)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])
        return xf, magnitude
    
    @staticmethod
    def normalize_data(data, method='minmax'):
        """
        Normalize data using specified method
        
        Args:
            data: input data array
            method: normalization method ('minmax', 'zscore', 'unit')
            
        Returns:
            numpy array: normalized data
        """
        data = np.asarray(data)
        
        if method == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max - data_min > 1e-6:
                return (data - data_min) / (data_max - data_min)
            else:
                return np.zeros_like(data)
        
        elif method == 'zscore':
            data_mean = np.mean(data)
            data_std = np.std(data)
            if data_std > 1e-6:
                return (data - data_mean) / data_std
            else:
                return np.zeros_like(data)
        
        elif method == 'unit':
            data_norm = np.linalg.norm(data)
            if data_norm > 1e-6:
                return data / data_norm
            else:
                return np.zeros_like(data)
        
        else:
            return data


# Global instance for convenience
analysis_utils = AnalysisUtils()