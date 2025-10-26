# --- SEARCHABLE COMMENT: Imports ---
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

# ============================================
# --- SEARCHABLE COMMENT: Audio Processing Helpers ---
# ============================================

# --- SEARCHABLE COMMENT: Resample Audio Function (Decimation/Interpolation) ---
def resample_audio_decimation(audio_data, original_sr, target_sr):
    """
    Resamples audio using simple decimation (point sampling) for downsampling
    and basic sinc interpolation for upsampling.
    ***This version explicitly DOES NOT use an anti-aliasing filter during decimation***
    to clearly demonstrate aliasing artifacts.

    Args:
        audio_data (np.ndarray): Input audio signal.
        original_sr (int): Original sampling rate.
        target_sr (int): Desired new sampling rate.

    Returns:
        np.ndarray: Resampled audio signal.
    """
    # Ensure audio is float32 and normalized roughly to [-1, 1] for processing consistency
    if not np.issubdtype(audio_data.dtype, np.floating):
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
        # Normalize assuming integer type if max_abs is large
        if max_abs > 1.5:
            if max_abs > 0: audio_data /= 32767.0 # Assume int16 range if large float

    # If rates are the same, no resampling needed
    if int(original_sr) == int(target_sr):
        return audio_data

    ratio = original_sr / target_sr

    # --- SEARCHABLE COMMENT: Downsampling (Decimation) ---
    if target_sr < original_sr: # Downsampling
        decimation_factor = int(np.round(ratio))
        # Ensure factor is at least 1
        if decimation_factor < 1:
            decimation_factor = 1

        print(f"Decimating audio by factor {decimation_factor} (taking every {decimation_factor}th sample)")
        print(f"⚠️ WARNING: No anti-aliasing filter applied before decimation - Aliasing artifacts are expected!")

        # Perform decimation by simple slicing
        resampled = audio_data[::decimation_factor]
        return resampled
    # --- SEARCHABLE COMMENT: Upsampling (Sinc Interpolation) ---
    else: # Upsampling
        print(f"Upsampling audio from {original_sr} Hz to {target_sr} Hz using basic sinc interpolation")
        new_length = int(len(audio_data) * target_sr / original_sr)
        # Avoid creating zero length array if input is very short
        if new_length < 1:
            return np.array([], dtype=np.float32)

        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)

        # Basic windowed sinc interpolation
        resampled = np.zeros(new_length, dtype=np.float32)
        window_size = 10 # Half-width of the sinc window

        for i, new_idx in enumerate(new_indices):
            center = int(np.round(new_idx))
            start = max(0, center - window_size)
            end = min(len(audio_data), center + window_size + 1)

            # Apply sinc function weighted by Hamming window
            for j in range(start, end):
                diff = new_idx - j
                if abs(diff) < 1e-6: # Avoid division by zero at the center
                    weight = 1.0
                else:
                    sinc_val = np.sin(np.pi * diff) / (np.pi * diff)
                    # Hamming window calculation (corrected denominator)
                    denominator = (end - 1) - start
                    if denominator <= 0: denominator = 1 # Avoid division by zero/negative
                    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * (j - start) / denominator)
                    weight = sinc_val * hamming

                # Ensure index j is valid
                if 0 <= j < len(audio_data):
                     resampled[i] += audio_data[j] * weight

        return resampled

# You can add other audio utility functions here if needed
