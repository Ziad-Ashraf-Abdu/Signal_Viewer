import numpy as np
from scipy.io.wavfile import write
import os


def generate_tone(frequencies, duration, samplerate=44100, filename="car_sound.wav"):
    """
    Generate a tone with one or multiple frequencies and save as a WAV file.

    :param frequencies: list of frequencies in Hz (e.g., [440, 660, 880])
    :param duration: length of the audio in seconds
    :param samplerate: samples per second (Hz)
    :param filename: output WAV file name
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None

    # Time array
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)

    # Start with silence
    waveform = np.zeros_like(t)

    # Add each frequency as a sine wave
    for freq in frequencies:
        waveform += np.sin(2 * np.pi * freq * t)

    # Normalize to avoid clipping (scale to -1 .. 1)
    waveform = waveform / len(frequencies)

    # Scale to 16-bit integer range
    waveform_integers = np.int16(waveform * 32767)

    # Save file
    write(filename, samplerate, waveform_integers)
    print(f"Generated {filename} with frequencies: {frequencies}")



generate_tone(frequencies=[400, 500, 600], duration=3, filename="car_sound.wav")
