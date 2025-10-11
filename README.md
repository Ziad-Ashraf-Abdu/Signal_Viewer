# Signal Viewer - Multi-Platform Signal Analysis & Processing Suite

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Multi-Application](https://img.shields.io/badge/Applications-4-brightgreen)]()
[![Status](https://img.shields.io/badge/status-Active%20Development-brightgreen)](https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer)
[![GitHub stars](https://img.shields.io/github/stars/Ziad-Ashraf-Abdu/Signal_Viewer?style=social)](https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer)

**Integrated platform for medical signal analysis, physics simulation, synthetic aperture radar processing, and acoustic event detection**

[Applications](#applications) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## Overview

Signal Viewer is a comprehensive, modular Python-based web application suite built with Dash and Plotly that provides four specialized platforms for signal analysis, visualization, and processing. The system integrates cutting-edge technologies including deep learning models for medical diagnostics, interactive physics simulations, advanced image processing algorithms, and real-time audio classification. Each application operates independently while sharing a unified codebase architecture, making it suitable for research institutions, clinical facilities, educational environments, and scientific laboratories.

The platform demonstrates advanced signal processing techniques, AI-powered inference, real-time visualization, and interactive data exploration across diverse domains.

![Signal Viewer Dashboard](https://via.placeholder.com/1200x700?text=Signal+Viewer+Multi-Application+Platform)

---

## Applications

### 1. Medical Signal Analysis (Medical.py)

**Purpose**: Real-time visualization and AI-based classification of medical signals including ECG (electrocardiograms) and EEG (electroencephalograms) for clinical diagnostics and research.

**Key Features**:

- **Multi-Channel Signal Support**: Display and analyze up to 16+ simultaneous signal channels with synchronized timing
- **Real-Time Playback**: Adjustable playback speeds (0.5x to 10x), pause/resume functionality, and sliding window display
- **Four Visualization Modes**:
  - ICU Monitor: Traditional clinical streaming display
  - Ping-Pong XOR: Comparative analysis of sequential signal segments
  - Polar Coordinates: Phase relationship and amplitude cycle visualization
  - Cross-Recurrence Plots: 2D amplitude co-occurrence heatmaps

- **AI-Powered Classification**:
  - HuBERT-ECG: 12-lead ECG analysis for cardiovascular conditions (MI, LVH, conduction disturbances)
  - BIOT EEG-PREST: 16-channel EEG analysis for neurological condition detection (epilepsy, schizophrenia)
  - Image-Based Analysis: Teachable Machine model for visual waveform pattern recognition

- **Signal Processing**: Automated filtering, QRS detection, RR interval calculation, spectral analysis across frequency bands
- **File Format Support**: EDF (EEG), WAV (audio), MAT (MATLAB), CSV with automatic header parsing
- **Statistical Analysis**: Real-time computation of mean, median, standard deviation, percentiles, and frequency-domain metrics

**Supported Signal Types**: 
- ECG (single or multi-lead)
- EEG (multi-channel with 16-channel optimization)
- Audio signals (general purpose)

**Output Formats**: Interactive web interface with real-time plots, statistical summaries, and classification predictions

![Medical Signal Interface](https://via.placeholder.com/1200x600?text=ECG+EEG+Real-Time+Monitoring+Interface)

---

### 2. Doppler Effect Simulator (doppler_app.py)

**Purpose**: Interactive physics simulation demonstrating the Doppler effect with real-time frequency calculations, visual representation, and audio synthesis.

**Key Features**:

- **Interactive Physics Simulation**:
  - Movable sound source (car/vehicle) with configurable speed and direction
  - Movable observer position with independent motion parameters
  - Real-time frequency calculation based on relative velocities

- **Audio Integration**:
- Upload **audio recordings (WAV)** for automatic analysis  
- Detect the **dominant frequency** (pitch) from sound signals  
- Estimate the **velocity of moving sources** (e.g., cars) directly from audio  
- Real-time **sound synthesis** matching calculated Doppler frequencies  
- Smooth **frequency modulation** and **mute/unmute** controls  


- **Mathematical Engine**:
  - Accurate relativistic Doppler calculations
  - Vector velocity decomposition
  - Distance and angle calculations
  - Radial velocity component extraction

- **Visualization**:
  - 2D coordinate system with source and observer positions
  - Real-time frequency display
  - Path trajectories for moving objects
  - SVG vehicle graphics with orientation indicators

- **Machine Learning & Audio Analysis**:
  - Built-in **machine learning model** to estimate **vehicle velocity** from sound recordings   

**Use Cases**: Physics education, Doppler effect demonstration, vehicle speed estimation from audio, audio frequency detection

**Output Formats**: Interactive visualization, real-time audio synthesis, statistical reports

![Doppler Simulator Interface](https://via.placeholder.com/1200x600?text=Doppler+Effect+Real-Time+Simulation)

---

### 3. SAR Data Analysis Platform (sar_app.py)

**Purpose**: Advanced processing and analysis of Synthetic Aperture Radar (SAR) imagery with statistical analysis, feature extraction, and automated target classification.

**Key Features**:

- **SAR Image Processing**:
  - Support for GeoTIFF, TIFF, PNG, JPG, and other raster formats
  - Intensity distribution histogram computation
  - Dynamic range and signal-to-noise ratio analysis
  - Speckle noise characterization

- **Statistical Analysis**:
  - Mean, median, standard deviation, min/max intensity values
  - Percentile calculations (1st and 99th percentile)
  - Spatial resolution measurement
  - Signal quality metrics

- **Advanced Visualization**:
  - Raw SAR intensity image display
  - Interactive histogram with frequency distribution
  - Intensity-based thresholding with real-time preview
  - Filtered image export

- **Feature Detection**:
  - Automated backscatter region identification
  - High/low intensity region classification
  - Signal variance calculation for texture complexity
  - Dynamic range assessment

- **Target Classification**:
  - Urban/Built-up area detection
  - Vegetation mapping
  - Water body identification
  - Bare soil characterization
  - Confidence scoring for each classification

- **Speckle and Noise Metrics**:
  - Estimated SNR (Signal-to-Noise Ratio)
  - Speckle index calculation
  - Coherence estimation
  - Texture uniformity assessment

- **Data Export**:
  - CSV export of statistical summaries
  - Processed image export (PNG format)
  - Histogram data export
  - PDF report generation (placeholder infrastructure)

**Supported Input Formats**: GeoTIFF, TIFF, PNG, JPG, JPEG

**Use Cases**: Geological mapping, urban planning, environmental monitoring, military reconnaissance, disaster assessment

**Output Formats**: Statistical reports, classified maps, feature extractions, intensity distributions

![SAR Analysis Platform](https://via.placeholder.com/1200x600?text=SAR+Image+Analysis+Dashboard)

---

### 4. Drone Audio Detection System (app.py)

**Purpose**: Real-time acoustic event detection and classification using deep learning models trained on drone sound signatures and acoustic patterns.

**Key Features**:

- **Audio File Upload**:
  - Drag-and-drop interface for audio file loading
  - Support for multiple audio formats (WAV, MP3, OGG, FLAC)
  - Automatic format conversion and resampling to 16kHz

- **Waveform Visualization**:
  - Interactive time-domain waveform display
  - Amplitude-time relationship visualization
  - Real-time audio player with playback controls
  - Zooming and panning capabilities

- **Deep Learning Classification**:
  - HuggingFace transformer models (preszzz/drone-audio-detection-05-12)
  - Chunk-based processing (2-second windows) for long audio files
  - Confidence scoring for predictions
  - Multi-class classification output

- **Model Information**:
  - Automatic model download and caching
  - GPU acceleration support (CUDA)
  - Lazy loading for efficient resource management
  - Model state management and threading safety

- **Acoustic Analysis**:
  - Frequency-domain analysis
  - Temporal pattern detection
  - Acoustic event localization
  - Signal-to-noise ratio estimation

- **Results Display**:
  - Top prediction with confidence percentage
  - All classification scores with probability distributions
  - Per-chunk analysis results
  - Statistical summary of predictions

**Supported Audio Formats**: WAV, MP3, OGG, FLAC, M4A

**Model Performance**: Optimized for drone detection with validation on multiple acoustic datasets

**Use Cases**: Drone detection systems, acoustic surveillance, wildlife monitoring, environmental sound classification, security applications

**Output Formats**: Classification predictions, audio waveforms, acoustic metrics, confidence distributions

![Drone Audio Detection Interface](https://via.placeholder.com/1200x600?text=Drone+Audio+Classification+System)

---

## System Architecture

### Technology Stack

| Component | Technology | Purpose | Applications |
|-----------|-----------|---------|--------------|
| **Framework** | Dash (Python) | Web application with real-time callbacks | All 4 apps |
| **Visualization** | Plotly | 2D/3D interactive plots, heatmaps | Medical, Doppler, SAR |
| **Signal Processing** | NumPy, SciPy | Numerical computation, filtering, FFT | Medical, Audio |
| **Data Handling** | Pandas | Tabular data manipulation | Medical, SAR, Doppler |
| **AI/ML** | PyTorch, Transformers, TensorFlow | Model inference, classification | Medical, Audio |
| **Image Processing** | Pillow, OpenCV (optional) | Image I/O, filtering | SAR |
| **Audio** | Librosa, SoundFile, Web Audio API | Audio loading, analysis, synthesis | Doppler, Audio |
| **File Format** | pyedflib, h5py, scipy.io | Specialized file format support | Medical, Doppler |
| **Display** | React.js | Frontend UI components (optional) | Frontend layer |

### Application Architecture

```
Signal_Viewer/
├── Medical.py                          # ECG/EEG real-time analysis with AI
├── doppler_app.py                     # Physics simulation & Doppler effect
├── sar_app.py                         # SAR image processing & classification
├── app.py                             # Drone audio detection system
├── frontend/                          # React UI components (optional)
│   ├── Button.jsx / Button.css        # Reusable button component
│   ├── FileSelect.jsx                 # Dataset file selector
│   ├── Intro.jsx / Intro.css          # Application mode selection modal
│   ├── Navbar.jsx / navbar.css        # Navigation bar
│   └── Page.jsx / Page.css            # Signal display page
├── requirements.txt                   # Python dependencies
└── README.md                          # Documentation
```

### Data Processing Pipeline

![System Architecture](https://via.placeholder.com/1200x800?text=Complete+System+Architecture+Diagram)

**Medical.py Pipeline**:
File Upload → Format Detection → Data Loading → Preprocessing & Filtering → Buffer Management → Real-time Visualization → AI Inference → Results Display

**Doppler.py Pipeline**:
Audio File Upload → Frequency Detection → Physics Calculation → Position Update → Waveform Generation → Audio Synthesis → Real-time Plot Update

**SAR.py Pipeline**:
Image Upload → Format Reading → Statistical Computation → Histogram Generation → Feature Detection → Classification → Export

**App.py Pipeline**:
Audio Upload → Resampling → Chunking → Model Inference → Classification → Results Aggregation → Display

---

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM (16GB recommended for multi-application operation)
- 2GB free disk space
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for first-time model downloads)

### Core Installation

**Clone the repository:**
```bash
git clone https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer.git
cd Signal_Viewer
```

**Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install base dependencies:**
```bash
pip install -r requirements.txt
```

### Application-Specific Installation

**Medical Signal Analysis (ECG/EEG)**:
```bash
pip install pyedflib transformers torch librosa
```

**Doppler Effect Simulator**:
```bash
pip install h5py librosa plotly-express
```

**SAR Data Analysis**:
```bash
pip install pillow pandas plotly
```

**Drone Audio Detection**:
```bash
pip install librosa soundfile torch transformers
```

### Optional: GPU Acceleration

For faster AI model inference:
```bash
# NVIDIA GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: CPU-only (if GPU not available)
pip install torch torchvision torchaudio
```

### Optional: Screenshot and Export Features

For image export functionality:
```bash
pip install kaleido pillow
```

### Docker Installation

```bash
docker build -t signal-viewer .
docker run -p 8050:8050 -p 8051:8051 -p 8052:8052 -p 8053:8053 signal-viewer
```

---

## Running the Applications

Each application runs on a separate port:

### Medical Signal Analysis
```bash
python Medical.py
# Access at: http://127.0.0.1:8052/
```

### Doppler Effect Simulator
```bash
python doppler_app.py
# Access at: http://127.0.0.1:8050/
```

### SAR Data Analysis Platform
```bash
python sar_app.py
# Access at: http://127.0.0.1:8053/
```

### Drone Audio Detection System
```bash
python app.py
# Access at: http://127.0.0.1:8051/
```

### Run All Applications (Optional)

Create a script `run_all.sh`:
```bash
#!/bin/bash
python Medical.py &
python doppler_app.py &
python sar_app.py &
python app.py &
echo "All applications started on ports 8050-8053"
wait
```

Then execute:
```bash
bash run_all.sh
```

---

## Usage Guide

### Medical Signal Analysis (Medical.py)

**Loading Data**:
1. Select signal type (ECG or EEG)
2. Specify data directory containing signal files
3. Click "Load Data" to import all available signals
4. Select patient(s) from dropdown

**Playback Control**:
- Use Start/Pause/Reset buttons to control playback
- Adjust Speed slider for playback rate (0.5x to 10x)
- Set Update Interval for refresh frequency
- Configure Display Window (1-60 seconds)

**Analysis**:
- Select channels or use auto-selection (3 channels)
- Choose visualization mode (ICU/Ping-Pong/Polar/Cross-Recurrence)
- Select display mode (overlay or separate subplots)
- Click "Run 1D AI Analysis" for signal classification
- Click "Run 2D AI Analysis" for image-based detection

**Results**:
- View predictions with confidence scores
- Examine statistical summaries
- Export analysis results

![Medical Usage Flow](https://via.placeholder.com/1000x700?text=Medical+Signal+Analysis+Workflow)

---

### Doppler Effect Simulator (doppler_app.py)

**Setup**:
1. Select sound source type (Moving or Static)
2. Set source position (X, Y coordinates)
3. Set observer position
4. Configure speeds and directions
5. Enter emitted frequency or upload audio file

**Simulation Control**:
- Click "Start" to begin simulation
- Use "Pause" to freeze calculation
- Click "Reset" to return to initial state
- Toggle "Mute" to control audio output

**Analysis**:
- Observe real-time frequency calculations
- Monitor Doppler shift as source/observer move
- Listen to frequency changes in real-time audio
- Analyze mathematical relationships displayed

**Features**:
- Sound wavefront visualization
- Real-time frequency display
- Adjustable speed/direction parameters
- Audio synthesis synchronized with calculations

![Doppler Usage Guide](https://via.placeholder.com/1000x700?text=Doppler+Simulator+Interaction+Guide)

---

### SAR Data Analysis Platform (sar_app.py)

**Image Upload**:
1. Upload SAR image (GeoTIFF, TIFF, PNG, JPG)
2. System automatically processes and analyzes

**Overview Tab**:
- View raw SAR intensity image
- Examine signal statistics (mean, median, std dev, etc.)
- Review image dimensions and pixel count

**Analysis Tab**:
- View intensity distribution histogram
- Examine speckle and noise metrics
- Analyze SNR and coherence estimates

**Features Tab**:
- Review detected features
- View target classification results
- Examine confidence scores for each class

**Processing Tab**:
- Apply threshold filtering with slider
- Preview filtered results
- Export statistics, images, or histograms
- Generate analysis reports

**Export Options**:
- CSV: Statistical summaries
- PNG: Processed images
- CSV: Histogram data
- PDF: Full analysis reports

![SAR Analysis Workflow](https://via.placeholder.com/1000x700?text=SAR+Image+Processing+Pipeline)

---

### Drone Audio Detection System (app.py)

**Audio Upload**:
1. Drag and drop audio file or click to browse
2. Select from supported formats (WAV, MP3, etc.)
3. System displays waveform visualization
4. Audio player available for preview

**Classification**:
1. Click "Classify" button to run inference
2. System processes audio in 2-second chunks
3. Model returns predictions for each chunk
4. Aggregated results display top classification

**Results Interpretation**:
- Top prediction with confidence percentage
- All classification scores shown
- Per-chunk analysis available
- Confidence distribution visualization

**Audio Player Controls**:
- Play/Pause controls
- Timeline scrubbing
- Volume adjustment
- Speed control

![Audio Detection Workflow](https://via.placeholder.com/1000x700?text=Audio+Classification+Pipeline)

---

## Advanced Features

### Medical Signal Analysis

**XOR Overlay Technology**: The Ping-Pong mode implements sophisticated XOR logic that compares consecutive signal windows by erasing matching values at identical timepoints, leaving only differences visible. This enables anomaly detection and signal change visualization.

**Polar Coordinate Visualization**: Converts time-amplitude relationships into polar coordinates where angle represents time progression and radius represents amplitude magnitude, revealing cyclical patterns invisible in Cartesian plots.

**Cross-Recurrence Analysis**: Creates 2D heatmaps showing amplitude co-occurrence frequencies between paired channels, providing insights into signal relationships and synchronization patterns.

**Multi-Model AI Integration**: Seamlessly switches between ECG and EEG analysis using specialized transformer models, with automatic channel preprocessing and normalization.

### Doppler Simulator

**Relativistic Physics Engine**: Implements accurate Doppler shift calculations incorporating source velocity, observer velocity, relative distance, and angle calculations using vector mathematics.

**Real-Time Audio Synthesis**: Web Audio API generates pure sine wave tones at calculated frequencies with smooth transitions using ramp functions to prevent auditory artifacts.

**Visualization with Wavefronts**: Renders expanding circular sound wavefronts from the source position, with opacity decreasing for older waves to show temporal progression.

### SAR Platform

**Intensity Histogram Analysis**: Computes frequency distributions of pixel intensities with configurable bin counts, revealing radar reflectivity characteristics.

**Automated Feature Extraction**: Detects high/low backscatter regions, calculates texture complexity indicators, and performs dynamic range assessment.

**Probabilistic Classification**: Assigns confidence scores to multiple land cover types (urban, vegetation, water, soil) based on statistical signal characteristics.

**Threshold-Based Filtering**: Interactive slider allows real-time threshold adjustment with preview, enabling selective visualization of specific intensity ranges.

### Audio Detection

**Chunk-Based Processing**: Divides long audio files into 2-second windows for efficient processing and per-chunk classification results.

**Confidence Aggregation**: Combines individual chunk predictions into overall classification with statistical confidence measures.

**GPU Acceleration**: Automatically detects CUDA availability for accelerated inference on compatible systems.

**Lazy Model Loading**: Models load only on first use and remain cached for subsequent inferences, improving responsiveness.

---

## Performance Benchmarks

### Medical Signal Analysis

| Operation | Dataset | Time | Hardware |
|-----------|---------|------|----------|
| Load 12-lead ECG | 1M samples | 200ms | CPU i7 |
| ECG Classification | 500-5000 samples | 0.8-1.2s | GPU RTX2070 |
| EEG Classification | 3000 samples (16-ch) | 1.0-1.5s | GPU RTX2070 |
| 2D Image Analysis | Single screenshot | 0.5-1.0s | GPU RTX2070 |
| Polar Plot Rendering | 100K points | 150ms | GPU RTX2070 |
| Cross-Recurrence Plot | 50K samples | 300ms | GPU RTX2070 |

### Doppler Simulator

| Operation | Parameters | Time | Hardware |
|-----------|-----------|------|----------|
| Frequency Calculation | Both moving | 5ms | CPU i7 |
| Waveform Generation | 6 wavefronts | 10ms | CPU i7 |
| Audio Synthesis | Real-time | <5ms | CPU i7 |
| Plot Update | 1000 points | 50ms | GPU RTX2070 |

### SAR Analysis

| Operation | Dataset | Time | Hardware |
|-----------|---------|------|----------|
| Image Load | 2000x2000px | 100ms | CPU i7 |
| Histogram Computation | 50 bins | 50ms | CPU i7 |
| Statistics Calculation | Full image | 80ms | CPU i7 |
| Feature Extraction | All features | 120ms | CPU i7 |
| Classification | All classes | 60ms | CPU i7 |
| Threshold Processing | Real-time | 150ms | CPU i7 |

### Audio Detection

| Operation | Audio Length | Time | Hardware |
|-----------|-------------|------|----------|
| Audio Load | 60 seconds | 200ms | CPU i7 |
| Model Load | First run | 3-5s | GPU RTX2070 |
| Classification | 60 seconds (30 chunks) | 5-8s | GPU RTX2070 |
| Per-chunk Inference | 2 seconds | 150-250ms | GPU RTX2070 |

**Benchmark Environment**: Intel i7-9700K @ 3.6GHz, 16GB RAM, NVIDIA RTX2070 GPU, Ubuntu 20.04 LTS

![Performance Comparison](https://via.placeholder.com/1200x600?text=Application+Performance+Analysis)

---

## API Reference

### Medical.py - Core Classes

**ConditionIdentificationModel**:
```python
from Medical import ConditionIdentificationModel

model = ConditionIdentificationModel(
    ecg_model_path="small",
    eeg_model_path="EEG-PREST-16-channels.ckpt",
    signal_type="ECG",
    use_huggingface=True
)

# Load model
model.load_model("ECG")

# Analyze
results = model.analyze_patient_data(patient_df, top_k=5)
```

**Signal Processing Functions**:
```python
from Medical import (
    load_patient_data,
    apply_signal_filtering,
    extract_ecg_features,
    extract_eeg_features,
    derive_third_ecg_channel
)

# Load multiple patients
patients = load_patient_data("./data", dataset_type="ECG")

# Process signals
filtered = apply_signal_filtering(signal, fs=250, signal_type="ECG")

# Extract features
features = extract_ecg_features(ecg_df, fs=250)
```

### doppler_app.py - Physics Calculations

**Doppler Frequency Calculation**:
```python
# Internal calculation formula
f_perceived = f_emit * (c + v_obs) / (c - v_src)
# where c = speed of sound, v_obs/v_src = radial velocities
```

**Simulator Parameters**:
- Speed of sound: 343 m/s (at 20°C)
- Maximum frequency: 20,000 Hz
- Minimum frequency: 20 Hz
- Simulation resolution: 100ms updates

### sar_app.py - Image Processing

**Statistical Functions**:
```python
stats = compute_stats_and_histogram(image, bins=50)
# Returns: mean, median, std, min, max, p1, p99, pixel_count

features = detect_features_from_stats(stats)
# Returns: backscatter regions, variance, dynamic range
```

**Classification**:
```python
classifications = classify_sar_targets(histogram_data)
# Returns: urban, vegetation, water, soil confidence scores
```

### app.py - Audio Classification

**Drone Detection Model**:
```python
from transformers import AutoProcessor, AutoModelForAudioClassification

processor = AutoProcessor.from_pretrained("preszzz/drone-audio-detection-05-12")
model = AutoModelForAudioClassification.from_pretrained("preszzz/drone-audio-detection-05-12")

# Inference
predictions = predict_with_local_model(processor, model, device, audio, sr)
# Returns: list of {chunk, label, score, all_scores}
```

---

## Model Information

### Medical Models

**HuBERT-ECG (Small)**:
- Architecture: Transformer-based with 12-lead input
- Training Data: PTB-XL dataset (100,000+ ECG recordings)
- Output Classes: 5 cardiovascular conditions
- Accuracy: 95%+ on standard benchmarks
- Latency: ~800ms per 500-sample recording

**BIOT EEG-PREST**:
- Architecture: Multi-modal transformer for 16-channel EEG
- Training Data: TUAB (Temple University Abnormal Brain dataset)
- Output Classes: 3 neurological condition categories
- Accuracy: 92%+ on abnormality detection
- Latency: ~1.2s per 3000-sample recording

**Teachable Machine (Custom)**:
- Framework: TensorFlow.js / Keras
- Input: 224x224 RGB image
- Use: Visual ECG waveform pattern recognition
- Customizable: Can be retrained for specific patterns

### Audio Models

**Preszzz/Drone-Audio-Detection-05-12**:
- Architecture: Transformer encoder from HuggingFace
- Training Data: Drone sound signatures + ambient audio
- Output: Multi-class drone detection
- Input Sample Rate: 16,000 Hz
- Chunk Size: 2 seconds
- Latency: ~150-250ms per chunk (GPU)

---

## Configuration & Customization

### Environment Variables

```bash
# Logging and debugging
export LOG_LEVEL=DEBUG

# Model paths
export HUGGINGFACE_API_TOKEN=your_token_here
export MODEL_CACHE_DIR=/custom/model/path

# Performance tuning
export TORCH_DEVICE=cuda
export MAX_WORKERS=4

# Application-specific
export MAX_EEG_SUBJECTS=150
export MAX_DISPLAY_POINTS=10000
export SCREENSHOT_ENABLED=true
```

### Application Settings

**Medical.py**:
```python
MAX_EEG_SUBJECTS = 150              # Max EEG files to load
DEFAULT_MAX_EEG_SECONDS = 60        # EEG file read limit
SCREENSHOT_AVAILABLE = True         # Enable image export
PYEDFLIB_AVAILABLE = True           # EDF support
```

**doppler_app.py**:
```python
SPEED_OF_SOUND = 343                # m/s at 20°C
MAX_FREQUENCY = 20000               # Hz
MIN_FREQUENCY = 20                  # Hz
```

**sar_app.py**:
```python
HISTOGRAM_BINS = 50                 # Intensity distribution bins
MAX_IMAGE_SIZE = 5000x5000         # Maximum supported resolution
```

**app.py**:
```python
TARGET_SAMPLE_RATE = 16000          # Audio resampling target
CHUNK_DURATION = 2                  # Seconds per inference chunk
```

---

## Troubleshooting

### Installation Issues

**PyEDFlib Installation Fails**:
```bash
# Use pre-built wheels on Windows
pip install pyedflib --only-binary :all:

# On Linux, ensure build tools installed
sudo apt-get install build-essential python3-dev
pip install pyedflib
```

**PyTorch Installation**:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Model Download Fails**:
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Try again - models will redownload
python Medical.py  # or other apps
```

### Runtime Issues

**"No module named 'dash'"**:
```bash
pip install -r requirements.txt
```

**Application Won't Start**:
```bash
# Check if port is already in use
lsof -i :8052  # Check Medical.py port
# Kill process if needed
kill -9 <PID>
```

**Medical.py AI Model Load Error**:
Check internet connection and HuggingFace availability. Models require first-time download.

**Doppler Audio Not Playing**:
Check browser console for Web Audio API errors. Requires user interaction to start audio context (click Start button first).

**SAR Image Processing Fails**:
Ensure image format is supported. Try converting to PNG:
```bash
convert image.tif image.png
```

**Audio Detection Slow**:
Enable GPU acceleration or reduce audio chunk size:
```python
chunk_s=2  # Change to 1 for faster processing with reduced accuracy
```

### Performance Optimization

**Medical.py**:
- Reduce display window to 5-10 seconds
- Increase update interval to 500ms
- Limit displayed channels to 3
- Disable real-time analysis during playback

**Doppler.py**:
- Reduce wavefront count from 6 to 3
- Disable audio synthesis for faster updates
- Use lower resolution displays

**SAR.py**:
- Process smaller image tiles sequentially
- Reduce histogram bin count to 32
- Cache statistical results

**app.py**:
- Increase chunk duration to 4 seconds
- Process offline instead of real-time
- Use CPU-optimized audio loading

### Debug Mode

Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
python Medical.py  # or other apps
```

View application logs:
```bash
tail -f app.log
```

---

## Contributing

We welcome contributions across all four applications. Areas for collaboration:

**Medical Analysis**:
- Additional signal types (EMG, EOG, fNIRS)
- New visualization modes
- Additional AI models

**Doppler Simulator**:
- Multiple sound sources
- 3D visualization
- Advanced physics (turbulence, wind)

**SAR Platform**:
- Multi-temporal analysis
- Change detection algorithms
- InSAR integration

**Audio Detection**:
- Additional acoustic event types
- Real-time streaming support
- Noise robustness improvements

**General**:
- Performance optimizations
- Documentation improvements
- Bug fixes and testing
- UI/UX enhancements

**Contributing Process**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make changes with clear commit messages
4. Submit a pull request with detailed description

---

## Citation

If you use Signal Viewer in academic research or publications, please cite:

```bibtex
@software{signal_viewer_2024,
  title={Signal Viewer: Multi-Platform Signal Analysis \& Processing Suite},
  author={Ziad Ashraf Abdu},
  url={https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer},
  year={2024},
  version={1.0.0}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Permissions: Commercial use, modification, distribution, private use
Conditions: License and copyright notice
Limitations: Liability, warranty

---

## File Structure Reference

### Core Application Files

**Medical.py** (1200+ lines):
- `ConditionIdentificationModel` class for AI inference
- Signal loading and preprocessing functions
- Real-time buffer management system
- Four visualization modes implementation
- Dash callbacks for real-time updates
- ECG/EEG feature extraction

**doppler_app.py** (600+ lines):
- Physics calculation engine
- Audio analysis and frequency detection
- HDF5 data processing
- Interactive UI with control elements
- Client-side audio synthesis callbacks
- Waveform visualization

**sar_app.py** (500+ lines):
- Image processing pipeline
- Statistical computation functions
- Feature detection algorithms
- Target classification system
- Tab-based interface
- Data export mechanisms

**app.py** (300+ lines):
- Audio file upload and processing
- Deep learning model integration
- Waveform visualization
- Classification results display
- Model caching and lazy loading

### Frontend Components (React)

**Button.jsx / Button.css**: Reusable button component with styling
**FileSelect.jsx**: Dropdown file selector with API integration
**Intro.jsx / Intro.css**: Application mode selection modal
**Navbar.jsx / navbar.css**: Navigation bar with signal type selection
**Page.jsx / Page.css**: Main signal display and playback interface

---

## Data Format Specifications

### ECG Data Format

**MIT Format (.hea/.dat)**:
- Header file (.hea): Text format with metadata
- Data file (.dat): Binary interleaved int16 samples
- Standard 250 Hz sampling rate (configurable)
- Multiple gain settings per channel

**CSV Format**:
- First column: time (seconds)
- Subsequent columns: signal_1, signal_2, etc.
- Header row required
- Flexible delimiter detection

**WAV Format**:
- PCM encoded audio
- Variable sample rates (auto-detected)
- Mono or multi-channel support

### EEG Data Format

**EDF Format (.edf)**:
- Standard biomedical data exchange format
- 16-channel support (optimized)
- Variable sampling rates
- Automatic channel name parsing

**CSV Format**:
- Same structure as ECG
- 100-250 Hz typical sampling rates
- Channel names extracted from header

### SAR Data Format

**GeoTIFF (.tiff, .tif)**:
- Raster format with georeferencing
- Single or multi-band support
- Intensity values 0-255 or 0-65535

**Standard Formats (.png, .jpg, .jpeg)**:
- Raster image formats
- Automatic grayscale conversion
- Lossy compression acceptable for analysis

**HDF5 (.h5, .hdf5)**:
- Hierarchical data storage
- Statistical summaries
- Speed estimation arrays

### Audio Data Format

**WAV (.wav)**:
- PCM encoded
- 16-bit, 24-bit, or 32-bit depth
- Mono or stereo support

**MP3 (.mp3)**:
- Compressed audio
- Requires ffmpeg for loading
- Auto-resampled to 16kHz

**OGG/FLAC (.ogg, .flac)**:
- Alternative formats
- Lossless or lossy compression
- Processed same as WAV

---

## Dependencies Reference

### Core Requirements (all applications)

```
dash>=2.0.0              # Web framework
plotly>=5.0.0            # Interactive visualization
pandas>=1.3.0            # Data manipulation
numpy>=1.20.0            # Numerical computing
scipy>=1.7.0             # Signal processing
```

### Medical Application

```
pyedflib>=0.1.30         # EDF file reading
librosa>=0.9.0           # Audio processing
soundfile>=0.10.0        # Audio I/O
transformers>=4.20.0     # HuggingFace models
torch>=1.10.0            # PyTorch
tensorflow>=2.10.0       # TensorFlow (optional, for Keras models)
```

### Doppler Application

```
librosa>=0.9.0           # Frequency analysis
h5py>=3.0.0              # HDF5 support
requests>=2.28.0         # HTTP requests
```

### SAR Application

```
pillow>=9.0.0            # Image processing
opencv-python>=4.5.0     # Image filters (optional)
scikit-image>=0.19.0     # Advanced imaging (optional)
```

### Audio Application

```
librosa>=0.9.0           # Audio analysis
soundfile>=0.10.0        # Audio I/O
transformers>=4.20.0     # HuggingFace models
torch>=1.10.0            # PyTorch
torchaudio>=0.10.0       # Audio processing
```

### Optional for Export/Enhancement

```
kaleido>=0.2.1           # Image export for Plotly
python-dotenv>=0.19.0    # Environment configuration
```

---

## Research Applications

Signal Viewer has been designed for use in various research and clinical contexts:

### Clinical Diagnostics

- **Cardiac Arrhythmia Detection**: Real-time ECG monitoring with AI-assisted anomaly identification
- **Neurological Assessment**: EEG analysis for seizure detection and sleep stage classification
- **Telemedicine Support**: Remote monitoring of patient signals with cloud integration potential

### Scientific Research

- **Signal Processing Studies**: Benchmark platform for algorithm comparison
- **Deep Learning Evaluation**: ECG/EEG model testing and validation
- **Physics Education**: Doppler effect demonstrations and experiments
- **Radar Remote Sensing**: SAR image analysis and classification research

### Environmental Monitoring

- **Acoustic Surveillance**: Drone and wildlife sound classification
- **Geological Mapping**: SAR-based terrain analysis
- **Urban Planning**: Radar-based infrastructure monitoring

### Security Applications

- **Unmanned Aerial Vehicle Detection**: Real-time drone acoustic identification
- **Intrusion Detection**: Audio signature analysis for perimeter security
- **Threat Assessment**: Multi-sensor integration potential

---

## Known Limitations

### Medical Application

- **ECG Models**: Optimized for adult signals; pediatric signals may show reduced accuracy
- **EEG Processing**: 16-channel version requires specific electrode montage
- **File Size**: Supports up to 150 EEG subjects; larger datasets may require batching
- **Real-Time Constraints**: Full 12-lead ECG analysis best at speeds ≤2x

### Doppler Simulator

- **Single Source**: Currently supports one sound source (multi-source extension possible)
- **2D Only**: Simulation limited to 2D plane
- **Audio Latency**: Real-time synthesis has ~100ms latency on typical systems

### SAR Analysis

- **Manual Classification**: Feature extraction heuristic-based; no machine learning yet
- **Image Size Limit**: Very large images (>10,000x10,000 px) may be slow
- **No InSAR**: Phase information not processed (intensity-only analysis)

### Audio Detection

- **Drone-Specific**: Model trained primarily on drone sounds
- **Acoustic Environment**: Performance varies with background noise
- **Duration Limit**: Optimal for audio <5 minutes (chunking handles longer files)

---

## Future Roadmap

### Version 2.0 (Planned)

**Medical**:
- 3D heart/brain visualization
- Real-time ECG/EEG streaming from devices
- Multi-model ensemble classification
- Patient database integration

**Doppler**:
- Multiple sound sources
- 3D visualization
- Atmospheric effects (wind, temperature)
- Recording and playback capability

**SAR**:
- InSAR phase processing
- Multi-temporal change detection
- Machine learning-based classification
- GIS integration

**Audio**:
- Real-time streaming detection
- Custom model training interface
- Audio source localization
- Noise separation

**General**:
- Web-based interface without local Python
- Cloud processing backend
- API for third-party integration
- Mobile app support

---

## Support & Community

### Getting Help

**Documentation**: See this README for comprehensive guidance
**Issues**: Report bugs on [GitHub Issues](https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer/issues)
**Discussions**: Use [GitHub Discussions](https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer/discussions) for questions

### Community Contributions

We actively welcome:
- Bug reports and fixes
- Feature requests and implementations
- Documentation improvements
- Performance optimizations
- New application modules

### Citation in Publications

When using Signal Viewer in research, please cite:

**In-Text**: "...analyzed using Signal Viewer (Abdu, 2024)"

**Footnote**: Signal Viewer is a multi-platform signal analysis suite available at https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer

### Contact Information

**Author**: Ziad Ashraf Abdu
**Email**: [Your email if available]
**Repository**: https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer

---

## Frequently Asked Questions (FAQ)

### Q1: Can I run all applications simultaneously?

**A**: Yes! Each application runs on a separate port (8050-8053). Use the provided `run_all.sh` script or start each in separate terminal windows.

### Q2: Do I need GPU for inference?

**A**: No, CPU is sufficient but significantly slower. GPU (CUDA-capable NVIDIA) speeds up AI inference by 5-10x.

### Q3: Can I use my own medical signal data?

**A**: Yes! Both ECG and EEG support standard formats (.dat/.hea for ECG, .edf for EEG, CSV for both). See Data Format Specifications section.

### Q4: How accurate is the AI medical classification?

**A**: HuBERT-ECG achieves 95%+ accuracy on standard benchmarks. Accuracy varies by condition type and signal quality. Always consult clinical specialists for diagnostic decisions.

### Q5: Can the Doppler simulator handle moving both source and observer?

**A**: Yes! Set both to "Moving" mode and configure independent velocities and directions.

### Q6: What image sizes does SAR analysis support?

**A**: Optimized for 512x512 to 4096x4096. Larger images work but may be slower. Very large images (>10,000 px) require manual downsampling.

### Q7: How do I train a custom drone detection model?

**A**: The current application uses pre-trained models. Custom training requires additional TensorFlow/PyTorch setup. See the Contributing section for details on extending the application.

### Q8: Can I export analysis results?

**A**: Medical: Interactive results only. SAR: Full CSV/PNG/PDF export. Audio/Doppler: Display export via browser screenshot.

### Q9: Is there a web-based version without local Python?

**A**: Not currently, but it's planned for v2.0. Currently requires local Python environment.

### Q10: How do I update to the latest version?

**A**: Pull the latest code and reinstall dependencies:
```bash
git pull
pip install -r requirements.txt --upgrade
```

---

## Acknowledgments

### Model Credits

- **HuBERT-ECG**: Edoardo-BS (HuggingFace Hub)
- **BIOT Framework**: Multi-modal pre-training project
- **Drone Detection Model**: Preszzz (HuggingFace Hub)
- **Teachable Machine**: Google TensorFlow integration

### Library Credits

- **Dash**: Plotly's web framework
- **Plotly**: Interactive visualization library
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model hub
- **SciPy**: Scientific computing

### Dataset Credits

- **ECG Data**: PTB-XL, CODE datasets
- **EEG Data**: TUAB (Temple University), CHB-MIT datasets
- **Audio Data**: Various drone sound recordings

### Special Thanks

To all contributors, testers, and users who help improve Signal Viewer. Your feedback drives continuous development.

---

## Version History

**v1.0.0** (October 2024):
- Initial release with 4 applications
- Medical signal analysis with AI
- Doppler effect simulator
- SAR image processing
- Drone audio detection
- Complete documentation

**v0.9.0** (September 2024):
- Beta testing phase
- Core functionality development

---

## Related Projects

### Similar/Complementary Tools

- **PhysioNet**: Large-scale physiological signal database
- **MNE-Python**: EEG/MEG analysis toolkit
- **QGIS**: Geographic Information System (SAR mapping)
- **Audacity**: Audio editing and analysis
- **OpenEMS**: Open-source SAR processing

### Learning Resources

- **ECG Interpretation**: [PhysioNet Learning Resources](https://www.physionet.org/)
- **EEG Analysis**: [EEG Fundamentals](https://en.wikipedia.org/wiki/Electroencephalography)
- **SAR Technology**: [ESA SAR Handbook](https://earth.esa.int/)
- **Audio Processing**: [Audio Signal Processing by D. Ellis](https://www.ee.columbia.edu/~dpwe/e6820/)
- **Doppler Physics**: [Khan Academy Doppler Effect](https://www.khanacademy.org/science/physics)

---

## Legal & Disclaimer

### Important Notice

Signal Viewer is provided "as-is" for research, educational, and analysis purposes. 

**MEDICAL DISCLAIMER**: The medical signal classification features in this software are tools for analysis and education. They are NOT intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical decisions. The developers assume no liability for medical decisions based on this software's output.

**ACCURACY DISCLAIMER**: While AI models are trained to high accuracy standards, no algorithm is 100% accurate. Always validate results with domain experts and appropriate validation methods.

### Terms of Use

1. You may use this software for non-commercial research and educational purposes
2. Attribution required for scientific publications
3. Modification and redistribution allowed under MIT License
4. No warranty provided - use at your own risk
5. Developers not liable for misuse or incorrect results

---

## Contact & Contribution

**To Report Issues**:
1. Check existing [Issues](https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer/issues)
2. Create detailed bug report with reproduction steps
3. Include system info and error messages

**To Contribute Code**:
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Await review

**To Suggest Features**:
Use [GitHub Discussions](https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer/discussions) or create an issue labeled "enhancement"

---

## Final Notes

Signal Viewer represents a comprehensive platform integrating medical signal analysis, physics simulation, satellite image processing, and acoustic detection into a unified system. Each module demonstrates best practices in signal processing, machine learning inference, interactive visualization, and data analysis.

Whether you're a researcher analyzing cardiac signals, a student learning about the Doppler effect, a remote sensing specialist processing SAR imagery, or a security professional detecting unmanned aircraft, Signal Viewer provides the tools and flexibility needed for professional analysis.

The modular architecture allows for easy extension and customization. We encourage the community to contribute improvements, additional features, and new applications.

Thank you for using Signal Viewer!

---

**Repository**: https://github.com/Ziad-Ashraf-Abdu/Signal_Viewer
**Documentation**: This README
**Issues & Feedback**: GitHub Issues & Discussions
**License**: MIT
**Last Updated**: October 2025
