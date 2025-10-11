# Signal Viewer - Multi-Platform Signal Analysis 

## Overview

Signal Viewer is a comprehensive, modular Python-based web application suite built with Dash and Plotly that provides four specialized platforms for signal analysis, visualization, and processing. The system integrates cutting-edge technologies including deep learning models for medical diagnostics, interactive physics simulations, advanced image processing algorithms, and real-time audio classification. Each application operates independently while sharing a unified codebase architecture.

The platform demonstrates advanced signal processing techniques, AI-powered inference, real-time visualization, and interactive data exploration across diverse domains.

---

## Applications

### 1. Medical Signal Analysis (Medical.py)

<img width="1580" height="722" alt="image" src="https://github.com/user-attachments/assets/9e57baa3-4522-4c71-ac03-3023adf41296" />

<img width="1585" height="726" alt="image" src="https://github.com/user-attachments/assets/c881c300-3fe2-43a8-a45f-15ace629ac05" />



**Purpose**: Real-time visualization and AI-based classification of medical signals including ECG (electrocardiograms) and EEG (electroencephalograms) for clinical diagnostics and research.

**Key Features**:

- **Multi-Channel Signal Support**: Display and analyze up to 16+ simultaneous signal channels with synchronized timing
- **Real-Time Playback**: Adjustable playback speeds (0.5x to 10x), pause/resume functionality, and sliding window display
- **Four Visualization Modes**:
  - ICU Monitor: Traditional clinical streaming display
  - XOR: Comparative analysis of sequential signal segments
  - Polar Coordinates: Phase relationship and amplitude cycle visualization
  - Cross-Recurrence Plots: 2D amplitude co-occurrence heatmaps

- **AI-Powered Classification**:
  - HuBERT-ECG: 12-lead ECG analysis for cardiovascular conditions (MI, LVH, Short QT Syndrome, Arrhythmia)
  - BIOT EEG-PREST: 16-channel EEG analysis for neurological condition detection (epilepsy, schizophrenia, Alzheimer's , Nacrolepsy)

- **Signal Processing**: Automated filtering, QRS detection, RR interval calculation

**Supported Signal Types**: 
- ECG (single or multi-lead)
- EEG (multi-channel with 16-channel optimization)

**Output Formats**: Interactive web interface with real-time plots, and classification predictions

![Recording 2025-10-11 195402](https://github.com/user-attachments/assets/d85412e3-07db-4655-8543-e2514be08e6f)
![Recording 2025-10-11 195927](https://github.com/user-attachments/assets/31704dbc-f738-43ec-bd16-255c77f0f0b6)
![Recording 2025-10-11 200656](https://github.com/user-attachments/assets/76ce778a-4873-44eb-abff-120298f34d3f)
![Recording 2025-10-11 200822](https://github.com/user-attachments/assets/1565259f-2ae9-4c47-a959-f96f38642579)





---

### 2. Doppler Effect Simulator (doppler_app.py)

<img width="1587" height="727" alt="image" src="https://github.com/user-attachments/assets/0db1cba0-fa48-47ff-83d3-f44196fbf980" />
<img width="1213" height="339" alt="image" src="https://github.com/user-attachments/assets/c8718ed9-4b4a-4ea7-80c8-0ad0562683b5" />
<img width="1581" height="723" alt="image" src="https://github.com/user-attachments/assets/b9b72dec-58cd-4981-a011-3573bb5de557" />



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

<img width="1225" height="335" alt="image" src="https://github.com/user-attachments/assets/8fc66c9a-7bf9-4336-8842-2384923bcbdb" />


![Dash-GoogleChrome2025-10-1119-10-25-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/30a2ba43-12e8-40df-94c3-7a16775957de)



---

### 3. SAR Data Analysis Platform (sar_app.py)

<img width="1585" height="729" alt="image" src="https://github.com/user-attachments/assets/8d11d46b-984f-405f-972d-bfd8589b757e" />

**Purpose**: Advanced processing and analysis of Synthetic Aperture Radar (SAR) imagery with statistical analysis, and feature extraction.

**Key Features**:

- **SAR Image Processing**:
  - Support for GeoTIFF, TIFF, PNG, JPG, and other raster formats
  - Intensity distribution histogram computation
  - Dynamic range and signal-to-noise ratio analysis
  - Speckle noise characterization

- **Statistical Analysis**:
  - Mean, median, standard deviation, min/max intensity values
  - Percentile calculations (1st and 99th percentile)
  - Signal quality metrics

- **Advanced Visualization**:
  - Raw SAR intensity image display
  - Interactive histogram with frequency distribution
  - Threshold Filter (Adjust the slider to highlight pixels below a certain intensity)

- **Feature Detection**:
  - Automated backscatter region identification
  - High/low intensity region classification
  - Signal variance calculation for texture complexity
  - Dynamic range assessment

- **Data Export**:
  - CSV export of statistical summaries

**Use Cases**: Geological mapping, urban planning, environmental monitoring, military reconnaissance, disaster assessment

**Output Formats**: Statistical reports, feature extractions, intensity distributions

<img width="1583" height="728" alt="image" src="https://github.com/user-attachments/assets/6fcffa78-d479-43f3-a09d-f623c2358026" />
<img width="1584" height="728" alt="image" src="https://github.com/user-attachments/assets/78c64810-97ec-4b68-a8b7-59191c0ad951" />




---

### 4. Drone Audio Detection System (app.py)

---<img width="1218" height="495" alt="Screenshot 2025-10-11 183244" src="https://github.com/user-attachments/assets/42216304-b75d-476e-a8b3-3e5dcf307a69" />

**Purpose**: Real-time acoustic event detection and classification using deep learning models trained on drone sound signatures and acoustic patterns.

**Key Features**:

- **Audio File Upload**:
  - Drag-and-drop interface for audio file loading
  - Support for audio format (WAV)
  - Automatic format conversion and resampling to 16kHz

- **Waveform Visualization**:
  - Interactive time-domain waveform display
  - Amplitude-time relationship visualization
  - Real-time audio player with playback controls

- **Deep Learning Classification**:
  - HuggingFace transformer models 
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
  - Acoustic event localization

- **Results Display**:
  - Top prediction with confidence percentage
  - All classification scores with probability distributions
  - Per-chunk analysis results

    <img width="1584" height="724" alt="Screenshot 2025-10-11 184942" src="https://github.com/user-attachments/assets/209c67ea-b6eb-43c0-bdd7-44db65675409" />

    <img width="1584" height="727" alt="Screenshot 2025-10-11 185006" src="https://github.com/user-attachments/assets/2c71241d-17c9-491c-89ce-47e85af0962c" />


**Model Performance**: Optimized for drone detection with validation on multiple acoustic datasets

**Use Cases**: Drone detection systems, acoustic surveillance, wildlife monitoring, environmental sound classification, security applications

**Output Formats**: Classification predictions, audio waveforms, acoustic metrics, confidence distributions







