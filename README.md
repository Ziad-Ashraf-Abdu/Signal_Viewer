# 🧠 Signal Viewer - Multi-Platform Signal Analysis  

## 🌍 Overview  

**Signal Viewer** is a comprehensive, modular Python-based web application suite built with **Dash** and **Plotly**, providing **four specialized platforms** for signal analysis, visualization, and processing.  

The system integrates cutting-edge technologies — including **deep learning models** for medical diagnostics 🩺, **interactive physics simulations** ⚛️, **advanced image processing** 🛰️, and **real-time audio classification** 🎧.  

Each application operates independently while sharing a **unified codebase architecture**, enabling seamless scalability and cross-domain data exploration.  

---

## 🧩 Applications  

### 1️⃣ Medical Signal Analysis (`Medical.py`) 🩺  

<img width="1580" height="722" alt="image" src="https://github.com/user-attachments/assets/9e57baa3-4522-4c71-ac03-3023adf41296" />  
<img width="1585" height="726" alt="image" src="https://github.com/user-attachments/assets/c881c300-3fe2-43a8-a45f-15ace629ac05" />  

**🎯 Purpose**: Real-time visualization and **AI-based classification** of medical signals including ECG and EEG for diagnostics and research.  

**✨ Key Features**:  
- 🧾 **Multi-Channel Signal Support** — Up to 16+ synchronized channels  
- 🕒 **Real-Time Playback** — Adjustable speed, pause/resume, and sliding window  
- 🎨 **Visualization Modes**:  
  - ICU Monitor 🏥  
  - XOR Comparison ⚡  
  - Polar Coordinates 🧭  
  - Cross-Recurrence Plots 🔁  
- 🤖 **AI-Powered Models**:  
  - **HuBERT-ECG** – Detects MI, LVH, Hypertrophy, Arrhythmia  
  - **BIOT EEG-PREST** – Identifies epilepsy, schizophrenia, Alzheimer’s, narcolepsy  
- 🔧 **Signal Processing** — Filtering, QRS detection, RR interval calculation  

**🧬 Supported Signals**: ECG / EEG  
**📊 Output**: Real-time interactive plots and classification predictions  

---

### ❤️ Myocardial Infarction  

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/55a9791c-51a9-48b0-9df8-ee4788489c04" width="450"></td>
    <td><img src="https://github.com/user-attachments/assets/fbe6e20e-5c19-4565-a782-75bd084962f0" width="450"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/fd03d3c1-571f-44a3-8cd7-0b5d7742b980" width="450"></td>
    <td><img src="https://github.com/user-attachments/assets/f16078be-9f50-4578-bd28-3aa7ca983a07" width="450"></td>
  </tr>
</table>

<img width="1896" height="903" alt="Screenshot 2025-10-11 202403" src="https://github.com/user-attachments/assets/12f5cc27-b9e3-4f11-aeee-b01f9a933a8f" />  

---

### 🧩 Schizophrenia  

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/304dccdd-2291-4ec5-bd8d-792c195d0751" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/8198c32f-39df-4591-888c-1f71ca057507" width="500"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9d6729aa-93f7-4cf3-9ce0-5589cc4c7553" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/3a633da6-c6af-4c35-8ba8-864d87f2e0ae" width="500"></td>
  </tr>
</table>

<img width="1883" height="911" alt="Screenshot 2025-10-11 202608" src="https://github.com/user-attachments/assets/93edf4f3-9a61-4b81-bf29-6213e43dd08c" />  

---

### 2️⃣ Doppler Effect Simulator (`doppler_app.py`) 🚗💨  

<img width="1587" height="727" alt="image" src="https://github.com/user-attachments/assets/0db1cba0-fa48-47ff-83d3-f44196fbf980" />  
<img width="1213" height="339" alt="image" src="https://github.com/user-attachments/assets/c8718ed9-4b4a-4ea7-80c8-0ad0562683b5" />  
<img width="1581" height="723" alt="image" src="https://github.com/user-attachments/assets/b9b72dec-58cd-4981-a011-3573bb5de557" />  

**🎯 Purpose**: Simulate the **Doppler effect** in real time with sound, visuals, and motion physics.  

**⚙️ Key Features**:  
- 🧠 **Interactive Simulation** — Movable sound source & observer  
- 🎵 **Audio Integration** — Upload WAV files, analyze frequency, estimate velocity  
- 🧮 **Mathematical Engine** — Relativistic Doppler equations  
- 📈 **Visualization** — Real-time trajectories & frequency display  
- 🤖 **Machine Learning** — Estimate vehicle speed from sound  

**Use Cases**: Physics education, Doppler demonstrations, acoustic analysis  

![Dash-GoogleChrome2025-10-1119-10-25-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/30a2ba43-12e8-40df-94c3-7a16775957de)  

---

### 3️⃣ SAR Data Analysis Platform (`sar_app.py`) 🛰️  

<img width="1585" height="729" alt="image" src="https://github.com/user-attachments/assets/8d11d46b-984f-405f-972d-bfd8589b757e" />  

**🎯 Purpose**: Process and analyze **Synthetic Aperture Radar (SAR)** images with detailed statistical insights.  

**📊 Key Features**:  
- 🖼️ **Image Processing** — Supports GeoTIFF, TIFF, PNG, JPG  
- 📉 **Statistical Analysis** — Mean, median, standard deviation  
- 🌈 **Visualization** — Interactive histograms and threshold filters  
- 🧭 **Feature Detection** — Identify backscatter regions & texture complexity  
- 💾 **Export** — CSV summaries of computed metrics  

**Use Cases**: Geological mapping, disaster monitoring, military reconnaissance  

<img width="1583" height="728" alt="image" src="https://github.com/user-attachments/assets/6fcffa78-d479-43f3-a09d-f623c2358026" />  
<img width="1584" height="728" alt="image" src="https://github.com/user-attachments/assets/78c64810-97ec-4b68-a8b7-59191c0ad951" />  

---

### 4️⃣ Drone Audio Detection System (`app.py`) 🚁🎙️  

<img width="1218" height="495" alt="Screenshot 2025-10-11 183244" src="https://github.com/user-attachments/assets/42216304-b75d-476e-a8b3-3e5dcf307a69" />  

**🎯 Purpose**: Detect and classify **drone sounds** in real time using deep learning.  

**🎵 Key Features**:  
- 🎧 **Audio Upload & Playback** — Drag-and-drop WAV support  
- 📈 **Waveform Visualization** — Interactive time-domain view  
- 🤖 **Deep Learning Models** — HuggingFace transformers for classification  
- ⚙️ **Model Info** — GPU acceleration, caching, threading safety  
- 🔊 **Acoustic Analysis** — Frequency spectrum and event localization  
- 🧾 **Results Display** — Confidence scores and per-chunk predictions  

<img width="1584" height="724" alt="Screenshot 2025-10-11 184942" src="https://github.com/user-attachments/assets/209c67ea-b6eb-43c0-bdd7-44db65675409" />  
<img width="1584" height="727" alt="Screenshot 2025-10-11 185006" src="https://github.com/user-attachments/assets/2c71241d-17c9-491c-89ce-47e85af0962c" />  

**📊 Model Performance**: Optimized for drone detection with cross-dataset validation  
**🧭 Use Cases**: Drone surveillance, wildlife monitoring, environmental sound detection  
**💾 Outputs**: Classification predictions, audio metrics, waveform plots  
