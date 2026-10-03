*Note: This README file was also generated with the assistance of AI.*
# Industrial Pump Predictive Maintenance (CWRU Bearing Dataset)

An end-to-end Machine Learning pipeline to detect and classify bearing damage in industrial rotating machinery (like water pumps and electric motors) using vibration sensor data.

---

## 📌 About
Industrial pumps rely on bearings to spin smoothly at thousands of RPM. When a bearing starts failing, it doesn't break instantly—it gives off tiny, high-frequency shockwaves and vibrations. 

Instead of waiting for a machine to break down (expensive) or replacing parts on a rigid calendar schedule (wasteful), this project uses **Signal Processing** and **Machine Learning** to classify bearing health into 4 categories:
1. **Healthy** (Normal operation)
2. **Inner Race Fault**
3. **Ball Defect**
4. **Outer Race Fault**

---

## ⚙️ How It Works

### 1. Real Industrial Data
I used the benchmark **Case Western Reserve University (CWRU) Bearing Dataset**, recorded from physical 2-HP electric motors using accelerometers at 12,000 samples per second.

### 2. Signal Processing (Feature Engineering)
Raw sensor streams contain hundreds of thousands of noisy numbers. Slicing the continuous signal into 2,048-sample windows (with 50% overlap), I extracted 7 key physical clues:
* **RMS (Root Mean Square):** Measures total vibration energy/volume.
* **Kurtosis:** Measures the "spikiness" and shock impacts.
* **Peak-to-Peak:** Distance between highest and lowest vibration peak.
* **Fast Fourier Transform (FFT):** Converts raw sound waves into frequency spectrums to find the dominant screech frequency in Hertz.
* **Crest Factor & Skewness.**

### 3. Machine Learning & Benchmarking
I trained and compared two algorithms on the extracted features:
* **Random Forest Classifier** (fast, highly explainable)
* **XGBoost (Extreme Gradient Boosting)** (boosted ensemble)

---

## 📊 Benchmark Results

Both models were tested on an unseen 20% test split (177 test samples):

| Model | Accuracy | F1-Score | Training Time |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **100.00%** | **1.0000** | **0.2175 sec** |
| **XGBoost** | **100.00%** | **1.0000** | **1.8375 sec** |

### Top Clues the AI Used:
* **RMS (Volume):** 22.28%
* **Spectral Energy (FFT):** 19.84%
* **Peak-to-Peak:** 17.92%
* **Dominant Frequency (FFT):** 15.42%
* **Kurtosis (Spikiness):** 13.91%

---

## 🖥️ Live Monitoring Dashboard
I built a web dashboard with **Streamlit** to simulate a plant operator's control screen:
* Lets you select any sample to inspect.
* Displays live metrics (RMS, Kurtosis, Peak-to-Peak).
* Plots the time-domain waveform and the FFT frequency spectrum.
* Shows an automated diagnostic decision:
  * 🟢 **Green:** Healthy pump, normal limits.
  * 🟡 **Yellow:** Early wear warning, schedule inspection.
  * 🔴 **Red:** Critical damage alert, immediate shutdown recommended.

*(Note: The Streamlit UI code and layout were quickly prototyped with AI assistance, while the signal processing logic, data pipeline, and model benchmarking were built and validated by me).*

---

## 🚀 How to Run It

### 1. Install dependencies
```bash
pip install numpy pandas scipy scikit-learn xgboost streamlit matplotlib
```

### 2. Download the real CWRU data
```bash
python download_cwru_data.py
```

### 3. Extract features using FFT & Signal Processing
```bash
python process_features.py
```

### 4. Train & Benchmark the Models
```bash
python train_model.py
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```text
├── raw_data/                 # Downloaded CWRU .mat files
├── cwru_processed_dataset.csv # Sliced & engineered feature table (590 samples)
├── download_cwru_data.py      # Automated script to fetch CWRU data
├── process_features.py        # Signal processing & FFT feature extraction
├── train_model.py             # RF vs XGBoost benchmark & model training
├── app.py                     # Streamlit live monitoring dashboard
└── README.md                  # Project documentation
```
