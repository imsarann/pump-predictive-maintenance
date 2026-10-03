import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Grundfos Pump Health Monitor", page_icon="💧", layout="wide")


@st.cache_resource
def load_model_and_data():
    data = pd.read_csv("cwru_processed_dataset.csv")
    with open("best_bearing_model.pkl", "rb") as file:
        model = pickle.load(file)
    return data, model


def draw_plots(dominant_freq, rms, peak_to_peak):
    # Reconstruct representative waveform and FFT peak from extracted features for UI preview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    time_pts = np.linspace(0, 0.05, 500)
    simulated_wave = (peak_to_peak / 2) * np.sin(2 * np.pi * dominant_freq * time_pts)
    
    ax1.plot(time_pts, simulated_wave, color="#007acc")
    ax1.set_title("Vibration Waveform (Time Domain)")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Acceleration (g)")
    ax1.grid(True, alpha=0.3)

    freqs = np.linspace(0, 1500, 300)
    spectrum = np.exp(-((freqs - dominant_freq) ** 2) / (2 * 25 ** 2)) * rms

    ax2.plot(freqs, spectrum, color="#ff7f0e", lw=2)
    ax2.axvline(dominant_freq, color="red", linestyle="--", label=f"Peak: {dominant_freq:.1f} Hz")
    ax2.set_title("FFT Frequency Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Spectral Energy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    st.title("💧 Industrial Pump Predictive Maintenance Monitor")
    st.markdown("Real-time fault diagnosis using **Signal Processing (FFT)** and **XGBoost AI**.")
    st.markdown("---")

    df, model = load_model_and_data()

    st.sidebar.header("🕹️ Sensor Telemetry Controls")
    sample_index = st.sidebar.slider("Select Sensor Sample Index", 0, len(df) - 1, 10)

    selected_row = df.iloc[sample_index]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMS (Volume)", f"{selected_row['RMS']:.4f} g")
    col2.metric("Kurtosis (Spikiness)", f"{selected_row['Kurtosis']:.2f}")
    col3.metric("Peak-to-Peak", f"{selected_row['Peak_to_Peak']:.3f} g")
    col4.metric("Dominant Frequency", f"{selected_row['Dominant_Freq_Hz']:.1f} Hz")

    st.markdown("---")

    feature_cols = [
        "RMS", 
        "Peak_to_Peak", 
        "Kurtosis", 
        "Skewness", 
        "Crest_Factor", 
        "Dominant_Freq_Hz", 
        "Spectral_Energy"
    ]
    
    input_features = pd.DataFrame([selected_row[feature_cols]])
    predicted_label = model.predict(input_features)[0]
    prediction_probabilities = model.predict_proba(input_features)[0]
    confidence = prediction_probabilities[predicted_label] * 100

    # CWRU fault class mapping
    fault_names = {
        0: "Healthy (Normal Operation)",
        1: "Inner Race Bearing Fault",
        2: "Ball Defect",
        3: "Outer Race Bearing Fault"
    }
    
    predicted_name = fault_names[predicted_label]
    actual_name = selected_row["Fault_Type"]

    st.subheader("🤖 AI Diagnostic Decision")

    if predicted_label == 0:
        st.success(f"🟢 **STATUS: {predicted_name}** | Confidence: {confidence:.1f}%\n\nPump operating within ISO 10816 vibration limits. No action required.")
    elif predicted_label in [1, 2]:
        st.warning(f"🟡 **STATUS: {predicted_name}** | Confidence: {confidence:.1f}%\n\nEarly mechanical fatigue detected. Schedule inspection during upcoming maintenance window.")
    else:
        st.error(f"🔴 **STATUS: {predicted_name}** | Confidence: {confidence:.1f}%\n\nCRITICAL: Severe bearing impact shocks detected. Immediate shutdown recommended to prevent motor damage!")

    st.info(f"📋 **Verification:** Actual Ground-Truth Label from CWRU Sensor: **{actual_name}**")

    st.subheader("📊 Signal Analysis (Time Waveform & FFT Spectrum)")
    plot_figure = draw_plots(
        selected_row['Dominant_Freq_Hz'], 
        selected_row['RMS'], 
        selected_row['Peak_to_Peak']
    )
    st.pyplot(plot_figure)


if __name__ == "__main__":
    main()