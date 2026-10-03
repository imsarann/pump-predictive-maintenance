import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

# Window configuration: 2048 samples with 50% overlap
WINDOW_SIZE = 2048
STEP_SIZE = 1024
SAMPLING_RATE = 12000


def find_sensor_data(mat_dictionary):
    """Extract Drive-End (DE) vibration signal. CWRU MAT keys vary by file (e.g. X097_DE_time)."""
    for key_name in mat_dictionary.keys():
        if "DE_time" in key_name:
            raw_array = mat_dictionary[key_name]
            return raw_array.flatten()
            
    print("Error: Could not find Drive-End sensor data in file!")
    return None


def calculate_features(window, sampling_rate):
    """Extract time and frequency-domain vibration features from a signal window."""
    # Time-domain metrics
    rms = np.sqrt(np.mean(window ** 2))
    peak_to_peak = np.ptp(window)
    kurt = kurtosis(window)
    skw = skew(window)
    crest_factor = (peak_to_peak / (2 * rms)) if rms > 0 else 0.0

    # Frequency-domain metrics 
    num_points = len(window)
    fft_result = np.fft.rfft(window)
    fft_magnitudes = np.abs(fft_result) / num_points
    fft_frequencies = np.fft.rfftfreq(num_points, d=1/sampling_rate)

    dominant_frequency = fft_frequencies[np.argmax(fft_magnitudes)]
    spectral_energy = np.sum(fft_magnitudes ** 2)

    return {
        "RMS": rms,
        "Peak_to_Peak": peak_to_peak,
        "Kurtosis": kurt,
        "Skewness": skw,
        "Crest_Factor": crest_factor,
        "Dominant_Freq_Hz": dominant_frequency,
        "Spectral_Energy": spectral_energy
    }


def main():
    print("Starting Feature Extraction on Real CWRU Data...")

    tasks = [
        ("raw_data/Healthy.mat", "Healthy", 0),
        ("raw_data/Inner_Race_Fault.mat", "Inner_Race_Fault", 1),
        ("raw_data/Ball_Fault.mat", "Ball_Fault", 2),
        ("raw_data/Outer_Race_Fault.mat", "Outer_Race_Fault", 3)
    ]

    all_data_rows = []

    for file_path, fault_name, label in tasks:
        print(f"\nProcessing: {fault_name} from {file_path}")

        mat_file = loadmat(file_path)
        signal = find_sensor_data(mat_file)
        total_points = len(signal)
        print(f"  • Total sensor readings: {total_points:,}")

        # Sliding window with 50% stride
        slices_created = 0
        current_start = 0

        while current_start + WINDOW_SIZE <= total_points:
            current_slice = signal[current_start : current_start + WINDOW_SIZE]
            slice_features = calculate_features(current_slice, SAMPLING_RATE)
            slice_features["Fault_Type"] = fault_name
            slice_features["Fault_Label"] = label

            all_data_rows.append(slice_features)
            current_start += STEP_SIZE
            slices_created += 1

        print(f"  • Slices extracted: {slices_created}")

    df = pd.DataFrame(all_data_rows)
    output_filename = "cwru_processed_dataset.csv"
    df.to_csv(output_filename, index=False)

    print("\n" + "=" * 50)
    print(f"SUCCESS! Created '{output_filename}' with {len(df)} total rows.")
    print("=" * 50)

    print("\nAverage Values per Fault Category:")
    summary_table = df.groupby("Fault_Type")[["RMS", "Kurtosis", "Peak_to_Peak"]].mean()
    print(summary_table)


if __name__ == "__main__":
    main()