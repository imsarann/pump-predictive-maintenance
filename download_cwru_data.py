import urllib.request
import os
from scipy.io import loadmat
import numpy as np

# CWRU benchmark dataset URLs (12kHz drive-end accelerometer)
CWRU_FILES = {
    "Healthy": "https://engineering.case.edu/sites/default/files/97.mat",
    "Inner_Race_Fault": "https://engineering.case.edu/sites/default/files/105.mat",
    "Ball_Fault": "https://engineering.case.edu/sites/default/files/118.mat",
    "Outer_Race_Fault": "https://engineering.case.edu/sites/default/files/130.mat"
}

os.makedirs("raw_data", exist_ok=True)

print("Downloading real CWRU bearing vibration files...")

for label, url in CWRU_FILES.items():
    destination = os.path.join("raw_data", f"{label}.mat")
    if not os.path.exists(destination):
        print(f"  Downloading {label}...")
        # CWRU server blocks requests without a browser User-Agent
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            out_file.write(response.read())
        print(f"   Saved to {destination}")
    else:
        print(f"   {label} already downloaded.")

print("\nAll real datasets downloaded! Inspecting files...")

# Sanity check MAT structure and sensor key
sample_data = loadmat("raw_data/Healthy.mat")
de_key = [k for k in sample_data.keys() if "DE_time" in k][0]
vibration_readings = sample_data[de_key].flatten()

print(f"\nInspection Report for Healthy Bearing:")
print(f"  • Variable Name: {de_key}")
print(f"  • Total Vibration Points Recorded: {len(vibration_readings):,} readings")
print(f"  • First 5 Raw Accelerometer Values (in g): {vibration_readings[:5]}")