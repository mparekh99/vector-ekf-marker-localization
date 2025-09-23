import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === CONFIG ===
folder_path = '.'  # Folder where your CSVs are
file_pattern = os.path.join(folder_path, '*.csv')

# === Find and Load CSVs ===
csv_files = glob.glob(file_pattern)
if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in the folder.")

print(f"📂 Found CSV files: {csv_files}\n")

results = []

# === Extract ΔY (forward distance) for each trial ===
for file in csv_files:
    df = pd.read_csv(file)

    if len(df) < 2:
        print(f"⚠️ Skipping {file} (not enough data)")
        continue

    # Extract start and end
    start_odom_y = df.iloc[0]['odom_y']
    end_odom_y = df.iloc[-1]['odom_y']
    delta_odom_y = end_odom_y - start_odom_y

    start_cam_y = df.iloc[0]['cam_y']
    end_cam_y = df.iloc[-1]['cam_y']
    delta_cam_y = end_cam_y - start_cam_y

    error = delta_cam_y - delta_odom_y

    results.append({
        'file': os.path.basename(file),
        'delta_odom_y': delta_odom_y,
        'delta_cam_y': delta_cam_y,
        'error': error
    })

# === Build Results DataFrame ===
df_results = pd.DataFrame(results)
df_results.sort_values(by='file', inplace=True)

# === Plot Bar Chart ===
x = np.arange(len(df_results))
width = 0.3

plt.figure(figsize=(12, 6))
plt.bar(x - width, df_results['delta_odom_y'], width, label='Odometry ΔY', color='gray')
plt.bar(x, df_results['delta_cam_y'], width, label='Camera ΔY', color='blue')
plt.bar(x + width, df_results['error'], width, label='Error (Cam - Odom)', color='red')

plt.xticks(x, df_results['file'], rotation=45)
plt.ylabel("Distance (mm)")
plt.title("Camera vs Odometry Movement (ΔY)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Save and Show ===
plt.savefig("camera_vs_odom_deltas.png")
print("\n✅ Plot saved as: camera_vs_odom_deltas.png")
plt.show()
