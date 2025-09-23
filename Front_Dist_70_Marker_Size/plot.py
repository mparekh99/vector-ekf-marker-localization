import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === CONFIG ===
folder_path = '.'  # Current folder
file_pattern = os.path.join(folder_path, '*.csv')
usable_error_threshold = 50  # mm — define "usable" error tolerance

# === Find and Load All CSVs ===
csv_files = glob.glob(file_pattern)
print(f"Found CSV files: {csv_files}")

dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    df['run_id'] = os.path.basename(file)
    dataframes.append(df)

# === Error if No CSVs Found ===
if not dataframes:
    raise FileNotFoundError("❌ No CSV files found. Make sure they are in the same folder as this script.")

# === Align Data by Frame Index ===
min_len = min(len(df) for df in dataframes)
for i in range(len(dataframes)):
    dataframes[i] = dataframes[i].iloc[:min_len].copy()
    dataframes[i]['frame'] = np.arange(min_len)

# === Concatenate All Runs ===
all_data = pd.concat(dataframes)

# === Compute Error Columns ===
all_data['error_x'] = all_data['cam_x'] - all_data['odom_x']
all_data['error_y'] = all_data['cam_y'] - all_data['odom_y']
all_data['euclidean_error'] = np.sqrt(all_data['error_x']**2 + all_data['error_y']**2)

# === Average Across Frames ===
grouped = all_data.groupby('frame')
avg_error = grouped['euclidean_error'].mean()
std_error = grouped['euclidean_error'].std()
avg_odom_x = grouped['odom_x'].mean()

# === Determine "Usable" Camera Region ===
usable_mask = avg_error < usable_error_threshold
usable_start = avg_odom_x[usable_mask].min() if usable_mask.any() else None
usable_end = avg_odom_x[usable_mask].max() if usable_mask.any() else None

# === Optional Debug Output ===
print(f"\nMax error: {avg_error.max():.2f} mm")
print(f"Min error: {avg_error.min():.2f} mm")

if usable_start is not None and usable_end is not None:
    print(f"\n✅ Camera pose is considered usable between:")
    print(f"→ {usable_start:.2f} mm and {usable_end:.2f} mm (based on {usable_error_threshold} mm threshold)\n")
else:
    print(f"\n❌ No usable range found — camera error never dropped below {usable_error_threshold} mm.\n"
          f"→ Try increasing the threshold or inspect individual error values.\n")

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(avg_odom_x, avg_error, label='Mean Error (Camera vs Odom)', color='blue')
plt.fill_between(avg_odom_x, avg_error - std_error, avg_error + std_error,
                 color='blue', alpha=0.2, label='±1 STD')

# Highlight usable region
if usable_start is not None and usable_end is not None:
    plt.axvspan(usable_start, usable_end, color='green', alpha=0.2, label='Usable Zone')

plt.axhline(y=usable_error_threshold, color='red', linestyle='--', label=f'{usable_error_threshold} mm Threshold')

plt.xlabel("Distance Forward (Odom X) [mm]")
plt.ylabel("Euclidean Error (Camera vs Odom) [mm]")
plt.title("Camera Pose Error vs Odometry (Averaged Over Runs)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
