import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === CONFIG ===
folder_path = '.'  # Folder with your 5 CSVs
usable_error_threshold = 50  # mm

# === Load CSVs ===
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
if not csv_files:
    raise FileNotFoundError("❌ No CSV files found.")

print(f"📂 Found CSV files: {csv_files}")

dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['run_id'] = os.path.basename(file)
    dataframes.append(df)

# === Align all runs to the shortest length ===
min_len = min(len(df) for df in dataframes)
for i in range(len(dataframes)):
    df = dataframes[i].iloc[:min_len].copy()
    df['frame'] = np.arange(min_len)
    dataframes[i] = df  # update list

# === Combine all runs ===
all_data = pd.concat(dataframes)

# === Compute Euclidean Error ===
all_data['error_x'] = all_data['cam_x'] - all_data['odom_x']
all_data['error_y'] = all_data['cam_y'] - all_data['odom_y']
all_data['euclidean_error'] = np.sqrt(all_data['error_x']**2 + all_data['error_y']**2)

# === Group by frame index ===
grouped = all_data.groupby('frame')
avg_error = grouped['euclidean_error'].mean()
std_error = grouped['euclidean_error'].std()
avg_odom_y = grouped['odom_y'].mean()

# === Determine usable error region ===
usable_mask = avg_error < usable_error_threshold
usable_indices = np.where(usable_mask)[0]

if len(usable_indices) > 0:
    usable_start_idx = usable_indices[0]
    usable_end_idx = usable_indices[-1]
    usable_start_y = avg_odom_y.iloc[usable_start_idx]
    usable_end_y = avg_odom_y.iloc[usable_end_idx]
    print(f"\n✅ Camera pose is trustworthy between {usable_start_y:.1f} mm and {usable_end_y:.1f} mm (backward from origin)")
else:
    print("\n❌ No usable region found under threshold of", usable_error_threshold, "mm")
    usable_start_y = usable_end_y = None

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(avg_odom_y, avg_error, label='Mean Error', color='blue')
plt.fill_between(avg_odom_y, avg_error - std_error, avg_error + std_error, color='blue', alpha=0.2, label='±1 STD')

if usable_start_y is not None:
    plt.axvspan(usable_start_y, usable_end_y, color='green', alpha=0.2, label='Trustworthy Range')
else:
    plt.axhline(usable_error_threshold, color='red', linestyle='--', label='Error Threshold')

plt.axhline(usable_error_threshold, color='red', linestyle='--')
plt.xlabel("Distance Backward (Odom Y) [mm]")
plt.ylabel("Camera vs Odometry Error [mm]")
plt.title("Camera Pose Accuracy When Moving Backward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("backward_pose_error_analysis.png")
print("📊 Saved plot as 'backward_pose_error_analysis.png'")
plt.show()
