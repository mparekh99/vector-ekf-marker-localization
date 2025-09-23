import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#         self.R = np.diag([
        #     25.0,                # 5 mm standard deviation -> 25 mm² variance
        #     25.0,                # Same for Y
        #     np.deg2rad(5.0)**2   # ≈ 0.0076 rad² variance (5° std dev)
        # ])

        # self.Q = np.array([
        #     [0.5, 0.0, 0.0],
        #     [0.0, 0.5, 0.0],
        #     [0.0, 0.0, np.deg2rad(1.5)**2]  # orientation might drift slowly
        # ])



# Directory containing your CSV files
data_dir = '.'  # Change this if your files are in a different folder

# Pattern to match csv files
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# List to hold dataframes
dfs = []

# Load all CSVs
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

# Combine all dataframes by concatenation
data = pd.concat(dfs, ignore_index=True)

# Sort by timestamp if needed
data = data.sort_values('timestamp')

# Extract columns for convenience
t = data['timestamp']
odom_x, odom_y, odom_theta = data['odom_x'], data['odom_y'], data['odom_theta']
cam_x, cam_y, cam_theta = data['cam_x'], data['cam_y'], data['cam_theta']
ekf_x, ekf_y, ekf_theta = data['ekf_x'], data['ekf_y'], data['ekf_theta']

# Wrap angles to [-pi, pi] to avoid jump plots
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

odom_theta = wrap_angle(odom_theta)
cam_theta = wrap_angle(cam_theta)
ekf_theta = wrap_angle(ekf_theta)

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(odom_x, odom_y, label='Odometry', linestyle='--')
plt.plot(cam_x, cam_y, label='Camera')
plt.plot(ekf_x, ekf_y, label='EKF', linewidth=2)
plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.title('Trajectory Comparison')
plt.legend()
plt.axis('equal')
plt.grid(True)

# Plot orientation over time
plt.figure(figsize=(10, 4))
plt.plot(t, odom_theta, label='Odometry Theta', linestyle='--')
plt.plot(t, cam_theta, label='Camera Theta')
plt.plot(t, ekf_theta, label='EKF Theta', linewidth=2)
plt.xlabel('Timestamp')
plt.ylabel('Orientation (radians)')
plt.title('Orientation over Time')
plt.legend()
plt.grid(True)

# Plot errors (optional)
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(t, odom_x - ekf_x, label='Odometry X - EKF X')
plt.plot(t, cam_x - ekf_x, label='Camera X - EKF X')
plt.ylabel('X Error (mm)')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t, odom_y - ekf_y, label='Odometry Y - EKF Y')
plt.plot(t, cam_y - ekf_y, label='Camera Y - EKF Y')
plt.xlabel('Timestamp')
plt.ylabel('Y Error (mm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
