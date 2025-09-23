import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load all CSV files from current directory
data_dir = '.'
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
dfs = [pd.read_csv(f) for f in csv_files]

# Combine all dataframes into one and sort by timestamp
data = pd.concat(dfs, ignore_index=True).sort_values('timestamp')

# Wrap angles helper function
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Extract and wrap angles
odom_theta = wrap_angle(data['odom_theta'].values)
cam_theta = wrap_angle(data['cam_theta'].values)
ekf_theta = wrap_angle(data['ekf_theta'].values)

# Compute measurement residuals: camera measurement minus EKF estimate
residual_x = data['cam_x'].values - data['ekf_x'].values
residual_y = data['cam_y'].values - data['ekf_y'].values
residual_theta = wrap_angle(cam_theta - ekf_theta)

# Plot residuals over time to visualize measurement noise characteristics
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(data['timestamp'], residual_x, label='Residual X (Camera - EKF)')
plt.ylabel('Error in X (mm)')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(data['timestamp'], residual_y, label='Residual Y (Camera - EKF)', color='orange')
plt.ylabel('Error in Y (mm)')
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(data['timestamp'], residual_theta, label='Residual Theta (Camera - EKF)', color='green')
plt.ylabel('Error in Theta (radians)')
plt.xlabel('Timestamp')
plt.legend()
plt.grid(True)

plt.suptitle('Measurement Residuals (Camera vs EKF)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Compute covariance matrix of residuals to estimate R
residuals = np.vstack([residual_x, residual_y, residual_theta])
R_estimated = np.cov(residuals)

print("Estimated measurement noise covariance R based on camera vs EKF residuals:\n", R_estimated)

# Print standard deviations for intuition
std_dev = np.sqrt(np.diag(R_estimated))
print(f"Estimated std devs: X = {std_dev[0]:.3f} mm, Y = {std_dev[1]:.3f} mm, Theta = {np.rad2deg(std_dev[2]):.3f} degrees")
