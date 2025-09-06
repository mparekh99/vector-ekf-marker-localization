import csv
import numpy as np
import matplotlib.pyplot as plt

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# Lists to store data
timestamps = []
odom_x, odom_y, odom_theta = [], [], []
cam_x, cam_y, cam_theta = [], [], []
ekf_x, ekf_y, ekf_theta = [], [], []

# Read CSV file
with open("big_test.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        timestamps.append(float(row['timestamp']))
        odom_x.append(safe_float(row['odom_x']))
        odom_y.append(safe_float(row['odom_y']))
        odom_theta.append(safe_float(row['odom_theta']))

        cam_x.append(safe_float(row['cam_x']))
        cam_y.append(safe_float(row['cam_y']))
        cam_theta.append(safe_float(row['cam_theta']))

        ekf_x.append(safe_float(row['ekf_x']))
        ekf_y.append(safe_float(row['ekf_y']))
        ekf_theta.append(safe_float(row['ekf_theta']))

# Convert to numpy arrays for easier handling
timestamps = np.array(timestamps)
odom_x = np.array(odom_x)
odom_y = np.array(odom_y)
odom_theta = np.array(odom_theta)

cam_x = np.array(cam_x)
cam_y = np.array(cam_y)
cam_theta = np.array(cam_theta)

ekf_x = np.array(ekf_x)
ekf_y = np.array(ekf_y)
ekf_theta = np.array(ekf_theta)

# Plot X and Y positions over time
plt.figure(figsize=(12, 6))
plt.plot(timestamps, odom_x, label='Odometry X', linestyle='--')
plt.plot(timestamps, cam_x, label='Camera X', linestyle=':')
plt.plot(timestamps, ekf_x, label='EKF X', linewidth=2)

plt.plot(timestamps, odom_y, label='Odometry Y', linestyle='--')
plt.plot(timestamps, cam_y, label='Camera Y', linestyle=':')
plt.plot(timestamps, ekf_y, label='EKF Y', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.title('Position Over Time')
plt.legend()
plt.grid(True)

# Plot theta (heading) over time (converted to degrees)
plt.figure(figsize=(12, 4))
plt.plot(timestamps, np.rad2deg(odom_theta), label='Odometry Theta', linestyle='--')
plt.plot(timestamps, np.rad2deg(cam_theta), label='Camera Theta', linestyle=':')
plt.plot(timestamps, np.rad2deg(ekf_theta), label='EKF Theta', linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Heading (degrees)')
plt.title('Heading Over Time')
plt.legend()
plt.grid(True)

# Plot XY trajectory
plt.figure(figsize=(8, 8))
plt.plot(odom_x, odom_y, label='Odometry', linestyle='--')
plt.plot(cam_x, cam_y, label='Camera', linestyle=':')
plt.plot(ekf_x, ekf_y, label='EKF', linewidth=2)

plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.title('XY Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True)

plt.show()
