import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("square_log.csv")

# Convert angles from radians to degrees for interpretability (optional)
df["odom_theta_deg"] = np.degrees(df["odom_theta"])
df["cam_theta_deg"] = np.degrees(df["cam_theta"])
df["ekf_theta_deg"] = np.degrees(df["ekf_theta"])

# Plot all trajectories
plt.figure(figsize=(10, 8))
plt.plot(df["odom_x"], df["odom_y"], label="Odometry", linestyle="--", color="gray")
plt.plot(df["cam_x"], df["cam_y"], 'rx', label="Camera Measurement", alpha=0.5)
plt.plot(df["ekf_x"], df["ekf_y"], label="EKF Estimate", linewidth=2)

plt.title("Trajectory Comparison")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()
