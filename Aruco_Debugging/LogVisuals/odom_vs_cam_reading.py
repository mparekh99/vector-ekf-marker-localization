import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("left_forward_wrap_angles.csv")

# Unwrap and convert angles from radians to degrees
df["odom_theta_deg"] = np.degrees(np.unwrap(df["odom_theta"]))
df["cam_theta_deg"] = np.degrees(np.unwrap(df["cam_theta"]))

# Filter valid camera readings
valid = ~df["cam_theta"].isna()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["odom_theta_deg"], label="Odometry θ", linewidth=2)
plt.plot(df["timestamp"][valid], df["cam_theta_deg"][valid], label="Camera θ", linestyle='--', marker='x', alpha=0.7)

plt.title("Heading (θ) Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Heading (degrees)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
