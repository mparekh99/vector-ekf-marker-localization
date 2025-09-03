import csv
import math
import matplotlib.pyplot as plt

# === Load data ===
timestamps = []
odom_xs, odom_ys = [], []
cam_xs, cam_ys = [], []

with open("pose_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            timestamps.append(float(row["timestamp"]))

            ox = float(row["odom_x"])
            oy = float(row["odom_y"])
            odom_xs.append(ox)
            odom_ys.append(oy)

            # Parse camera data, which may be None
            cx_raw = row["cam_x"]
            cy_raw = row["cam_y"]

            if cx_raw.strip() == "" or cy_raw.strip() == "":
                cam_xs.append(None)
                cam_ys.append(None)
            else:
                cx = float(cx_raw)
                cy = float(cy_raw)
                cam_xs.append(cx)
                cam_ys.append(cy)

        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

# === Normalize time ===
time_zero = [t - timestamps[0] for t in timestamps]

# === Compute Distance to Target (0, 165 mm) ===
target_x, target_y = 0, 165

def compute_distance(xs, ys):
    distances = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            distances.append(None)
        else:
            dx = target_x - x
            dy = target_y - y
            distances.append(math.sqrt(dx**2 + dy**2))
    return distances

odom_dists = compute_distance(odom_xs, odom_ys)
cam_dists = compute_distance(cam_xs, cam_ys)

# === Filter out None values for camera before plotting ===
def filter_valid(xs, ys, times):
    return zip(*[(x, y, t) for x, y, t in zip(xs, ys, times) if x is not None and y is not None])

cam_xs_filtered, cam_times_x = zip(*[(x, t) for x, t in zip(cam_xs, time_zero) if x is not None])
cam_ys_filtered, cam_times_y = zip(*[(y, t) for y, t in zip(cam_ys, time_zero) if y is not None])
cam_dists_filtered, cam_times_dist = zip(*[(d, t) for d, t in zip(cam_dists, time_zero) if d is not None])

# === Plot ===
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- Plot 1: X over time ---
axs[0].plot(time_zero, odom_xs, label='Odometry X', color='orange')
axs[0].plot(cam_times_x, cam_xs_filtered, label='Camera X (valid only)', color='green')
axs[0].axhline(y=0, color='red', linestyle=':', linewidth=1.5, label='Target X = 0 mm')
axs[0].set_ylabel("X Position (mm)")
axs[0].set_title("X Position Over Time")
axs[0].legend()
axs[0].grid(True)

# --- Plot 2: Y over time ---
axs[1].plot(time_zero, odom_ys, label='Odometry Y', color='orange')
axs[1].plot(cam_times_y, cam_ys_filtered, label='Camera Y (valid only)', color='green')
axs[1].axhline(y=165, color='red', linestyle=':', linewidth=1.5, label='Target Y = 165 mm')
axs[1].set_ylabel("Y Position (mm)")
axs[1].set_title("Y Position Over Time")
axs[1].legend()
axs[1].grid(True)

# --- Plot 3: Distance to Target ---
axs[2].plot(time_zero, odom_dists, label='Odometry → Target', color='orange')
axs[2].plot(cam_times_dist, cam_dists_filtered, label='Camera → Target (valid only)', color='green')
axs[2].set_ylabel("Distance to [0, 165] (mm)")
axs[2].set_xlabel("Time (s)")
axs[2].set_title("Distance to Target Over Time")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
