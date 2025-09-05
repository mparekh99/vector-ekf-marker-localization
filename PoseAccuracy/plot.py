import pandas as pd
import matplotlib.pyplot as plt

# Files
baseline_file = "pose_log.csv"
comparison_files = ["undistorted.csv", "subpix_test1.csv", "subpix_test2.csv"]
titles = ["Undistorted", "Subpix Test 1", "Subpix Test 2"]

# Load baseline data once
baseline_df = pd.read_csv(baseline_file).sort_values("frame")

# Loop through each comparison and create a separate window
for file, title in zip(comparison_files, titles):
    comparison_df = pd.read_csv(file).sort_values("frame")

    # New figure (new window)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Comparison: pose_log.csv vs {file}", fontsize=16)

    # Left plot: Baseline
    axes[0].plot(baseline_df["frame"], baseline_df["estimated_distance_mm"], label="Camera", marker='o')
    axes[0].plot(baseline_df["frame"], baseline_df["hypot_solvepnp_mm"], label="SolvePnP", marker='x')
    axes[0].plot(baseline_df["frame"], baseline_df["odom_hyp"], label="Odometry", marker='s')
    axes[0].set_title("Baseline: pose_log.csv")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Distance (mm)")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Right plot: Comparison file
    axes[1].plot(comparison_df["frame"], comparison_df["estimated_distance_mm"], label="Camera", marker='o')
    axes[1].plot(comparison_df["frame"], comparison_df["hypot_solvepnp_mm"], label="SolvePnP", marker='x')
    axes[1].plot(comparison_df["frame"], comparison_df["odom_hyp"], label="Odometry", marker='s')
    axes[1].set_title(f"Comparison: {title}")
    axes[1].set_xlabel("Frame")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # Show the figure in a new window
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust for suptitle
    plt.show()
