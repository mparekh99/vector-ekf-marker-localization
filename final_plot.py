import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_interpolate(file_path, time_base):
    df = pd.read_csv(file_path)
    df_interp = pd.DataFrame({'timestamp': time_base})
    for col in df.columns:
        if col != 'timestamp':
            df_interp[col] = np.interp(time_base, df['timestamp'], df[col])
    return df_interp

def plot_innovations(df, save_path=None):
    time = df['timestamp']

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, df['innovation_x'], label='Innovation X')
    plt.ylabel('Innovation X (mm)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, df['innovation_y'], label='Innovation Y')
    plt.ylabel('Innovation Y (mm)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, np.degrees(df['innovation_theta']), label='Innovation Theta')
    plt.ylabel('Innovation Theta (deg)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()

    plt.suptitle('Innovation Residuals Over Time')
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(os.path.join(save_path, "innovation_residuals.png"))
            print(f"Saved innovation residuals plot to {os.path.join(save_path, 'innovation_residuals.png')}")
        except Exception as e:
            print("Failed to save innovation residuals plot:", e)
    plt.show()

def main():
    file_paths = ['final_log.csv']
    save_dir = "./plots"
    os.makedirs(save_dir, exist_ok=True)

    dfs = [pd.read_csv(fp) for fp in file_paths]

    # Interpolate onto common time base
    min_end_time = min(df['timestamp'].max() for df in dfs)
    max_start_time = max(df['timestamp'].min() for df in dfs)
    common_time = np.linspace(max_start_time, min_end_time, 500)
    interpolated = [load_and_interpolate(fp, common_time) for fp in file_paths]

    # --- Plot Position X over Time ---
    plt.figure(figsize=(10, 5))
    for i, df in enumerate(interpolated):
        plt.plot(common_time, df['pred_x'], 'r--', alpha=0.6, label='Prediction X (Odometry)' if i == 0 else "")
        plt.plot(common_time, df['cam_x'], 'b:', alpha=0.6, label='Camera X' if i == 0 else "")
        plt.plot(common_time, df['ekf_x'], 'g-', alpha=0.9, label='EKF X' if i == 0 else "")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (mm)")
    plt.title("X Position Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(save_dir, "x_position_over_time.png"))
        print(f"Saved X position plot to {os.path.join(save_dir, 'x_position_over_time.png')}")
    except Exception as e:
        print("Failed to save X position plot:", e)
    plt.show()

    # --- Plot Position Y over Time ---
    plt.figure(figsize=(10, 5))
    for i, df in enumerate(interpolated):
        plt.plot(common_time, df['pred_y'], 'r--', alpha=0.6, label='Prediction Y (Odometry)' if i == 0 else "")
        plt.plot(common_time, df['cam_y'], 'b:', alpha=0.6, label='Camera Y' if i == 0 else "")
        plt.plot(common_time, df['ekf_y'], 'g-', alpha=0.9, label='EKF Y' if i == 0 else "")
    plt.xlabel("Time (s)")
    plt.ylabel("Y Position (mm)")
    plt.title("Y Position Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(save_dir, "y_position_over_time.png"))
        print(f"Saved Y position plot to {os.path.join(save_dir, 'y_position_over_time.png')}")
    except Exception as e:
        print("Failed to save Y position plot:", e)
    plt.show()

    # --- Plot Measurement Usage and Mahalanobis Distance ---
    plt.figure(figsize=(12, 5))
    for i, df in enumerate(interpolated):
        m_dist = df['mahalanobis_distance']
        used = df['measurement_used'].astype(bool)

        plt.plot(common_time, m_dist, 'k-', label='Mahalanobis Distance' if i == 0 else "")

        accepted_and_below = used & (m_dist <= 60)
        plt.scatter(common_time[accepted_and_below], m_dist[accepted_and_below],
                    c='green', s=15, alpha=0.7, label='Accepted Measurement' if i == 0 else "", marker='o')

        rejected = ~accepted_and_below
        plt.scatter(common_time[rejected], m_dist[rejected],
                    c='gray', s=10, alpha=0.3, label='Rejected Measurement' if i == 0 else "", marker='x')

    plt.axhline(y=60, color='red', linestyle='--', label='Acceptance Threshold (60)')
    plt.ylabel("Mahalanobis Distance")
    plt.xlabel("Time (s)")
    plt.title("Measurement Usage & Mahalanobis Distance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(save_dir, "mahalanobis_usage.png"))
        print(f"Saved Mahalanobis usage plot to {os.path.join(save_dir, 'mahalanobis_usage.png')}")
    except Exception as e:
        print("Failed to save Mahalanobis usage plot:", e)
    plt.show()

    # --- Plot Innovations ---
    plot_innovations(interpolated[0], save_path=save_dir)

if __name__ == "__main__":
    main()
