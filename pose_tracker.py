import numpy as np
import time
import csv
import math
from marker_processor import MarkerProcessor
from world import Marker_World
from kalman import KalmanFilter


class PoseTracker:
    def __init__(self):
        self.kalman = KalmanFilter()
        self.world = Marker_World()
        self.marker_processor = MarkerProcessor(self.world)

        self.position = np.array([0, 0])
        self.heading = math.pi / 2
        self.last_update_time = time.time()

        # LOG
        self.logs = []
        self.last_odom = (0, 0, 0)
        self.last_cam = None

    def update_pose(self, raw_image, robot):
        frame = self.marker_processor.preprocess_frame(raw_image)
        pose, _ = self.marker_processor.process_frame(frame)

        current_time = time.time()
        dt = current_time - self.last_update_time  # in seconds
        self.last_update_time = current_time

        v_l = robot.left_wheel_speed_mmps
        v_r = robot.right_wheel_speed_mmps

        v = (v_l + v_r) / 2

        x, y, theta = self.kalman.initial_predict(v, dt, robot.gyro.z)

        if pose is not None:
            pos = pose[:3, 3]
            x = pos[0]
            y = pos[1]
            theta = math.atan2(pose[1, 0], pose[0, 0])
            theta += math.pi / 2  # Correction for heading
            x, y, theta = self.kalman.update(x, y, theta)

        # UPDATE:
        self.position[0] = x
        self.position[1] = y
        self.heading = theta

        return frame

    # def log(self, timestamp):
    #     x_odom, y_odom, t_odom = self.last_odom
    #     if self.last_cam is not None:
    #         x_cam, y_cam, t_cam = self.last_cam
    #     else:
    #         x_cam = y_cam = t_cam = None

    #     x_ekf, y_ekf, t_ekf = self.position[0].item(), self.position[1].item(), self.yaw

    #     self.logs.append({
    #         "timestamp": timestamp,
    #         "odom_x": x_odom,
    #         "odom_y": y_odom,
    #         "odom_theta": t_odom,
    #         "cam_x": x_cam,
    #         "cam_y": y_cam,
    #         "cam_theta": t_cam,
    #         "ekf_x": x_ekf,
    #         "ekf_y": y_ekf,
    #         "ekf_theta": t_ekf
    #     })

    # def save_logs(self, filename="pose_log.csv"):
    #     if not self.logs:
    #         print("No logs to save.")
    #         return
    #     keys = self.logs[0].keys()
    #     with open(filename, "w", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=keys)
    #         writer.writeheader()
    #         writer.writerows(self.logs)

    # def get_x(self):
    #     return self.position[0].item()

    # def get_y(self):
    #     return self.position[1].item()

    # def get_yaw(self):
    #     return self.yaw
