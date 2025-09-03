import numpy as np
import time
import csv
import math
from marker_processor import MarkerProcessor
from world import Marker_World
from kalman import KalmanFilter
from utils import wrap_angle_pi


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
        self.frame_number = 0
        
        #ODOM Tracking
        self.odom_x = 0
        self.odom_y = 0
        self.odom_theta = math.pi/2


    def update_pose(self, raw_image, robot):
        frame = self.marker_processor.preprocess_frame(raw_image)
        pose, _ = self.marker_processor.process_frame(frame)
        # test = self.marker_processor.process_frame(frame)
        current_time = time.time()
        dt = current_time - self.last_update_time  # in seconds
        self.last_update_time = current_time

        v_l = robot.left_wheel_speed_mmps
        v_r = robot.right_wheel_speed_mmps

        v = (v_l + v_r) / 2

        x_pred, y_pred, theta_pred = self.kalman.initial_predict(v, dt, robot.gyro.z)

        x_cam, y_cam, theta_cam = None, None, None
        x, y, theta = None, None, None



        # ODOM
        theta_k = self.odom_theta + robot.gyro.z * dt
        # theta_k = wrap_angle_pi(theta_k)

        self.odom_x = self.odom_x + v * math.cos(theta_k) * dt
        self.odom_y = self.odom_y + v * math.sin(theta_k) * dt

        if pose is not None:
            pos = pose[:3, 3]
            x_cam = pos[0]
            y_cam = pos[1]
            print("CAMERA READ:")
            print(x_cam, y_cam)
            theta_cam = math.atan2(pose[1, 0], pose[0, 0]) + math.pi / 2
            x, y, theta = self.kalman.update(x_cam, y_cam, theta_cam)
        else:
            x, y, theta = x_pred, y_pred, theta_pred # UPDATE WITH ODOMOETRY 


        # UPDATE:
        self.position[0] = x
        self.position[1] = y
        self.heading = theta

        # print("KALMAN")
        # print(x, y)


        self.logs.append({
            "timestamp": current_time,
            "odom_x": self.odom_x,
            "odom_y": self.odom_y,
            "odom_theta": self.odom_theta,
            "cam_x": x_cam,
            "cam_y": y_cam,
            "cam_theta": theta_cam,
            "ekf_x": x,
            "ekf_y": y,
            "ekf_theta": theta
        })

        return frame

    def save_logs(self, filename="pose_log.csv"):
        if not self.logs:
            print("No logs to save.")
            return
        keys = self.logs[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.logs)
        print(f"Logs saved to {filename}")
