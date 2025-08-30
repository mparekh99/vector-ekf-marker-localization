import numpy as np
import time
import csv
import math
from marker_processor import MarkerProcessor
from world import Marker_World
from kalman import KalmanFilter
from utils import wrap_angle


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
    # def is_camera_pose_valid(self, cam_x, cam_y, cam_theta, pred_x, pred_y, pred_theta, 
    #                         pos_thresh_mm=100.0, theta_thresh_deg=30.0):
    #     dx = cam_x - pred_x
    #     dy = cam_y - pred_y
    #     dist_mm = math.sqrt(dx**2 + dy**2)

    #     # print(cam_theta, pred_theta)
    #     dtheta = wrap_angle(cam_theta - pred_theta)
    #     dtheta_deg = math.degrees(abs(dtheta))

    #     if dist_mm > pos_thresh_mm:
    #         print(f"[Reject] Pose too far: Δpos = {dist_mm:.1f} mm")
    #         return False
    #     if dtheta_deg > theta_thresh_deg:
    #         print(f"[Reject] Angle diff too large: Δθ = {dtheta_deg:.1f}°")
    #         return False

    #     return True



    def update_pose(self, raw_image, robot):
        frame = self.marker_processor.preprocess_frame(raw_image)
        # pose, _ = self.marker_processor.process_frame(frame)
        test = self.marker_processor.process_frame(frame)

        # current_time = time.time()
        # dt = current_time - self.last_update_time  # in seconds
        # self.last_update_time = current_time

        # v_l = robot.left_wheel_speed_mmps
        # v_r = robot.right_wheel_speed_mmps

        # v = (v_l + v_r) / 2

        # x_pred, y_pred, theta_pred = self.kalman.initial_predict(v, dt, robot.gyro.z)

        # x_cam, y_cam, theta_cam = None, None, None
        # x, y, theta = None, None, None

        # if pose is not None:
        #     pos = pose[:3, 3]
        #     x_cam = pos[0]
        #     y_cam = pos[1]
        #     theta_cam = math.atan2(pose[1, 0], pose[0, 0]) + math.pi / 2
        #     # print(theta_cam)


        #     if -200 <= x_cam <= 200 and -200 <= y_cam <= 200:
        #         x, y, theta = self.kalman.update(x_cam, y_cam, theta_cam)

        #     else:
        #         x, y, theta = x_pred, y_pred, theta_pred # UPDATE WITH ODOMOETRY 

        # else:
        #     x, y, theta = x_pred, y_pred, theta_pred

        # UPDATE:
        # self.position[0] = x
        # self.position[1] = y
        # self.heading = theta

        # self.logs.append({
        #     "timestamp": current_time,
        #     "odom_x": x_pred,
        #     "odom_y": y_pred,
        #     "odom_theta": theta_pred,
        #     "cam_x": x_cam,
        #     "cam_y": y_cam,
        #     "cam_theta": theta_cam,
        #     "ekf_x": x,
        #     "ekf_y": y,
        #     "ekf_theta": theta
        # })

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
