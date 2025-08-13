import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
import pickle


class MarkerProcessor:
    def __init__(self, marker_world):
        self.marker_world = marker_world
        self.marker_transforms = marker_world.marker_transforms  

        # LOAD IN CALIBRATED STUFF
        calib = pickle.load(open("camera_calib_pickle.p", "rb"))
        self.mtx = calib["mtx"]
        self.dist = calib["dist"]

        #
        self.obj_points = np.float32([
            [0, 0, 0],   # top left
            [0.05, 0, 0],   # top right
            [0.05, 0.05, 0],  # bottom right
            [0, 0.05, 0]   # bottom left
        ])

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def preprocess_frame(self, frame_pil):
        frame_np = np.array(frame_pil)
        return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    def invert_homogeneous(self, T):
        R_mat = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_mat.T
        T_inv[:3, 3] = -R_mat.T @ t
        return T_inv

    def process_frame(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is None:
            return None, frame

        camera_poses = []

        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id not in self.marker_transforms:
                continue

            image_points = corner.reshape((4, 2)).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(self.obj_points, image_points, self.mtx, self.dist)
            if not success:
                continue

            R_cm, _ = cv2.Rodrigues(rvec)
            marker_camera = np.eye(4)
            marker_camera[:3, :3] = R_cm
            marker_camera[:3, 3] = tvec.flatten()
            camera_marker = self.invert_homogeneous(marker_camera)

            marker = self.marker_transforms[marker_id]
            fixed_position, fixed_rotation, translation = marker["pos"], marker["rot"], marker["translation"]

            marker_global = np.eye(4)
            marker_global[:3, :3] = fixed_rotation @ R_cm
            marker_global[:3, 3] = fixed_position

            vector_pos = marker_global @ camera_marker

            pos = vector_pos[:3, 3] * 1000 / 3.1
            pos[0] += translation[0]
            pos[1] += translation[1]
            vector_pos[:3, 3] = pos

            camera_poses.append(vector_pos)

        if not camera_poses:  # No poses were calculated 
            return None, frame

        avg_translation = np.mean([pose[:3, 3] for pose in camera_poses], axis=0)
        avg_rotation = np.mean([pose[:3, :3] for pose in camera_poses], axis=0)
        U, _, VT = np.linalg.svd(avg_rotation)
        rot_avg = U @ VT

        camera_global_avg = np.eye(4)
        camera_global_avg[:3, :3] = rot_avg
        camera_global_avg[:3, 3] = avg_translation

        return camera_global_avg, frame
    
