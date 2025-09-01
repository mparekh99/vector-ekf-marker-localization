import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
import pickle
import csv
import os
from pupil_apriltags import Detector

# Constants
MARKER_LENGTH = 50 #MM



class MarkerProcessor:
    def __init__(self, marker_world):
        self.marker_world = marker_world
        self.marker_transforms = marker_world.marker_transforms  


        self.mtx = np.array(
            [[344.99536199,   0.,         335.18457768],
            [  0.,         340.95784312, 174.66841336],
            [  0.,           0.,           1.        ]])
        
        self.dist = np.array([[-0.08150416, -0.10299143,  0.00519719, -0.0057255,   0.04514907]])


        self.obj_points = np.float32([
            [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
            [ MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
            [ MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
            [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]
        ])

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # self.prev_rotation = None
        self.prev_quat = None  # Store last quaternion for continuity
        self.frame_number = 0

    def preprocess_frame(self, frame_pil):
        self.frame_number += 1
        np_frame = np.array(frame_pil)
        return np_frame

    
    def invert_homogeneous(self, T):
        R_mat = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_mat.T
        T_inv[:3, 3] = -R_mat.T @ t
        return T_inv

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = self.detector.detectMarkers(frame)
        # print(len(corners))
        
        if ids is None:
            return None, frame
        
        poses = []
        
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]

            if marker_id not in self.marker_transforms:
                continue

            image_points = corner.reshape((4, 2)).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(self.obj_points, image_points, self.mtx, self.dist)

            if not success:
                print(f"Failed to solvePnP for marker ID {marker_id}")
                continue



            R_cm, _ = cv2.Rodrigues(rvec)
            rot = R.from_matrix(R_cm)
            q = rot.as_quat()
            
            marker_camera = np.eye(4)
            marker_camera[:3, :3] = R_cm
            marker_camera[:3, 3] = tvec.flatten()

            T_camera_in_marker = self.invert_homogeneous(marker_camera)

            marker = self.marker_transforms[marker_id]
            fixed_position, fixed_rotation, translation = marker["pos"], marker["rot"], marker["translation"]

            marker_global = np.eye(4)
            marker_global[:3, :3] = fixed_rotation

            marker_global[:3, 3] = fixed_position

            vector_pos = marker_global @ T_camera_in_marker

            pos = vector_pos[:3, 3]


            pos[0] += translation[0]
            pos[1] += translation[1]

            vector_pos[:3, 3] = pos

            np.set_printoptions(suppress=True, precision=6)
            
            # print(vector_pos[:3,3])
   

            poses.append(vector_pos)
     
            # print(vector_pos[:3,3])

        if len(poses) == 0:
            return None, frame
         
        avg_translation = np.mean([pose[:3, 3] for pose in poses], axis=0)

        # Create a final pose using the first marker's rotation and averaged position
        final_pose = np.eye(4)
        final_pose[:3, 3] = avg_translation
        final_pose[:3, :3] = poses[0][:3, :3]  # Use the first marker's rotation
      


        return final_pose, frame
        # return camera_global_avg, frame

    def log_rotation_matrix(self, frame_number, R_cm, log_file='rotation_log.csv'):
        # Check if file exists to write header
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header row: frame + 9 elements of rotation matrix
                header = ['frame'] + [f'R_{i}{j}' for i in range(3) for j in range(3)]
                writer.writerow(header)

            # Flatten 3x3 matrix to 1D list
            R_flat = R_cm.flatten().tolist()
            writer.writerow([frame_number] + R_flat)
    

    def log_quaternion(self, frame_number, quat, log_file='quaternion_log.csv'):
        # Convert rotation matrix to quaternion
        # quat = R.from_matrix(R_cm).as_quat()  # [x, y, z, w]

        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['frame', 'qx', 'qy', 'qz', 'qw'])  # header
            writer.writerow([frame_number] + quat.tolist())