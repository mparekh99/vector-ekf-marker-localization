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

        self.opt = np.array([
            [267.95426624,   0.,         325.68665971],
            [  0.,         262.35441111, 179.63134443],
            [  0.,           0.,           1.        ]])



        self.at_detector = Detector(
            families=" tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )


        self.last_pose = None
        self.frame_number = 0
        self.last_frame_number = None



    def preprocess_frame(self, frame_pil):
        self.last_frame_number = self.frame_number
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
    
    # def estimate_distance(self, box_height_px, frame):
    #     frame_height, frame_width = frame.shape[:2]

    #     focal_length_px = (frame_height / 2) / math.tan(math.radians(FOV_VERTICAL / 2))

    #     if box_height_px <= 0:
    #         return float('inf')

    #     distance_mm = (focal_length_px * MARKER_LENGTH) / box_height_px
    #     return distance_mm


    def process_frame(self, frame):
        marker_logs = []

        R_diag = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = self.at_detector.detect(gray)

        fx = self.mtx[0, 0]
        fy = self.mtx[1, 1]
        cx = self.mtx[0, 2]
        cy = self.mtx[1, 2]

        # Detect tags and estimate pose
        tags = self.at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=MARKER_LENGTH
        )

        pose = None
        poses = []
        pose_distance_list = []


        for tag in tags:

            rvec = tag.pose_R
            tvec = tag.pose_t

            # print(tag.tag_id)

            # print(rvec)
            marker_in_camera = np.eye(4)
            marker_in_camera[:3, :3] = rvec
            marker_in_camera[:3, 3] = tvec.flatten()

            marker = self.marker_transforms[tag.tag_id]
            fixed_pos, fixed_rot, translation = marker["pos"], marker["rot"], marker["translation"]

            marker_global = np.eye(4)
            marker_global[:3, :3] = fixed_rot
            marker_global[:3, 3] = fixed_pos

            camera_in_marker = self.invert_homogeneous(marker_in_camera)


            camera_global = marker_global @ camera_in_marker

            pos = camera_global[:3, 3]

            pos[0] += translation[0]
            pos[1] += translation[1]

            camera_global[:3, 3] = pos

            # Compute Euclidean distance between current pose position and fixed marker position
            distance_to_tag = np.linalg.norm(pos - fixed_pos)

            if distance_to_tag <= 678.0:   # PAST TRUST
                pose_key = tuple(camera_global.flatten())
                pose_distance_list.append({
                    'pose_key': pose_key,
                    'distance': distance_to_tag,
                    'tag_id': tag.tag_id
                })

        if not pose_distance_list:
            # No valid poses found
            return None, frame, None
            
        # Find the pose info with smallest distance
        best_pose_info = min(pose_distance_list, key=lambda x: x['distance'])

        best_pose = np.array(best_pose_info['pose_key']).reshape(4, 4)
        best_tag_id = best_pose_info['tag_id']
        # best_dist = best_pose_info['distance']
        
        
        return best_pose, frame, best_tag_id
    



