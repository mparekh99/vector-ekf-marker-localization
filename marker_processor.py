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
FOV_HORIZONTAL = 90
FOV_VERTICAL = 50
MAX_DISTANCE_ERROR = 30      # mm
MAX_POSE_JUMP = 100          # mm
ANGLE_THRESHOLD_RADIANS = np.radians(15)
FRAME_TO_FRAME_ANGLE_THRESHOLD = np.radians(30)

def angle_difference(angle1, angle2):
    diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def average_poses(poses_with_cov):
    # Separate poses and covariances
    poses = [p[0] for p in poses_with_cov]
    covariances = [p[1] for p in poses_with_cov]

    # Average translations
    translations = np.array([p[:3, 3] for p in poses])
    mean_translation = np.mean(translations, axis=0)

    # Average rotations using quaternions
    rotations = [R.from_matrix(p[:3, :3]) for p in poses]
    quats = np.array([r.as_quat() for r in rotations])

    # Normalize quaternions and flip to the same hemisphere
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]

    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    mean_rot = R.from_quat(mean_quat).as_matrix()

    # Average covariance matrices (simple arithmetic mean)
    mean_covariance = np.mean(covariances, axis=0)

    # Compose averaged pose matrix
    averaged_pose = np.eye(4)
    averaged_pose[:3, :3] = mean_rot
    averaged_pose[:3, 3] = mean_translation

    return averaged_pose, mean_covariance



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


        self.obj_points = np.float32([
            [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
            [ MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
            [ MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
            [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]
        ])

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.R = np.array([
            [25.0,  0.0,  0.0],                         # x: 5 mm std dev → 5^2
            [0.0,  25.0,  0.0],                         # y: 5 mm std dev
            [0.0,   0.0, np.deg2rad(1.0)**2]            # yaw: ~1° std dev
        ])

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
    
    def estimate_distance(self, box_height_px, frame):
        frame_height, frame_width = frame.shape[:2]

        focal_length_px = (frame_height / 2) / math.tan(math.radians(FOV_VERTICAL / 2))

        if box_height_px <= 0:
            return float('inf')

        distance_mm = (focal_length_px * MARKER_LENGTH) / box_height_px
        return distance_mm

    # CHATGPT
    def get_marker_size_in_px(self, corners):
        # corners is a 4x2 array: [top-left, top-right, bottom-right, bottom-left]
        # Compute width and height in pixels
        top = np.linalg.norm(corners[0][0] - corners[0][1])
        bottom = np.linalg.norm(corners[0][2] - corners[0][3])
        left = np.linalg.norm(corners[0][0] - corners[0][3])
        right = np.linalg.norm(corners[0][1] - corners[0][2])

        width_px = (top + bottom) / 2.0
        height_px = (left + right) / 2.0

        return width_px, height_px
    
    # CHATGPT
    def reprojection_error(self, rvec, tvec, image_points):
        projected_points, _ = cv2.projectPoints(self.obj_points, rvec, tvec, self.mtx, self.dist)
        projected_points = projected_points.reshape(-1, 2)
        error = np.linalg.norm(projected_points - image_points, axis=1).mean()
        return error
    


    def process_frame(self, frame):
        marker_logs = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

        # undistorted = cv2.undistort(frame, self.mtx, self.dist, None, self.opt)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        # print(len(corners))
        
        if ids is None:
            return None, frame, self.R
        
        angular_poses = []
        head_on_poses = []
        
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]

            if marker_id not in self.marker_transforms:
                continue

            refined_corner = cv2.cornerSubPix(gray, corner, (11,11), (-1,-1), criteria)

            image_points = refined_corner.reshape((4, 2)).astype(np.float32)

            retval, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                self.obj_points,
                image_points,
                self.mtx,
                self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )


            best_error = float('inf')

            for i in range(len(rvecs)):
                rvec = rvecs[i]
                tvec = tvecs[i]

                error = self.reprojection_error(rvec, tvec, image_points)

                if error < best_error:
                    best_error = error


                R_cm, _ = cv2.Rodrigues(rvec)
                marker_in_camera = np.eye(4)
                marker_in_camera[:3, :3] = R_cm
                marker_in_camera[:3, 3] = tvec.flatten()

                camera_in_marker = self.invert_homogeneous(marker_in_camera)

                # Convert marker → global
                marker = self.marker_transforms[marker_id]
                fixed_pos, fixed_rot, translation = marker["pos"], marker["rot"], marker["translation"]

                marker_global = np.eye(4)
                marker_global[:3, :3] = fixed_rot
                marker_global[:3, 3] = fixed_pos

                camera_global = marker_global @ camera_in_marker

                pos = camera_global[:3, 3]

                pos[0] += translation[0]
                pos[1] += translation[1]

                camera_global[:3, 3] = pos
                
                is_valid, is_perpendicular = self.filter_pose(camera_global, corner, frame, marker)


                if is_valid: 

                    is_corner, yaw_diff, distance = self.dynamic_vals(camera_global, corner, frame, marker, marker_id)

                    dynamic_R = self.compute_dynamic_R(distance, yaw_diff, is_corner, error)

                    if is_perpendicular:
                        head_on_poses.append((camera_global, dynamic_R))  # include R if needed
                    else:
                        angular_poses.append((error, camera_global, dynamic_R))

        if len(angular_poses) == 0 and len(head_on_poses) == 0:
            return None, frame, self.R # NO POSE
        
        if len(angular_poses) > 0:

            best_tuple = min(angular_poses, key=lambda x: x[0])
            best_error = best_tuple[0]
            best_pose = best_tuple[1]
            best_dynamic_R = best_tuple[2]

            self.last_pose = best_pose
            return best_pose, frame, best_dynamic_R
        
        
        # AVERAGE
        camera_global_avg, dynamic_R = average_poses(head_on_poses)

        self.last_pose = camera_global_avg

        # print(camera_global_avg[:3, 3])

        return camera_global_avg, frame, dynamic_R



# CHATGPT
    def filter_pose(self, pose, corner, frame, marker_info):

        fixed_pos = marker_info["pos"]
        expected_yaw = marker_info["angle"]

        width_px, height_px = self.get_marker_size_in_px(corner)
        avg_px_size = (width_px + height_px) / 2
        est_distance = self.estimate_distance(avg_px_size, frame) - 25.4  # offset
        pos = pose[:3, 3]
        euclidean_dist = np.linalg.norm(fixed_pos[:2] - pos[:2])

        camera_yaw = math.atan2(pose[1, 0], pose[0, 0]) + math.pi /2 
        yaw_diff = angle_difference(camera_yaw, expected_yaw)

        if yaw_diff < ANGLE_THRESHOLD_RADIANS:
            # Perpendicular 
            return True, True

        if abs(est_distance - euclidean_dist) > MAX_DISTANCE_ERROR:
            return False, False
        

        # Consecutive Readings
        if self.last_pose is not None: 
            if self.last_frame_number + 1 == self.frame_number:
                delta_translation = np.linalg.norm(pos - self.last_pose[:3, 3])

                # COmpare YAW's 
                last_yaw = math.atan2(self.last_pose[1, 0], self.last_pose[0, 0]) + math.pi/ 2
                curr_yaw = math.atan2(pose[1, 0], pose[0, 0]) + math.pi/2 

                yaw_diff = angle_difference(last_yaw, curr_yaw)
    

                if delta_translation > MAX_POSE_JUMP or yaw_diff > FRAME_TO_FRAME_ANGLE_THRESHOLD:
                    return False, False

        return True, False
    

    def dynamic_vals(self, camera_global, corner, frame, marker_info, marker_id):
        fixed_pos = marker_info["pos"]
        expected_yaw = marker_info["angle"]

        width_px, height_px = self.get_marker_size_in_px(corner)
        avg_px_size = (width_px + height_px) / 2

        distance = self.estimate_distance(avg_px_size, frame) - 25.4

        pos = camera_global[:3, 3]
        euclidean_dist = np.linalg.norm(fixed_pos[:2] - pos[:2])

        avg_dist = (distance + euclidean_dist) / 2

        camera_yaw = math.atan2(camera_global[1, 0], camera_global[0, 0]) + math.pi / 2
        yaw_diff = angle_difference(camera_yaw, expected_yaw)

        corner_marker_ids = [4, 3, 6, 7] 
        is_corner = marker_id in corner_marker_ids

        return is_corner, yaw_diff, avg_dist


    # CHATGPT
    def compute_dynamic_R(self, distance, yaw_diff, is_corner, reprojection_error):
        base_R = self.R.copy()

        scale = 1.0

        # Increase scale based on distance (quadratic growth)
        scale *= (1 + (distance / 2000)**2)

        # Increase scale based on yaw difference (quadratic)
        scale *= (1 + (yaw_diff / np.radians(15))**2)

        # Increase scale if it's a corner marker (less reliable)
        if is_corner:
            scale *= 2.0

        # Increase scale based on reprojection error (normalize and scale)
        reproj_scale = max(1.0, reprojection_error / 2.0)
        scale *= reproj_scale

        # Return scaled covariance matrix
        R_matrix = base_R * scale

        return R_matrix





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