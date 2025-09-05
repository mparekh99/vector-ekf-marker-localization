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
MAX_POSE_JUMP = 50          # mm
ANGLE_THRESHOLD_RADIANS = np.radians(15)



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

        self.last_pose = None
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
    
    def estimate_distance(self, box_height_px, frame):
        frame_height, frame_width = frame.shape[:2]

        focal_length_px = (frame_height / 2) / math.tan(math.radians(FOV_VERTICAL / 2))

        if box_height_px <= 0:
            return float('inf')

        distance_mm = (focal_length_px * MARKER_LENGTH) / box_height_px
        return distance_mm

    # CHATGPTS    
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
            return None, frame
        
        poses = []
        fallback_poses = []  # Store all poses for fallback
        
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
                flags=cv2.SOLVEPNP_IPPE_SQUARE,  #EPnP, etc.
            )


            best_error = float('inf')
            best_index = -1

            for i in range(len(rvecs)):
                rvec = rvecs[i]
                tvec = tvecs[i]

                error = self.reprojection_error(rvec, tvec, image_points)

                if error < best_error:
                    best_error = error
                    best_index = i


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

                marker_normal = -R_cm[:, 2]  # Flip to point toward the camera
                cos_angle = marker_normal[2]  # dot with [0, 0, 1]
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                # print(f"Angle (radians): {angle_rad:.3f}")

                if self.filter_pose(camera_global, corner, frame, fixed_pos):
                    if angle_rad < ANGLE_THRESHOLD_RADIANS:
                        print("Accepted: marker viewed head-on")
                        poses.append(camera_global)
                    else:
                        fallback_poses.append((error, camera_global))

        if len(poses) == 0:
            if len(fallback_poses) == 0:
                return None, frame
            
            best_fallback = min(fallback_poses, key=lambda x: x[0])[1]

            
            return best_fallback, frame
        
        # Average translations
        translations = np.array([p[:3, 3] for p in poses])
        mean_translation = np.mean(translations, axis=0)

        # Average rotations using quaternions
        rotations = [R.from_matrix(p[:3, :3]) for p in poses]
        quats = np.array([r.as_quat() for r in rotations])
        
        # Normalize quaternions and flip to same hemisphere
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]

        mean_quat = np.mean(quats, axis=0)
        mean_quat /= np.linalg.norm(mean_quat)
        mean_rot = R.from_quat(mean_quat).as_matrix()

        # Compose averaged pose matrix
        camera_global_avg = np.eye(4)
        camera_global_avg[:3, :3] = mean_rot
        camera_global_avg[:3, 3] = mean_translation

        print(mean_translation)

        return camera_global_avg, frame
    

    def filter_pose(self, pose, corner, frame, fixed_pos):

        width_px, height_px = self.get_marker_size_in_px(corner) 
        avg_px_size = (width_px + height_px) / 2 
        distance = self.estimate_distance(avg_px_size, frame) - 25.4
        pos = pose[:3,3]
        euclidian_dist = math.sqrt(((fixed_pos[0] -pos[0])**2 + ((fixed_pos[1] - pos[1])**2)))

        if abs(distance - euclidian_dist) < MAX_DISTANCE_ERROR:
            return True
        
        
        return False
        # print("DIST", distance - 25.4)

    def compute_pose_score(self, reproj_error, angle_rad, distance_error_mm):
        # Lower reprojection error, lower angle, and lower distance error → higher score
        score = 1.0

        # Normalize and penalize
        score -= min(reproj_error / 10.0, 1.0)  # reproj error over 10px is heavily penalized
        score -= min(angle_rad / np.radians(45), 1.0)  # angles beyond 45° are bad
        score -= min(abs(distance_error_mm) / 50.0, 1.0)  # more than 50mm error → reduce confidence

        return max(score, 0.0)




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
    

    # def log_quaternion(self, frame_number, quat, log_file='quaternion_log.csv'):
    #     # Convert rotation matrix to quaternion
    #     # quat = R.from_matrix(R_cm).as_quat()  # [x, y, z, w]

    #     file_exists = os.path.isfile(log_file)

    #     with open(log_file, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         if not file_exists:
    #             writer.writerow(['frame', 'qx', 'qy', 'qz', 'qw'])  # header
    #         writer.writerow([frame_number] + quat.tolist())


    