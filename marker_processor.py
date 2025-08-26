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
        # self.mtx = calib["mtx"]
        self.dist = calib["dist"]
        self.optimal_camera_matrix = calib["optimal_camera_matrix"]

        # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        # Because Square is 50mm x 50mm
        self.obj_points = np.float32([
            [-0.025,  0.025, 0],  # top-left
            [ 0.025,  0.025, 0],  # top-right
            [ 0.025, -0.025, 0],  # bottom-right
            [-0.025, -0.025, 0],  # bottom-left
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
    
    # CHATGPT!!!
    # def orthonormalize_rotation(self, R_mat):
    #     # Using SVD to find the closest orthonormal rotation matrix
    #     U, _, Vt = np.linalg.svd(R_mat)
    #     R_ortho = U @ Vt
    #     # Ensure a proper rotation (determinant == 1)
    #     if np.linalg.det(R_ortho) < 0:
    #         Vt[-1, :] *= -1
    #         R_ortho = U @ Vt
    #     return R_ortho


    def process_frame(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is None:
            return None, frame

        camera_poses = []
        frame_height, frame_width = frame.shape[:2]
        # center_x, center_y = frame_width / 2, frame_height / 2
        # max_center_offset_px = 100  # tweak this threshold for "centered" markers

        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id not in self.marker_transforms:
                continue

            image_points = corner.reshape((4, 2)).astype(np.float32)
            # marker_center = np.mean(image_points, axis=0)
            # x_img, y_img = marker_center

            # if abs(x_img - center_x) > max_center_offset_px or abs(y_img - center_y) > max_center_offset_px:
            #     print(f"Skipping marker {marker_id} due to image position: ({x_img:.1f}, {y_img:.1f}) not near center")
            #     continue

            # success, rvec, tvec = cv2.solvePnP(self.obj_points, image_points, self.mtx, self.dist)
            
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                image_points,
                self.optimal_camera_matrix,
                self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if not success:
                continue

            # REFINE 
            # estimating the rotation and translation that minimizes the reprojection error
            #  using a non-linear minimization method and starting from an initial estimate of the solution. 
            rvec, tvec = cv2.solvePnPRefineLM(
                self.obj_points,
                image_points,
                self.optimal_camera_matrix,
                self.dist,
                rvec,
                tvec
            )



            R_cm, _ = cv2.Rodrigues(rvec)




            # CHATGPT Tried to make it so that it would filter out stuff that its not directly facing. But this won't work because if the reading itself is bad saying its directly ahead when it isn't then this filter won't do anything
            # marker_z_cam = R_cm[:, 2]  # Marker Z-axis in camera frame
            # cos_angle = marker_z_cam[2]  # Dot product with camera Z-axis
            # angle_deg = math.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


            # # Accept only if viewing angle is close to expected "frontal" angle
            # if 73.0 > angle_deg or angle_deg > 75.0:
            #     print(f"Skipping marker {marker_id} due to viewing angle: {angle_deg}°")
            #     continue

            marker_camera = np.eye(4)
            marker_camera[:3, :3] = R_cm
            marker_camera[:3, 3] = tvec.flatten()


            np.set_printoptions(suppress=True, precision=3)

            print("MARKER IN CAMERA FRAME", marker_camera[:3, 3] * 1000)
            # print("MARKER IN CAMERA FRAME", marker_camera[:3, :3])
            print()

            camera_marker = self.invert_homogeneous(marker_camera)


            temp = camera_marker[:3, 3]
            print(f'ROBOT IN MARKER FRAME', temp * 1000)
            # print("ROBOT IN MARKER FRAME", camera_marker[:3, :3])
            print()



            marker = self.marker_transforms[marker_id]
            fixed_position, fixed_rotation, translation = marker["pos"], marker["rot"], marker["translation"]

            marker_global = np.eye(4)
            marker_global[:3, :3] = fixed_rotation    # @ R_cm  # THIS IS MAYBE WHERE I'm MESSING UP??
            marker_global[:3, 3] = fixed_position

            vector_pos = marker_global @ camera_marker

            pos = vector_pos[:3, 3] * 1000 / 3.1
            pos[0] += translation[0]
            pos[1] += translation[1]
            vector_pos[:3, 3] = pos

            # print(pos)
            # print(marker_id)

            camera_poses.append(vector_pos)

        if not camera_poses:  # No poses were calculated 
            return None, frame

        camera_global_avg = camera_poses[0] 

        return camera_global_avg, frame
    
