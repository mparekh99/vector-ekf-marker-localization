import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
import pickle
import csv
import os


# https://www.chiefdelphi.com/t/finding-camera-location-with-solvepnp/159685/6

# CONSTANTS
FOV_HORIZONTAL = 90
FOV_VERTICAL = 50
REAL_VECTOR_HIEGHT = 50  # MM


# https://www.youtube.com/watch?v=bs81DNsMrnM
def drawAxes(img, corners, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    # print("CRASHING????")
    corner = tupleOfInts(corners[0].ravel())
    print("FEOFOIEWJFOIWEJO")
    img = cv2.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 5)
    print("FWASTTTTTTT")
    img = cv2.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 5)
    return img

class MarkerProcessor:
    def __init__(self, marker_world):
        self.marker_world = marker_world
        self.marker_transforms = marker_world.marker_transforms  

        # LOAD IN CALIBRATED STUFF
        calib = pickle.load(open("camera_calib_data.pkl", "rb"))
        self.mtx = calib["camera_matrix"]
        self.dist = calib["dist_coeff"]
        self.rvecs = calib["rvecs"]
        self.tvecs = calib["tvecs"]

        self.optimal = calib["new_camera_matrix"]


        # self.prev_rotation = None
        self.prev_quat = None  # Store last quaternion for continuity
        self.frame_number = 0


        # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        L = 50
        # Because Square is 50mm x 50mm
        self.obj_points = np.float32([
            [-L/2,  L/2, 0],  # point 0 - top-left
            [ L/2,  L/2, 0],  # point 1 - top-right
            [ L/2, -L/2, 0],  # point 2 - bottom-right
            [-L/2, -L/2, 0],  # point 3 - bottom-left
        ])


        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def preprocess_frame(self, frame_pil):
        self.frame_number += 1
        frame_np = np.array(frame_pil)
        un_distort = cv2.undistort(frame_np, self.mtx, self.dist, None, self.optimal)
        fix_frame = cv2.cvtColor(un_distort, cv2.COLOR_RGB2BGR)
        return fix_frame
    
    def estimate_distance(self, box_height_px, frame):
        frame_height, frame_width = frame.shape[:2]

        focal_length_px = (frame_height / 2) / math.tan(math.radians(FOV_VERTICAL / 2))

        if box_height_px <= 0:
            return float('inf')

        distance_mm = (focal_length_px * REAL_VECTOR_HIEGHT) / box_height_px
        return distance_mm


  

    
    def invert_homogeneous(self, T):
        R_mat = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_mat.T
        T_inv[:3, 3] = -R_mat.T @ t
        return T_inv

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        

        if ids is None:
            return None, frame

        # center_x, center_y = frame_width / 2, frame_height / 2
        # max_center_offset_px = 100  # tweak this threshold for "centered" markers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # worldPtsCur = ?

        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id not in self.marker_transforms:
                    continue

            cornersRefined = cv2.cornerSubPix(gray, corner[0],(11, 11),(-1, -1),criteria)
            _ , rvecs, tvecs = cv2.solvePnP(self.obj_points, cornersRefined, self.mtx, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)




            R_cm, _ = cv2.Rodrigues(rvecs)
            rot = R.from_matrix(R_cm)
            q = rot.as_quat()




            if self.prev_quat is not None:
                
                print(q, self.prev_quat, np.dot(q, self.prev_quat))
                if np.dot(q, self.prev_quat)  < 0.7:
                #     print("ENTERING")
                    continue


            #     # if np.sign(q[2]) != np.sign(self.prev_quat[2]):  # IF qz flips 

            #     # if np.dot(q, self.prev_quat) < 0:
            #     #     print("FLIPPING at Frame", self.frame_number)
            #     #     q *= -1  # Flip quaternion

            # self.prev_quat = q.copy()


            # R_flipped = R.from_quat(q).as_matrix()

            # self.log_rotation_matrix(self.frame_number, R_flipped)





            # self.log_rotation_matrix(self.frame_number, R_cm)
            # self.log_quaternion(self.frame_number, q)


            marker_camera = np.eye(4)
            marker_camera[:3, :3] = R_cm
            marker_camera[:3, 3] = tvecs.flatten()

            # rot = R.from_matrix(marker_camera[:3, :3])
            # euler = rot.as_euler('xyz', degrees=True)
            # print("Euler angles (XYZ, degrees):", euler)

            # # Invert it to get camera->marker (i.e. camera pose in marker's frame)
            T_camera_in_marker = self.invert_homogeneous(marker_camera)

            # # Euler angles for visualization (in degrees

            marker = self.marker_transforms[marker_id]
            fixed_position, fixed_rotation, translation = marker["pos"], marker["rot"], marker["translation"]

            marker_global = np.eye(4)
            marker_global[:3, :3] = fixed_rotation

            marker_global[:3, 3] = fixed_position

            vector_pos = marker_global @ T_camera_in_marker

            np.set_printoptions(suppress=True, precision=6)


            print(vector_pos[:3,3])


        return 0
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