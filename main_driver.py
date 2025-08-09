import anki_vector 
import cv2 
import numpy as np
from PIL import Image 
import time
from anki_vector.util import degrees
from scipy.spatial.transform import Rotation as R
import pickle
import keyboard
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




calib_result_pickle = pickle.load(open("camera_calib_pickle.p", "rb"))
mtx = calib_result_pickle["mtx"]
optimal_camera_matrix = calib_result_pickle["optimal_camera_matrix"]
dist = calib_result_pickle["dist"]

# === ArUco marker parameters ===
aruco_marker_side_length = 0.0254  # meters (1 inch)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()

obj_points = np.float32([
    [0,0,0], # top left
    [ 0.0254 ,0,0],  # top right
    [ 0.0254  , 0.0254  ,0], # bottom right
    [0, 0.0254  ,0]  # bottom left
])

# === Keyboard control state ===
control_state = {
    "left": False,
    "right": False,
    "forward": False,
    "backward": False
}

# === Keyboard teleoperation ===
def teleop_listener():
    while True:
        control_state["forward"] = keyboard.is_pressed("up")
        control_state["backward"] = keyboard.is_pressed("down")
        control_state["left"] = keyboard.is_pressed("left")
        control_state["right"] = keyboard.is_pressed("right")
        time.sleep(0.01)

# === Rotation helper ===
def rotation_matrix_y(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,              1]
    ])


def rotation_matrix_x(angle_degrees):
    theta = np.radians(angle_degrees)
    return np.array([
        [1, 0,            0           ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

# === Invert transform ===
def invert_homogeneous(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv

# === Compute robot pose in global frame ===
def get_robot_pose_in_global(rvec, tvec, marker_global_pose):
    R_cm, _ = cv2.Rodrigues(rvec)
    T_cm = np.eye(4)
    T_cm[:3, :3] = R_cm
    T_cm[:3, 3] = tvec.flatten()
    camera_in_marker = invert_homogeneous(T_cm)
    T_cg = marker_global_pose @ camera_in_marker
    return T_cg


def main():
    with anki_vector.Robot("00603f86") as robot:
        # Setup robot and camera as before
        robot.behavior.set_head_angle(degrees(7.0))
        robot.behavior.set_lift_height(0.0)
        robot.camera.init_camera_feed()

        listener_thread = threading.Thread(target=teleop_listener, daemon=True)
        listener_thread.start()
        # Set up plot once before loop:

        # print('HOOOWLWMWOJDOW')


        while True:
            frame_pil = robot.camera.latest_image.raw_image
            frame_np = np.array(frame_pil)
            frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, rejected = detector.detectMarkers(frame)
            # print("HOHEFOHEWOFHOIEF")
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # print("bkjsbfkjbwvkje cj")

                for i, corner in enumerate(corners):
                    image_points = corner.reshape((4, 2)).astype(np.float32)

                    success, rvec, tvec = cv2.solvePnP(obj_points, image_points, mtx, dist)

                    if not success:
                        continue

                    # cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.015)
                    # imgpts, _ = cv2.projectPoints(obj_points, rvec, tvec, mtx, dist)
                    # imgpts = imgpts.reshape(-1, 2).astype(int)

                    # origin = tuple(imgpts[0])
                    # cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 2, cv2.LINE_AA)   # X - Red
                    # cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 2, cv2.LINE_AA)   # Y - Green
                    # cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 2, cv2.LINE_AA)  
                    # # print(f"Marker ID: {marker_id}")
                    # print("Rotation Vector (rvec):", rvec.flatten())
                    # print("Translation Vector (tvec):", tvec.flatten())

                    # R_marker_to_global = np.array([
                    #     [1,  0,  0],   # X stays X
                    #     [0,  0,  1],   # Z becomes Y
                    #     [0, -1,  0]    # -Y becomes Z
                    # ])





                    R_cm, _ = cv2.Rodrigues(rvec)
                    r = R.from_matrix(R_cm)
                    marker_camera = np.eye(4)
                    marker_camera[:3, 3] = tvec.flatten()
                    marker_camera[:3, :3] = r.as_matrix()
                    # print(f'Marker in Camera Frame: {marker_camera[:3, :3]}')

                    camera_marker = invert_homogeneous(marker_camera)

                    # Rotation to align marker frame to global frame:
                    # R_marker_to_global = np.array([
                    #     [1, 0, 0],   # X stays X (right)
                    #     [0, 0, 1],   # marker Z (forward) becomes global Y (forward)
                    #     [0, -1, 0]   # marker Y (down) becomes global -Z (up)
                    # ])


                    # rot = camera_marker[:3, :3]
                    # inv = np.linalg.inv(rot)
                    # print(inv)
                    #inv is the local rotation of the front marker 
                    
                    # Front 
                    # rot = rotation_matrix_x(-90) @ R_cm
                    # Left 
                    # rot =  rotation_matrix_z(90) @ rotation_matrix_x(-90) @ R_cm
                    # Right 
                    # rot =  rotation_matrix_z(-90) @ rotation_matrix_x(-90) @ R_cm
                    
                    
                    # print("HELLo")
                    marker_global = np.eye(4)
                    marker_global[:3, :3] = rot   # Because of opencv layout x-> right, y -> down, z ->forwards
                    # marker_global[:3, 3] = np.array([0, 0.2, 0])
                    marker_global[:3, 3] = np.array([-0.2, 0, 0])

                    camera_global = marker_global @ camera_marker # markers cancel out


                    vector_pos = camera_global[:3,3]
                    vector_pos *= 1000
                    print(f'Vector Pos {vector_pos}')
                    np.set_printoptions(suppress=True, precision=6)
                    print(f'Camera in GlobalPose: \n{camera_global[:3, :3]}')


                    print()
                    # print("X axis vector:", camera_global[:3, 0])
                    # print("Y axis vector:", camera_global[:3, 1])
                    # print("Z axis vector:", camera_global[:3, 2])








                            # euler_deg = r.as_euler('xyz', degrees=True)
                            # print(f"Euler Angles (degrees): {euler_deg}")


            cv2.imshow("Vector Camera View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if control_state["forward"]:
                robot.motors.set_wheel_motors(100, 100)
            elif control_state["backward"]:
                robot.motors.set_wheel_motors(-100, -100)
            elif control_state["left"]:
                robot.motors.set_wheel_motors(-50, 50)
            elif control_state["right"]:
                robot.motors.set_wheel_motors(50, -50)
            else:
                robot.motors.set_wheel_motors(0, 0)

        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
