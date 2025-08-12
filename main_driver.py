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
from enum import Enum
import matplotlib.pyplot as plt
import math



class Marker(Enum):
    FRONT_RIGHT = 4
    FRONT_LEFT = 3
    RIGHT = 2
    FRONT = 1
    LEFT = 0
    BOTTOM = 5
    BOTTOM_LEFT = 6
    BOTTOM_RIGHT = 7




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
    [ 0.05 ,0,0],  # top right
    [ 0.05  , 0.05  ,0], # bottom right
    [0, 0.05  ,0]  # bottom left
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

# PLOT 
def plot_scene(ax, pose, marker_transforms):
    ax.clear()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Vector and Marker Positions")
    ax.grid(True)

    # Plot each marker
    for marker_id in marker_transforms:

        pos, _, _ = marker_transforms[marker_id]
        x = pos[0] * 1000
        y = pos[1] * 1000
        ax.plot(x, y, 'ro')
        ax.text(x + 5, y + 5, f'Marker {marker_id}', color='red', fontsize=8)

    # Plot the robot's position
    pos = pose[:3, 3]
    x = pos[0]
    y = pos[1]
    ax.plot(x, y, 'bo')
    ax.text(x + 5, y + 5, "Vector", color='blue')

    # Calulate YAW
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    yaw = yaw + math.pi / 2 # ROTATE BY 90 To Correct it

    # Draw direction arrow
    length = 30
    dx = length * math.cos(yaw)
    dy = length * math.sin(yaw)
    ax.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')




def main():

    marker_transforms = {
        Marker.LEFT.value: (np.array([-0.2, 0, 0]), rotation_matrix_z(90) @ rotation_matrix_x(-90), np.array([-188, -1.9])),
        Marker.FRONT_LEFT.value: (np.array([-0.2, 0.2, 0]), rotation_matrix_z(45) @ rotation_matrix_x(-90), np.array([-200, 200])), 
        Marker.FRONT.value: (np.array([0, 0.2, 0]), rotation_matrix_x(-90), np.array([-2, 200])),
        Marker.FRONT_RIGHT.value: (np.array([0.2, 0.2, 0]), rotation_matrix_z(-45) @ rotation_matrix_x(-90), np.array([197, 196])),
        Marker.RIGHT.value: (np.array([0.2, 0, 0]), rotation_matrix_z(-90) @ rotation_matrix_x(-90), np.array([164, 2.7])),
        Marker.BOTTOM.value: (np.array([0, -0.2, 0]), rotation_matrix_z(180) @ rotation_matrix_x(-90), np.array([0, -207])),
        Marker.BOTTOM_LEFT.value: (np.array([-0.2, -0.2, 0]), rotation_matrix_z(135) @ rotation_matrix_x(-90), np.array([-202, -205])), 
        Marker.BOTTOM_RIGHT.value: (np.array([0.2, -0.2, 0]), rotation_matrix_z(-135) @ rotation_matrix_x(-90), np.array([209, -207])),
    }


    with anki_vector.Robot("00603f86") as robot:
        # Setup robot and camera as before
        robot.behavior.set_head_angle(degrees(7.0))
        robot.behavior.set_lift_height(0.0)
        robot.camera.init_camera_feed()

        listener_thread = threading.Thread(target=teleop_listener, daemon=True)
        listener_thread.start()

        plt.ion()
        fig, ax = plt.subplots()

        last_known_pose = None

        while True:
            frame_pil = robot.camera.latest_image.raw_image
            frame_np = np.array(frame_pil)
            frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, rejected = detector.detectMarkers(frame)
            # print("HOHEFOHEWOFHOIEF")

            camera_poses = []

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]

                if marker_id not in marker_transforms:
                    continue

                image_points = corner.reshape((4, 2)).astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(obj_points, image_points, mtx, dist)
                
                if not success:
                    continue

                # BUILD HOMOGENOUS TRANSFORMATION FOR POSE READING 
                R_cm, _ = cv2.Rodrigues(rvec)
                r = R.from_matrix(R_cm)
                marker_camera = np.eye(4)
                marker_camera[:3, 3] = tvec.flatten()
                marker_camera[:3, :3] = r.as_matrix()

                # INVERT TRANSFORM TO GET in terms of MARKER FRAME
                camera_marker = invert_homogeneous(marker_camera)
                marker_global = np.eye(4)
                # translation = [0,0] # TEMP TRANSLATION WILL BE CHANGED 

                fixed_position, fixed_rotation, translation = marker_transforms[marker_id]
                marker_global[:3, :3] = fixed_rotation @ R_cm
                marker_global[:3, 3] = fixed_position

                vector_pos = marker_global @ camera_marker

                pos = vector_pos[:3, 3] * 1000 / 3.1 

                pos[0] += translation[0]
                pos[1] += translation[1]

                # print(pos)

                #UPDATE x
                vector_pos[:3, 3] = pos

                np.set_printoptions(suppress=True, precision=4)

                # print(pos[:2])


                camera_poses.append(vector_pos)

            
            if camera_poses:
                # I CAN DO THIS BECAUSE ALL IN SAME FRAME NOW!!!
                avg_translation = np.mean([pose[:3, 3] for pose in camera_poses], axis=0)

                # AVG ROT
                avg_rotation = np.mean([pose[:3, :3] for pose in camera_poses], axis=0)

                # USE SVD-- from CHATGPT -- But basically gets me the averaged matrix as a matrix, since averaging matrixies could mess up matrix in general.
                U, U, VT = np.linalg.svd(avg_rotation)

                camera_global_avg = np.eye(4)
                camera_global_avg[:3, :3] = avg_rotation
                camera_global_avg[:3, 3] = avg_translation

                final_vector_pos = camera_global_avg[:3, 3]
                print(final_vector_pos)

                last_known_pose = camera_global_avg
                



            if last_known_pose is not None:
                print(last_known_pose)
                plot_scene(ax, last_known_pose, marker_transforms)

                plt.draw()
                plt.pause(0.01)


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


if __name__ == '__main__':
    main()
