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
from utils import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
from world import Marker_World
from pose_tracker import PoseTracker


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

# PLOT 
def plot_scene(ax, pose_tracker):
    ax.clear()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Vector and Marker Positions")
    ax.grid(True)

    # Plot each marker
    for marker_id in pose_tracker.world.marker_transforms:

        marker = pose_tracker.world.marker_transforms[marker_id]
        pos = marker['pos']
        x = pos[0] * 1000
        y = pos[1] * 1000
        ax.plot(x, y, 'ro')
        ax.text(x + 5, y + 5, f'Marker {marker_id}', color='red', fontsize=8)


    # Plot the robot's position
    x = pose_tracker.position[0]
    y = pose_tracker.position[1]
    ax.plot(x, y, 'bo')
    ax.text(x + 5, y + 5, "Vector", color='blue')

    # Calulate YAW
    heading = pose_tracker.heading

    # Draw direction arrow
    length = 30
    dx = length * math.cos(heading)
    dy = length * math.sin(heading)
    ax.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')

def setup(robot_serial: str):
    """Setup robot, plotting, threading."""
    robot = anki_vector.Robot(robot_serial)
    robot.connect()

    # Camera + head settings
    robot.behavior.set_head_angle(degrees(7.0))
    robot.behavior.set_lift_height(0.0)
    robot.camera.init_camera_feed()

    # Start keyboard listener in background
    listener_thread = threading.Thread(target=teleop_listener, daemon=True)
    listener_thread.start()

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots()

    return robot, fig, ax


def main():

    pose_tracker = PoseTracker()
    robot, fig, ax = setup("00603f86")

    try:
        while True:
            frame_pil = robot.camera.latest_image.raw_image
            frame = pose_tracker.update_pose(frame_pil, robot)
            plot_scene(ax, pose_tracker)
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

    finally:
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
