import anki_vector
from anki_vector.util import degrees
import cv2
import numpy as np
import pickle

# Load camera calibration
calib = pickle.load(open("camera_calib_data.pkl", "rb"))
camera_matrix = calib['camera_matrix']
dist_coeffs = calib['dist_coeff']

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.05  # meters (50mm)

def drawAxes(img, corners, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    # corners is shape (1,4,2), get rid of outer dim:
    corners = corners.reshape(-1, 2)  # shape (4, 2)
    
    origin = tupleOfInts(corners[0])  # first corner point (x, y)
    
    img = cv2.line(img, origin, tupleOfInts(imgpts[0].ravel()), (255, 0, 0), 5)  # X axis in red
    img = cv2.line(img, origin, tupleOfInts(imgpts[1].ravel()), (0, 255, 0), 5)  # Y axis in green
    img = cv2.line(img, origin, tupleOfInts(imgpts[2].ravel()), (0, 0, 255), 5)  # Z axis in blue
    return img


def main(robot_serial):
    with anki_vector.Robot(robot_serial) as robot:
        robot.behavior.set_head_angle(degrees(7.0))
        robot.behavior.set_lift_height(0.0)
        robot.camera.init_camera_feed()

        print("Starting camera feed...")

        # Define 3D axes points (5 cm length)
        axis_3d_points = np.float32([
            [marker_length, 0, 0],  # X axis
            [0, marker_length, 0],  # Y axis
            [0, 0, marker_length]   # Z axis
        ])

        while True:
            img = robot.camera.latest_image.raw_image  # PIL image
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i in range(len(ids)):
                    retval, rvec, tvec = cv2.solvePnP(
                        np.array([
                            [-marker_length / 2, marker_length / 2, 0],
                            [marker_length / 2, marker_length / 2, 0],
                            [marker_length / 2, -marker_length / 2, 0],
                            [-marker_length / 2, -marker_length / 2, 0],
                        ], dtype=np.float32),
                        corners[i].reshape(-1, 2),
                        camera_matrix,
                        dist_coeffs,
                        cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if retval:
                        imgpts, _ = cv2.projectPoints(axis_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
                        print(corners[i])
                        print(imgpts)


                        frame = drawAxes(frame, corners[i], imgpts)
                        print("CEHQWJDKQWJN")

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.imshow("Vector Aruco Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    robot_serial = "00806b78"
    main(robot_serial)
