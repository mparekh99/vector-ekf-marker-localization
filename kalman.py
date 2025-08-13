import numpy as np
import math
from utils import wrap_angle

class KalmanFilter:

    def __init__(self):
        # self.P_last = np.diag([0.5, 0.5, np.deg2rad(5.0)])**2
        self.P_last = np.diag([1.0, 1.0, np.deg2rad(10.0)])**2

        # self.Q = np.array([
        #     [0.1, 0.0, 0.0],  # position drift
        #     [0.0, 0.1, 0.0],
        #     [0.0, 0.0, np.deg2rad(1.5)**2]  # orientation drift
        # ])
        # self.Q = np.array([
        #     [0.3, 0.0, 0.0],
        #     [0.0, 0.3, 0.0],
        #     [0.0, 0.0, np.deg2rad(3.0)**2]
        # ])

        self.Q = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, np.deg2rad(1.0)**2]
        ])

        self.R = np.array([
            [25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
            [0.0, 0.0, np.deg2rad(6.0)**2]
        ])




        self.H = np.eye(3)


        # self.R = np.array([
        #     [0.01, 0.0, 0.0],        # X noise ~10cm
        #     [0.0, 0.01, 0.0],        # Y noise
        #     [0.0, 0.0, np.deg2rad(5.0)**2]  # Orientation ~5 deg
        # ])


        self.x_last = 0
        self.y_last = 0
        self.theta_last = math.pi / 2


    def initial_predict(self, velocity, timestep, theta):
        
        # ODOMETRY PREDICTOIN -- using the Motion Model Taken From: https://www.youtube.com/watch?v=LrsTBWf6Wsc
        
        theta_k = self.theta_last + theta * timestep
        x_k = self.x_last + velocity * math.cos(theta_k) * timestep
        y_k = self.y_last + velocity * math.sin(theta_k) * timestep

        # COMPUTE JACOBIAN MATRIX 
        # Look at notebook on how I got this: Partial Derivate with Jacobian Matrix
        F_k = np.array([[1, 0, -velocity * math.sin(theta_k) * timestep],
                        [0, 1, -velocity * math.cos(theta_k) * timestep], 
                        [0, 0, 1]])

        # Predict New Covariance 
        self.P_last = F_k @ self.P_last @ F_k.T + self.Q


        # UPDATE
        self.x_last = x_k
        self.y_last = y_k
        self.theta_last = theta_k

        return x_k, y_k, theta_k


    def update(self, x, y, theta):
        x_est = np.array([self.x_last, self.y_last, self.theta_last])
        z_k = np.array([x , y, theta]) # camera measurement
        y_k = z_k - x_est # difference between measurement and prediction

        y_k[2] = wrap_angle(y_k[2])


        # MINI FILTER
        threshold = 150
        if np.any(np.abs(y_k[:2]) > threshold):
            print("Rejected CAM Reading: ", z_k)
            # print("Measurement rejected due to large innovation:", y_k)
            return self.x_last, self.y_last, self.theta_last
        


        print(f"\n[CAM Update] Measurement z_k:    x={z_k[0]:.4f}, y={z_k[1]:.4f}, theta={z_k[2]:.4f}")
        print(f"[Prediction] x_est:              x={x_est[0]:.4f}, y={x_est[1]:.4f}, theta={x_est[2]:.4f}")
        print(f"[Residual]   y_k:                dx={y_k[0]:.4f}, dy={y_k[1]:.4f}, dtheta={y_k[2]:.4f}")


        # KALMAN GAIN
        S = self.H @ self.P_last @ self.H.T + self.R
        K = self.P_last @ self.H.T @ np.linalg.inv(S)
        # print("Kalman Gain K:\n", K)

 
        # Update state
        x_updated = x_est + K @ y_k

        # print(f"Updated state before covariance update: {x_updated}")

        # Update Error
        I = np.eye(3)
        self.P_last = (I - K @ self.H) @ self.P_last


        # Save updated state
        self.x_last = x_updated[0]
        self.y_last = x_updated[1]
        self.theta_last = x_updated[2]

        print(f"[Updated State] x={self.x_last:.4f}, y={self.y_last:.4f}, theta={self.theta_last:.4f}")
        # print(f"Updated covariance P_last:\n{self.P_last}")

        return self.x_last, self.y_last, self.theta_last
