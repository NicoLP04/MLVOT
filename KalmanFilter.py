"""Kalman filter for a point moving in 2D (constant-velocity model).

State  x = [x, y, vx, vy]^T      Measurement z = [x, y]^T
"""
import numpy as np


class KalmanFilter:
    def __init__(self, dt: float, u_x: float, u_y: float, std_acc: float,
                 x_std_meas: float, y_std_meas: float, x0: float, y0: float) -> None:
        # control input (accelerations), 0 for a constant-velocity model
        self.u = np.array([[u_x], [u_y]])

        # initial state: position = (x0, y0), velocity = 0
        self.x = np.array([[x0], [y0], [0.0], [0.0]])

        # system model: next state = A @ state + B @ u
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[dt ** 2 / 2, 0],
                           [0, dt ** 2 / 2],
                           [dt, 0],
                           [0, dt]])

        # we only measure the position
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # process noise
        self.Q = std_acc ** 2 * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                                          [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                                          [dt ** 3 / 2, 0, dt ** 2, 0],
                                          [0, dt ** 3 / 2, 0, dt ** 2]])

        # measurement noise (x and y independent)
        self.R = np.array([[x_std_meas ** 2, 0],
                           [0, y_std_meas ** 2]])

        # covariance of the prediction error
        self.P = np.eye(4)

    def predict(self) -> None:
        """Move the state one time step forward."""
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z: tuple[float, float]) -> None:
        """Correct the prediction with a measurement z = (x, y)."""
        z = np.array([[z[0]], [z[1]]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
