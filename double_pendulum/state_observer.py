import numpy as np
import scipy.signal
from enum import Enum
from typing import Protocol

from double_pendulum.model import DoublePendulum


class SensingMode(Enum):
    FullOrderStateObserverButNotUseForFeedback = 1
    FullOrderStateObserverMode = 2
    MinimalOrderStateObserverMode = 3


class ObserverProtocol(Protocol):
    def calc_next_x_hat(self, u: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def get_x_hat(self) -> np.ndarray: ...

    def get_A_closed_loop(self) -> np.ndarray: ...


class FullOrderStateObserver(ObserverProtocol):
    """同一次元状態オブザーバー"""

    def __init__(
        self,
        model: DoublePendulum,
        C: np.ndarray,
        L: np.ndarray,
        dt: float,
        x_hat0: np.ndarray,
    ):
        A, b = model.calc_continuous_linear_system()
        self.dim_x: int = A.shape[0]
        assert C.shape[1] == self.dim_x
        self.dim_y: int = C.shape[0]
        assert L.shape == (self.dim_x, self.dim_y)

        self.A = A
        self.b = b
        self.C = C
        self.L = L
        self.dt = dt

        self.A_closed_loop = self.A - self.L @ self.C

        # 初期値
        self.x_hat = x_hat0

    def get_x_hat(self) -> np.ndarray:
        return self.x_hat

    def get_A_closed_loop(self) -> np.ndarray:
        return self.A_closed_loop

    def _calc_x_hat_dot(
        self, x_hat: np.ndarray, u: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        assert y.shape == (self.dim_y,)
        # オブザーバーの計算
        x_hat_dot = self.A @ x_hat + self.b @ u + self.L @ (y - self.C @ x_hat)
        return x_hat_dot

    def calc_next_x_hat(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_hat = self.x_hat
        dt = self.dt

        # RK4 で計算
        # y と u は固定でいいんか？
        k1 = self._calc_x_hat_dot(x_hat, u, y)
        k2 = self._calc_x_hat_dot(x_hat + 0.5 * dt * k1, u, y)
        k3 = self._calc_x_hat_dot(x_hat + 0.5 * dt * k2, u, y)
        k4 = self._calc_x_hat_dot(x_hat + dt * k3, u, y)

        x_hat_next = x_hat + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        self.x_hat = x_hat_next
        return x_hat_next

    def print_observer_info(self):
        poles = np.linalg.eig(self.A_closed_loop)[0]
        print("Full-Order Observer ---")
        print(f"L: \n{self.L}")
        print(f"A_closed_loop: \n{self.A_closed_loop}")
        print(f"poles: {poles}")

    @classmethod
    def calc_pole_placement(
        cls, model: DoublePendulum, C: np.ndarray, target_poles: np.ndarray
    ) -> np.ndarray:
        """極配置関数"""
        A, _ = model.calc_continuous_linear_system()
        assert target_poles.shape == (A.shape[0],)

        # 双対問題
        A = A.T
        B = C.T

        # 極配置
        # [place_poles — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html)
        full_state_feedback_obj = scipy.signal.place_poles(A, B, target_poles)
        F: np.ndarray = getattr(full_state_feedback_obj, "gain_matrix")

        # 双対問題
        L = F.T

        return L


class MinimalOrderStateObserver:
    """最小次元オブザーバー"""

    def __init__(
        self,
        model: DoublePendulum,
        C: np.ndarray,
        L: np.ndarray,
        dt: float,
        z0: np.ndarray,
    ):
        A, b = model.calc_continuous_linear_system()
        self.dim_x: int = A.shape[0]
        self.dim_y: int = C.shape[0]
        self.dim_z: int = self.dim_x - self.dim_y
        assert C.shape == (self.dim_y, self.dim_x)
        assert L.shape == (self.dim_z, self.dim_y)
        assert z0.shape == (self.dim_z,)

        self.C = C
        self.L = L
        self.dt = dt

        # 初期値
        self.z = z0

        # 簡略化
        # C は [1, 0, 0, 0] or [[1, 0, 0, 0]; [0, 1, 0, 0]]
        # T = np.identity(self.dim_x)

        A_11 = A[: self.dim_y, : self.dim_y]
        A_12 = A[: self.dim_y, self.dim_y :]
        A_21 = A[self.dim_y :, : self.dim_y]
        A_22 = A[self.dim_y :, self.dim_y :]

        B_1 = b[: self.dim_y]
        B_2 = b[self.dim_y :]

        self.F = A_22 - L @ A_12
        self.G = A_22 @ L + A_21 - L @ (A_12 @ L + A_11)
        self.H = B_2 - L @ B_1
        self.W = np.block(
            [
                [np.zeros((self.dim_y, self.dim_z))],
                [np.eye(self.dim_z)],
            ]
        )
        self.V = np.block(
            [
                [np.eye(self.dim_y)],
                [L],
            ]
        )

        self.x_hat = self.W @ self.z

    def get_x_hat(self) -> np.ndarray:
        return self.x_hat

    def get_A_closed_loop(self) -> np.ndarray:
        return self.F

    def _calc_z_dot(self, z: np.ndarray, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        z_dot = self.F @ z + self.G @ y + self.H @ u
        return z_dot

    def calc_next_x_hat(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = self.z
        dt = self.dt

        k1 = self._calc_z_dot(z, u, y)
        k2 = self._calc_z_dot(z + 0.5 * dt * k1, u, y)
        k3 = self._calc_z_dot(z + 0.5 * dt * k2, u, y)
        k4 = self._calc_z_dot(z + dt * k3, u, y)
        next_z = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt

        self.x_hat = self.W @ next_z + self.V @ y

        self.z = next_z
        return self.x_hat

    def print_observer_info(self):
        poles = np.linalg.eig(self.F)[0]
        print("Minimal-Order Observer ---")
        print(f"C: \n{self.C}")
        print(f"L: \n{self.L}")
        print(f"F: \n{self.F}")
        print(f"G: \n{self.G}")
        print(f"H: \n{self.H}")
        print(f"W: \n{self.W}")
        print(f"V: \n{self.V}")
        print(f"poles: {poles}")

    @classmethod
    def calc_pole_placement(
        cls, model: DoublePendulum, C: np.ndarray, target_poles: np.ndarray
    ) -> np.ndarray:
        A, _ = model.calc_continuous_linear_system()
        dim_x = A.shape[0]
        dim_y = C.shape[0]

        assert target_poles.shape == (dim_x - dim_y,)

        A_12 = A[:dim_y, dim_y:]
        A_22 = A[dim_y:, dim_y:]

        # 極配置
        full_state_feedback_obj = scipy.signal.place_poles(A_22.T, A_12.T, target_poles)
        L: np.ndarray = getattr(full_state_feedback_obj, "gain_matrix").T

        return L
