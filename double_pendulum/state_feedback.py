import numpy as np
import scipy.signal
import scipy.linalg
from double_pendulum.model import DoublePendulum


class FullStateFeedback:
    """完全状態フィードバック"""

    def __init__(self, A: np.ndarray, b: np.ndarray, F: np.ndarray):
        self.F = F
        self.A = A
        self.b = b

        assert F.shape == (b.shape[1], A.shape[0])

        self.A_closed_loop = A - b @ F

    def calc_u(self, x: np.ndarray) -> float:
        return (-self.F @ x)[0]

    def poles(self):
        return np.linalg.eig(self.A_closed_loop)[0]

    def print_feedback_system_info(self):
        poles = self.poles()
        print("Feedback System ---")
        print(f"F: \n{self.F}")
        print(f"A_closed_loop: \n{self.A_closed_loop}")
        print(f"poles: {poles}")


def calc_pole_placement(
    A: np.ndarray, b: np.ndarray, target_poles: np.ndarray
) -> np.ndarray:
    """極配置関数"""
    assert target_poles.shape == (A.shape[0],)

    # 極配置
    # [place_poles — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html)
    full_state_feedback_obj = scipy.signal.place_poles(A, b, target_poles)
    F: np.ndarray = getattr(full_state_feedback_obj, "gain_matrix")

    return F


def calc_continuous_lqr(
    A: np.ndarray, b: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """連続リカッチを解いて最適フィードバックゲインを求める"""
    # リカッチ方程式を解く
    P = scipy.linalg.solve_continuous_are(A, b, Q, R)

    F = np.linalg.solve(R, b.T @ P)
    return F


def calc_descrete_lqr(
    A: np.ndarray, b: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """離散リカッチを解いて最適フィードバックゲインを求める"""
    # リカッチ方程式を解く
    P = scipy.linalg.solve_discrete_are(A, b, Q, R)

    F = np.linalg.solve(R + b.T @ P @ b, b.T @ P @ A)
    return F
