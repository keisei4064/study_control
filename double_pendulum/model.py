import numpy as np


default_g = 9.81
default_M_1 = 0.1
default_J_1 = 0.01
default_L_1 = 0.1
default_l_1 = 0.05
default_b_1 = 0.001
default_M_2 = 0.1
default_J_2 = 0.01
default_L_2 = 0.1
default_l_2 = 0.05
default_b_2 = 0.001
default_k = 1.0


class DoublePendulum:
    def __init__(
        self,
        *,
        g: float = default_g,
        M_1: float = default_M_1,
        J_1: float = default_J_1,
        L_1: float = default_L_1,
        l_1: float = default_l_1,
        b_1: float = default_b_1,
        M_2: float = default_M_2,
        J_2: float = default_J_2,
        L_2: float = default_L_2,
        l_2: float = default_l_2,
        b_2: float = default_b_2,
        k: float = default_k,
    ) -> None:
        self.g = g
        self.M_1 = M_1
        self.J_1 = J_1
        self.L_1 = L_1
        self.l_1 = l_1
        self.b_1 = b_1
        self.M_2 = M_2
        self.J_2 = J_2
        self.L_2 = L_2
        self.l_2 = l_2
        self.b_2 = b_2
        self.k = k

    def f(self, x: np.ndarray, u: float) -> np.ndarray:
        assert x.size == 4
        
        theta_1, theta_2, theta_dot_1, theta_dot_2 = x

        g = self.g
        M_1 = self.M_1
        J_1 = self.J_1
        L_1 = self.L_1
        l_1 = self.l_1
        b_1 = self.b_1
        M_2 = self.M_2
        J_2 = self.J_2
        L_2 = self.L_2
        l_2 = self.l_2
        b_2 = self.b_2
        k = self.k

        M2L1l2cost1t2 = M_2 * L_1 * l_2 * np.cos(theta_1 + theta_2)
        M2L1l2sint1t2 = M_2 * L_1 * l_2 * np.sin(theta_1 + theta_2)

        M = np.array(
            [
                [J_1 + M_1 * l_1**2 + M_2 * L_1**2, M2L1l2cost1t2],
                [M2L1l2cost1t2, J_2 + M_2 * l_2**2],
            ]
        )
        h = np.array(
            [
                [
                    -b_1 * theta_dot_1
                    - b_2 * (theta_dot_1 + theta_dot_2)
                    + M2L1l2sint1t2 * theta_dot_2**2
                    - ((M_1 * l_1 + M_2 * L_1) * np.sin(theta_1)) * g
                    + k * u,
                ],
                [
                    -b_2 * (theta_dot_1 + theta_dot_2)
                    + M2L1l2sint1t2 * theta_dot_1**2
                    + (M_2 * l_2 * np.sin(theta_2)) * g
                ],
            ]
        )

        theta_2dot = np.linalg.solve(M, h).reshape(-1)
        return np.array([theta_dot_1, theta_dot_2, theta_2dot[0], theta_2dot[1]])
