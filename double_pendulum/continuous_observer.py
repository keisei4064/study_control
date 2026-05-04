import numpy as np
import scipy.signal
import scipy.linalg
from model import DoublePendulum

import continuous_controller


class FullOrderStateObserver:
    """同一次元状態オブザーバー"""

    def __init__(self, model: DoublePendulum, C: np.ndarray, L: np.ndarray):
        A, b = model.calc_continuous_linear_system()
        self.dim_x: int = A.shape[0]
        assert C.shape[1] == self.dim_x
        self.dim_y: int = C.shape[0]
        assert L.shape == (self.dim_x, self.dim_y)

        self.A = A
        self.b = b
        self.C = C
        self.L = L

    def calc_x_hat_dot(
        self, x_hat: np.ndarray, u: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        assert y.shape == (self.dim_y,)
        # オブザーバーの計算
        x_hat_dot = self.A @ x_hat + self.b @ u + self.L @ (y - self.C @ x_hat)
        return x_hat_dot

    def calc_next_x_hat(
        self, x_hat: np.ndarray, u: np.ndarray, y: np.ndarray, dt: float
    ) -> np.ndarray:
        # RK4 で計算
        # y と u は固定でいいんか？
        k1 = self.calc_x_hat_dot(x_hat, u, y)
        k2 = self.calc_x_hat_dot(x_hat + 0.5 * dt * k1, u, y)
        k3 = self.calc_x_hat_dot(x_hat + 0.5 * dt * k2, u, y)
        k4 = self.calc_x_hat_dot(x_hat + dt * k3, u, y)

        x_hat_next = x_hat + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        return x_hat_next

    def print_observer_info(self):
        A_closed_loop = self.A - self.L @ self.C
        poles = np.linalg.eig(A_closed_loop)[0]
        print("Full-Order Observer ---")
        print(f"L: \n{self.L}")
        print(f"A_closed_loop: \n{A_closed_loop}")
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
        # [place_poles — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html?utm_source=chatgpt.com)
        full_state_feedback_obj = scipy.signal.place_poles(A, B, target_poles)
        F: np.ndarray = getattr(full_state_feedback_obj, "gain_matrix")

        # 双対問題
        L = F.T

        return L


class MinimalOrderStateObserver:
    pass


def simulate(
    model: DoublePendulum,
    full_state_feedback: continuous_controller.FullStateFeedback,
    C: np.ndarray,
    observer: FullOrderStateObserver,
    x0: np.ndarray,
    x_hat0: np.ndarray,
    dt: float,
    sim_time: float,
    use_observer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert x0.shape == (4,)

    step_num = int(sim_time / dt) + 1
    t_vec = np.linspace(0.0, sim_time, step_num, dtype=np.float64)
    x_vec = np.zeros((t_vec.size, x0.shape[0]))
    x_hat_vec = np.zeros((t_vec.size, x0.shape[0]))
    u_vec = np.zeros((t_vec.size, 1))

    # 初期状態
    x_vec[0] = x0
    x_hat_vec[0] = x_hat0

    # 4次ルンゲクッタ
    for i in range(1, t_vec.size):
        # オブザーバー計算
        y = C @ x_vec[i - 1]
        x_hat_vec[i] = observer.calc_next_x_hat(x_hat_vec[i - 1], u_vec[i - 1], y, dt)

        # フィードバック入力
        if use_observer:
            u = full_state_feedback.calc_u(x_hat_vec[i])  # 推定値
        else:
            u = full_state_feedback.calc_u(x_vec[i - 1])  # 真値

        x_vec[i] = model.rk4(x_vec[i - 1], u, dt)
        u_vec[i] = u

    return x_vec, t_vec, u_vec, x_hat_vec


def main():
    import matplotlib.pyplot as plt
    import visualize

    # 実験条件パラメータ設定 ----------------------------------------
    model = DoublePendulum()
    sim_dt = 0.0001
    sim_time = 2.0
    # anim_dt = 0.1
    anim_dt = 0.01

    # x0 = np.array([0.01, 0.01, 0, 0])
    # x0 = np.array([0.05, 0, 0, 0])
    # x0 = np.array([0.05, 0.0, 0.05, 0.05])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(-10), 0, 0])
    # x0 = np.array([np.deg2rad(20), np.deg2rad(10), np.deg2rad(-30), 0])
    x0 = np.array([np.deg2rad(30), np.deg2rad(20), 0, -10])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(30), np.deg2rad(-30), np.deg2rad(30)])

    # オブザーバー設定 ------------------------------------------------
    C = np.block([np.identity(2), np.zeros((2, 2))])
    # C = np.array([[1.0, 0.0, 0.0, 0.0]])
    x_hat0 = np.array([0.0, 0.0, 0.0, 0.0])

    # =================================================
    # LQR でFを決定 ----------------------------------

    Q = np.diag([10.0, 100.0, 0.1, 0.1])
    R = np.array([[1.0]])
    F = continuous_controller.calc_lqr(model, Q, R)

    full_state_feedback = continuous_controller.FullStateFeedback(model, F)
    full_state_feedback.print_feedback_system_info()
    # =================================================
    # オブザーバーの構築

    controller_poles = full_state_feedback.poles()
    # 一番遅いフィードバック極を基準に
    observer_base = 5.0 * (-np.max(np.real(controller_poles)))
    observer_poles = -observer_base * np.array([1.0, 1.2, 1.4, 1.6])  # バラバラにする必要があるらしい
    L = FullOrderStateObserver.calc_pole_placement(model, C, observer_poles)
    observer = FullOrderStateObserver(model, C, L)
    observer.print_observer_info()
    # =================================================

    # シミュレーション ----------------------------------------------
    x_vec, t_vec, u_vec, x_hat_vec = simulate(
        model=model,
        full_state_feedback=full_state_feedback,
        C=C,
        observer=observer,
        x0=x0,
        x_hat0=x_hat0,
        dt=sim_dt,
        sim_time=sim_time,
        use_observer=True,
        # use_observer=False,
    )

    # 結果のプロット ---------------------------------------------
    state_labels = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\dot{\theta}_1$",
        r"$\dot{\theta}_2$",
    ]

    state_hat_labels = [
        r"$\hat{\theta}_1$",
        r"$\hat{\theta}_2$",
        r"$\hat{\dot{\theta}}_1$",
        r"$\hat{\dot{\theta}}_2$",
    ]
    error_vec = x_vec - x_hat_vec

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(8, 8),
    )

    ax_x = axes[0]
    ax_e = axes[1]
    ax_u = axes[2]

    for i, (label, hat_label) in enumerate(
        zip(state_labels, state_hat_labels, strict=True)
    ):
        ax_x.plot(t_vec, x_vec[:, i], label=label)
        ax_x.plot(t_vec, x_hat_vec[:, i], linestyle="--", label=hat_label)

    for i, label in enumerate(state_labels):
        ax_e.plot(t_vec, error_vec[:, i], label=rf"$e_{i + 1}$")

    ax_x.set_ylabel("state")
    ax_x.grid()
    ax_x.legend(ncol=2)

    ax_e.set_ylabel("estimation error")
    ax_e.grid()
    ax_e.legend(ncol=4)

    ax_u.plot(t_vec, u_vec.squeeze(), label=r"$u$")
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("input")
    ax_u.grid()
    ax_u.legend()

    # 凡例は右上固定
    ax_x.legend(ncol=2, loc="upper right")
    ax_e.legend(ncol=4, loc="upper right")
    ax_u.legend(loc="upper right")
    plt.tight_layout()
    plt.show(block=False)

    # アニメーション ---------------------------------------------
    plotter = visualize.DoublePendulumPlotter(model=model)
    split_num = int(anim_dt / sim_dt)
    animation = plotter.animate(
        x_history=x_vec[::split_num],
        t_history=t_vec[::split_num],
        x_hat_history=x_hat_vec[::split_num],
        interval_ms=anim_dt * 1000,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
