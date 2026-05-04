import numpy as np
import scipy.signal
import scipy.linalg
from model import DoublePendulum


class FullStateFeedback:
    """完全状態フィードバック"""

    def __init__(self, model: DoublePendulum, F: np.ndarray):
        self.F = F
        A, b = model.calc_continuous_linear_system()
        self.A = A
        self.b = b

        assert F.shape == (b.shape[1], A.shape[0])

        self.A_closed_loop = A - b @ F

    def calc_u(self, x: np.ndarray) -> float:
        return (-self.F @ x)[0]

    def print_feedback_system_info(self):
        poles = np.linalg.eig(self.A_closed_loop)[0]
        print(f"A_closed_loop: \n{self.A_closed_loop}")
        print(f"closed-loop poles: {poles}")


def calc_pole_placement(model: DoublePendulum, target_poles: np.ndarray) -> np.ndarray:
    """極配置関数"""
    A, b = model.calc_continuous_linear_system()
    assert target_poles.shape == (A.shape[0],)

    # 極配置
    # [place_poles — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html?utm_source=chatgpt.com)
    full_state_feedback_obj = scipy.signal.place_poles(A, b, target_poles)
    F: np.ndarray = getattr(full_state_feedback_obj, "gain_matrix")

    return F


def calc_lqr(model: DoublePendulum, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """LQRで最適フィードバックを求める"""
    # リカッチ方程式を解く
    A, b = model.calc_continuous_linear_system()
    P = scipy.linalg.solve_continuous_are(A, b, Q, R)

    F = np.linalg.solve(R, b.T @ P)
    return F


def simulate_feedback_response(
    model: DoublePendulum,
    full_state_feedback: FullStateFeedback,
    x0: np.ndarray,
    dt: float,
    sim_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert x0.shape == (4,)

    step_num = int(sim_time / dt) + 1
    t_vec = np.linspace(0.0, sim_time, step_num, dtype=np.float64)
    x_vec = np.zeros((t_vec.size, x0.shape[0]))
    u_vec = np.zeros((t_vec.size, 1))

    # 初期状態
    x_vec[0] = x0

    # 4次ルンゲクッタ
    for i in range(1, t_vec.size):
        # フィードバック入力
        u = full_state_feedback.calc_u(x_vec[i - 1])
        x_vec[i] = model.rk4(x_vec[i - 1], u, dt)
        u_vec[i] = u

    return x_vec, t_vec, u_vec


def main():
    import matplotlib.pyplot as plt
    import visualize

    # 実験条件パラメータ設定 ----------------------------------------
    model = DoublePendulum()
    sim_dt = 0.0001
    sim_time = 3
    # anim_dt = 0.1
    anim_dt = 0.01

    # x0 = np.array([0.05, 0.0, 0.05, 0.05])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(10), 0, 0])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(-30), 0])
    x0 = np.array([np.deg2rad(30), np.deg2rad(20), 0, -10])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(30), np.deg2rad(-30), np.deg2rad(30)])

    # =================================================
    # 極配置 でFを決定 --------------------------------

    # target_poles = np.array([-5.0, -6.0, -7.0, -8.0])
    # target_poles = np.array([-1.0, -2.0, -3.0, -4.0])
    # target_poles = np.array([-0.3, -0.4, -0.1, -0.2])

    # F = calc_pole_placement(model, target_poles)

    # LQR でFを決定 ----------------------------------

    # Q = np.diag([1.0, 1.0, 1.0, 1.0])
    # Q = np.diag([1.0, 1.0, 0, 0])
    # R = np.diag([0.1])
    Q = np.diag([10.0, 100.0, 0.1, 0.1])
    R = np.array([[1.0]])
    F = calc_lqr(model, Q, R)

    # =================================================
    
    # シミュレーション ----------------------------------------------

    full_state_feedback = FullStateFeedback(model, F)
    full_state_feedback.print_feedback_system_info()

    x_vec, t_vec, u_vec = simulate_feedback_response(
        model=model,
        full_state_feedback=full_state_feedback,
        x0=x0,
        dt=sim_dt,
        sim_time=sim_time,
    )

    # 結果のプロット ---------------------------------------------
    state_labels = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\dot{\theta}_1$",
        r"$\dot{\theta}_2$",
    ]
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 6),
    )

    ax_x = axes[0]
    ax_u = axes[1]

    for i, label in enumerate(state_labels):
        ax_x.plot(t_vec, x_vec[:, i], label=label)

    ax_x.set_ylabel("state")
    ax_x.grid()
    ax_x.legend()

    ax_u.plot(t_vec, u_vec, label=r"$u$")
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("input")
    ax_u.grid()
    ax_u.legend()

    plt.tight_layout()
    plt.show(block=False)

    # アニメーション ---------------------------------------------
    plotter = visualize.DoublePendulumPlotter(model=model)
    split_num = int(anim_dt / sim_dt)
    animation = plotter.animate(
        x_history=x_vec[::split_num],
        t_history=t_vec[::split_num],
        interval_ms=anim_dt * 1000,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
