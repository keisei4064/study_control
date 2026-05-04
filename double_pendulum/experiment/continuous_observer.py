from double_pendulum.model import DoublePendulum
import double_pendulum.state_feedback as state_feedback
import double_pendulum.state_observer as state_observer

import numpy as np

def simulate(
    model: DoublePendulum,
    full_state_feedback: state_feedback.FullStateFeedback,
    C: np.ndarray,
    observer: state_observer.ObserverProtocol,
    x0: np.ndarray,
    dt: float,
    sim_time: float,
    use_observer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_num = int(sim_time / dt) + 1
    t_vec = np.linspace(0.0, sim_time, step_num, dtype=np.float64)
    x_vec = np.zeros((t_vec.size, x0.shape[0]))
    x_hat_vec = np.zeros((t_vec.size, x0.shape[0]))
    u_vec = np.zeros((t_vec.size, 1))

    # 初期状態
    x_vec[0] = x0
    x_hat_vec[0] = observer.get_x_hat()

    # 4次ルンゲクッタ
    for i in range(1, t_vec.size):
        # オブザーバー計算
        y = C @ x_vec[i - 1]

        # オブザーバー計算
        x_hat_vec[i] = observer.calc_next_x_hat(u_vec[i - 1], y)

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
    import double_pendulum.visualize as visualize
    import double_pendulum.control_common as control_common

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
    x0 = np.array([np.deg2rad(20), np.deg2rad(10), np.deg2rad(-30), 0])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(20), 0, -10])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(30), np.deg2rad(-30), np.deg2rad(30)])

    # オブザーバー設定 ------------------------------------------------
    C = np.block([np.identity(2), np.zeros((2, 2))])
    # C = np.array([[1.0, 0.0, 0.0, 0.0]])
    x_hat0 = np.array([0.0, 0.0, 0.0, 0.0])

    # =================================================
    # LQR でFを決定 ----------------------------------

    Q = np.diag([10.0, 100.0, 0.1, 0.1])
    R = np.array([[1.0]])
    F = state_feedback.calc_lqr(model, Q, R)

    full_state_feedback = state_feedback.FullStateFeedback(model, F)
    full_state_feedback.print_feedback_system_info()
    # =================================================
    # オブザーバーの構築

    controller_poles = full_state_feedback.poles()
    # 一番遅いフィードバック極を基準に
    # observer_base = 5.0 * (-np.max(np.real(controller_poles)))
    observer_base = 3.0 * (-np.mean(np.real(controller_poles)))
    observer_poles = -observer_base * np.array(
        [1.0, 1.2, 1.4, 1.6]
    )  # バラバラにする必要があるらしい

    # 同一次元オブザーバー
    # L = state_observer.FullOrderStateObserver.calc_pole_placement(model, C, observer_poles)
    # observer = state_observer.FullOrderStateObserver(model, C, L, sim_dt, x_hat0)

    # 最小次元オブザーバー
    L = state_observer.MinimalOrderStateObserver.calc_pole_placement(
        model, C, observer_poles[: -C.shape[0]]
    )
    observer = state_observer.MinimalOrderStateObserver(model, C, L, sim_dt, z0=x_hat0[C.shape[0] :])

    observer.print_observer_info()
    # =================================================

    fig, ax = plt.subplots()
    control_common.plot_eigenvalues_on_complex_plane(
        ax, full_state_feedback.A_closed_loop, "Feedback"
    )
    control_common.plot_eigenvalues_on_complex_plane(
        ax, observer.get_A_closed_loop(), "Observer"
    )
    plt.tight_layout()
    plt.show(block=False)

    # シミュレーション ----------------------------------------------
    x_vec, t_vec, u_vec, x_hat_vec = simulate(
        model=model,
        full_state_feedback=full_state_feedback,
        C=C,
        observer=observer,
        x0=x0,
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
    _animation = plotter.animate(
        x_history=x_vec[::split_num],
        t_history=t_vec[::split_num],
        x_hat_history=x_hat_vec[::split_num],
        interval_ms=anim_dt * 1000,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
