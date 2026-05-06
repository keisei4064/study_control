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
    ctrl_dt: float,
    sim_dt: float,
    sim_time: float,
    use_observer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_num = int(sim_time / sim_dt) + 1
    t_vec = np.linspace(0.0, sim_time, step_num, dtype=np.float64)
    x_vec = np.zeros((t_vec.size, x0.shape[0]))
    x_hat_vec = np.zeros((t_vec.size, x0.shape[0]))
    u_vec = np.zeros((t_vec.size, 1))

    # 初期状態
    x_vec[0] = x0
    x_hat_vec[0] = observer.get_x_hat()
    if use_observer:
        u_vec[0] = full_state_feedback.calc_u(x_hat_vec[0])
    else:
        u_vec[0] = full_state_feedback.calc_u(x_vec[0])

    # サンプラー
    ctrl_step = round(ctrl_dt / sim_dt)
    if not np.isclose(ctrl_step * sim_dt, ctrl_dt):
        raise ValueError("ctrl_dt must be an integer multiple of sim_dt")

    # 4次ルンゲクッタ
    for i in range(1, t_vec.size):
        x_vec[i] = model.rk4(x_vec[i - 1], float(u_vec[i - 1][0]), sim_dt)

        if i % ctrl_step == 0:
            sampled_x = x_vec[i]
            sampled_y = C @ sampled_x

            # オブザーバー計算
            x_hat_vec[i] = observer.calc_next_x_hat(u_vec[i - 1], sampled_y)

            # フィードバック入力
            if use_observer:
                u_vec[i] = full_state_feedback.calc_u(x_hat_vec[i])
            else:
                u_vec[i] = full_state_feedback.calc_u(sampled_x)
        else:
            x_hat_vec[i] = x_hat_vec[i - 1]
            u_vec[i] = u_vec[i - 1]

    return x_vec, t_vec, u_vec, x_hat_vec


def main():
    import matplotlib.pyplot as plt
    import double_pendulum.visualize as visualize
    import double_pendulum.control_common as control_common

    # 実験条件パラメータ設定 ----------------------------------------
    model = DoublePendulum()
    sim_dt = 0.0001
    sim_time = 2.0
    # ctrl_dt = 0.005
    # ctrl_dt = 0.01
    ctrl_dt = 0.02
    # ctrl_dt = 0.05
    anim_dt = 0.01

    # x0 = np.array([0.01, 0.01, 0, 0])
    # x0 = np.array([0.05, 0, 0, 0])
    # x0 = np.array([0.05, 0.0, 0.05, 0.05])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(-10), 0, 0])
    x0 = np.array([np.deg2rad(20), np.deg2rad(10), np.deg2rad(-30), 0])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(20), 0, -10])
    # x0 = np.array([np.deg2rad(30), np.deg2rad(30), np.deg2rad(-30), np.deg2rad(30)])

    # =================================================
    A, b = model.calc_discrete_linear_system(ctrl_dt)  # 離散システム

    # オブザーバー設定 ------------------------------------------------
    C = np.block([np.identity(2), np.zeros((2, 2))])
    # C = np.array([[1.0, 0.0, 0.0, 0.0]])
    x_hat0 = np.array([0.0, 0.0, 0.0, 0.0])

    # LQR でフィードバックを構築 ----------------------------------
    Q = np.diag([10.0, 100.0, 0.1, 0.1])
    R = np.array([[1.0]])
    F = state_feedback.calc_descrete_lqr(A, b, Q, R)

    full_state_feedback = state_feedback.FullStateFeedback(A, b, F)
    full_state_feedback.print_feedback_system_info()
    controller_poles = full_state_feedback.poles()
    # =================================================
    # オブザーバーの構築

    # 離散系では「一番遅い極」は絶対値が最大の極
    # オブザーバーを制御器より早く収束させたい
    slowest_controller_radius = np.max(np.abs(controller_poles))
    observer_base_radius = slowest_controller_radius**3  # 適当に強める

    # 同じ極を重複させないように少しばらす
    observer_poles = observer_base_radius ** np.array([1.0, 1.2, 1.4, 1.6])

    # 同一次元オブザーバー
    # L = state_observer.FullOrderStateObserver.calc_pole_placement(A, C, observer_poles)
    # observer = state_observer.FullOrderStateObserver(A, b, C, L, "discrete", x_hat0)

    # # 最小次元オブザーバー
    L = state_observer.MinimalOrderStateObserver.calc_pole_placement(
        A, C, observer_poles[: -C.shape[0]]
    )
    y0 = C @ x0
    x2_hat0 = x_hat0[C.shape[0] :]
    z0 = x2_hat0 - L @ y0
    observer = state_observer.MinimalOrderStateObserver(
        A, b, C, L, "discrete", z0=z0, y0=y0
    )

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
        ctrl_dt=ctrl_dt,
        sim_dt=sim_dt,
        sim_time=sim_time,
        # use_observer=True,
        use_observer=False,
    )

    # 結果のプロット ---------------------------------------------
    plotter = visualize.DoublePendulumPlotter(model=model)
    plotter.plot_time_series(t_vec, x_vec, u_vec, x_hat_vec=x_hat_vec)

    # アニメーション ---------------------------------------------
    split_num = int(anim_dt / sim_dt)

    x_video = x_vec[::split_num]
    t_video = t_vec[::split_num]
    u_video = u_vec[::split_num]
    x_hat_video = x_hat_vec[::split_num]
    interval_ms = anim_dt * 1000

    # _animation = plotter.animate(
    #     x_history=x_video,
    #     t_history=t_video,
    #     interval_ms=interval_ms,
    #     x_hat_history=x_hat_video,
    #     repeat=True,
    # )
    # plt.show(block=False)

    # _phase_animation = plotter.animate_phase_space(
    #     x_history=x_video,
    #     t_history=t_video,
    #     interval_ms=interval_ms,
    #     repeat=True,
    # )
    # plt.show()

    _both_animation = plotter.animate_both(
        x_history=x_video,
        t_history=t_video,
        u_history=u_video,
        x_hat_history=x_hat_video,
        interval_ms=interval_ms,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
