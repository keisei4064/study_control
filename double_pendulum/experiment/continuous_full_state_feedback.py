from double_pendulum.model import DoublePendulum
import double_pendulum.state_feedback as state_feedback

import numpy as np


def simulate_feedback_response(
    model: DoublePendulum,
    full_state_feedback: state_feedback.FullStateFeedback,
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
    import double_pendulum.visualize as visualize
    import double_pendulum.control_common as control_common

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
    F = state_feedback.calc_lqr(model, Q, R)

    full_state_feedback = state_feedback.FullStateFeedback(model, F)
    full_state_feedback.print_feedback_system_info()

    # =================================================

    fig, ax = plt.subplots()
    control_common.plot_eigenvalues_on_complex_plane(
        ax, full_state_feedback.A_closed_loop, "Feedback"
    )
    plt.tight_layout()
    plt.show(block=False)

    # シミュレーション ----------------------------------------------

    x_vec, t_vec, u_vec = simulate_feedback_response(
        model=model,
        full_state_feedback=full_state_feedback,
        x0=x0,
        dt=sim_dt,
        sim_time=sim_time,
    )

    # 結果のプロット ---------------------------------------------
    plotter = visualize.DoublePendulumPlotter(model=model)
    plotter.plot_time_series(t_vec, x_vec, u_vec)

    # アニメーション ---------------------------------------------
    split_num = int(anim_dt / sim_dt)
    x_video = x_vec[::split_num]
    t_video = t_vec[::split_num]
    interval_ms = anim_dt * 1000

    # _animation = plotter.animate(
    #     x_history=x_video,
    #     t_history=t_video,
    #     interval_ms=interval_ms,
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
        interval_ms=interval_ms,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
