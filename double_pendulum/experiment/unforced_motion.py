from double_pendulum.model import DoublePendulum

import numpy as np


def unforced_motion(
    model: DoublePendulum,
    x0: np.ndarray,
    dt: float,
    sim_time: float,
):
    assert x0.shape == (4,)

    step_num = int(sim_time / dt) + 1
    t_vec = np.linspace(0.0, sim_time, step_num, dtype=np.float64)
    x_vec = np.zeros((t_vec.size, x0.shape[0]))
    x_vec[0] = x0

    # 4次ルンゲクッタ
    for i in range(1, t_vec.size):
        x_vec[i] = model.rk4(x_vec[i - 1], 0, dt)

    return x_vec, t_vec


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from double_pendulum.visualize import DoublePendulumPlotter

    model = DoublePendulum(b_1=0.001, b_2=0.001)
    plotter = DoublePendulumPlotter(model=model)
    sim_dt = 0.001

    x_history, t_history = unforced_motion(
        model=model,
        x0=np.array(
            [
                np.deg2rad(30.0),
                np.deg2rad(10.0),
                15.0,
                15.0,
            ],
            dtype=np.float64,
        ),
        dt=sim_dt,
        sim_time=50.0,
    )

    print(x_history.shape)
    print(t_history.shape)
    plt.plot(t_history, x_history)
    plt.show()

    # アニメーションのプロット
    interval_ms = sim_dt * 1000
    stride = 50
    x_video = x_history[::stride]
    t_video = t_history[::stride]
    interval_ms = interval_ms * stride
    animation = plotter.animate(
        x_history=x_video,
        t_history=t_video,
        interval_ms=interval_ms,
    )
    plt.show()
    # plotter.save_animation(animation, interval_ms, "gif")

    # 相空間アニメーションのプロット
    phase_animation = plotter.animate_phase_space(
        x_history=x_video,
        t_history=t_video,
        interval_ms=interval_ms,
    )
    plt.show()

    # 1フレームプロット
    plotter.plot(x=x_history[0], t=0.0)
    plt.show()
