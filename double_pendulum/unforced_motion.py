from model import DoublePendulum

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

    model = DoublePendulum()
    x_vec, t_vec = unforced_motion(
        model=model,
        x0=np.array([np.pi / 4, 0, 0, 0]).T,
        dt=0.01,
        sim_time=10,
    )

    print(x_vec.shape)
    print(t_vec.shape)
    plt.plot(t_vec, x_vec)
    plt.show()
