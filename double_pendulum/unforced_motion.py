from model import DoublePendulum

import numpy as np


def unforced_motion(
    model: DoublePendulum,
    x0: np.ndarray,
    dt: float,
    sim_time: float,
):
    assert x0.shape == (4,)

    t_vec = np.arange(0, sim_time, dt)
    x_vec = np.zeros((t_vec.size, x0.shape[0]))
    x_vec[0] = x0

    # 4次ルンゲクッタ
    for i in range(1, t_vec.size):
        k1 = model.f(x_vec[i - 1], 0) / 2
        k2 = model.f(x_vec[i - 1] + k1 * dt, 0) / 2
        k3 = model.f(x_vec[i - 1] + k2 * dt, 0) / 2
        k4 = model.f(x_vec[i - 1] + k3 * 2 * dt, 0) / 2
        delta_x = (k1 + 2 * k2 + 2 * k3 + k4) / 3 * dt
        x_vec[i] = x_vec[i - 1] + delta_x

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
