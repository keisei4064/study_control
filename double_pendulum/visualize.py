import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.text import Text
from typing import TypeAlias

from model import DoublePendulum
from unforced_motion import unforced_motion

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(slots=True)
class DoublePendulumArtists:
    line_1: Line2D
    line_2: Line2D
    point_line: Line2D
    time_text: Text


class DoublePendulumPlotter:
    def __init__(self, model):
        self.model = model

    def setup_axes(self, ax: Axes) -> None:
        length = self.model.L_1 + self.model.L_2
        margin = 0.05 * length

        ax.set_aspect("equal")
        ax.set_xlim(-length - margin, length + margin)
        ax.set_ylim(-length - margin, length + margin)
        ax.grid(True)

    def _calc_points(
        self,
        x: FloatArray,
    ) -> tuple[
        list[float], list[float], list[float], list[float], list[float], list[float]
    ]:
        assert x.shape == (4,)

        theta_1 = x[0]
        theta_2 = x[1]

        L_1 = self.model.L_1
        L_2 = self.model.L_2

        pivot_x = 0.0
        pivot_y = 0.0

        joint_1_x = L_1 * np.sin(theta_1)
        joint_1_y = -L_1 * np.cos(theta_1)

        tip_x = joint_1_x + L_2 * np.sin(theta_2)
        tip_y = joint_1_y + L_2 * np.cos(theta_2)

        link_1_x = [pivot_x, joint_1_x]
        link_1_y = [pivot_y, joint_1_y]

        link_2_x = [joint_1_x, tip_x]
        link_2_y = [joint_1_y, tip_y]

        point_x = [pivot_x, joint_1_x] #, tip_x]
        point_y = [pivot_y, joint_1_y] #, tip_y]

        return link_1_x, link_1_y, link_2_x, link_2_y, point_x, point_y

    def plot(
        self,
        ax: Axes,
        x: FloatArray,
        t: float,
    ) -> tuple[Artist, ...]:
        link_1_x, link_1_y, link_2_x, link_2_y, point_x, point_y = self._calc_points(x)

        (line_1,) = ax.plot(
            link_1_x,
            link_1_y,
            linewidth=12.0,
            solid_capstyle="round",
        )
        (line_2,) = ax.plot(
            link_2_x,
            link_2_y,
            linewidth=12.0,
            solid_capstyle="round",
        )
        (point_line,) = ax.plot(
            point_x,
            point_y,
            "o",
            markersize=8.0,
        )
        time_text = ax.text(
            0.02,
            0.95,
            f"t = {t:.2f} s",
            transform=ax.transAxes,
            va="top",
        )

        return line_1, line_2, point_line, time_text

    def init_artists(self, ax: Axes) -> DoublePendulumArtists:
        (line_1,) = ax.plot(
            [],
            [],
            linewidth=12.0,
            solid_capstyle="round",
        )
        (line_2,) = ax.plot(
            [],
            [],
            linewidth=12.0,
            solid_capstyle="round",
        )
        (point_line,) = ax.plot(
            [],
            [],
            "o",
            markersize=8.0,
        )
        time_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            va="top",
        )

        return DoublePendulumArtists(
            line_1=line_1,
            line_2=line_2,
            point_line=point_line,
            time_text=time_text,
        )

    def update_artists(
        self,
        artists: DoublePendulumArtists,
        x: FloatArray,
        t: float,
    ) -> tuple[Artist, ...]:
        link_1_x, link_1_y, link_2_x, link_2_y, point_x, point_y = self._calc_points(x)

        artists.line_1.set_data(link_1_x, link_1_y)
        artists.line_2.set_data(link_2_x, link_2_y)
        artists.point_line.set_data(point_x, point_y)
        artists.time_text.set_text(f"t = {t:.2f} s")

        return (
            artists.line_1,
            artists.line_2,
            artists.point_line,
            artists.time_text,
        )

    def animate(
        self,
        x_history: FloatArray,
        t_history: FloatArray,
        interval_ms: float = 20.0,
    ) -> FuncAnimation:
        if x_history.ndim != 2 or x_history.shape[1] != 4:
            raise ValueError(f"x_history must have shape (N, 4), got {x_history.shape}")
        if t_history.ndim != 1:
            raise ValueError(f"t_history must have shape (N,), got {t_history.shape}")
        if x_history.shape[0] != t_history.shape[0]:
            raise ValueError(
                f"x_history and t_history length mismatch: "
                f"{x_history.shape[0]} != {t_history.shape[0]}"
            )

        fig, ax = plt.subplots()
        self.setup_axes(ax)
        artists = self.init_artists(ax)

        def init() -> tuple[Artist, ...]:
            return self.update_artists(
                artists=artists,
                x=x_history[0],
                t=float(t_history[0]),
            )

        def update(frame_index: int) -> tuple[Artist, ...]:
            return self.update_artists(
                artists=artists,
                x=x_history[frame_index],
                t=float(t_history[frame_index]),
            )

        animation = FuncAnimation(
            fig=fig,
            func=update,
            frames=len(t_history),
            init_func=init,
            interval=interval_ms,
            blit=True,
            repeat=False,
        )

        return animation

def main() -> None:
    model = DoublePendulum(b_1=0.001, b_2=0.001)
    plotter = DoublePendulumPlotter(model=model)

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
        dt=0.01,
        sim_time=50.0,
    )

    animation = plotter.animate(
        x_history=x_history,
        t_history=t_history,
        interval_ms=10.0,
    )

    plt.show()


if __name__ == "__main__":
    main()
