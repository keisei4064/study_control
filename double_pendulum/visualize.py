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
from typing import Literal

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
    def __init__(self, model: DoublePendulum):
        self.model = model

    def setup_axes(self, ax: Axes) -> None:
        """Axes設定"""

        length = self.model.L_1 + self.model.L_2
        margin = 0.05 * length  # 全長の5%マージン

        ax.set_aspect("equal")
        ax.set_xlim(-length - margin, length + margin)
        ax.set_ylim(-length - margin, length + margin)
        ax.grid(True)

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

        # タイトルとして時間表示（blit=False にする必要あり）
        # time_title_text = ax.set_title("")

        # axes内にテキストで時間表示
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
        """アニメーションを作成"""
        # 配列shapeチェック
        if x_history.ndim != 2 or x_history.shape[1] != 4:
            raise ValueError(f"x_history must have shape (N, 4), got {x_history.shape}")
        if t_history.ndim != 1:
            raise ValueError(f"t_history must have shape (N,), got {t_history.shape}")
        if x_history.shape[0] != t_history.shape[0]:
            raise ValueError(
                f"x_history and t_history length mismatch: "
                f"{x_history.shape[0]} != {t_history.shape[0]}"
            )

        # axes と artistの初期設定
        fig, ax = plt.subplots()
        self.setup_axes(ax)
        artists = self.init_artists(ax)

        # 初期化関数
        def init() -> tuple[Artist, ...]:
            # 単にインデックス0を指定するだけ
            return self.update_artists(
                artists=artists,
                x=x_history[0],
                t=float(t_history[0]),
            )

        # 更新関数
        def update(frame_index: int) -> tuple[Artist, ...]:
            return self.update_artists(
                artists=artists,
                x=x_history[frame_index],
                t=float(t_history[frame_index]),
            )

        # アニメーションオブジェクトの作成
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

    def _calc_points(
        self,
        x: FloatArray,
    ) -> tuple[
        list[float], list[float], list[float], list[float], list[float], list[float]
    ]:
        """描画に必要な座標を計算する"""
        assert x.shape == (4,)

        theta_1 = x[0]
        theta_2 = x[1]

        L_1 = self.model.L_1
        L_2 = self.model.L_2

        # 固定回転軸座標
        pivot_x = 0.0
        pivot_y = 0.0

        # リンク継ぎ目の回転軸座標
        joint_1_x = L_1 * np.sin(theta_1)
        joint_1_y = -L_1 * np.cos(theta_1)

        # リンク2の先端
        tip_x = joint_1_x + L_2 * np.sin(theta_2)
        tip_y = joint_1_y + L_2 * np.cos(theta_2)

        # リンク1の両端座標
        link_1_x = [pivot_x, joint_1_x]
        link_1_y = [pivot_y, joint_1_y]

        # リンク2の両端座標
        link_2_x = [joint_1_x, tip_x]
        link_2_y = [joint_1_y, tip_y]

        # 回転軸点の座標群
        point_x = [pivot_x, joint_1_x]
        point_y = [pivot_y, joint_1_y]

        return link_1_x, link_1_y, link_2_x, link_2_y, point_x, point_y

    def plot(
        self,
        x: FloatArray,
        t: float,
    ) -> tuple[Artist, ...]:
        """1フレーム単体をプロット"""
        fig, ax = plt.subplots()
        self.setup_axes(ax)
        artists = self.init_artists(ax)
        return self.update_artists(
            artists=artists,
            x=x,
            t=t,
        )

    def save_animation(
        self,
        animation: FuncAnimation,
        interval_ms: float,
        format: Literal["mp4", "gif"] = "mp4",
    ):
        fps = int(1000 / interval_ms)

        match format:
            case "mp4":
                animation.save(
                    "double_pendulum.mp4",
                    writer="ffmpeg",
                    fps=fps,
                    dpi=150,
                )
            case "gif":
                animation.save(
                    "double_pendulum.gif",
                    writer="pillow",
                    fps=fps,
                    dpi=120,
                )


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

    # アニメーションのプロット
    interval_ms = 10
    stride = 5
    x_video = x_history[::stride]
    t_video = t_history[::stride]
    interval_ms = interval_ms * stride
    animation = plotter.animate(
        x_history=x_video,
        t_history=t_video,
        interval_ms=interval_ms,
    )
    plt.show()
    plotter.save_animation(animation, interval_ms, "gif")

    # 1フレームプロット
    plotter.plot(x=x_history[0], t=0.0)
    plt.show()


if __name__ == "__main__":
    main()
