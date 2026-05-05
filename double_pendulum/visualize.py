import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.text import Text
from typing import TypeAlias
from typing import Literal

from double_pendulum.model import DoublePendulum

FloatArray: TypeAlias = npt.NDArray[np.float64]


def darken_color(color, factor: float = 0.8) -> tuple[float, float, float]:
    """Matplotlib の色指定を少し暗くした RGB に変換する。"""
    r, g, b = mcolors.to_rgb(color)
    return (factor * r, factor * g, factor * b)


@dataclass(slots=True)
class DoublePendulumArtists:
    line_1: Line2D
    line_2: Line2D
    point_line: Line2D
    x_hat_line_1: Line2D
    x_hat_line_2: Line2D
    time_text: Text


@dataclass(slots=True)
class PhaseSpaceArtists:
    theta_1_line: Line2D
    theta_2_line: Line2D
    theta_1_point: Line2D
    theta_2_point: Line2D
    time_text: Text


@dataclass(slots=True)
class BothViewArtists:
    pendulum: DoublePendulumArtists
    pendulum_ax: Axes
    phase_space: PhaseSpaceArtists
    phase_space_ax: Axes
    x_history: FloatArray
    t_history: FloatArray
    x_hat_history: FloatArray | None
    theta_limit: float
    theta_dot_limit: float


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
            label="true link 1",
            zorder=2,
        )
        (line_2,) = ax.plot(
            [],
            [],
            linewidth=12.0,
            solid_capstyle="round",
            label="true link 2",
            zorder=2,
        )
        (point_line,) = ax.plot(
            [],
            [],
            "o",
            markersize=8.0,
            label="true joints",
            zorder=3,
            color="gray",
        )

        x_hat_line_1_color = darken_color(line_1.get_color(), factor=0.85)
        x_hat_line_2_color = darken_color(line_2.get_color(), factor=0.85)

        # 推定値 x_hat: 真値と同系色の濃い破線リンクだけを重ねる
        (x_hat_line_1,) = ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=3.0,
            color=x_hat_line_1_color,
            alpha=0.75,
            solid_capstyle="round",
            dash_capstyle="round",
            label=r"$\hat{x}$ link 1",
        )
        (x_hat_line_2,) = ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=3.0,
            color=x_hat_line_2_color,
            alpha=0.75,
            solid_capstyle="round",
            dash_capstyle="round",
            label=r"$\hat{x}$ link 2",
        )

        # デフォルトで非表示
        x_hat_line_1.set_visible(False)
        x_hat_line_2.set_visible(False)

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
            x_hat_line_1=x_hat_line_1,
            x_hat_line_2=x_hat_line_2,
            time_text=time_text,
        )

    def update_artists(
        self,
        artists: DoublePendulumArtists,
        x: FloatArray,
        t: float,
        x_hat: FloatArray | None = None,
    ) -> tuple[Artist, ...]:
        link_1_x, link_1_y, link_2_x, link_2_y, point_x, point_y = self._calc_points(x)

        artists.line_1.set_data(link_1_x, link_1_y)
        artists.line_2.set_data(link_2_x, link_2_y)
        artists.point_line.set_data(point_x, point_y)
        artists.time_text.set_text(f"t = {t:.2f} s")

        # 推定値 x_hat があるならそれも描写
        if x_hat is None:
            # 非表示
            artists.x_hat_line_1.set_visible(False)
            artists.x_hat_line_2.set_visible(False)
        else:
            (
                x_hat_link_1_x,
                x_hat_link_1_y,
                x_hat_link_2_x,
                x_hat_link_2_y,
                _,
                _,
            ) = self._calc_points(x_hat)

            artists.x_hat_line_1.set_data(x_hat_link_1_x, x_hat_link_1_y)
            artists.x_hat_line_2.set_data(x_hat_link_2_x, x_hat_link_2_y)

            # 表示
            artists.x_hat_line_1.set_visible(True)
            artists.x_hat_line_2.set_visible(True)

        return (
            artists.line_1,
            artists.line_2,
            artists.point_line,
            artists.x_hat_line_1,
            artists.x_hat_line_2,
            artists.time_text,
        )

    def animate(
        self,
        x_history: FloatArray,
        t_history: FloatArray,
        interval_ms: float,
        repeat: bool = False,
        x_hat_history: FloatArray | None = None,
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

        if x_hat_history is not None:
            # 推定値 x_hat がある場合はその shape もチェック
            if x_hat_history.ndim != 2 or x_hat_history.shape[1] != 4:
                raise ValueError(
                    f"x_hat_history must have shape (N, 4), got {x_hat_history.shape}"
                )
            if x_hat_history.shape[0] != t_history.shape[0]:
                raise ValueError(
                    f"x_hat_history and t_history length mismatch: "
                    f"{x_hat_history.shape[0]} != {t_history.shape[0]}"
                )

        fig, ax = plt.subplots()
        self.setup_axes(ax)
        artists = self.init_artists(ax)

        # 更新関数
        def update(frame_index: int) -> tuple[Artist, ...]:
            x_hat = None if x_hat_history is None else x_hat_history[frame_index]
            return self.update_artists(
                artists=artists,
                x=x_history[frame_index],
                x_hat=x_hat,
                t=float(t_history[frame_index]),
            )

        # アニメーションオブジェクトの作成
        animation = FuncAnimation(
            fig=fig,
            func=update,
            frames=len(t_history),
            init_func=lambda: update(0),
            interval=interval_ms,
            blit=True,
            repeat=repeat,
        )

        return animation

    def init_phase_space_artists(
        self,
        ax: Axes,
        x_history: FloatArray,
        t_history: FloatArray,
    ) -> tuple[PhaseSpaceArtists, float, float]:
        """位相空間のプロットを初期化し、artists と表示範囲を返す"""
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
        ax.grid(True)

        # 変位の表示範囲（ゼロ対称、±5%マージン）
        theta_max_abs = float(np.max(np.abs(x_history[:, 0:2])))
        theta_dot_max_abs = float(np.max(np.abs(x_history[:, 2:4])))
        theta_limit = max(theta_max_abs, 1.0e-6) * 1.05
        theta_dot_limit = max(theta_dot_max_abs, 1.0e-6) * 1.05

        # 軸範囲を [-1, 1] に正規化（正方形表示のため）
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")

        # 目盛り位置を固定（5個：-1, -0.5, 0, 0.5, 1）
        n_ticks = 5
        tick_positions = np.linspace(-1, 1, n_ticks).tolist()
        ax.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
        ax.yaxis.set_major_locator(ticker.FixedLocator(tick_positions))

        # 目盛りラベルを元のスケールで表示
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v * theta_limit:.2f}")
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v * theta_dot_limit:.2f}")
        )

        # 線
        (theta_1_line,) = ax.plot([], [], label=r"$\theta_1$", linewidth=1.5)
        (theta_2_line,) = ax.plot([], [], label=r"$\theta_2$", linewidth=1.5)
        theta_1_color = theta_1_line.get_color()
        theta_2_color = theta_2_line.get_color()
        # 点
        (theta_1_point,) = ax.plot([], [], "o", color=theta_1_color, markersize=6.0)
        (theta_2_point,) = ax.plot([], [], "o", color=theta_2_color, markersize=6.0)
        # 時刻
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

        ax.legend(loc="upper right")

        artists = PhaseSpaceArtists(
            theta_1_line=theta_1_line,
            theta_2_line=theta_2_line,
            theta_1_point=theta_1_point,
            theta_2_point=theta_2_point,
            time_text=time_text,
        )
        return artists, theta_limit, theta_dot_limit

    def update_phase_space_artists(
        self,
        artists: PhaseSpaceArtists,
        theta_limit: float,
        theta_dot_limit: float,
        frame_index: int,
        x_history: FloatArray,
        t_history: FloatArray,
    ) -> tuple[Artist, ...]:
        """位相空間のアートをフレーム更新"""
        end = frame_index + 1
        # theta1（正規化された座標で描画）
        artists.theta_1_line.set_data(
            x_history[:end, 0] / theta_limit,
            x_history[:end, 2] / theta_dot_limit,
        )
        artists.theta_1_point.set_data(
            [x_history[frame_index, 0] / theta_limit],
            [x_history[frame_index, 2] / theta_dot_limit],
        )
        # theta2
        artists.theta_2_line.set_data(
            x_history[:end, 1] / theta_limit,
            x_history[:end, 3] / theta_dot_limit,
        )
        artists.theta_2_point.set_data(
            [x_history[frame_index, 1] / theta_limit],
            [x_history[frame_index, 3] / theta_dot_limit],
        )
        # 時刻
        artists.time_text.set_text(f"t = {t_history[frame_index]:.2f} s")

        return (
            artists.theta_1_line,
            artists.theta_2_line,
            artists.theta_1_point,
            artists.theta_2_point,
            artists.time_text,
        )

    def animate_phase_space(
        self,
        x_history: FloatArray,
        t_history: FloatArray,
        interval_ms: float,
        repeat: bool = False,
    ) -> FuncAnimation:
        """theta1/theta2 の相空間アニメーションを作成"""
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
        artists, theta_limit, theta_dot_limit = self.init_phase_space_artists(
            ax, x_history, t_history
        )

        def update(frame_index: int) -> tuple[Artist, ...]:
            return self.update_phase_space_artists(
                artists,
                theta_limit,
                theta_dot_limit,
                frame_index,
                x_history,
                t_history,
            )

        animation = FuncAnimation(
            fig=fig,
            func=update,
            frames=len(t_history),
            init_func=lambda: update(0),
            interval=interval_ms,
            blit=True,
            repeat=repeat,
        )

        return animation

    def animate_both(
        self,
        x_history: FloatArray,
        t_history: FloatArray,
        interval_ms: float,
        repeat: bool = False,
        x_hat_history: FloatArray | None = None,
    ) -> FuncAnimation:
        """振り子 + 位相空間の両方を並べて表示するアニメーション"""
        if x_history.ndim != 2 or x_history.shape[1] != 4:
            raise ValueError(f"x_history must have shape (N, 4), got {x_history.shape}")
        if t_history.ndim != 1:
            raise ValueError(f"t_history must have shape (N,), got {t_history.shape}")
        if x_history.shape[0] != t_history.shape[0]:
            raise ValueError(
                f"x_history and t_history length mismatch: "
                f"{x_history.shape[0]} != {t_history.shape[0]}"
            )
        if x_hat_history is not None:
            if x_hat_history.ndim != 2 or x_hat_history.shape[1] != 4:
                raise ValueError(
                    f"x_hat_history must have shape (N, 4), got {x_hat_history.shape}"
                )
            if x_hat_history.shape[0] != t_history.shape[0]:
                raise ValueError(
                    f"x_hat_history and t_history length mismatch: "
                    f"{x_hat_history.shape[0]} != {t_history.shape[0]}"
                )

        # 横並びで axes を作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.setup_axes(ax1)

        # 各Axes と描画artists を対応付け
        # 振り子
        pendulum_artists = self.init_artists(ax1)
        # 位相空間
        phase_space_artists, theta_limit, theta_dot_limit = (
            self.init_phase_space_artists(ax2, x_history, t_history)
        )

        # Artists をまとめる
        state = BothViewArtists(
            pendulum=pendulum_artists,
            pendulum_ax=ax1,
            phase_space=phase_space_artists,
            phase_space_ax=ax2,
            x_history=x_history,
            t_history=t_history,
            x_hat_history=x_hat_history,
            theta_limit=theta_limit,
            theta_dot_limit=theta_dot_limit,
        )

        # 更新関数
        def update(frame_index: int) -> tuple[Artist, ...]:
            x_hat = None if x_hat_history is None else x_hat_history[frame_index]
            pendulum_updates = self.update_artists(
                state.pendulum,
                x_history[frame_index],
                float(t_history[frame_index]),
                x_hat,
            )
            ps_updates = self.update_phase_space_artists(
                state.phase_space,
                theta_limit,
                theta_dot_limit,
                frame_index,
                x_history,
                t_history,
            )
            return (*pendulum_updates, *ps_updates)

        # アニメーションを作成
        animation = FuncAnimation(
            fig=fig,
            func=update,
            frames=len(t_history),
            init_func=lambda: update(0),
            interval=interval_ms,
            blit=True,
            repeat=repeat,
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
