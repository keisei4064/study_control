from __future__ import annotations

import math
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def _arc_points(
    *,
    center: tuple[float, float],
    radius: float,
    theta_start_rad: float,
    theta_end_rad: float,
    num_points: int,
) -> list[tuple[float, float]]:
    """円弧上の点列を作る。"""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if num_points < 2:
        raise ValueError("num_points must be greater than or equal to 2")

    points: list[tuple[float, float]] = []
    for index in range(num_points):
        ratio: float = index / (num_points - 1)
        theta_rad: float = theta_start_rad + ratio * (theta_end_rad - theta_start_rad)
        points.append(
            (
                center[0] + radius * math.cos(theta_rad),
                center[1] + radius * math.sin(theta_rad),
            )
        )
    return points


def build_curved_arrow_path(
    *,
    center: tuple[float, float],
    radius: float,
    theta_start_deg: float,
    theta_end_deg: float,
    tail_width: float,
    head_width: float,
    head_length_angle_deg: float,
    num_points: int = 80,
) -> Path:
    """塗りつぶし曲線矢印を表す Path を作る。

    Args:
        center: 回転軸の位置。
        radius: 矢印の中心線半径。
        theta_start_deg: 矢印の開始角度 [deg]。
        theta_end_deg: 矢印の終了角度 [deg]。
        tail_width: 矢印胴体の太さ。
        head_width: 矢印頭の幅。
        head_length_angle_deg: 矢印頭が占める角度 [deg]。
        num_points: 胴体片側の円弧を近似する点数。

    Returns:
        塗りつぶし曲線矢印の Path。
    """
    if tail_width <= 0.0:
        raise ValueError("tail_width must be positive")
    if head_width < tail_width:
        raise ValueError("head_width must be greater than or equal to tail_width")
    if head_length_angle_deg <= 0.0:
        raise ValueError("head_length_angle_deg must be positive")

    theta_start_rad: float = math.radians(theta_start_deg)
    theta_end_rad: float = math.radians(theta_end_deg)
    sweep_rad: float = theta_end_rad - theta_start_rad
    if math.isclose(sweep_rad, 0.0, abs_tol=1e-12):
        raise ValueError("theta_start_deg and theta_end_deg must be different")

    direction: float = 1.0 if sweep_rad > 0.0 else -1.0
    head_length_rad: float = math.radians(head_length_angle_deg) * direction

    if abs(head_length_rad) >= abs(sweep_rad):
        head_length_rad = 0.35 * sweep_rad

    theta_head_base_rad: float = theta_end_rad - head_length_rad

    outer_tail_radius: float = radius + tail_width / 2.0
    inner_tail_radius: float = radius - tail_width / 2.0
    outer_head_radius: float = radius + head_width / 2.0
    inner_head_radius: float = radius - head_width / 2.0

    outer_tail_points: list[tuple[float, float]] = _arc_points(
        center=center,
        radius=outer_tail_radius,
        theta_start_rad=theta_start_rad,
        theta_end_rad=theta_head_base_rad,
        num_points=num_points,
    )
    inner_tail_points: list[tuple[float, float]] = _arc_points(
        center=center,
        radius=inner_tail_radius,
        theta_start_rad=theta_head_base_rad,
        theta_end_rad=theta_start_rad,
        num_points=num_points,
    )

    outer_head_base: tuple[float, float] = (
        center[0] + outer_head_radius * math.cos(theta_head_base_rad),
        center[1] + outer_head_radius * math.sin(theta_head_base_rad),
    )
    tip: tuple[float, float] = (
        center[0] + radius * math.cos(theta_end_rad),
        center[1] + radius * math.sin(theta_end_rad),
    )
    inner_head_base: tuple[float, float] = (
        center[0] + inner_head_radius * math.cos(theta_head_base_rad),
        center[1] + inner_head_radius * math.sin(theta_head_base_rad),
    )

    vertices: list[tuple[float, float]] = [
        *outer_tail_points,
        outer_head_base,
        tip,
        inner_head_base,
        *inner_tail_points,
        outer_tail_points[0],
    ]
    codes: list[int] = (
        [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    )
    return Path(vertices, codes)


def draw_torque_input(
    ax: Axes,
    *,
    center: tuple[float, float],
    radius: float,
    torque: float,
    torque_scale: float,
    theta1_deg: float = 40,
    theta2_deg: float = 140,
    min_tail_width: float = 0.04,
    max_tail_width: float = 0.14,
    color: str = "red",
) -> None:
    """回転軸へのトルク入力を塗りつぶし曲線矢印で描く。

    Args:
        ax: 描画先 Axes。
        center: 回転軸の位置。
        radius: トルク矢印を描く中心線半径。
        torque: トルク入力。正なら反時計回り、負なら時計回り。
        torque_scale: この値以上の絶対値を最大太さとして扱う正のスケール。
        theta1_deg: 正方向に描くときの開始角度 [deg]。
        theta2_deg: 正方向に描くときの終了角度 [deg]。
        min_tail_width: |torque| が小さいときの矢印胴体の太さ。
        max_tail_width: |torque| >= torque_scale の矢印胴体の太さ。
        color: 矢印の色。
    """
    if torque_scale <= 0.0:
        raise ValueError("torque_scale must be positive")
    if max_tail_width < min_tail_width:
        raise ValueError(
            "max_tail_width must be greater than or equal to min_tail_width"
        )

    if math.isclose(torque, 0.0, abs_tol=1e-12):
        return

    normalized_magnitude: float = min(abs(torque) / torque_scale, 1.0)
    tail_width: float = min_tail_width + normalized_magnitude * (
        max_tail_width - min_tail_width
    )
    head_width: float = 2.6 * tail_width
    head_length_angle_deg: float = 18.0 + 8.0 * normalized_magnitude

    theta_start_deg: float = theta1_deg if torque >= 0.0 else theta2_deg
    theta_end_deg: float = theta2_deg if torque >= 0.0 else theta1_deg
    path: Path = build_curved_arrow_path(
        center=center,
        radius=radius,
        theta_start_deg=theta_start_deg,
        theta_end_deg=theta_end_deg,
        tail_width=tail_width,
        head_width=head_width,
        head_length_angle_deg=head_length_angle_deg,
    )

    arrow = PathPatch(
        path,
        facecolor=color,
        edgecolor=color,
        linewidth=0.0,
        joinstyle="miter",
    )
    ax.add_patch(arrow)


def setup_axis_panel(ax: Axes, *, title: str) -> None:
    """各 subplot の見た目を整える。"""
    ax.scatter([0.0], [0.0], s=50)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    torque_values: list[float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
    torque_values: list[float] = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
    radius: float = 1.0
    torque_scale: float = 1.0

    fig: Figure
    fig, axes_array = plt.subplots(
        1,
        len(torque_values),
        figsize=(15, 3.5),
        squeeze=False,
    )
    axes: list[Axes] = [cast(Axes, ax) for ax in axes_array[0]]

    for ax, torque in zip(axes, torque_values, strict=True):
        setup_axis_panel(ax, title=rf"$u = {torque:.1f}$")
        draw_torque_input(
            ax,
            center=(0.0, 0.0),
            radius=radius,
            torque=torque,
            torque_scale=torque_scale,
        )

    fig.suptitle("Torque input around a rotation axis", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
