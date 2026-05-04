import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

def calc_controlability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """可制御性行列を計算"""
    n = B.shape[0]
    l = B.shape[1]
    assert A.shape == (n, n)

    U_c = np.zeros((n, n * l))
    U_c[:, 0:l] = B
    for i in range(1, n):
        U_c[:, l * i : l * (i + 1)] = A @ U_c[:, l * (i - 1) : l * i]

    return U_c


def calc_observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """可観測性行列を計算"""
    n = C.shape[1]
    k = C.shape[0]
    assert A.shape == (n, n)

    U_o = np.zeros((k * n, n))
    U_o[0:k] = C
    for i in range(1, n):
        U_o[k * i : k * (i + 1)] = U_o[k * (i - 1) : k * i] @ A

    return U_o


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def plot_eigenvalues_on_complex_plane(ax: Axes, A: FloatArray, label: str) -> Axes:
    eigvals: ComplexArray = np.linalg.eigvals(A)

    # 描画
    ax.scatter(eigvals.real, eigvals.imag, label=label)

    # 実軸・虚軸
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Eigenvalues")
    ax.grid(True)
    ax.set_aspect("equal")
    ax.legend()
    
    return ax


def main():
    from model import DoublePendulum

    model = DoublePendulum()
    # model = DoublePendulum(b_1=0.0, b_2=0.0)
    # model = DoublePendulum(b_1=0.0, b_2=0.0, k=0.0)
    sample_T = 0.01

    A_c, b_c = model.calc_continuous_linear_system()
    A_d, b_d = model.calc_discrete_linear_system(sample_T=sample_T)
    C = np.block([np.eye(2), np.zeros((2, 2))])
    # C = np.block([np.eye(1), np.zeros((1, 3))])
    dim_x = A_c.shape[0]

    # 連続と離散，両方でチェック
    systems: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("continuous linear system", A_c, b_c),
        (f"discrete linear system (T={sample_T})", A_d, b_d),
    ]

    for name, A, b in systems:
        print(f"{name} ===============================================")
        print(f"A:\n{A}")
        print(f"b:\n{b}")
        print(f"C:\n{C}")
        print("\n---\n")

        # ---
        # 可制御性判定
        U_c = calc_controlability_matrix(A, b)
        rank_U_c = np.linalg.matrix_rank(U_c)
        print(U_c)
        print(f"rank(U_c): {rank_U_c}")
        print(f"controlability: {rank_U_c == dim_x}")
        print("\n---\n")

        # 可観測性判定
        U_o = calc_observability_matrix(A, C)
        rank_U_o = np.linalg.matrix_rank(U_o)
        print(U_o)
        print(f"rank(U_o): {rank_U_o}")
        print(f"observability: {rank_U_o == dim_x}")

        # ---
        # 条件数チェック
        print("\n---\n")
        cond_c = np.linalg.cond(U_c)
        cond_o = np.linalg.cond(U_o)

        print(f"cond(U_c): {cond_c:.3e}")
        print(f"cond(U_o): {cond_o:.3e}")

        sv_c = np.linalg.svd(U_c, compute_uv=False)
        sv_o = np.linalg.svd(U_o, compute_uv=False)

        print("singular values of U_c:", sv_c)
        print("singular values of U_o:", sv_o)

        print("\n---\n")


if __name__ == "__main__":
    main()
