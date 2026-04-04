import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_positions(file_path: Path) -> np.ndarray:
    data = np.loadtxt(file_path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(
            f"Formato inválido en {file_path}: se esperaban al menos 4 columnas"
        )
    return data[:, 1:4]


def umeyama_alignment(source: np.ndarray, target: np.ndarray):
    if source.shape != target.shape:
        raise ValueError("source y target deben tener la misma forma")

    n = source.shape[0]
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)

    source_centered = source - mean_source
    target_centered = target - mean_target

    cov = (target_centered.T @ source_centered) / n
    u, d, vt = np.linalg.svd(cov)

    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1

    r = u @ s @ vt
    var_source = np.sum(source_centered**2) / n
    scale = np.trace(np.diag(d) @ s) / var_source
    t = mean_target - scale * (r @ mean_source)

    return scale, r, t


def apply_similarity(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Normaliza dos trayectorias de cámara y las grafica en 3D"
    )
    parser.add_argument(
        "--groundtruth",
        type=Path,
        default=script_dir / "MH_05_difficult_groundTruth.txt",
        help="Archivo ground truth (timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "--estimacion",
        type=Path,
        default=script_dir / "MH_05_estimacion.txt",
        help="Archivo de estimación (timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Ruta opcional para guardar la figura (png)",
    )
    args = parser.parse_args()

    gt_file = args.groundtruth.resolve()
    est_file = args.estimacion.resolve()

    if not gt_file.exists():
        raise FileNotFoundError(f"No existe el archivo groundtruth: {gt_file}")
    if not est_file.exists():
        raise FileNotFoundError(f"No existe el archivo de estimación: {est_file}")

    gt = load_positions(gt_file)
    est = load_positions(est_file)

    n = min(len(gt), len(est))
    if n < 2:
        raise ValueError("No hay suficientes muestras para normalizar y graficar")

    gt_n = gt[:n]
    est_n = est[:n]

    scale, rotation, translation = umeyama_alignment(est_n, gt_n)
    est_aligned = apply_similarity(est_n, scale, rotation, translation)

    rmse = np.sqrt(np.mean(np.sum((est_aligned - gt_n) ** 2, axis=1)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_n[:, 0], gt_n[:, 1], gt_n[:, 2], label="Ground Truth", linewidth=2)
    ax.plot(
        est_aligned[:, 0],
        est_aligned[:, 1],
        est_aligned[:, 2],
        label="Estimación (normalizada)",
        linewidth=2,
    )

    ax.scatter(*gt_n[0], c="green", s=45, marker="o", label="Inicio GT")
    ax.scatter(*est_aligned[0], c="red", s=45, marker="^", label="Inicio Est")

    ax.set_title(f"MH_05 - Trayectorias 3D (RMSE: {rmse:.4f} m)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()

    print(f"Ground Truth: {gt_file}")
    print(f"Estimación:   {est_file}")
    print(f"Muestras usadas: {n}")
    print(f"Escala aplicada a estimación: {scale:.6f}")
    print(f"RMSE tras alineación: {rmse:.6f} m")

    if args.save is not None:
        save_path = args.save.resolve()
        fig.savefig(save_path, dpi=200)
        print(f"Figura guardada en: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
