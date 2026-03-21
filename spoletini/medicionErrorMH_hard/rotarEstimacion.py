import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_tum_file(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, vals[:8])
            rows.append([t, tx, ty, tz, qx, qy, qz, qw])

    if not rows:
        raise ValueError(f"No se pudieron leer poses válidas en: {path}")

    return np.asarray(rows, dtype=np.float64)


def write_tum_file(path: Path, data: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            t, tx, ty, tz, qx, qy, qz, qw = row
            f.write(
                f"{t:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def get_axis_rotation_candidates():
    candidates = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            r = np.zeros((3, 3), dtype=np.float64)
            for i, p in enumerate(perm):
                r[i, p] = signs[i]
            if np.isclose(np.linalg.det(r), 1.0):
                candidates.append(r)
    return candidates


def sync_by_timestamp(est: np.ndarray, ref: np.ndarray, max_dt: float = 0.02):
    t_est = est[:, 0]
    t_ref = ref[:, 0]

    pairs = []
    for i, t in enumerate(t_est):
        j = np.searchsorted(t_ref, t)
        best_j = None
        best_dt = None
        for cand in (j - 1, j):
            if 0 <= cand < len(t_ref):
                dt = abs(t_ref[cand] - t)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_j = cand
        if best_j is not None and best_dt is not None and best_dt <= max_dt:
            pairs.append((i, best_j))

    if len(pairs) < 10:
        n = min(len(est), len(ref))
        return est[:n, 1:4], ref[:n, 1:4], n

    est_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    ref_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    return est[est_idx, 1:4], ref[ref_idx, 1:4], len(pairs)


def fit_scale_and_translation(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_c = source - source_mean
    target_c = target - target_mean

    denom = np.sum(source_c**2)
    if denom < 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(source_c * target_c) / denom)

    translation = target_mean - scale * source_mean
    return scale, translation


def score_rotation(r_fix: np.ndarray, est_xyz: np.ndarray, ref_xyz: np.ndarray):
    est_rot = (r_fix @ est_xyz.T).T
    scale, translation = fit_scale_and_translation(est_rot, ref_xyz)
    aligned = scale * est_rot + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - ref_xyz) ** 2, axis=1))))
    return rmse, scale, translation


def select_best_rotation(est_data: np.ndarray, ref_data: np.ndarray):
    est_xyz, ref_xyz, n_sync = sync_by_timestamp(est_data, ref_data)

    best = None
    for r_fix in get_axis_rotation_candidates():
        rmse, scale, translation = score_rotation(r_fix, est_xyz, ref_xyz)
        candidate = (rmse, scale, translation, r_fix)
        if best is None or rmse < best[0]:
            best = candidate

    rmse, scale, translation, r_fix = best
    return r_fix, rmse, scale, translation, n_sync


def rotate_estimation(data: np.ndarray, r_fix: np.ndarray):
    out = data.copy()

    positions = data[:, 1:4]
    quats = data[:, 4:8]  # qx qy qz qw

    rot_fix = Rotation.from_matrix(r_fix)
    rot_old = Rotation.from_quat(quats)

    # Rotación global de ejes: p' = R p, R' = R R_old
    positions_new = (r_fix @ positions.T).T
    rot_new = rot_fix * rot_old
    quats_new = rot_new.as_quat()

    out[:, 1:4] = positions_new
    out[:, 4:8] = quats_new
    return out


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Rota una trayectoria estimada (TUM) para alinear ejes con EuRoC. "
            "Si se provee --ref, busca automáticamente la mejor rotación de ejes."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "estim.txt",
        help="Archivo de entrada en formato TUM (default: estim.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "estim_euroc_axes.txt",
        help="Archivo de salida en formato TUM (default: estim_euroc_axes.txt)",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=script_dir / "gt_norm.txt",
        help=(
            "Ground truth para elegir automáticamente la mejor rotación de ejes "
            "(default: gt_norm.txt). Si no existe, usa una conversión fija cam->EuRoC."
        ),
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    ref_path = args.ref.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No existe archivo de entrada: {input_path}")

    data = read_tum_file(input_path)

    if ref_path.exists():
        ref_data = read_tum_file(ref_path)
        r_fix, rmse, scale, translation, n_sync = select_best_rotation(data, ref_data)
        print(f"[Auto] Ground truth: {ref_path}")
        print(f"[Auto] Poses sincronizadas: {n_sync}")
        print(f"[Auto] RMSE (con solo escala+traslación): {rmse:.6f} m")
        print(f"[Auto] Escala de evaluación: {scale:.6f}")
        print(f"[Auto] Traslación de evaluación: {translation}")
    else:
        r_fix = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        print("[Info] No se encontró --ref, usando conversión fija cam->EuRoC.")

    rotated = rotate_estimation(data, r_fix)

    write_tum_file(output_path, rotated)

    print(f"Entrada : {input_path}")
    print(f"Salida  : {output_path}")
    print(f"Poses   : {len(rotated)}")
    print("Rotación de ejes aplicada:")
    print(r_fix)


if __name__ == "__main__":
    main()
