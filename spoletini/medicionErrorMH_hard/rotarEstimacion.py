import argparse
from pathlib import Path

import numpy as np


def read_tum_file(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            rows.append(list(map(float, vals[:8])))

    if not rows:
        raise ValueError(f"No se pudieron leer poses válidas en: {path}")
    return np.asarray(rows, dtype=np.float64)


def write_tum_file(path: Path, data: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            t, tx, ty, tz, qx, qy, qz, qw = row
            f.write(
                f"{t:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def rotate_points_90deg_x(data: np.ndarray, clockwise: bool) -> np.ndarray:
    out = data.copy()

    theta = -np.pi / 2.0 if clockwise else np.pi / 2.0

    c = np.cos(theta)
    s = np.sin(theta)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )

    xyz = data[:, 1:4]
    xyz_rot = (rx @ xyz.T).T
    out[:, 1:4] = xyz_rot

    return out


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Rota en 3D los puntos (tx, ty, tz) de un archivo de poses TUM "
            "90° sobre X y genera dos salidas: clockwise y counter-clockwise."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "estim_nueva.txt",
        help="Archivo de entrada en formato TUM (default: estim_nueva.txt)",
    )
    parser.add_argument(
        "--output-cw",
        type=Path,
        default=script_dir / "estim_nueva_rotX90_cw.txt",
        help="Salida con rotación X -90° (clockwise).",
    )
    parser.add_argument(
        "--output-ccw",
        type=Path,
        default=script_dir / "estim_nueva_rotX90_ccw.txt",
        help="Salida con rotación X +90° (counter-clockwise).",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_cw_path = args.output_cw.resolve()
    output_ccw_path = args.output_ccw.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No existe archivo de entrada: {input_path}")

    data = read_tum_file(input_path)
    rotated_cw = rotate_points_90deg_x(data, clockwise=True)
    rotated_ccw = rotate_points_90deg_x(data, clockwise=False)

    write_tum_file(output_cw_path, rotated_cw)
    write_tum_file(output_ccw_path, rotated_ccw)

    print(f"Entrada : {input_path}")
    print(f"Salida CW  : {output_cw_path}")
    print(f"Salida CCW : {output_ccw_path}")
    print("Rotaciones aplicadas: X -90° (CW) y X +90° (CCW)")
    print(f"Poses: {len(data)}")


if __name__ == "__main__":
    main()
