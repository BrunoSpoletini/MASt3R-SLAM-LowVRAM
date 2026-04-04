#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def quat_to_rot(qx: float, qy: float, qz: float, qw: float):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def rot_to_quat(r):
    trace = r[0][0] + r[1][1] + r[2][2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def swap_yz_rotation(r):
    # Change of basis with axis permutation P that swaps y <-> z.
    # R' = P R P
    idx = [0, 2, 1]
    return [[r[idx[i]][idx[j]] for j in range(3)] for i in range(3)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intercambia Y/Z en posición y orientación de un archivo TUM."
    )
    parser.add_argument("input", type=Path, help="Archivo TUM de entrada")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Archivo de salida (por defecto: <input>_yz_pose.tum)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"No existe el archivo: {args.input}")

    out_path = args.output or args.input.with_name(
        f"{args.input.stem}_yz_pose{args.input.suffix}"
    )

    rows_out = []
    with args.input.open("r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(
                    "Formato TUM inválido: se esperaban al menos 8 columnas"
                )

            t = float(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            qx = float(parts[4])
            qy = float(parts[5])
            qz = float(parts[6])
            qw = float(parts[7])

            r = quat_to_rot(qx, qy, qz, qw)
            r_swapped = swap_yz_rotation(r)
            qx2, qy2, qz2, qw2 = rot_to_quat(r_swapped)

            rows_out.append((t, x, z, y, qx2, qy2, qz2, qw2))

    with out_path.open("w", encoding="utf-8") as outfile:
        for row in rows_out:
            outfile.write(
                f"{row[0]:.9f} {row[1]:.15g} {row[2]:.15g} {row[3]:.15g} "
                f"{row[4]:.15g} {row[5]:.15g} {row[6]:.15g} {row[7]:.15g}\n"
            )

    print(f"Entrada: {args.input}")
    print(f"Salida:  {out_path}")
    print("Transformación aplicada: swap Y<->Z en posición y orientación")


if __name__ == "__main__":
    main()
