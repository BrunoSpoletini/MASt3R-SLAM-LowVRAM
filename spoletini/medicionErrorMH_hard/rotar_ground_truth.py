import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_tum(path: Path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError("Formato TUM inválido: se esperaban al menos 8 columnas")
    return data  # [t, tx, ty, tz, qx, qy, qz, qw]


def tum_to_SE3(row):
    t = row[1:4]
    q = row[4:8]  # qx qy qz qw
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def SE3_to_tum(timestamp, T):
    t = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # qx qy qz qw
    return [timestamp, *t, *q]


# T_BS del sensor.yaml de EuRoC (cam0): sensor(camara) -> body(IMU)
T_BS = np.array(
    [
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def convert_imu_gt_to_camera_gt(T_wb: np.ndarray) -> np.ndarray:
    # EuRoC GT está en IMU/body: T_WB
    # Queremos GT en cámara: T_WS = T_WB * T_BS
    return T_wb @ T_BS


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Convierte ground truth de EuRoC del frame IMU/body al frame de cámara "
            "usando T_BS de cam0/sensor.yaml."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "gt_norm.txt",
        help="Archivo de entrada TUM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "gt_cam.txt",
        help="Archivo de salida TUM en frame de cámara.",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"No existe archivo de entrada: {input_path}")

    trajectory = load_tum(input_path)

    converted = []
    for row in trajectory:
        T_in = tum_to_SE3(row)
        T_out = convert_imu_gt_to_camera_gt(T_in)
        converted.append(SE3_to_tum(row[0], T_out))

    np.savetxt(output_path, converted, fmt="%.9f")

    print(f"Entrada: {input_path}")
    print(f"Salida : {output_path}")
    print("Conversión: IMU/body -> cámara (T_WS = T_WB * T_BS)")
    print(f"Poses  : {len(converted)}")


if __name__ == "__main__":
    main()
