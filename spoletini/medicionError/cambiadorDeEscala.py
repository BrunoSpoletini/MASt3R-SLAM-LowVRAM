import numpy as np

input_file = "./easy/gt.tum"
output_file = "./easy/gt_norm.tum"

timestamps = []
poses = []

# Leer archivo
with open(input_file, "r") as f:
    for line in f:
        if line.strip() == "" or line.startswith("#"):
            continue

        parts = line.strip().split()

        # formato esperado:
        # timestamp tx ty tz qx qy qz qw
        t = float(parts[0])
        pose = parts[1:]

        timestamps.append(t)
        poses.append(pose)

# Convertir a numpy
timestamps = np.array(timestamps)


def infer_time_divisor(ts: np.ndarray) -> tuple[float, str]:
    if ts.size < 2:
        return 1.0, "s"

    dt = np.diff(ts)
    median_step = float(np.median(np.abs(dt)))

    if median_step >= 1e6:
        return 1e9, "ns"
    if median_step >= 1e3:
        return 1e6, "us"
    if median_step >= 1:
        return 1e3, "ms"
    return 1.0, "s"


# Normalizar tiempo a segundos, arrancando en 0
t0 = timestamps[0]
divisor, detected_unit = infer_time_divisor(timestamps)
timestamps = (timestamps - t0) / divisor

# Guardar archivo TUM
with open(output_file, "w") as f:
    for t, pose in zip(timestamps, poses):
        f.write(f"{t:.20f} " + " ".join(pose) + "\n")

print(f"Unidad detectada en timestamp: {detected_unit}")
print(f"Archivo guardado como: {output_file}")
