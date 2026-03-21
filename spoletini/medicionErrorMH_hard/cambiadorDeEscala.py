import numpy as np

input_file = "gt.txt"
output_file = "gt_norm.txt"

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

# Normalizar tiempo
t0 = timestamps[0]
timestamps = (timestamps - t0) / 1e9  # ns → s y arranca en 0

# Guardar archivo TUM
with open(output_file, "w") as f:
    for t, pose in zip(timestamps, poses):
        f.write(f"{t:.6f} " + " ".join(pose) + "\n")

print(f"Archivo guardado como: {output_file}")
