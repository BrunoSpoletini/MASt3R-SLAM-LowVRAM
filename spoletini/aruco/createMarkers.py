import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os

# ===== CONFIG =====
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_size = 400  # tamaño del código (sin borde blanco)
border_size = 100  # borde blanco alrededor

# ===== ARGUMENTOS =====
parser = argparse.ArgumentParser(description="Generar marcadores ArUco")
parser.add_argument(
    "-n",
    "--num_markers",
    type=int,
    default=5,
    help="Número de marcadores a generar (default: 5)",
)
parser.add_argument(
    "-s",
    "--start_id",
    type=int,
    default=0,
    help="ID inicial de los marcadores (default: 0)",
)
parser.add_argument(
    "-od",
    "--output_dir",
    type=str,
    default="markers",
    help="Directorio de salida (default: markers)",
)
args = parser.parse_args()

# ===== CREAR DIRECTORIO DE SALIDA =====
os.makedirs(args.output_dir, exist_ok=True)

# ===== GENERAR MARCADORES =====
for i in range(args.num_markers):
    marker_id = args.start_id + i
    print(f"Generando marcador ID {marker_id}...")

    marker = aruco.drawMarker(aruco_dict, marker_id, marker_size)

    # Crear fondo blanco
    total_size = marker_size + 2 * border_size
    image = np.ones((total_size, total_size), dtype=np.uint8) * 255

    # Insertar marcador
    image[
        border_size : border_size + marker_size, border_size : border_size + marker_size
    ] = marker

    # Guardar
    filename = os.path.join(args.output_dir, f"aruco_{marker_id:03d}.png")
    cv2.imwrite(filename, image)
    print(f"  ✓ Guardado: {filename}")

print(f"\n✓ {args.num_markers} marcadores generados en '{args.output_dir}'")
