import argparse
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description="Visualizador de nube de puntos y poses")
parser.add_argument(
    "ply",
    nargs="?",
    default="data.ply",
    help="Archivo PLY con la nube de puntos (default: data.ply)",
)
parser.add_argument(
    "poses", nargs="?", default="data.txt", help="Archivo de poses (default: data.txt)"
)
args = parser.parse_args()

# Cargar nube de puntos
pcd = o3d.io.read_point_cloud(args.ply)


def make_camera_frustum(T, size=0.05, color=[1, 0.5, 0]):
    """Crea un frustum de cámara (pirámide) en la pose T."""
    # Vértices del frustum en espacio de cámara
    # Eje Z apunta hacia adelante (hacia donde mira la cámara)
    w, h, d = size * 0.8, size * 0.5, size
    corners_cam = np.array(
        [
            [0, 0, 0],  # apex (centro óptico)
            [w, h, d],  # esquina superior derecha
            [-w, h, d],  # esquina superior izquierda
            [-w, -h, d],  # esquina inferior izquierda
            [w, -h, d],  # esquina inferior derecha
        ]
    )
    # Transformar al mundo
    R = T[:3, :3]
    t = T[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    # Líneas: apex -> 4 esquinas + rectángulo frontal
    points = corners_world.tolist()
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


# Cargar poses (formato: timestamp tx ty tz qx qy qz qw)
transforms = []
with open(args.poses) as f:
    for line in f:
        if line.startswith("#"):
            continue
        vals = list(map(float, line.split()))
        tx, ty, tz = vals[1], vals[2], vals[3]
        qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]

        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        transforms.append(T)

# Frustums de cámara
frustums = [make_camera_frustum(T) for T in transforms]

# Trayectoria: línea que conecta los centros de cámara
centers = np.array([T[:3, 3] for T in transforms])
traj = o3d.geometry.LineSet()
traj.points = o3d.utility.Vector3dVector(centers)
traj.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centers) - 1)])
traj.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * max(len(centers) - 1, 0))

# Visualizar todo junto
o3d.visualization.draw_geometries([pcd, traj] + frustums)
