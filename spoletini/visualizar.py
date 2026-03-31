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
parser.add_argument(
    "--line-width",
    type=float,
    default=4.0,
    help="Grosor de líneas para trayectoria y frustums",
)
args = parser.parse_args()

# Cargar nube de puntos
pcd = o3d.io.read_point_cloud(args.ply)


def make_cylinder_segment(p0, p1, radius, color):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-9:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)

    # Open3D cylinder axis is +Z by default; rotate it onto segment direction.
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dir_vec = v / length
    cross = np.cross(z_axis, dir_vec)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(z_axis, dir_vec), -1.0, 1.0)

    if cross_norm < 1e-9:
        if dot < 0.0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([np.pi, 0.0, 0.0])
            )
            cyl.rotate(R, center=(0.0, 0.0, 0.0))
    else:
        axis = cross / cross_norm
        angle = np.arccos(dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=(0.0, 0.0, 0.0))

    mid = 0.5 * (p0 + p1)
    cyl.translate(mid)
    return cyl


def line_set_to_cylinders(line_set, radius):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    colors = np.asarray(line_set.colors)
    meshes = []
    for i, (a, b) in enumerate(lines):
        color = colors[i].tolist() if i < len(colors) else [1.0, 1.0, 1.0]
        segment = make_cylinder_segment(points[a], points[b], radius, color)
        if segment is not None:
            meshes.append(segment)
    return meshes


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

line_radius = max(float(args.line_width), 1e-4)
traj_meshes = line_set_to_cylinders(traj, line_radius)
frustum_meshes = []
for frustum in frustums:
    frustum_meshes.extend(line_set_to_cylinders(frustum, line_radius))

# Visualizar todo junto con líneas más gruesas
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="MASt3R-SLAM Visualizer")
vis.add_geometry(pcd)
for mesh in traj_meshes:
    vis.add_geometry(mesh)
for mesh in frustum_meshes:
    vis.add_geometry(mesh)

vis.run()
vis.destroy_window()
