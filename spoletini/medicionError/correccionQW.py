import numpy as np

data = np.loadtxt("gt_cam.txt")
# Columnas actuales: t x y z qw qx qy qz
# TUM necesita:      t x y z qx qy qz qw

gt_tum = np.column_stack(
    [
        data[:, 0],  # timestamp
        data[:, 1:4],  # x y z
        data[:, 5:8],  # qx qy qz
        data[:, 4],  # qw  (lo movemos al final)
    ]
)

np.savetxt("gt_cam_tum.txt", gt_tum, fmt="%.9f")
