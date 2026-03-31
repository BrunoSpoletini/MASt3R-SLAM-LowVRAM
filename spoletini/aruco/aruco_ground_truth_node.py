import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


class GroundTruthNode(Node):

    def __init__(self):
        # Inicialización del nodo
        super().__init__('ground_truth_node')

        # Suscripción a las detecciones de ArUco
        self.sub = self.create_subscription(
            PoseArray,
            '/aruco_detections',
            self.callback,
            10
        )

        # Publicador de poses individuales estimadas en el mundo
        self.pub = self.create_publisher(
            PoseArray,
            '/ground_truth_raw',
            10
        )

        # Carga de posiciones conocidas de los markers
        self.markers = self.load_markers("markers.txt")

    def load_markers(self, filename):
        # Carga desde archivo las posiciones de los markers en el mundo
        markers = {}
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                marker_id = int(parts[0])
                pos = np.array(list(map(float, parts[1:4])))
                markers[marker_id] = pos
        return markers

    def create_transform(self, R_mat, t_vec):
        # Construcción de matriz homogénea 4x4
        T = np.eye(4)
        T[0:3, 0:3] = R_mat
        T[0:3, 3] = t_vec
        return T

    def callback(self, msg):

        pose_array = PoseArray()
        pose_array.header = msg.header

        # Se procesa cada marker detectado
        for pose in msg.poses:

            marker_id = int(pose.orientation.w)

            # Se ignoran markers desconocidos
            if marker_id not in self.markers:
                continue

            # Vector de traslación cámara->marker
            tvec = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])

            # Vector de rotación Rodrigues
            rvec = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z
            ])

            # Conversión a matriz de rotación
            R_cam_marker, _ = cv2.Rodrigues(rvec)

            # Transformación cámara->marker
            T_cam_marker = self.create_transform(R_cam_marker, tvec)

            # Inversión para obtener marker->cámara
            T_marker_cam = np.linalg.inv(T_cam_marker)

            # Transformación mundo->marker
            T_world_marker = np.eye(4)
            T_world_marker[0:3, 3] = self.markers[marker_id]

            # Composición: mundo->cámara
            T_world_cam = T_world_marker @ T_marker_cam

            # Extracción de posición y rotación
            pos = T_world_cam[0:3, 3]
            rot = T_world_cam[0:3, 0:3]

            # Conversión a quaternion
            quat = R.from_matrix(rot).as_quat()

            # Normalización del quaternion
            quat = quat / np.linalg.norm(quat)

            out_pose = Pose()
            out_pose.position.x = float(pos[0])
            out_pose.position.y = float(pos[1])
            out_pose.position.z = float(pos[2])

            out_pose.orientation.x = float(quat[0])
            out_pose.orientation.y = float(quat[1])
            out_pose.orientation.z = float(quat[2])
            out_pose.orientation.w = float(quat[3])

            pose_array.poses.append(out_pose)

        # Publicación solo si hay resultados
        if pose_array.poses:
            self.pub.publish(pose_array)

            self.get_logger().info(
                f"Publicadas {len(pose_array.poses)} poses (una por marker)"
            )


def main():
    rclpy.init()
    node = GroundTruthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()