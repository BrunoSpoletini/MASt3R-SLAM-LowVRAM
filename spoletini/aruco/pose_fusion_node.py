import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
import numpy as np


class PoseFusionNode(Node):

    def __init__(self):
        # Inicialización del nodo
        super().__init__('pose_fusion_node')

        # Suscripción a poses individuales
        self.sub = self.create_subscription(
            PoseArray,
            '/ground_truth_raw',
            self.callback,
            10
        )

        # Publicador de la pose final fusionada
        self.pub = self.create_publisher(
            PoseStamped,
            '/ground_truth',
            10
        )

        # Umbral para detección de outliers
        self.outlier_threshold = 2.0

    def callback(self, msg):

        # Si no hay poses, no se procesa
        if not msg.poses:
            return

        # Caso trivial: un solo marker
        if len(msg.poses) == 1:
            out = PoseStamped()
            out.header = msg.header
            out.pose = msg.poses[0]
            self.pub.publish(out)
            return

        positions = []
        quats = []

        # Extracción de datos
        for pose in msg.poses:
            positions.append([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])

            quats.append([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])

        positions = np.array(positions)
        quats = np.array(quats)

        # Cálculo de media y desviación estándar
        mean = np.mean(positions, axis=0)
        std = np.std(positions, axis=0)

        filtered_positions = []
        filtered_quats = []

        # Eliminación de outliers
        for i, pos in enumerate(positions):

            dist = np.linalg.norm(pos - mean)
            std_norm = np.linalg.norm(std)

            if std_norm < 1e-6:
                keep = True
            else:
                keep = dist < self.outlier_threshold * std_norm

            if keep:
                filtered_positions.append(pos)
                filtered_quats.append(quats[i])
            else:
                self.get_logger().warn(f"Outlier removido: {pos}")

        if not filtered_positions:
            return

        filtered_positions = np.array(filtered_positions)
        filtered_quats = np.array(filtered_quats)

        self.get_logger().info(f"Usando {len(filtered_positions)} markers")

        # Promedio ponderado por distancia
        weighted_pos = np.zeros(3)
        total_weight = 0
        weights = []

        for pos in filtered_positions:

            dist = np.linalg.norm(pos)

            if dist < 1e-6:
                continue

            # Se asigna mayor peso a markers cercanos
            w = 1.0 / dist

            weighted_pos += w * pos
            total_weight += w
            weights.append(w)

        if total_weight == 0:
            return

        avg_pos = weighted_pos / total_weight

        weights = np.array(weights).reshape(-1, 1)

        # Promedio de quaternions
        avg_quat = np.sum(weights * filtered_quats, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)

        # Construcción del mensaje final
        out = PoseStamped()
        out.header = msg.header

        out.pose.position.x = float(avg_pos[0])
        out.pose.position.y = float(avg_pos[1])
        out.pose.position.z = float(avg_pos[2])

        out.pose.orientation.x = float(avg_quat[0])
        out.pose.orientation.y = float(avg_quat[1])
        out.pose.orientation.z = float(avg_quat[2])
        out.pose.orientation.w = float(avg_quat[3])

        # Publicación
        self.pub.publish(out)

        self.get_logger().info(
            f"FUSED -> x={avg_pos[0]:.3f}, y={avg_pos[1]:.3f}, z={avg_pos[2]:.3f}"
        )


def main():
    rclpy.init()
    node = PoseFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()