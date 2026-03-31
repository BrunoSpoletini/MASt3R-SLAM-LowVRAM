import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoDetector(Node):

    def __init__(self):
        # Inicializa el nodo ROS2
        super().__init__('aruco_detector')

        # Se utiliza CvBridge para convertir imágenes ROS a OpenCV
        self.bridge = CvBridge()

        # Inicialmente no se conoce la calibración de la cámara
        self.camera_matrix = None
        self.dist_coeffs = None

        # Suscripciones a imagen y parámetros de cámara
        self.create_subscription(Image, '/image_raw', self.image_cb, 10)
        self.create_subscription(CameraInfo, '/camera_info', self.info_cb, 10)

        # Publicador de poses de markers detectados
        self.pub = self.create_publisher(PoseArray, '/aruco_detections', 10)

        # Diccionario de ArUco utilizado
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        # Tamaño físico del marker en metros
        self.marker_size = 0.10

    def info_cb(self, msg):
        # Se recibe la calibración de la cámara
        # Se construye la matriz intrínseca 3x3
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

        # Coeficientes de distorsión
        self.dist_coeffs = np.array(msg.d)

    def image_cb(self, msg):

        # Si aún no se recibió la calibración, no se procesa la imagen
        if self.camera_matrix is None:
            self.get_logger().warn("Esperando CameraInfo...")
            return

        # Conversión de imagen ROS a OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Conversión a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección de markers
        corners, ids, _ = aruco.detectMarkers(gray, self.dict)

        # Se prepara el mensaje de salida
        pose_array = PoseArray()
        pose_array.header = msg.header

        if ids is not None:

            # Estimación de pose de cada marker respecto a la cámara
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self.camera_matrix,
                self.dist_coeffs
            )

            # Se recorre cada marker detectado
            for i, marker_id in enumerate(ids.flatten()):

                pose = Pose()

                # Vector de traslación (posición relativa cámara->marker)
                tvec = tvecs[i][0]

                # Vector de rotación en formato Rodrigues
                rvec = rvecs[i][0]

                # Se asigna la posición
                pose.position.x = float(tvec[0])
                pose.position.y = float(tvec[1])
                pose.position.z = float(tvec[2])

                # Se almacena el rvec en la orientación (uso temporal)
                pose.orientation.x = float(rvec[0])
                pose.orientation.y = float(rvec[1])
                pose.orientation.z = float(rvec[2])

                # Se utiliza el campo w para transportar el ID del marker
                pose.orientation.w = float(marker_id)

                pose_array.poses.append(pose)

                # Se dibuja el eje del marker en la imagen
                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    0.05
                )

        # Publicación de las detecciones
        self.pub.publish(pose_array)

        # Visualización
        cv2.imshow("Aruco Detector PRO", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()