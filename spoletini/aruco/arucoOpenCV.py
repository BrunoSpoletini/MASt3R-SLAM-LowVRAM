import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_detector")

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/image_raw", self.callback, 10)

        self.dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.marker_length_m = 0.05
        self.axis_length_m = 0.03
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        corners, ids, _ = aruco.detectMarkers(frame, self.dict)

        if ids is not None:
            print("Detected:", ids.flatten())
            aruco.drawDetectedMarkers(frame, corners, ids)

            if self.camera_matrix is None:
                height, width = frame.shape[:2]
                focal_length = float(width)
                self.camera_matrix = np.array(
                    [
                        [focal_length, 0.0, width / 2.0],
                        [0.0, focal_length, height / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_length_m,
                self.camera_matrix,
                self.dist_coeffs,
            )

            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.axis_length_m,
                )

        cv2.imshow("Aruco", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
