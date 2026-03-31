import argparse
import time
import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoDetector:
    def __init__(self):
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.marker_length_m = 0.05
        self.axis_length_m = 0.03
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def process_frame(self, frame):

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

        return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Aruco detector sobre video o webcam")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Ruta al archivo de video. Si no se indica, se usa la webcam.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice de cámara a usar cuando no se pasa --video.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detector = ArucoDetector()

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        cap = cv2.VideoCapture(args.camera_index)
        source_name = f"camera index {args.camera_index}"

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source_name}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = 30.0
    frame_period_s = 1.0 / source_fps

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            output = detector.process_frame(frame)
            cv2.imshow("Aruco", output)

            elapsed_s = time.perf_counter() - loop_start
            remaining_s = max(0.0, frame_period_s - elapsed_s)
            wait_ms = max(1, int(round(remaining_s * 1000.0)))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
