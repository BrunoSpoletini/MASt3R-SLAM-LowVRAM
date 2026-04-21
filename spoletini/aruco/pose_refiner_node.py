import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import argparse
import sys

# Nodo encargado de refinar las poses de la camara segun que le mundo que se van publicando para que el ground truth no quede erratico

class PoseRefiner(Node):
    def __init__(self, max_velocity, alpha_pos, alpha_rot):
        super().__init__('pose_refiner')
        
        # Asignamos los argumentos de python a variables de la clase
        self.max_velocity = max_velocity
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot

        self.sub = self.create_subscription(PoseArray, '/ground_truth_raw', self.callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/ground_truth', 10)
        
        # Variables de estado para llevar refinar la funcion
        self.last_pose = None
        self.last_time = None

    def callback(self, msg):
        if not msg.poses:
            return

        current_time = self.get_clock().now()
        
        # Para las traslaciones de las poses obtenidas en el mismo instante obtenemos la mediana,
        # ya que es mucho más robusto que la media frente a un marker outlier
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
        raw_pos = np.median(positions, axis=0)
        
        # Para las rotaciones de las poses obtenidas en el mismo instante, obtenemos la rotacion 
        # del marcador mas cercano, ya que promediar cuaterniones es complejo, usaremos el primero para esta lógica
        raw_quat = [msg.poses[0].orientation.x, msg.poses[0].orientation.y, 
                    msg.poses[0].orientation.z, msg.poses[0].orientation.w]

        # Si la nueva pose se ha movido demasiado en poco tiempo, lo rechazamos
        if self.last_pose is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            dist = np.linalg.norm(raw_pos - self.last_pose['pos'])
            
            if dt > 0 and (dist / dt) > self.max_velocity:
                self.get_logger().warn(f"Salto detectado ({dist/dt:.2f} m/s). Ignorando pose.")
                return

        # Realizamos un suavizado Exponential Moving Average (EMA)
        if self.last_pose is None:
            refined_pos = raw_pos
            refined_quat = raw_quat
        else:
            # Suavizamos la posición
            refined_pos = self.alpha_pos * raw_pos + (1.0 - self.alpha_pos) * self.last_pose['pos']
            
            # Suavizamos la rotación usando Spherical Linear Interpolation (SLERP)
            key_times = [0, 1]
            key_rots = R.from_quat([self.last_pose['quat'], raw_quat])
            slerp = Slerp(key_times, key_rots)
            refined_quat = slerp([self.alpha_rot])[0].as_quat()

        # Guardamos el nuevo estado
        self.last_pose = {'pos': refined_pos, 'quat': refined_quat}
        self.last_time = current_time

        # Publicamos el ground truth refinado
        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = "world"
        out.pose.position.x, out.pose.position.y, out.pose.position.z = refined_pos
        out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = refined_quat
        self.pub.publish(out)

def main():
    # Obtenemos los argumentos
    parser = argparse.ArgumentParser(description='Refinador de pose con filtros EMA y SLERP')
    parser.add_argument('--vel', type=float, default=3.0, help='Velocidad máxima permitida (m/s)')
    parser.add_argument('--apos', type=float, default=0.2, help='Alpha para suavizado de posición')
    parser.add_argument('--arot', type=float, default=0.1, help='Alpha para suavizado de rotación')
    args, _ = parser.parse_known_args()

    rclpy.init()
    
    # Pasamos los argumentos capturados al constructor del nodo
    node = PoseRefiner(args.vel, args.apos, args.arot)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()