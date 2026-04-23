<p align="center">
  <h1 align="center">Fork de MASt3R-SLAM: SLAM denso en tiempo real con priors de reconstrucción 3D</h1>

  <p align="center">Adaptaciones para ejecutar en entornos con poca VRAM y herramientas de visualización añadidas</p>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2412.12392">Paper</a> | <a href="https://youtu.be/wozt71NBFTQ">Video</a> | <a href="https://edexheim.github.io/mast3r-slam/">Project Page</a></h3>
  <div align="center"></div>
 
<p align="center">
    <img src="./media/timelapsePieza.gif" alt="teaser" width="100%">
</p>
<br>

# Introducción

Este repositorio tiene como finalidad mostrar la experimentación realizada con MAST3R-SLAM para la materia de Robótica Móvil de la Facultad de Ciencias Exactas, Ingeniería y Agrimensura de la Universidad Nacional de Rosario.
El proceso completo se detalla en <a href="./informe.pdf">este informe</a>.


## Instalación 

```
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```
Revisar la version de CUDA del sistema con nvcc
```
nvcc --version
```
Instalar la versión de pytorch correspondiente:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Clonar el repo e instalar las dependencias.
```
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

Descargar los checkpoints.
```
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## Usuarios de WSL
Si estás usando WSL, haz checkout a la rama windows y sigue la instalación anterior.
```
git checkout windows
```
Esto desactiva el multiprocesamiento, que causa un problema con la memoria compartida, como se comenta [aquí](https://github.com/rmurai0610/MASt3R-SLAM/issues/21).


## Ejecución sobre un video
El sistema puede procesar videos MP4 o carpetas que contengan imágenes RGB.
```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml
```
Si se conocen los parámetros de calibración, puedes especificarlos en intrinsics.yaml.
```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml --calib config/intrinsics.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml --calib config/intrinsics.yaml
```

En nuestro caso, bajamos y corrimos el programa sobre el dataset de Freiburg1 Room, y sobre un video mp4 grabado por nosotros mismos.

## Reproducibilidad
En el repositorio original de MAST3R-SLAM se ejecutaron todos los experimentos en una RTX 4090 (24 GB de VRAM).
Esta versión puede ejecutarse en una RTX 2070 Super (8 GB de VRAM).

# Explicación del código propio
## Nodos de ArUco
### Aruco Marker Detector

Nodo encargado de leer las imagenes del dataset y publicar las poses de los marcadores ArUco detectados. Tambien puede mostrar la deteccion de los marcadores en tiempo real si se desea.

Para inicializar el nodo es necesario correr el siguiente comando:
```bash
python3 aruco_marker_detector --dict <aruco_dictionary> --size <marker_size> --viz <visualize>
```

Los argumentos del comando son:
- aruco_dictionary: indica el diccionario de marcadores de ArUco a utilizar
- marker_size: indica el tamano de los marcadores ArUco
- visualize: indica si visualizar o no los marcadores ArUco sobre las imagenes de dataset en tiempo real

Los valores por defecto de los argumentos son:
- aruco_dictionary: "DICT_4X4_50"
- marker_size: 0.133
- visualize: true

### Aruco Ground Truth

Nodo encargado de leer las poses de los marcadores ArUco y publicar la pose de la camara segun la pose del marcador y sus coordenadas en el mundo

Para inicializar el nodo es necesario correr el siguiente comando:
```bash
python3 aruco_ground_truth --file <markers_file>
```

Los argumentos del comando son:
- markers_file: archivo donde se encuentran las poses de los marcadores ArUco segun el mundo

Los valores por defecto de los argumentos son:
- markers_file: "markers.txt"

El archivo de las posiciones de los marcadores debe seguir el siguiente formato
```text
# id   x      y      z      roll   pitch  yaw
0      1.50   0.00   0.80   0.0    90.0   0.0
1      3.25   2.10   0.80   0.0    90.0   90.0
```

### Pose Refiner

Nodo encargado de refinar las poses de la camara segun que le mundo que se van publicando para que el ground truth no quede erratico

Para inicializar el nodo es necesario correr el siguiente comando:
```bash
python3 nombre_del_paquete aruco_marker_detector --vel <max_velocity> --apos <alpha_positions> --arot <alpha_rotations>
```

Los argumentos del comando sin:
- max_velocity: indica la velocidad maxima que el ground truth se puede mover
- alpha_positions: indica el alpha de las posiciones para el EMA
- alpha_rotations: indica si alpha de las rotaciones para el EMA

Los valores por defecto de los argumentos son:
- max_velocity: 3.0
- alpha_positions: 0.2
- alpha_rotations: 0.1
