FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget curl git build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    libglfw3-dev libglew-dev \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Forzar python3.11 como default (la imagen trae 3.10)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2 \
    && python3.11 -m pip install --upgrade pip

# Instalar torch PRIMERO, antes de copiar el repo
# (curope lo necesita disponible en build time)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY . .

RUN pip install -e thirdparty/mast3r
RUN pip install -e thirdparty/in3d
RUN pip install --no-build-isolation -e .

CMD ["bash"]