FROM debian:bullseye-slim as yolox

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    git \
    wget \
    emacs \
    protobuf-compiler \
    cmake \
    libgl1-mesa-dev

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install loguru
RUN pip install tabulate
RUN pip install opencv_python
RUN pip install onnx==1.8.1
RUN pip install onnx-simplifier==0.3.5

RUN git clone https://github.com/Megvii-BaseDetection/YOLOX

WORKDIR /YOLOX

RUN wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
COPY export_onnx.sh /YOLOX

ENTRYPOINT ["/bin/bash", "export_onnx.sh"]
