#FROM python:3.11-slim
#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
#
#WORKDIR /app
#
## Create model data directory
##RUN mkdir -p /app/model-data
#
## Install system dependencies
##RUN apt-get update && apt-get install -y \
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libsm6 \
#    libxext6 \
#    libxrender-dev \
#    libopencv-dev \
#    tesseract-ocr \
#    && rm -rf /var/lib/apt/lists/*
#
## Install Python dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy application files
#COPY src ./src
#COPY images ./images
#
##CMD [ "python", "./src/main.py" ]
##CMD [ "python", "/app/rapidocr_test.py" ]
##CMD [ "python", "trocr-test.py" ]
##CMD [ "python", "./src/testtrocr.py" ]
##CMD [ "python", "./src/ocr.py" ]
#CMD [ "python", "./src/test.py" ]
#

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 or newer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3 the default
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub[hf_xet]

# Copy application files
COPY src ./src
COPY images ./images
COPY .env .

# Create directories
RUN mkdir -p /app/model-data /app/output

#CMD ["python", "./src/test.py"]
CMD [ "python", "./src/main.py" ]
