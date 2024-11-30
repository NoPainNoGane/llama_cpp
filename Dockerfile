FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

COPY ./src .
COPY ./requirements.txt .

ENV CMAKE_ARGS="-DGGML_CUDA=on"

RUN apt-get update -y && apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev -y && \
ln -s /usr/bin/python3 /usr/bin/python && \    
pip install --no-cache-dir -r requirements.txt 

RUN pip install llama-cpp-python==0.2.88 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/122

EXPOSE 8000

CMD ["python", "main.py"]


# FROM python:3.11-slim

# # Установите необходимые зависимости для сборки
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY ./src .
# COPY ./requirements.txt .

# ENV CMAKE_ARGS="-DGGML_CUDA=on"

# RUN pip install --no-cache-dir -r requirements.txt 
    
# RUN pip install llama-cpp-python==0.2.80 \
#     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/122

# CMD ["python", "main.py"]