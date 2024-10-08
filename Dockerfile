FROM python:3.11-slim

# Установите необходимые зависимости для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./src .
COPY ./requirements.txt .

ENV CMAKE_ARGS="-DGGML_CUDA=on"

RUN pip install --no-cache-dir -r requirements.txt 
    
RUN pip install llama-cpp-python==0.2.69 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/122

CMD ["python", "main.py"]