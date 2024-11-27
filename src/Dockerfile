FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    tesseract-ocr \
    libtesseract-dev \
    libpthread-stubs0-dev \
    pkg-config \
    tesseract-ocr-ell \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN g++ -o server recognizer.cpp api_server.cpp -std=c++17 -I. $(pkg-config --cflags --libs opencv4) -ltesseract -lpthread

EXPOSE 8000

CMD ["./server"]
