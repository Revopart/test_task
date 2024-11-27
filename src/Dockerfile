# Используем официальный образ Ubuntu в качестве базового
FROM ubuntu:22.04

# Устанавливаем основные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    libopencv-dev \
    g++ \
    libleptonica-dev \
    tesseract-ocr \
    tesseract-ocr-ell \
    libtesseract-dev

# Устанавливаем OpenCV (если требуется последняя версия, можно использовать более сложный процесс сборки)
RUN apt-get install -y libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev

# Создаём рабочую директорию
WORKDIR /app

# Копируем исходный код в контейнер
COPY . /app

# Скачиваем библиотеку cpp-httplib
RUN git clone https://github.com/yhirose/cpp-httplib.git /app/cpp-httplib

# Компилируем сервер
RUN g++ -o server main.cpp -I./cpp-httplib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -ltesseract -lpthread

# Указываем порт для API
EXPOSE 8000

# Запускаем сервер
CMD ["./server"]
