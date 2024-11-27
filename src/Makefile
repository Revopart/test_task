TARGET = api_server
SRC = recognizer.cpp api_server.cpp
CXX = g++
CXXFLAGS = -std=c++17 -I. $(shell pkg-config --cflags --libs opencv4) -ltesseract -lpthread

# Имя Docker образа
DOCKER_IMAGE = server

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS)

clean:
	rm -f $(TARGET)

rebuild: clean build

docker-build:
	docker build -t $(DOCKER_IMAGE) .


docker-run:
	docker run -p 8000:8000 $(DOCKER_IMAGE)

docker-rebuild: docker-build docker-run
