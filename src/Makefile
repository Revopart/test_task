# Имя Docker-образа
IMAGE_NAME = cpp-ocr-api
CONTAINER_NAME = cpp-ocr-container

# Основные команды
all: build run

# Сборка Docker-образа
build:
	docker build -t $(IMAGE_NAME) .

# Запуск контейнера
run:
	docker run --name $(CONTAINER_NAME) -p 8000:8000 -d $(IMAGE_NAME)

# Остановка контейнера
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Перезапуск контейнера
restart: stop run

# Очистка: удаление образа и контейнера
clean: stop
	docker rmi $(IMAGE_NAME) || true

# Просмотр логов контейнера
logs:
	docker logs -f $(CONTAINER_NAME)

# Тестовый запрос (зависит от наличия curl)
test:
	curl -X 'POST' \
	  'http://127.0.0.1:8000/predict' \
	  -H 'accept: application/json' \
	  -H 'Content-Type: multipart/form-data' \
	  -F 'file=@letters.png'
