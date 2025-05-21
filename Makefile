# Names of the code directory and the Docker image, change them to match your project
DOCKER_IMAGE_NAME := modeling_activity_pace
DOCKER_CONTAINER_NAME := modeling_activity_pace
CODE_DIRECTORY := modeling_activity_pace

DOCKER_PARAMS=  -it --rm --name=$(DOCKER_CONTAINER_NAME)
# Specify GPU device(s) to use. Comment out this line if you don't have GPUs available
# DOCKER_PARAMS+= --gpus '"device=0,1"'
# A command we can reuse to Docker container while mounting the local directory
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_NAME)

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the Docker image"
	@echo "	run-bash	Run the Docker image in a container, after building it"
	@echo "	stop		Stop the container if it is running"
	@echo "	logs		Display logs"

build:
	docker build -t $(DOCKER_IMAGE_NAME) .

run-bash: build
	$(DOCKER_RUN_MOUNT) /bin/bash || true

stop:
	docker stop $(DOCKER_IMAGE_NAME) || true && docker rm $(DOCKER_IMAGE_NAME) || true

logs:
	docker logs -f $(DOCKER_CONTAINER_NAME)
