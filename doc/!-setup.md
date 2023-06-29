# Installation

The pipeline has been tested on Ubuntu 22.04 LTS using and NVIDIA GPU P100/V100 with Driver Version: 515.86.01.

The full environment can be found in the Dockerfile, below we describe how it can be built using docker container. Note that it is possible to directly setup the environment without the use of docker.

## Container


### Installs on the Host system

This is a Docker container that contains necessary software for performing MVFlow crowd tracking. Please ensure that you have the necessary Nvidia drivers installed on your host environment before proceeding. The drivers can be downloaded from the NVIDIA Drivers page: https://www.nvidia.com/Download/index.aspx. It is recommended to download the latest version of the drivers that are compatible with the user's hardware and operating system.

You will also need the NVIDIA Container Toolkit installed on your host environment. This can be done by executing:

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
And installing nvidia-docker2

```
apt-get update
apt-get install -y nvidia-docker2
```

Then restart the Docker daemon:

```
systemctl restart docker
```

### Building the Docker Image

To build the Docker image, navigate to the root directory and run the following command:

`docker build --progress=plain -t container:latest . &> docker_build.log`


This command will create a Docker image named `container:latest` and log the build output to a file named `docker_build.log`. 

### Running the Docker Container

To run the Docker container, use the following command from the root directory:

`docker run --gpus all -v .:/root -p 8080:8080 -p 5000:5000 -it container /bin/bash`

Once the container is running, you can interact with it using a Bash shell. Any changes you make to the container's file system will be lost when the container is stopped.

The -v flag mounts the host directory `$/data` into the container. You can modify this path as necessary.

The -p flag maps the host port 8080 to the container port 8080. This allows you to access the Flask server running in the container from your host environment.
The -p flag maps the host port 5000 to the container port 5000. This allows you to access the jupyter server running in the container from your host environment.

### Starting the Container

`docker start $container_id`

### Attaching to the Container

`docker attach $container_id`
