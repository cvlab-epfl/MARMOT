FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
LABEL maintainer="Your Name <youremail@example.com>"

ARG DEBIAN_FRONTEND=noninteractive

# Install apt-getable dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libopencv-dev \
        libceres-dev \
        python3-dev \
        python3-numpy \
        python3-opencv \
        python3-pip \
        python3-pyproj \
        python3-scipy \
        python3-yaml \
        python-is-python3 \
        curl \
        sudo \
        htop \
        wget \
        tmux \
        nano \
        neovim \
        sshfs \
        locate \
        mc \
        less \
        libgtk-3-dev \
        libboost-all-dev \
        libboost-python-dev \
        expat \
        libcgal-dev \
        libsparsehash-dev \
        unzip \
        dos2unix \
        exiftool \
        libmagic-dev \
        ffmpeg \
        libzmq3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade pip
# RUN pip install install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117


WORKDIR /

RUN git clone --recursive https://github.com/mapillary/OpenSfM.git /OpenSfM

# set up OpenSfM
WORKDIR /OpenSfM
# RUN pip install "cython==0.29.35"
RUN git checkout 8887d336cdc305427d59d02a96ef9396aa197ad3
#RUN cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install
RUN pip install cloudpickle==0.4.0 \
exifread==2.1.2 \
flask==2.3.2 \
fpdf2==2.4.6 \
joblib==0.14.1 \
matplotlib \
networkx==2.5 \
numpy>=1.19 \
Pillow>=8.1.1 \
pyproj>=1.9.5.1 \
pytest==3.0.7 \
python-dateutil>=2.7 \
pyyaml==6.0.1 \
scipy>=1.10.0 \
Sphinx==4.2.0 \
six \
xmltodict==0.10.2 \
wheel \
opencv-python
#RUN pip install -r requirements.txt && \
RUN python3 setup.py build

# install the viewer dependencies for OpenSfM
RUN bash /OpenSfM/viewer/node_modules.sh

ENV PYTHONPATH="/OpenSfM/viewer:$PYTHONPATH"

# OpenSfM setup is complete

WORKDIR /
# Install mot3d
RUN git clone https://github.com/cvlab-epfl/mot3d.git /mot3d

ENV PYTHONPATH="/mot3d:$PYTHONPATH"
# RUN echo 'export PYTHONPATH="/mot3d:$PYTHONPATH"' >> ~/.bashrc

WORKDIR /mot3d

RUN unzip mussp-master-6cf61b8.zip && \
    rm mussp-master-6cf61b8.zip

WORKDIR /mot3d/mot3d/solvers/wrappers/muSSP/

RUN ./cmake_and_build.sh


# Start setting up the crowd tracking environment





RUN git clone https://github.com/cvlab-epfl/MVFlow.git /MVFlow

ENV HOME /

WORKDIR /

COPY requirements.txt  .
RUN pip install --no-cache-dir -r requirements.txt 

WORKDIR /root

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install scikit-spatial shapely pyransac3d imageio[ffmpeg]
RUN python3 -m pip install django-bootstrap-form

# RUN python setupenv.py
