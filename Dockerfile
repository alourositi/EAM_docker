# What you need to build image and use it with this Dockerfile
# ### installation
# sudo chmod 776 nvidia-container-runtime-script.sh
# sudo ./nvidia-container-runtime-script.sh 
# sudo docker build -t docker_eam .
# sudo docker run -it --rm --gpus all --net=host --privileged --volume=/dev:/dev docker_eam

ARG CUDA="10.2"
ARG CUDNN="7"
ARG distro="18.04"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${distro}

ENV TZ=Europe/Athens
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install basics
RUN apt-get -y update \
 && apt-get -y install --no-install-recommends apt-utils software-properties-common \
 && apt-add-repository -y ppa:deadsnakes/ppa -u

# htop curl tree vim iotop bmon
RUN apt-get -y install --no-install-recommends wget git ca-certificates bzip2 cmake vim \
  && apt-get -y install --no-install-recommends libsm6 libglib2.0-0 ffmpeg \
  && apt-get -y install --no-install-recommends libxext6 libxrender-dev virtualenv python3.6-dev g++\
  && apt-get -y install python3-pip

#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE \
# || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
#RUN add-apt-repository -y "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

#RUN apt-get -y install --no-install-recommends librealsense2-dkms

RUN python3 -m virtualenv --python=/usr/bin/python3.6 /opt/venv

# git demo
RUN git clone --branch reverse https://github.com/alourositi/EAM_docker.git \
 && cd EAM_docker/modules/detector/ \
 && wget --progress=bar:force:noscroll https://www.dropbox.com/s/n97xv8sqrseg2oi/e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth

WORKDIR EAM_docker

#install dependencies
#COPY requirements.txt .
RUN . /opt/venv/bin/activate
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# copy aplication
#COPY maskrcnn_benchmark/ maskrcnn_benchmark/
#COPY setup.py .

# install pycocotools
WORKDIR modules/detector
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python3 setup.py build_ext install
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN python3 setup.py build develop


RUN rm -rf /var/lib/apt/lists/*

#ENV IP_KAFKA=195.251.117.126
#ENV PORT_KAFKA=9091
WORKDIR /EAM_docker
#WORKDIR demo
#CMD ["python3","object_detections_EAM.py"]
