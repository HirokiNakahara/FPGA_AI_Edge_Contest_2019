FROM pytorch/pytorch:nightly-devel-cuda9.2-cudnn7

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# python and cuda
RUN pip install torchvision tensorboardX pandas pillow matplotlib scikit-learn scikit-image opencv-python pydot GraphViz numpy

# add user
ARG UID
RUN useradd docker -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker
