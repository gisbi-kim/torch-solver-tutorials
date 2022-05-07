FROM ubuntu:22.04 
LABEL maintainer="giseop.kim@naverlabs.com"

RUN echo "Build a Ubuntu 22.04 env for DL environment"

RUN apt-get update 
RUN apt-get install -y g++
RUN apt-get install -y cmake
RUN apt-get install -y libgoogle-glog-dev libgflags-dev
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libsuitesparse-dev
