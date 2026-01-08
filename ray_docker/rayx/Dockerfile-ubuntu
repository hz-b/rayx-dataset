FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Setup the required capabilities for the container runtime
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive

# Install vulkan
RUN apt update
RUN apt install -y wget
RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
RUN apt update
RUN apt install -y vulkan-sdk

# Install RAY-X dependencies
RUN apt install -y cmake git gcc gdb libblas-dev liblapack-dev libhdf5-dev mc ninja-build build-essential
RUN apt clean all

# Install python3.10
RUN apt install -y python3.10 python3-pip
RUN pip install virtualenv
RUN virtualenv --python="/usr/bin/python3.10" "/venv/"
ENV PATH="/venv/bin:$PATH"

# Build RAY-X
RUN git clone --recurse-submodules https://github.com/hz-b/RAY-X.git -b development
RUN pip install -U numpy
RUN cd RAY-X && sh compile.sh
RUN cd RAY-X/build/bin/debug

# Create RAY-X workdir
RUN mkdir -p /RAY-X-workdir

ENTRYPOINT ["tail", "-f", "/dev/null"]