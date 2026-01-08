# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Prevent interactive dialogs
ENV DEBIAN_FRONTEND=noninteractive

# ---------- Build arguments for versioning ----------
# RAYX version number and deb package filename
ARG RAYX_VERSION="1.1.0"
ARG RAYX_DEB="RAYX-${RAYX_VERSION}-Linux.deb"

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    libgomp1 \
    libhdf5-dev \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt

# ---------- Download selected RAYX release ----------
RUN wget "https://github.com/hz-b/rayx/releases/download/v${RAYX_VERSION}/${RAYX_DEB}"

# ---------- Install the package ----------
RUN apt-get update && apt-get install -y ./${RAYX_DEB} \
    && rm ${RAYX_DEB} \
    && rm -rf /var/lib/apt/lists/*

# ---------- Volume for data ----------
RUN mkdir -p /opt/rayx-data
VOLUME ["/opt/rayx-data"]

# Optional environment variable
ENV RAYX_DATA_DIR=/opt/ray-workdir

# ---------- Default command: automatically run RAYX ----------
# Adjust the binary path if needed.
WORKDIR /opt/ray-workdir
