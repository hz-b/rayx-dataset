FROM --platform=linux/amd64 ubuntu:24.04

USER root

RUN apt update && apt -y install wget
RUN wget https://github.com/hz-b/ray-installers/releases/download/1.160/Ray-UI-r1.160.56-Ubuntu24.04-x86_64-installer
RUN chmod +x Ray-UI-r1.160.56-Ubuntu24.04-x86_64-installer
RUN apt install -y qtcreator qt6-base-dev qt6-tools-dev qmake6 qtcreator assistant-qt6 linguist-qt6 && rm -rf /var/lib/apt/lists/*
RUN ./Ray-UI-r1.160.56-Ubuntu24.04-x86_64-installer in --al -c --da
#RUN apt install -y libicu
# install locales and Xvfb; ensure Qt plugins and xcb libs already installed
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# generate and set UTF-8 locale
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# environment variables for runtime
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV QT_PLUGIN_PATH=/usr/lib/qt6/plugins
# don't force platform in the image; you can override at runtime, but a sensible default:
ENV QT_QPA_PLATFORM=xcb

# write the wrapper
RUN cat > /usr/local/bin/rayui <<'EOF'
#!/bin/bash
# wrapper to run Ray UI inside container
# ensures locale, Xvfb, and QT env vars; then execs the real binary

# ensure UTF-8 locale
if ! locale | grep -q "UTF-8"; then
  if command -v locale-gen >/dev/null 2>&1; then
    locale-gen en_US.UTF-8 >/dev/null 2>&1 || true
    update-locale LANG=en_US.UTF-8 >/dev/null 2>&1 || true
  fi
  export LANG=en_US.UTF-8
  export LC_ALL=en_US.UTF-8
fi

# Ensure QT plugin path (adjust if your plugins are elsewhere)
export QT_PLUGIN_PATH=${QT_PLUGIN_PATH:-/usr/lib/qt6/plugins}

# If no DISPLAY is set, start Xvfb on :99 and export DISPLAY
if [ -z "${DISPLAY:-}" ]; then
  # try to find xvfb-run first (preferred because it cleans up)
  if command -v xvfb-run >/dev/null 2>&1; then
    exec xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" /Applications/RAY-UI/rayui.sh "$@"
  else
    # start Xvfb in background and give it a moment to spin up
    if command -v Xvfb >/dev/null 2>&1; then
      Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
      sleep 0.3
      export DISPLAY=:99
    else
      echo "Warning: Xvfb not installed and no DISPLAY set; rayui may fail." >&2
    fi
  fi
fi

# Ensure platform plugin selection (use xcb by default when using Xvfb/X11)
: ${QT_QPA_PLATFORM:=xcb}
export QT_QPA_PLATFORM

# Exec the real script (replace with full path if different)
exec /Applications/RAY-UI/rayui.sh "$@"
EOF

# make wrapper executable
RUN chmod +x /usr/local/bin/rayui

RUN apt update && apt install -y python3 python3-venv python3-pip build-essential  && rm -rf /var/lib/apt/lists/*
RUN rayui --version


## RAY-X
ARG RAYX_VERSION="1.1.0"
ARG RAYX_DEB="RAYX-${RAYX_VERSION}-Linux.deb"
# ---------- Download selected RAYX release ----------
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    libhdf5-dev \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

RUN wget "https://github.com/hz-b/rayx/releases/download/v${RAYX_VERSION}/${RAYX_DEB}"



# ---------- Install the package ----------
RUN apt-get update && apt-get install -y ./${RAYX_DEB} \
    && rm ${RAYX_DEB} \
    && rm -rf /var/lib/apt/lists/*
    
RUN rayx --version

WORKDIR /App
COPY . /App

# create venv and install the package into it
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install /App

# make the venv python available via PATH for interactive runs
ENV PATH="/opt/venv/bin:${PATH}"

RUN cat > /usr/local/bin/rayx-database <<'EOF'
#!/bin/bash
exec python /App/generate.py "$@"
EOF
RUN chmod +x /usr/local/bin/rayx-database

CMD ["/bin/bash"]
