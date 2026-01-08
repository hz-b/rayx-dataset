FROM --platform=linux/amd64 stateoftheartio/qt6:6.4-gcc-aqt
USER root

RUN apt update && apt -y install wget
RUN wget https://github.com/hz-b/ray-installers/releases/download/1.160/Ray-UI-r1.160.56-Ubuntu24.04-x86_64-installer
RUN chmod +x Ray-UI-r1.160.56-Ubuntu24.04-x86_64-installer
RUN ./Ray-UI-r1.160.56-Ubuntu24.04-x86_64-installer in --al -c --da

ENV QT_QPA_PLATFORM=offscreen
RUN apt update && apt install -y python3
ADD script_rayui_bg.py /opt/script_rayui_bg.py
