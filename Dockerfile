FROM tensorflow/tensorflow:2.8.0-gpu

COPY requirements.txt tmp/requirements.txt

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && apt update && apt install -y ffmpeg \
    libsm6 libxext6

RUN pip install -U pip && pip install -r tmp/requirements.txt && rm tmp/requirements.txt
