FROM openvino/ubuntu20_runtime:2022.1.0

USER root

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /requirements.txt
COPY ./models.tgz /models.tgz

WORKDIR /

RUN pip3 install -r requirements.txt
RUN tar xvf models.tgz

COPY . /

CMD ["python3", "multi_channel_yolox.py"]


