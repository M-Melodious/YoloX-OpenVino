#FROM openvino/ubuntu20_runtime:2021.4.2
FROM public.ecr.aws/lts/ubuntu:latest

USER root

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 wget  -y

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt

COPY . /

RUN wget https://github.com/M-Melodious/YoloX-Models/raw/main/yolox-x/INT8/yolox-x_INT8.bin
RUN wget https://raw.githubusercontent.com/M-Melodious/YoloX-Models/main/yolox-x/INT8/yolox-x_INT8.xml
RUN mv yolox-x_INT8.bin ./models/INT8/yolox-x_INT8.bin
RUN mv yolox-x_INT8.xml ./models/INT8/yolox-x_INT8.xml

CMD ["python3", "multi_channel_yolox.py"]


