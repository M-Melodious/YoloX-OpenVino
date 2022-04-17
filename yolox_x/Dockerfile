FROM openvino/ubuntu20_runtime:2021.4.2

USER root

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt

COPY . /

#CMD ["python3", "multi_channel_yolox.py"]
CMD ["run.sh"]


