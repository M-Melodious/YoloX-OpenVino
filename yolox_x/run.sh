#!/bin/bash
#source /opt/intel/openvino/bin/setupvars.sh && python3 multi_channel_yolox.py
wget https://github.com/M-Melodious/YoloX-Models/raw/main/yolox-x/INT8/yolox-x_INT8.bin
wget https://raw.githubusercontent.com/M-Melodious/YoloX-Models/main/yolox-x/INT8/yolox-x_INT8.xml
mv yolox-x_INT8.bin ./models/INT8/yolox-x_INT8.bin
mv yolox-x_INT8.xml ./models/INT8/yolox-x_INT8.xml
echo "Running inference with OpenVINO"
python3 multi_channel_yolox.py