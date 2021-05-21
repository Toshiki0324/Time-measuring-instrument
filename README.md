# Time-measuring-instrument
## DEMO

### youtube Link

# Features 特徴
In the time measuring instrument project, the position of the hand is estimated and the time from taking a part to taking the next part is measured and displayed. By visualizing time, you can discover the bottleneck areas and find opportunities for improvement.



The training model can be detected by replacing it with a model created by yourself. It detects that the hand has entered the area prepared in advance and sends the detection by MQTT to node-red to visualize it.
### Prerequisites
It is assumed that node-red is pre-installed. If you do not have it installed, you can install it from the Software Center.
# Story
We will work to reduce assembly time and production costs, but dedicated software is expensive and will not reach the people who need it. The purpose of this project is to contribute to many industries including manufacturing with the power of AI by providing an open source solution using cheap and high-speed AI edge devices with NVIDIA Jetson Nano.

# Requirement

 - NVIDIA Jetson device (veirfied on Jetson Nano Developer Kit)
 - Logcool 270 (USB Camera) <br> https://www.amazon.com/Logitech-C270-720pixels-Black-webcam/dp/B01BGBJ8Y0/ref=sr_1_3?dchild=1&keywords=C270&qid=1605453031&sr=8-3

# Installation
Clone github and install the required libraries.
```
$ https://github.com/Toshiki0324/Time-measuring-instrument.git
$ cd Time-measuring-instrument.git
$ sudo bash install-tf-v45.sh
```

# How to use
```
python3 time_measuring.py -l='label.txt' -m='frozen_inference_graph.pb' --host <your IPaddres> --t test
```
