# Time-measuring-instrument
## DEMO

### youtube Link

# Features 特徴
In the time measuring instrument project, the position of the hand is estimated and the time from taking a part to taking the next part is measured and displayed. By visualizing time, you can discover the bottleneck areas and find opportunities for improvement.
<img src="https://i.gyazo.com/4fbadf16b230abd6e81138c46a153981.jpg" width=50%>
<img src="https://i.gyazo.com/69ed50192bc7492e679caf857067161f.jpg" width=50%>

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
$ git clone https://github.com/Toshiki0324/Time-measuring-instrument.git
$ cd Time-measuring-instrument.git
$ sudo bash install-tf-v45.sh
```

# How to use
```
$ cd src
$ python3 time_measuring.py -l='label.txt' -m='frozen_inference_graph.pb' --host <your IP address> --t test
```

## How to know the IP address
 Check the IP address of JETSON NANO before starting it.
eth0 is a wired LAN adapter, wlan0 is a wireless LAN adapter
 
```
ifconfig
```
<img src="https://i.gyazo.com/b4d2200608dfc419d47d61213d49ec68.png" width=50%>

All you have to do to configure the mqtt in node is enter the IP address you looked up.


systemctl status mosquitto

## How to set MQTT of Node-red
After starting Node-red, start the browser and access the following address
```
http://localhost:1880
```
Use mqtt in node, with drag-and-drop add.

<img src="https://i.gyazo.com/c80d381ef86ee0669d4ef6b9462f7634.png" width=10%>

For the settings in the mqtt in node, enter the IP address and Topic you checked.

<img src="https://i.gyazo.com/706e7ec74752cb73ff78f73cfb44f39a.png" width=30%>

<img src="https://i.gyazo.com/40df2e4a05a4e673818247484eb8931d.png" width=30%>



 
