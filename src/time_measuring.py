import argparse
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf
import paho.mqtt.client as mqtt

from distutils.version import StrictVersion

try:
  if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
except:
  pass

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='Time measuring instrument')
parser.add_argument('-l', '--labels', default='./labels.txt', help="default: './exported_graphs/labels.txt'")
parser.add_argument('-m', '--model', default='./exported_graphs/frozen_inference_graph.pb', help="default: './exported_graphs/frozen_inference_graph.pb'")
parser.add_argument('-d', '--device', default='normal_cam', help="normal_cam, jetson_nano_web_cam, or video_file. default: 'normal_cam'") # normal_cam / jetson_nano_raspi_cam / jetson_nano_web_cam
parser.add_argument('-i', '--input_video_file', default='', help="Input video file")
# ///////////////////////////////
parser.add_argument('--host', type=str, default='localhost', metavar='MQTT_HOST', help='MQTT remote broker IP address')
parser.add_argument('-p', '--port', type=int, default=1883, metavar='MQTT_PORT', help='MQTT port number')
parser.add_argument('-t', '--topic',type=str, metavar='MQTT_TOPIC', help='MQTT topic to be published on')

# //////////////////////////////
args = parser.parse_args()

detection_graph = tf.Graph()

mode = 'bbox'

colors = [

  (192, 255, 0),
  (255, 255, 0),
  (255, 192, 0),
  (255, 128, 0),
  (255, 64, 0),
  (255, 0, 0),
  (255, 0, 64),
  (255, 0, 128),
  (255, 0, 192),
  (255, 0, 255),
  (192, 0, 255),
  (128, 0, 255),
  (64, 0, 255),
  (0, 0, 255),
]

# ブローカーに接続できたときの処理
def on_connect(client, userdata, flag, rc):
    print("Connected with result code " + str(rc))   # 接続結果表示

# ブローカーが切断したときの処理
def on_disconnect(client, userdata, flag, rc):
    if rc != 0:
        print("Unexpected disconnection.")

# publishが完了したときの処理
def on_publish(client, userdata, mid):
    print("publish: {0}".format(mid))

def init_mqtt(host, port=1883):
    client = mqtt.Client() #clientオブジェクト作成
    client.on_connect = on_connect # 接続時に実行するコールバック関数設定
    client.on_disconnect = on_disconnect # 切断時のコールバックを登録
    #client.on_publish = on_publish # メッセージ送信時のコールバック
    client.connect(host, port, 60) # MQTT broker接続
    client.loop_start()  # 処理開始
    return client

def publish_bboxes(client, topic, frame_num, bboxes):
    # for box in zip(bboxes):
    if bboxes is not None:
        info = '{0},{1}'.format(
            frame_num,
            boxe,
        )
        # print(boxe)
        # print(info)
        if client is not None:
            client.publish(topic, info) # topic名=topicに infoというメッセージを送


client = None
if args.topic is not None:
    client = init_mqtt(args.host, args.port)

def load_graph():
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# def mosaic(src, ratio=0.1):
#     small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
#     return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

# def mosaic_area(src, x_min, y_min, x_max, y_max, ratio=0.1):
#     dst = src.copy()
#     dst[y_min:y_max, x_min:x_max] = mosaic(dst[y_min:y_max, x_min:x_max], ratio)
#     return dst

# Load a (frozen) Tensorflow model into memory.
print('Loading graph...')
detection_graph = load_graph()
print('Graph is loaded')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
  tf_sess = tf.Session(config = tf_config)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

def run_inference_for_single_image(image, graph):
  # Run inference
  output_dict = tf_sess.run(tensor_dict,
                          feed_dict={image_tensor: image})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

if args.input_video_file != "":
  # WORKAROUND
  print("[Info] --input_video_file has an argument. so --device was replaced to 'video_file'.")
  args.device = "video_file"

# Switch camera according to device
if args.device == 'normal_cam':
  cam = cv2.VideoCapture(0)
# elif args.device == 'jetson_nano_raspi_cam':
#   GST_STR = 'nvarguscamerasrc \
#     ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
#     ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
#     ! videoconvert \
#     ! appsink drop=true sync=false'
#   cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
elif args.device == 'jetson_nano_web_cam':
  cam = cv2.VideoCapture(1)
# elif args.device == 'raspi_cam':
#   from picamera.array import PiRGBArray
#   from picamera import PiCamera
#   cam = PiCamera()
#   cam.resolution = (1280, 900)
#   stream = PiRGBArray(cam)
# elif args.device == 'video_file':
#   cam = cv2.VideoCapture(args.input_video_file)
else:
  print('[Error] --device: wrong device')
  parser.print_help()
  sys.exit()

count_max = 0

if __name__ == '__main__':
  count = 0

  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())
  
  frame_count = 0

  while True:
    # if args.device == 'raspi_cam':
    #   cam.capture(stream, 'bgr', use_video_port=True)
    #   img = stream.array
    # else:
    ret, img = cam.read()
    if not ret:
      print('error')
      break

    key = cv2.waitKey(1)
    # if key == 77 or key == 109: # when m or M key is pressed, go to mosaic mode
    #   mode = 'mosaic'
    # elif key == 66 or key == 98: # when b or B key is pressed, go to bbox mode
    #   mode = 'bbox'
    if key == 27: # when ESC key is pressed break
        break

    count += 1
    if count > count_max:
      img_bgr = cv2.resize(img, (300, 300))

      # convert bgr to rgb
      image_np = img_bgr[:,:,::-1]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      start = time.time()
      output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      elapsed_time = time.time() - start

      #Colors
      color_Red = (0, 0, 225)
      color_Lime = (0,255,0)
      color_Blue = (255,0,0)

      #Part1  
      xmin1, ymin1 = 10, 10
      xmax1, ymax1 = 350, 350
      #Part2
      xmin2, ymin2 = 360, 10
      xmax2, ymax2 = 700, 350
      #Part3  
      xmin3, ymin3 = 710, 10
      xmax3, ymax3 = 1050, 350
      #Part4
      xmin4, ymin4 = 1060, 360
      xmax4, ymax4 = 1270, 800


      cv2.rectangle(img, (xmin1, ymin1), (xmax1, ymax1), color_Blue, 3)
      cv2.rectangle(img, (xmin2, ymin2), (xmax2, ymax2), color_Blue, 3)
      cv2.rectangle(img, (xmin3, ymin3), (xmax3, ymax3), color_Blue, 3)
      cv2.rectangle(img, (xmin4, ymin4), (xmax4, ymax4), color_Blue, 3)


      for i in range(output_dict['num_detections']):
        class_id = output_dict['detection_classes'][i]
        if class_id < len(labels):
          label = labels[class_id]
        else:
          label = 'unknown'

        detection_score = output_dict['detection_scores'][i]
        # print('box[1], box[0]), (box[3], box[2]), color, 3')
        if detection_score > 0.5:
            # Define bounding box
            h, w, c = img.shape
            box = output_dict['detection_boxes'][i] * np.array( \
              [h, w,  h, w])
            box = box.astype(np.int)

            speed_info = '%s: %.3f' % ('fps', 1.0/elapsed_time)
            cv2.putText(img, speed_info, (10,50), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            if mode == 'bbox':
              class_id = class_id % len(colors)
              color = colors[class_id]

              # Draw circle
              cx = (box[1]+ box[3])//2
              cy = (box[0]+ box[2])//2
              cv2.circle(img,(cx,cy),6,(0,0,225),cv2.FILLED)
              # test
              if cx >= xmin1 and cy >= ymin1 and cx <= xmax1 and cy <= ymax1:
                boxe = "flag1"
                print(boxe)
                frame_count += 1
                cv2.rectangle(img, (xmin1, ymin1), (xmax1, ymax1), color_Red, 4)
                publish_bboxes(client, args.topic, frame_count, boxe)
              elif cx >= xmin2 and cy >= ymin2 and cx <= xmax2 and cy <= ymax2:
                boxe = "flag2"
                print(boxe)
                frame_count += 1
                cv2.rectangle(img, (xmin2, ymin2), (xmax2, ymax2), color_Red, 4)
                publish_bboxes(client, args.topic, frame_count, boxe)
              elif cx >= xmin3 and cy >= ymin3 and cx <= xmax3 and cy <= ymax3:
                boxe = "flag3"
                print(boxe)
                frame_count += 1
                cv2.rectangle(img, (xmin3, ymin3), (xmax3, ymax3), color_Red, 4)
                publish_bboxes(client, args.topic, frame_count, boxe)
              elif cx >= xmin4 and cy >= ymin4 and cx <= xmax4 and cy <= ymax4:
                boxe = "flag4"
                print(boxe)
                frame_count += 1
                cv2.rectangle(img, (xmin4, ymin4), (xmax4, ymax4), color_Red, 4)
                publish_bboxes(client, args.topic, frame_count, boxe)
              else :
                print("no")

              # Draw bounding box
              cv2.rectangle(img, \
                (box[1], box[0]), (box[3], box[2]), color, 3)
                
              # Put label near bounding box
              information = '%s: %.1f%%' % (label, output_dict['detection_scores'][i] * 100.0)
              cv2.putText(img, information, (box[1] + 15, box[2] - 15), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

              # print(str(box[1]) + ',' + str(box[0]), label)
            # elif mode == 'mosaic':
            #   img = mosaic_area(img, box[1], box[0], box[3], box[2], ratio=0.05)

            # frame_count += 1

      cv2.imshow('detection result', img)
      count = 0
    # if args.device == 'raspi_cam':
    #   stream.seek(0)
    #   stream.truncate()

  tf_sess.close()
  cam.release()
  cv2.destroyAllWindows()

  if client is not None:
      client.disconnect()
  
