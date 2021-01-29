from py_src.yolov4.tf import YOLOv4
from py_src.yolov4.model.common import YOLOConv2D
import cv2
import numpy as np

yolo = YOLOv4()
yolo.classes = "/home/hhk7734/NN/coco.names"
yolo.load_cfg("config/yolov4.cfg")
yolo.load_cfg("config/yolov4-tiny.cfg")
yolo.load_cfg("config/yolov4-tiny-relu.cfg")
yolo.make_model()
if yolo._model.name == "YOLOv4":
    yolo.load_weights("/home/hhk7734/NN/yolov4.weights", weights_type="yolo")
else:
    yolo.load_weights(
        "/home/hhk7734/NN/yolov4-tiny.weights", weights_type="yolo"
    )

# yolo.summary()

frame = cv2.imread("test/kite.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(frame.shape)
print(yolo._config)