"""
test script for edge tpu model
"""

from yolov4.tflite import YOLOv4
import cv2

IMAGE_PATH = "kite.jpg"
MODEL_PATH = "quant_model_edgetpu.tflite"

yolo = YOLOv4(tiny=True, tpu=True)

yolo.classes = "dataset/coco.names"

yolo.load_tflite(MODEL_PATH)

yolo.inference(IMAGE_PATH)
