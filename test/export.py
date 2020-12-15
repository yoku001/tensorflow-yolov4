""" 
export model as keras, onnx and edge tpu model

onnx export requires tensorflow-onnx package
"""

import tensorflow as tf
import subprocess
from yolov4.tf import YOLOv4

MODEL_PATH = "tiny_yolov4_relu/"
WEIGHT_PATH = "yolov4-tiny-relu.weights"
DATASET_PATH = "path/to/dataset"

# create model
yolov4 = YOLOv4(tiny=True, tpu=True)
yolov4.classes = "dataset/coco.names"
yolov4.make_model(activation1="relu")
yolov4.load_weights(WEIGHT_PATH, weights_type="yolo")

# save as keras model
yolov4.model.save(MODEL_PATH)

# save as onnx model
try:
    subprocess.run(
        [
            "python",
            "-m",
            "tf2onnx.convert",
            "--opset",
            "12",
            "--saved-model",
            MODEL_PATH,
            "--output",
            MODEL_PATH + "model.onnx",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
except:
    pass

# callibrate with coco dataset
dataset = yolov4.load_dataset(
    dataset_path="dataset/train2017.txt",
    training=False,
    image_path_prefix=DATASET_PATH,
)

# save quantization aware model
yolov4.save_as_tflite(
    MODEL_PATH + "quant_model.tflite",
    quantization="full_int8",
    data_set=dataset,
    num_calibration_steps=250,
)
