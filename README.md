![license](https://img.shields.io/github/license/hhk7734/tensorflow-yolov4)
![pypi](https://img.shields.io/pypi/v/yolov4)
![language](https://img.shields.io/github/languages/top/hhk7734/tensorflow-yolov4)

# tensorflow-yolov4

```shell
python3 -m pip install yolov4
```

YOLOv4 Implemented in Tensorflow 2.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

## Dependencies

```shell
python3 -m pip install -U pip setuptools wheel
```

```shell
python3 -m pip install numpy
```

Install OpenCV (cv2)

### Tensorflow 2

```shell
python3 -m pip install tensorflow
```

### TFlite

Ref: [https://www.tensorflow.org/lite/guide/python](https://www.tensorflow.org/lite/guide/python)

## Objective

- [x] Train and predict using TensorFlow 2 only
- [ ] Run yolov4 on Coral board(TPU).

## Performance

<p align="center"><img src="test/performance.png" width="640"\></p>

## Help

```python
>>> from yolov4.tf import YOLOv4
>>> help(YOLOv4)
```

## Inference

### tensorflow

```python
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "coco.names"

yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")

yolo.inference(media_path="kite.jpg")

yolo.inference(media_path="road.mp4", is_image=False)
```

[Object detection test jupyter notebook](./test/object_detection_in_image.ipynb)

### tensorflow lite

```python
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "coco.names"

yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")

yolo.save_as_tflite("yolov4.tflite")
```

```python
from yolov4.tflite import YOLOv4

yolo = YOLOv4()

yolo.classes = "coco.names"

yolo.load_tflite("yolov4.tflite")

yolo.inference("kite.jpg")
```

## Training

```python
import tensorflow.keras import optimizers
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "coco.names"
yolo.input_size = 608
yolo.batch_size = 32
yolo.subdivision = 16

yolo.make_model()
yolo.load_weights("yolov4.conv.137", weights_type="yolo")

data_set = yolo.load_dataset("val2017.txt")
# data_set = yolo.load_dataset(
#     "/home/hhk7734/darknet/data/train.txt",
#     dataset_type="yolo",
# )

optimizer = optimizers.Adam(learning_rate=1e-4)
yolo.compile(optimizer=optimizer, loss_iou_type="ciou")

yolo.fit(data_set, epochs=1500)
yolo.model.save_weights("checkpoints")
```

[Custom training on Colab jupyter notebook](./test/custom_training_on_colab.ipynb)
