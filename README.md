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

1. [ ] Train and predict using TensorFlow 2 only
1. [ ] Run yolov4 on Coral board(TPU).

## Performance

<p align="center"><img src="data/performance.png" width="640"\></p>

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

yolo.classes = "/home/hhk7734/tensorflow-yolov4/data/classes/coco.names"

yolo.make_model()
yolo.load_weights("/home/hhk7734/Desktop/yolov4.weights", weights_type="yolo")

yolo.inference(
    media_path="/home/hhk7734/tensorflow-yolov4/data/kite.jpg",
)

yolo.inference(
    media_path="/home/hhk7734/tensorflow-yolov4/data/road.mp4",
    is_image=False
)
```

### tensorflow lite

`tf.keras.layers.UpSampling2D()` seems to be in TensorFlow >= 2.3.0

## Training

**Not successful yet.**

```python

from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "/home/hhk7734/tensorflow-yolov4/data/classes/coco.names"

yolo.input_size = 608
yolo.batch_size = 32
yolo.subdivision = 16
yolo.make_model()
yolo.load_weights("/home/hhk7734/Desktop/yolov4.conv.137", weights_type="yolo")

data_set = yolo.load_dataset(
    "/home/hhk7734/tensorflow-yolov4/data/dataset/val2017.txt"
)

yolo.compile(iou_type="ciou", learning_rate=1e-4)
yolo.fit(data_set, epochs=2000)
yolo.model.save_weights("checkpoints")
```

```python
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "/home/hhk7734/tensorflow-yolov4/data/classes/coco.names"

yolo.input_size = 416
yolo.make_model()
yolo.load_weights("/home/hhk7734/Desktop/yolov4.conv.137", weights_type="yolo")

data_set = yolo.load_dataset(
    "/home/hhk7734/darknet/data/train.txt",
    dataset_type="yolo",
)

yolo.compile(iou_type="ciou", learning_rate=1e-4)
yolo.fit(data_set, epochs=2000)
yolo.model.save_weights("checkpoints")
```
