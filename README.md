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
```

### tensorflow lite

```python
import yolov4.tflite as yolo

detector = yolo.YOLOv4(
    names_path="/home/hhk7734/tensorflow-yolov4/data/classes/coco.names",
    tflite_path="/home/hhk7734/Desktop/yolov4.tflite",
)

detector.inference(
    media_path="/home/hhk7734/tensorflow-yolov4/data/road.mp4",
    is_image=False,
    cv_waitKey_delay=1,
)
```

## Training

```python
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "/home/hhk7734/tensorflow-yolov4/data/classes/coco.names"

yolo.input_size = 416
yolo.make_model()
yolo.load_weights("/home/hhk7734/Desktop/yolov4.conv.137", weights_type="yolo")

datasets = yolo.load_datasets("/home/hhk7734/tensorflow-yolov4/data/dataset/val2017.txt")

yolo.compile(learning_rate=4e-7)
yolo.fit(datasets, epochs=4000, batch_size=4)

yolo.model.save_weights("checkpoints")
```

```python
from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.classes = "/home/hhk7734/tensorflow-yolov4/data/classes/coco.names"

yolo.input_size = 416
yolo.make_model()
yolo.load_weights("/home/hhk7734/Desktop/yolov4.conv.137", weights_type="yolo")

datasets = yolo.load_datasets(
    "/home/hhk7734/darknet/data/train.txt",
    datasets_type="yolo",
)

yolo.compile(learning_rate=4e-7)
yolo.fit(datasets, epochs=4000, batch_size=4)

yolo.model.save_weights("checkpoints")
```
