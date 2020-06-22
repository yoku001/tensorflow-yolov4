![license](https://img.shields.io/github/license/hhk7734/tensorflow-yolov4)
![pypi](https://img.shields.io/pypi/v/yolov4)
![language](https://img.shields.io/github/languages/top/hhk7734/tensorflow-yolov4)

# tensorflow-yolov4

```shell
python3 -m pip install yolov4
```

YOLOv4 Implemented in Tensorflow 2.
Convert YOLOv4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

## Performance

<p align="center"><img src="data/performance.png" width="640"\></p>

## tensorflow

```python
import yolov4.tf as yolo

detector = yolo.YoloV4(
    names_path="/home/hhk7734/tensorflow-yolov4/data/classes/coco.names",
    weights_path="/home/hhk7734/Desktop/yolov4.weights",
)

detector.inference(
    media_path="/home/hhk7734/tensorflow-yolov4/data/kite.jpg",
    cv_waitKey_delay=1000,
)
```

## tensorflow lite

```python
import yolov4.tflite as yolo

detector = yolo.YoloV4(
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
import yolov4.tf as yolo

detector = yolo.YoloV4(
    names_path="/home/hhk7734/tensorflow-yolov4/data/classes/coco.names"
)

detector.train(
    train_annote_path="/home/hhk7734/tensorflow-yolov4/data/dataset/val2017.txt",
    test_annote_path="/home/hhk7734/tensorflow-yolov4/data/dataset/val2017.txt",
    pre_trained_weights="/home/hhk7734/Desktop/yolov4.weights",
)
```

```python
import yolov4.tf as yolo

detector = yolo.YoloV4(
    names_path="/home/hhk7734/darknet/data/class.names"
)

detector.train(
    train_annote_path="/home/hhk7734/darknet/data/train.txt",
    test_annote_path="/home/hhk7734/darknet/data/train.txt",
    dataset_type="yolo",
)
```
