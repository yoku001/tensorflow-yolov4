![license](https://img.shields.io/github/license/hhk7734/tensorflow-yolov4)
![pypi](https://img.shields.io/pypi/v/yolov4)
![language](https://img.shields.io/github/languages/top/hhk7734/tensorflow-yolov4)

# tensorflow-yolov4

YOLOv4 Implemented in Tensorflow 2.
Convert YOLOv4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

## Performance

<p align="center"><img src="data/performance.png" width="640"\></p>

## Demo

```python
import yolov4

detector = yolov4.YoloV4()
detector.inference(
    names_path="/home/hhk7734/tensorflow-yolov4/data/classes/coco.names",
    media_path="/home/hhk7734/tensorflow-yolov4/data/road.mp4",
    weights_path="/home/hhk7734/Desktop/yolov4.weights",
    isImage=False,
)
```
