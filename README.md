![license](https://img.shields.io/github/license/hhk7734/tensorflow-yolov4)
![pypi](https://img.shields.io/pypi/v/yolov4)
![language](https://img.shields.io/github/languages/top/hhk7734/tensorflow-yolov4)

# tensorflow-yolov4

```shell
python3 -m pip install yolov4
```

YOLOv4 Implemented in Tensorflow 2.

## Download Weights

- [yolov4-tiny.conv.29](https://drive.google.com/file/d/1WtOuGfUgNyNfALo5_VhQ1kb5QenRE0Gt/view?usp=sharing)
- [yolov4-tiny.weights](https://drive.google.com/file/d/1GJwGiR7rizY_19c_czuLN8p31BwkhWY5/view?usp=sharing)
- [yolov4.conv.137](https://drive.google.com/file/d/1li1pUtqpXj_-ZXxA8wJq-nzW8h2HWsrP/view?usp=sharing)
- [yolov4.weights](https://drive.google.com/file/d/15P4cYyZ2Sd876HKAEWSmeRdFl_j-0upi/view?usp=sharing)
- [coco.names](https://github.com/hhk7734/tensorflow-yolov4/tree/master/test/dataset)

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
- [x] Run yolov4-tiny-relu on Coral board(TPU).
- [ ] Train tiny-relu with coco 2017 dataset
- [ ] Update Docs
- [ ] Optimize model and operations

## Performance

![performance](https://github.com/hhk7734/tensorflow-yolov4/blob/master/test/performance.png)

![performance-tiny](https://github.com/hhk7734/tensorflow-yolov4/blob/master/test/performance-tiny.png)

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

```python
from yolov4.tf import YOLOv4

yolo = YOLOv4(tiny=True)

yolo.classes = "coco.names"

yolo.make_model()
yolo.load_weights("yolov4-tiny.weights", weights_type="yolo")

yolo.inference(media_path="kite.jpg")

yolo.inference(media_path="road.mp4", is_image=False)
```

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

[Custom training on Colab jupyter notebook](https://github.com/hhk7734/tensorflow-yolov4/blob/master/test/custom_training_on_colab.ipynb)
