"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Any, Callable, Dict, List

import tensorflow as tf
from tensorflow import keras

from .layers import YOLOConv2D
from ..common.config import YOLOConfig


class YOLOv4Model(keras.Model):
    def __init__(self, config: YOLOConfig):
        self._model_config: YOLOConfig = config
        super().__init__(
            name="YOLOv4Tiny" if self._model_config.tiny else "YOLOv4"
        )
        _l2 = None

        self._model_layers = []
        layer_name: str
        layer_option: Dict[str, Any]
        for layer_name, layer_option in self._model_config.items():
            if layer_option["type"] == "convolutional":
                self._model_layers.append(
                    YOLOConv2D(
                        activation=layer_option["activation"],
                        filters=layer_option["filters"],
                        kernel_regularizer=_l2,
                        kernel_size=layer_option["size"],
                        name=layer_name,
                        strides=layer_option["stride"],
                    )
                )

            elif layer_option["type"] == "route":
                if "groups" in layer_option:
                    self._model_layers.append(
                        _split_and_get(
                            layer_option["groups"], layer_option["group_id"]
                        )
                    )
                else:
                    if len(layer_option["layers"]) == 1:
                        self._model_layers.append(lambda x: x)
                    else:
                        self._model_layers.append(
                            keras.layers.Concatenate(axis=-1)
                        )

            elif layer_option["type"] == "shortcut":
                self._model_layers.append(keras.layers.Add())

            elif layer_option["type"] == "maxpool":
                self._model_layers.append(
                    keras.layers.MaxPooling2D(
                        pool_size=(layer_option["size"], layer_option["size"]),
                        strides=(
                            layer_option["stride"],
                            layer_option["stride"],
                        ),
                        padding="same",
                    )
                )

            elif layer_option["type"] == "upsample":
                self._model_layers.append(
                    keras.layers.UpSampling2D(interpolation="bilinear")
                )

            elif layer_option["type"] == "yolo":
                self._model_layers.append(lambda x: x)

            elif layer_option["type"] == "net":
                _l2 = keras.regularizers.L2(
                    l2=self._model_config["net"]["decay"]
                )

    def call(self, x):
        output = []
        return_val = []
        layer_option: Dict[str, Any]
        for layer_option in self._model_config.values():
            layer_number = layer_option["count"]
            if layer_number == -1:
                continue
            layer_function = self._model_layers[layer_number]

            if layer_option["type"] == "route":
                if "groups" in layer_option:
                    index = layer_option["layers"][0]
                    output.append(layer_function(output[index]))
                else:
                    if len(layer_option["layers"]) == 1:
                        index = layer_option["layers"][0]
                        output.append(layer_function(output[index]))
                    else:
                        output.append(
                            layer_function(
                                [output[i] for i in layer_option["layers"]],
                            )
                        )

            elif layer_option["type"] == "shortcut":
                indexes: List[int] = layer_option["from"]
                indexes.append(layer_number - 1)
                output.append(layer_function([output[i] for i in indexes]))

            else:
                if layer_number == 0:
                    output.append(layer_function(x))
                else:
                    output.append(layer_function(output[layer_number - 1]))

                if layer_option["type"] == "yolo":
                    return_val.append(output[layer_number])

        return return_val


def _split_and_get(groups: int, group_id: int) -> Callable:
    return lambda x: tf.split(
        x,
        groups,
        axis=-1,
    )[group_id]
