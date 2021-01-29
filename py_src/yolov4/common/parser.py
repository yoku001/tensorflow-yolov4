"""
MIT License

Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>

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
from typing import Any, Dict


def parse_cfg(cfg_path: str) -> Dict[str, Any]:
    layer_meta = {
        "net": {
            "batch": "int",
            "subdivisions": "int",
            "width": "int",
            "height": "int",
            "channels": "int",
            "angle": "bool",
            "saturation": "float",
            "exposure": "float",
            "hue": "float",
            "mosaic": "bool",
            "momentum": "float",
            "decay": "float",
            "learning_rate": "float",
            "burn_in": "int",
            "max_batches": "int",
            "policy": "str",
            "steps": "int_list",
            "scales": "float_list",
        },
        "convolutional": {
            "batch_normalize": "int",
            "filters": "int",
            "size": "int",
            "stride": "int",
            "pad": "int",
            "activation": "str",
        },
        "maxpool": {
            "size": "int",
            "stride": "int",
        },
        "route": {
            "layers": "index_list",
            "groups": "int",
            "group_id": "int",
        },
        "shortcut": {"from": "index_list", "activation": "str"},
        "upsample": {
            "stride": "int",
        },
        "yolo": {
            "mask": "int_list",
            "anchors": "other",
            "num": "int",
            "classes": "int",
            "ignore_thresh": "float",
            "truth_thresh": "float",
            "jitter": "float",
            "random": "bool",
            "resize": "float",
            "scale_x_y": "float",
            "iou_thresh": "float",
            "iou_loss": "str",
            "iou_normalizer": "float",
            "cls_normalizer": "float",
            "max_delta": "int",
            "nms_kind": "str",
            "beta_nms": "float",
        },
    }

    config: Dict[str, Any] = {}
    count: Dict[str, int] = {
        "convolutional": 0,
        "maxpool": 0,
        "net": 0,
        "route": 0,
        "shortcut": 0,
        "upsample": 0,
        "yolo": 0,
    }
    layer_count: int = -1
    layer_type: str = "net"
    layer_name: str = ""

    with open(cfg_path, "r") as fd:
        for line in fd:
            line = line.strip().split("#")[0]
            if line == "":
                continue
            if line[0] == "[":
                # layer name
                count[layer_type] += 1
                layer_type = line[1:-1]
                if layer_type == "net":
                    layer_name = layer_type
                else:
                    layer_count += 1
                    layer_name = layer_type + str(count[layer_type])
                config[layer_name] = {"count": layer_count, "type": layer_type}
            else:
                # layer option
                option, value = line.split("=")
                option = option.strip()
                value = value.strip()

                try:
                    if layer_meta[layer_type][option] == "int":
                        value = int(value)
                    elif layer_meta[layer_type][option] == "float":
                        value = float(value)
                    elif layer_meta[layer_type][option] == "str":
                        pass
                    elif layer_meta[layer_type][option] == "bool":
                        value = bool(int(value))
                    elif layer_meta[layer_type][option] == "index_list":
                        value = [
                            int(i)
                            if int(i) >= 0
                            else config[layer_name]["count"] + int(i)
                            for i in value.split(",")
                        ]
                    elif layer_meta[layer_type][option] == "int_list":
                        value = [int(i.strip()) for i in value.split(",")]
                    elif layer_meta[layer_type][option] == "float_list":
                        value = [float(i.strip()) for i in value.split(",")]
                    elif layer_meta[layer_type][option] == "other":
                        if layer_type == "yolo":
                            if option == "anchors":
                                value = [
                                    int(i.strip()) for i in value.split(",")
                                ]
                                _value = []
                                for i in range(len(value) // 2):
                                    _value.append(
                                        (value[2 * i], value[2 * i + 1])
                                    )
                                value = _value
                except KeyError as error:
                    raise RuntimeError(
                        f"cfg_parser: [{layer_name}] '{option}' is not"
                        " supported."
                    ) from error

                config[layer_name][option] = value

    # count[layer_type] += 1
    # print(count)

    return config
