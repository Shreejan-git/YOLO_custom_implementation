from yolov7_detector import Yolov7_detector

_YOLO_V7 = None


def get_yolov7_ob():
    global _YOLO_V7

    if _YOLO_V7 is None:
        model_weight = "yolov7-tiny.pt"
        _YOLO_V7 = Yolov7_detector(weights=model_weight)

    return _YOLO_V7
