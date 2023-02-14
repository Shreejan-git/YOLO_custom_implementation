from object_detector.yolov7_object_detector import Yolov7_detector
from object_tracker.yolov7_bytetrack import Yolov7_bytetrack

_YOLO_V7 = None
_YOLO_V7_BYTETRACK = None


def get_yolov7_ob():
    global _YOLO_V7

    if _YOLO_V7 is None:
        model_weight = "yolov7-tiny.pt"
        _YOLO_V7 = Yolov7_detector(weights=model_weight)

    return _YOLO_V7


def get_yolov7_bytetrack():
    global _YOLO_V7_BYTETRACK
    if _YOLO_V7_BYTETRACK is None:
        model_weight = "yolov7-tiny.pt"
        _YOLO_V7_BYTETRACK = Yolov7_bytetrack(weights=model_weight)

    return _YOLO_V7_BYTETRACK
