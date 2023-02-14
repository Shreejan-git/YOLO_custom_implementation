import cv2
import numpy as np
import numpy.typing as npt
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class Yolov7_detector:

    def __init__(self, weights):
        # Initialize
        set_logging()
        self.weights = weights
        self.device = select_device('')  # leaving '' would result in CPU
        # print('This is device', self.device)
        self.model = attempt_load(self.weights, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, img: npt.NDArray, conf_thres=0.25, iou_thres=0.45, imgsz=640, agnostic_nms=True, trace=False,
               augment=True):
        object_count = {}
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, 640)  # [yesle k garxa?]

        img0 = letterbox(img, imgsz, stride=stride)[0]
        # Convert
        img0 = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img0 = np.ascontiguousarray(img0)  # (path, img, img0, cap)
        img0 = torch.from_numpy(img0).to(self.device)
        img0 = img0.float()
        img0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img0.ndimension() == 3:
            img0 = img0.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img0, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)[0]
        pred[:, :4] = scale_coords(img0.shape[2:], pred[:, :4], img.shape).round()

        for c in pred[:, -1].unique():
            n = (pred[:, -1] == c).sum()  # counting the total number of time a particular class has arrived
            key = self.names[int(c)]
            value = int(n)
            object_count[key] = value

        print(object_count)

        return reversed(pred), self.names, object_count


if __name__ == '__main__':
    pass
    # img = 'inference/images/bus.jpg'
    # img = 'cricket.jpg'
    #
    # img = cv2.imread(img)
    # y7 = Yolov7_detector()
    #
    # d = y7.detect(img)