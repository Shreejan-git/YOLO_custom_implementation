import warnings
import cv2
import numpy as np
import numpy.typing as npt
import torch
import yaml
import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from byte_tracker import BYTETracker
from utils.detections import Detections

warnings.filterwarnings('ignore')

from botsort_tracker.mc_bot_sort import BoTSORT


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


class Yolov7_botsort:
    def __init__(self, weights, conf_thres=0.50, iou_thres=0.45, imgsz=640, device="cpu", classes='coco.yaml',
                 track_high_thresh=0.3, track_low_thresh=0.05, new_track_thresh=0.4, proximity_thresh=0.5,
                 appearance_thresh=0.25, fast_reid_config=r"fast_reid/configs/MOT17/sbs_S50.yml",
                 fast_reid_weights=r"pretrained/mot17_sbs_S50.pth",
                 cmc_method="sparseOptFlow", name="exp", ablation=False, track_buffer=30, with_reid=False,
                 match_thresh=0.7, min_box_area=10, fuse_score="mot20", hide_labels_name=True):
        """
        parameters:
        track_high_thresh: tracking confidence threshold
        track_low_thresh: lowest detection threshold
        new_track_thresh: new track thresh
        proximity_thresh: threshold for rejecting low overlap reid matches
        appearance_thresh: threshold for rejecting low appearance similarity reid matches
        fast_reid_config: reid config file path
        fast_reid_weights: reid config file path
        cmc_method: "cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc"
        name: "save results to project/name
        albation:
        with_reid: with ReID module.
        track_buffer: the frames for keep lost tracks
        match_thresh: matching threshold for tracking
        min_box_area: filter out tiny boxes
        fuse_score: fuse score and iou for association
        hide_labels_name: displaying the label name while detecting
        """
        # Initialize
        set_logging()  # what does this do?
        self.weights = weights
        self.settings = {
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'imgsz': imgsz,

        }
        self.classes = classes
        self.device = select_device(device)  # leaving '' would result in CPU
        self.model = attempt_load(self.weights, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

        self.opt = {
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "proximity_thresh": proximity_thresh,
            "appearance_thresh": appearance_thresh,
            "fast_reid_config": fast_reid_config,
            "fast_reid_weights": fast_reid_weights,
            "device": device,
            "cmc_method": cmc_method,
            "name": name,
            "ablation": ablation,
            "with_reid": with_reid,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "min_box_area": min_box_area,
            'fuse_score': fuse_score,
            'hide_labels_name': hide_labels_name
        }

        if device != "cpu":
            self.model.half()
            self.model.to(self.device).eval()

        self.tracker = BoTSORT(self.opt, frame_rate=30.0)

    def detect(self, img: npt.NDArray, track=True):
        # object_count = {}

        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.settings['imgsz'], s=stride)  # check img_size
        classes = yaml.load(open(self.classes), Loader=yaml.SafeLoader)['classes']

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
            pred = self.model(img0)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.settings["conf_thres"], self.settings["iou_thres"])

        results = []
        labels = None

        for i, det in enumerate(pred):
            detections = []
            if len(det) > 0:
                # det[:, :4] = scale_coords(img0.shape[2:], det[:, :4], img.shape).round()
                boxes = scale_coords(img0.shape[2:], det[:, :4], img.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            online_targets = self.tracker.update(detections, img)

            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                if tlwh[2] * tlwh[3] > self.opt['min_box_area']:
                    if self.opt["hide_labels_name"]:
                        label = f'{tid}, {self.names[int(tcls)]}'
                        #haven't sent t.score
                        a = [tid, tlbr, label]
                        results.append(a)
                    else:
                        label = f'{tid}, {int(tcls)}'
                        a = [tid, tlbr, label]
                        results.append(a)

        return results


if __name__ == '__main__':
    pass
    # img = 'inference/images/bus.jpg'
    # img = 'cricket.jpg'
    #
    # img = cv2.imread(img)
    # y7 = Yolov7_detector()
    #
    # d = y7.detect(img)
