def yolov7_bytetrack_example(img='/home/shreejan/Documents/yolov7/woman.jpg', webcam=False):
    """
    Args:
        img: To detect the objects in an image, webcam must not be False. If link to an image is not provided,
        the default image path will be used.
        webcam: If this is set to True, object detection will be done in real-time. Else, via: given image

    Returns: Object detection
    """
    line_thickness = 1
    yolov7_bytetrack_instance = get_yolov7_bytetrack()

    if webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size
        assert cap.isOpened(), f'Failed to open the webcam'
        while True:
            ret, img = cap.read()
            # a = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            # print(a)
            # align_bottom = img.shape[0]
            # align_right = (img.shape[1] / 1.45)
            if ret:
                img = cv2.flip(img, 1)  # flip left-right
                # coordinates, names, object_counts = yolov7_bytetrack_instance.detect(img, conf_thres=0.5)
                detections = yolov7_bytetrack_instance.detect(img, track=True)
                print('this is the value of detections', detections)
                detected_frame = draw(img, detections)
                print('Detected Frame', json.dumps(detections, indent=4))

                cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
                cv2.imshow('Webcam', detected_frame)
                #
                if cv2.waitKey(1) == ord('q'):
                    break

                # break
                # img = draw_bboxes(coordinates, names, img, line_thickness)
                # for key, value in object_counts.items():
                #     kv = f"{key}={value}"
                #     align_bottom = align_bottom - 35
                #     cv2.putText(img, str(kv), (int(align_right), align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                 (45, 255, 26), 1)
                #
                # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                # cv2.imshow('img', img)
                # if cv2.waitKey(1) == ord('q'):
                #     cv2.destroyAllWindows()  # what will happen when adding these two here
                #     cap.release()
                #     break
            else:
                print('Could not find a frame')
                break

        cv2.destroyAllWindows()
        cap.release()

    else:  # for the input image
        img = cv2.imread(img)
        coordinates, names, object_counts = yolov7_bytetrack_instance.detect(img)
        img = draw_bboxes(coordinates, names, img, line_thickness)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_bboxes(coor, names, img, line_thickness):
    """

    Args:
        coor: It contains the xy coordinates, confidence score and class
        names: All the classes of the cooo dataset
        img:Every frame of the video
        line_thickness: bounding box thickness

    Returns: Frame consisting bounding box, class and the confidence score

    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    tl = line_thickness or round(0.002 * (img_height + img_width) / 2) + 1  # line/font thickness

    for *x, conf, cls in coor:
        r = random.randint(1, 255)
        g = random.randint(1, 255)
        b = random.randint(1, 255)

        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

        cv2.rectangle(img, c1, c2, (r, g, b), thickness=tl, lineType=cv2.LINE_AA)

        label = f'{names[int(cls)]}:{conf:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, (r, g, b), -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [r, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


if __name__ == '__main__':

    import json
    from utils.get_yolov7_utils import get_yolov7_bytetrack
    from utils.detections import draw
    import cv2
    import random
    yolov7_bytetrack_example(webcam=True)
