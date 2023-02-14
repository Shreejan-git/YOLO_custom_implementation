import cv2


def opencv_object_tracking(tracker="csrt"):
    """
    opencv-contrib-python = 4.7.0.68

    Args:
        tracker: name of the object tracker to be used.

    Returns:

    """
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.legacy.TrackerCSRT_create(),
        "kcf": cv2.legacy.TrackerKCF_create(),
        "boosting": cv2.legacy.TrackerBoosting_create(),
        "mil": cv2.legacy.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create()
    }

    tracker = OPENCV_OBJECT_TRACKERS[tracker]  # default is csrt

    # initialize the bounding box coordinates of the object we are going to track
    print("[INFO] starting the webcam")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    assert cap.isOpened(), f"could not open the webcam"

    ok, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    bbox = cv2.selectROI(frame, False)
    # bbox = (287, 23, 86, 320)
    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = cap.read()
        ok, bbox = tracker.update(frame)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.namedWindow('WEBCAM FRAME', cv2.WINDOW_NORMAL)
        cv2.imshow('WEBCAM FRAME', frame)

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == "__main__":
    opencv_object_tracking("mosse")
