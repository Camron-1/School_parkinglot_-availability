import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


mask_path = "./mask_1920_1080.png"
video_path = "./samples/video.mp4"

mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
if mask is None:
    raise SystemExit(f"Could not load mask image: {mask_path}")
mask_h, mask_w = mask.shape

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(
        f"Could not open video: {video_path}\n"
        "Create app/samples/ and add the file, or set video_path to a valid .mp4."
    )

connected_components = cv.connectedComponentsWithStats(mask, 4, cv.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (mask_w, mask_h))

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]

            diffs[spot_indx] = calc_diff(
                spot_crop, previous_frame[y1 : y1 + h, x1 : x1 + w, :]
            )

        print([diffs[j] for j in np.argsort(diffs)][::-1])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            amax = float(np.amax(diffs))
            if amax > 0:
                arr_ = [
                    j for j in np.argsort(diffs) if diffs[j] / amax > 0.4
                ]
            else:
                arr_ = range(len(spots))
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]

            spot_status = empty_or_not(spot_crop)

            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv.putText(
        frame,
        "Available spots: {} / {}".format(
            str(sum(spots_status)), str(len(spots_status))
        ),
        (100, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.imshow("frame", frame)
    if cv.waitKey(25) & 0xFF == ord("q"):
        break

    frame_nmr += 1

cap.release()
cv.destroyAllWindows()
