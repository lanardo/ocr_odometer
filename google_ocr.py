import numpy as np
import cv2
from api_utils import ApiUtils


def process(img):
    red_range_1 = [(0, 150, 0), (10, 255, 255)]
    red_range_2 = [(140, 150, 0), (179, 255, 255)]

    black_range = [(0, 50, 0), (179, 255, 50)]

    # 149, 40, 80

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, red_range_1[0], red_range_1[1])
    red2 = cv2.inRange(hsv, red_range_2[0], red_range_2[1])
    red_mask = cv2.bitwise_or(red1, red2)

    black_mask = cv2.inRange(hsv, black_range[0], black_range[1])
    cv2.imshow("black", black_mask)
    cv2.imshow("red", red_mask)

    mask = cv2.bitwise_or(red_mask, black_mask)

    mask = cv2.blur(mask, (11, 11))
    mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]
    dilate = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)

    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    if len(contours) == 0:
        return []
    else:
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return boxes


def odo_value(boxes, annos):
    candis = []
    for box in boxes:
        (x0, y0, w, h) = box
        value_annos = []
        x1, y1 = x0 + w, y0 + h
        for anno in annos:
            pt0 = anno['boundingPoly']['vertices'][0]
            pt1 = anno['boundingPoly']['vertices'][2]

            xx0, yy0 = pt0['x'], pt0['y']
            xx1, yy1 = pt1['x'], pt1['y']
            text_rect_sz = (yy1 - yy0) * (xx1 - xx0)

            _x0 = max(x0, xx0)
            _y0 = max(y0, yy0)
            _x1 = min(x1, xx1)
            _y1 = min(y1, yy1)

            if (_x0 < _x1) and (_y0 < _y1):
                _sz = (_y1 - y0) * (_x1 - _x0)
            else:
                _sz = 0

            ratio = _sz / text_rect_sz

            if ratio > 0.7:
                value_annos.append(anno)
        if len(value_annos) != 0:
            candis.append([value_annos, box])
    return candis


if __name__ == '__main__':
    util = ApiUtils()

    img_path = "./images/PCE28-.jpg"
    # img_path = "./images/PCE28-Pose2.jpg"
    img = cv2.imread(img_path)

    annos, img = util.img2text(img_path)
    rects = process(img)

    candis = odo_value(rects, annos)
    for [values, box] in candis:
        (x0, y0, w, h) = box
        cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (0, 255, 0), 2)
        str = ""
        for anno in values:
            str += anno['description'].encode("utf-8")
        # cv2.putText(img, str, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("result2.jpg", img)
    cv2.imwrite("result2.jpg", img)
    cv2.waitKey(0)





