import csv
import cv2

from pytesseract import pytesseract as pt

img_path = './images/PCE28-.jpg'

pt.run_tesseract(img_path, 'output', lang=None, boxes=True, config='hocr')
boxes = []
chars = []

with open('output.box', encoding="utf8") as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        if len(row) == 6:
            boxes.append(row)

img = cv2.imread(img_path)
h, w = img.shape[:2]
for b in boxes:
    cv2.rectangle(img, (int(b[1]), h-int(b[2])), (int(b[3]), h - int(b[4])), (255, 0, 0), 2)
    cv2.putText(img, b[0], (int(b[1]), h - int(b[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imshow("result", img)
cv2.imwrite("result.jpg", img)
cv2.waitKey(0)
