import cv2
import base64
import json
import requests
import numpy as np
from PIL import Image, ExifTags


ORIENTATION_NORMAL = 3
ORIENTATION_90_DEGREE = 2
ORIENTATION_180_DEGREE = 1
ORIENTATION_270_DEGREE = 0

ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

MAXIMUM_SIZE = 4 * 1024 * 1024  # google could api limitation 4 MB


def load_image(image_path):
    try:
        image = Image.open(image_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

        cv_img = np.array(image)
        cv_img = cv_img[:, :, ::-1].copy()
        return cv_img
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        cv_img = cv2.imread(image_path)
        return cv_img


def get_orientation(points):
    cen_x = .0
    cen_y = .0
    for i in range(4):
        if 'x' not in points[i].keys():
            points[i]['x'] = 0
        if 'y' not in points[i].keys():
            points[i]['y'] = 0
        cen_x += points[i]['x']
        cen_y += points[i]['y']

    cen_x /= 4
    cen_y /= 4

    x0 = points[0]['x']
    y0 = points[0]['y']

    if x0 < cen_x:
        if y0 < cen_y:
            """ 
            0 --------- 1
            |           |
            3 --------- 2 """
            return ORIENTATION_NORMAL  # 3
        else:
            """
            1 --------- 2
            |           |
            0 --------- 3 """
            return ORIENTATION_270_DEGREE  # 2

    else:
        if y0 < cen_y:
            """
            3 --------- 0
            |           |
            2 --------- 1 """
            return ORIENTATION_90_DEGREE  # 0
        else:
            """
            2 --------- 3
            |           |
            1 --------- 0 """
        return ORIENTATION_180_DEGREE  # 1


def correlate_orientation(points, orientation, img_width, img_height):
    for i in range(4):
        point = points[i]
        if 'x' not in point.keys():
            point['x'] = 0
        if 'y' not in point.keys():
            point['y'] = 0

        if orientation == ORIENTATION_NORMAL:
            new_x = point['x']
            new_y = point['y']
        elif orientation == ORIENTATION_270_DEGREE:
            new_x = img_height - point['y']
            new_y = point['x']
        elif orientation == ORIENTATION_90_DEGREE:
            new_x = point['y']
            new_y = img_width - point['x']
        elif orientation == ORIENTATION_180_DEGREE:
            new_x = img_width - point['x']
            new_y = img_height - point['y']

        points[i]['x'] = new_x
        points[i]['y'] = new_y


def make_request(cv_img, feature_types):
    request_list = []

    # Read the image and convert to json
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _ratio = float(MAXIMUM_SIZE) / float(h * w)
    _quality = min(int(_ratio * 10) * 10, 100)

    content_obj = {'content': base64.b64encode(
        cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, _quality])[1].tostring()).decode('UTF-8')}

    feature_obj = []
    for feature_type in feature_types:
        feature_obj.append({'type': feature_type})

    request_list.append(
        {'image': content_obj,
         'features': feature_obj
         }
    )

    return json.dumps({'requests': request_list}).encode()


class ApiUtils:
    def __init__(self):
        self.endpoint_url = 'https://vision.googleapis.com/v1/images:annotate'
        self.api_key = 'XXX'

    def __get_response(self, json_data):
        try:
            response = requests.post(
                url=self.endpoint_url,
                data=json_data,
                params={'key': self.api_key},
                headers={'Content-Type': 'application/json'})

            # print(response)
            ret_json = json.loads(response.text)
            return ret_json['responses'][0]

        except Exception as e:
            print(e)
            return None

    def img2text(self, path):
        try:
            img = load_image(path)
            response = self.__get_response(make_request(cv_img=img,
                                                        feature_types=['TEXT_DETECTION',
                                                                       'DOCUMENT_TEXT_DETECTION']))
            if response is None:
                return None
            else:

                annos = response['textAnnotations']

                # recognize the orientation
                first_rect = annos[0]['boundingPoly']['vertices']
                ori = get_orientation(points=first_rect)
                height, width = img.shape[:2]
                if ori != ORIENTATION_NORMAL:
                    img = cv2.rotate(img, rotateCode=ori)

                for i in range(1, len(annos)):
                    correlate_orientation(points=annos[i]['boundingPoly']['vertices'], orientation=ori,
                                          img_width=width, img_height=height)
                    pt0 = annos[i]['boundingPoly']['vertices'][0]
                    pt1 = annos[i]['boundingPoly']['vertices'][2]
                    # cv2.rectangle(img, (pt0['x'], pt0['y']), (pt1['x'], pt1['y']), (255, 0, 0), 1)
                    print(annos[i]['description'])

                return annos, img

        except Exception as e:
            print("\t exception as " + str(e))
            pass
