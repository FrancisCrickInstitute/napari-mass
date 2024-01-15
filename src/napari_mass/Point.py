import cv2 as cv
import numpy as np


class Point:
    # coordinates: [x, y]

    def __init__(self, value):
        if isinstance(value, dict):
            self.location = value['location']
        else:
            self.location = value

    def draw(self, image, pixel_size, label=None, draw_size=1,
             color=(0, 255, 0), label_color=(192, 192, 192)):
        font_size = int(np.ceil(draw_size * 0.5))
        thickness = draw_size

        position = np.round(np.divide(self.location, pixel_size)).astype(int)
        cv.drawMarker(image, position, color, cv.MARKER_CROSS, draw_size * 2, thickness, line_type=cv.LINE_AA)
        if label is not None:
            cv.putText(image, str(label), position, cv.FONT_HERSHEY_SIMPLEX, font_size, label_color,
                       thickness, lineType=cv.LINE_AA)

    def to_dict(self):
        return {
            'location': [float(self.location[0]), float(self.location[1])],
        }
