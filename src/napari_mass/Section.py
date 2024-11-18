import logging
import numpy as np

from napari_mass.image.util import *
from napari_mass.util import *


class Section:
    # coordinates: [x, y]

    def __init__(self, value):
        if isinstance(value, dict):
            self.polygon = np.asarray(value['polygon'], dtype=np.float32)
            self.center = value.get('center')
            self.angle = value.get('angle')
            self.confidence = value.get('confidence', 1)
        else:
            self.polygon = np.asarray(value, dtype=np.float32)
            self.center = None
            self.angle = None
            self.confidence = 1
        center, lengths, angle, skewness = get_shape_stats(self.polygon)
        self.lengths = lengths
        self.skewness = skewness
        if self.center is None:
            self.center = center
        if self.angle is None:
            self.angle = angle
        self.image = None

    def get_size(self, pixel_size=1):
        lengths = np.divide(np.flip(self.lengths), pixel_size)
        return np.ceil(lengths).astype(int)

    def init_image(self, source, pixel_size=None):
        if pixel_size is None:
            pixel_size = source.get_pixel_size_micrometer()[:2]
        if self.image is None:
            self.image = self.extract_image(source, pixel_size)
        self.image_pixel_size = pixel_size

    def extract_image(self, source, pixel_size):
        origin = get_value_units_micrometer(source.position[:2])
        polygon = (self.polygon - origin) / pixel_size
        polygon_min, polygon_max = np.min(polygon, 0), np.max(polygon, 0)

        w, h = np.ceil(polygon_max - polygon_min).astype(int)
        x, y = polygon_min

        #w, h = np.ceil(np.max([polygon_max - self.center, self.center - polygon_min], 0) * 2).astype(int)
        #x, y = np.round(self.center - np.array([w, h]) / 2)
        #if x < 0:
        #    x = 0
        #    w = self.center[0] * 2
        #if y < 0:
        #    y = 0
        #    h = self.center[1] * 2

        cropped = color_image(norm_image_minmax(get_image_crop(source, x, y, w, h, pixel_size=pixel_size)))
        # improved crop using mask
        polygon1 = polygon - (x, y)
        mask = get_contour_mask(polygon1, shape=(h, w))
        image = cropped * np.atleast_3d(mask)
        alpha_image = np.dstack([image, mask])
        return alpha_image

    def create_rotated_image(self, image, pixel_size):
        # center of new image corresponds to custom center
        polygon = self.polygon / pixel_size
        center_local = np.divide(self.center, pixel_size) - np.min(polygon, 0)
        #center_local = get_image_size(image) / 2
        rotated_image = rotate_image(image, -self.angle, center_local)
        rotated_image_alt = cv.rotate(rotated_image, cv.ROTATE_180)
        return rotated_image, rotated_image_alt

    def extract_rotated_image_raw(self, source, pixel_size, size, use_alpha=True):
        rotated_rect = np.divide(self.center, pixel_size[:2]), np.float32(size), -self.angle
        box = cv.boxPoints(rotated_rect)
        boxmin, boxmax = np.min(box, 0), np.max(box, 0)
        w, h = np.ceil(boxmax - boxmin).astype(int)
        x, y = boxmin
        cropped = color_image(int2float_image(get_image_crop(source, x, y, w, h, pixel_size=pixel_size)))
        if use_alpha:
            polygon1 = self.polygon / pixel_size[:2] - boxmin
            mask = get_contour_mask(polygon1, shape=(h, w))
            image = np.dstack([cropped, mask])
        else:
            image = cropped
        rotated_image = rotate_image(image, -self.angle + 90)
        image = reshape_image(rotated_image, np.flip(size))
        return image

    def get_rotated_image(self, source, pixel_size, size=None):
        rotated_image, _ = self.create_rotated_image(self.extract_image(source, pixel_size), pixel_size)
        if size is not None:
            rotated_image = reshape_image(rotated_image, size)
        return rotated_image

    def draw(self, image, pixel_size, label=None, draw_size=1,
             color=(0, 255, 0), arrow_color=(127, 127, 127), label_color=(192, 192, 192), draw_arrow=True):
        thickness = draw_size
        cx, cy = np.divide(self.center, pixel_size)
        points = np.round(self.polygon / pixel_size).astype(int)
        cv.polylines(image, [points], True, color, thickness, lineType=cv.LINE_AA)

        if draw_arrow:
            arrow_length = self.lengths[1] / pixel_size * 0.5
            angle_rad = np.deg2rad(-self.angle)  # image y vs vector domain -> invert y
            x0 = int(cx)
            y0 = int(cy)
            x1 = int(cx + arrow_length * math.cos(angle_rad))
            y1 = int(cy + arrow_length * math.sin(angle_rad))
            cv.arrowedLine(image, (x0, y0), (x1, y1), arrow_color, thickness, line_type=cv.LINE_AA)

        if label is not None:
            font_size = draw_size * 0.5
            draw_text(image, str(label), (cx, cy), font_size, thickness, label_color)

    def draw_points(self, source, pixel_size, size_points=[], labels=None, point_color=None):
        rotated_image = grayscale_image(self.get_rotated_image(source, pixel_size))
        if len(size_points) == 0:
            size_points = self.size_points
        draw_size = np.linalg.norm(rotated_image.shape[:2]) / 1000
        font_size = int(np.ceil(draw_size))
        thickness = int(np.ceil(draw_size / 2))
        colors = create_color_table(1000)
        out_image = color_image(float2int_image(rotated_image))
        nchannels = out_image.shape[2] if len(out_image.shape) > 2 else 1
        if labels is None:
            labels = range(len(size_points))
        image_center = get_image_size(out_image) / 2
        for label, size_point in zip(labels, size_points):
            center0, area = size_point
            center = (center0 + image_center).astype(int)
            radius = math.sqrt(area / math.pi) * 2
            if point_color is None:
                color = color_float_to_cv(colors[int(label) % len(colors)])
            else:
                color = point_color
            if len(color) < nchannels:
                color = list(color) + [255]
            cv.circle(out_image, center, int(radius), color, thickness=thickness)
            #draw_text(out_image, str(label), center, font_size, thickness, color)
        draw_text(out_image, f'N={len(size_points)}', None, font_size, thickness, [127, 127, 127])
        return out_image

    def calc_signal_ratio(self, source, pixel_size):
        image = self.extract_image(source, pixel_size)
        self.signal_ratio = np.count_nonzero(image) / image.size
        return self.signal_ratio

    def to_dict(self):
        return {
            'polygon': self.polygon.tolist(),
            'center': [float(self.center[0]), float(self.center[1])],
            'angle': float(self.angle),
            'confidence': float(self.confidence),
        }


def get_section_images(sections, source, pixel_size=None):
    images = []
    logging.info('Obtaining all section images')
    if pixel_size is None:
        pixel_size = source.get_pixel_size_micrometer()[:2]
    crop_size, max_size = get_section_sizes(sections, pixel_size)
    for section in sections:
        rotated_image = section.get_rotated_image(source, pixel_size, max_size)
        images.append(uint8_image(norm_image_minmax(rotated_image)))
    return images


def get_section_sizes(sections, pixel_size=None):
    sizes = [section.get_size() for section in sections]
    if pixel_size:
        sizes = np.divide(sizes, pixel_size[:2])
    mean_size = np.median(sizes, 0).astype(int)
    padded_size = np.round(mean_size * 1.25).astype(int)
    return mean_size, padded_size
