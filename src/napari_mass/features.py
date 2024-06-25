import cv2 as cv
import logging
from sklearn.neighbors import KDTree
from tqdm import tqdm

from napari_mass.image.util import *
from napari_mass.util import *


def get_features(image, keypoints):
    #feature_model = cv.ORB_create()
    feature_model = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64)
    _, descriptors = feature_model.compute(float2int_image(image), keypoints)
    return descriptors


def init_section_features(section, image_function=None,
                          size_range=None, min_npoints=1):
    pixel_size = section.image_pixel_size
    rotated_image, rotated_image_alt = section.create_rotated_image(section.image, pixel_size)
    size_range_px = convert_size_to_pixels(size_range, pixel_size) if size_range is not None else None
    area_range_px = diameter2area(size_range_px) if size_range_px is not None else None

    if image_function:
        bin_image = image_function(rotated_image, pixel_size=pixel_size)
    else:
        bin_image = simple_detection_image(rotated_image)
    bin_image_alt = rotate_image(bin_image, 180)

    section.points, section.size_points, section.keypoints, section.descriptors = \
        get_image_features(bin_image, area_range_px)
    section.points_alt, section.size_points_alt, section.keypoints_alt, section.descriptors_alt = \
        get_image_features(bin_image_alt, area_range_px)

    # print(len(self.points))
    if len(section.points) < min_npoints:  # or len(self.points_alt) < min_npoints:
        message = 'Insufficient key points in section'
        raise ValueError(message)
    if len(section.points) >= 2:
        tree = KDTree(section.points, leaf_size=2)
        dist, ind = tree.query(section.points, k=2)
        section.nn_distance = np.median(dist[:, 1])
    else:
        section.nn_distance = 1

    section.bin_image = bin_image
    section.bin_image_alt = bin_image_alt


def get_image_features(image, area_range=None):
    area_points = get_contour_points(image, area_range=area_range)
    center = np.flip(image.shape[:2]) / 2
    # center points around (0, 0)
    points = [point - center for point, size in area_points]
    size_points = [(point - center, 2 * area2radius(area)) for point, area in area_points]
    # convert area to diameter
    keypoints = [cv.KeyPoint(point[0], point[1], 2 * area2radius(area)) for point, area in area_points]
    descriptors = get_features(image, keypoints)
    return points, size_points, keypoints, descriptors


def init_sections_features(sections, source, pixel_size, image_function=None,
                           size_range=None, min_npoints=1,
                           show_stats=False, out_filename=None):
    if len(sections) > 0:
        for section in tqdm(sections):
            section.init_image(source, pixel_size)
            init_section_features(section, image_function, size_range, min_npoints)
        sections_npoints = [min(len(section.points), len(section.points_alt)) for section in sections]
        mean_npoints = np.median(sections_npoints)
        for sectioni, npoints in enumerate(sections_npoints):
            if abs(npoints - mean_npoints) / mean_npoints > 0.5:
                logging.warning(f'Section {sectioni} #keypoints: {npoints}')

        show_histogram(sections_npoints, range=None, title='Mag sections #keypoints',
                       show=show_stats, out_filename=out_filename)
