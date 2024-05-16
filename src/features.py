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


def init_section_features(section, image_function=None, detection_params=None):
    pixel_size = section.image_pixel_size
    rotated_image, rotated_image_alt = section.create_rotated_image(section.image, pixel_size)
    if detection_params:
        slice_thickness = get_dict_value(detection_params, 'size_slice_thickness_um')
        size_range0 = deserialise(get_dict_value(detection_params, 'size_range_um'))
        size_range = [float(x) for x in size_range0]
        size_range_px = convert_size_to_pixels(size_range, pixel_size)
        min_area, max_area = radius_to_area_range(size_range, slice_thickness, pixel_size)
        min_npoints = get_dict_value(detection_params, 'min_npoints', 1)
    else:
        min_area, max_area = 1, None
        size_range_px = convert_size_to_pixels([1, 10], pixel_size)
        min_npoints = 1

    if image_function:
        bin_image = image_function(rotated_image, size_range=size_range_px)
    else:
        bin_image = simple_detection_image(rotated_image)
    bin_image_alt = rotate_image(bin_image, 180)

    section.points, section.size_points, section.keypoints, section.descriptors = \
        get_image_features(bin_image, min_area, max_area)
    section.points_alt, section.size_points_alt, section.keypoints_alt, section.descriptors_alt = \
        get_image_features(bin_image_alt, min_area, max_area)

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


def get_image_features(image, min_area=1, max_area=None):
    area_points = get_contour_points(image, min_area=min_area, max_area=max_area)
    center = np.flip(image.shape[:2]) / 2
    # center points around (0, 0)
    points = [point - center for point, size in area_points]
    size_points = [(point - center, 2 * area2radius(area)) for point, area in area_points]
    # convert area to diameter
    keypoints = [cv.KeyPoint(point[0], point[1], 2 * area2radius(area)) for point, area in area_points]
    descriptors = get_features(image, keypoints)
    return points, size_points, keypoints, descriptors


def init_sections_features(sections, source, pixel_size, show_stats=True, out_filename=None, **params):
    if len(sections) > 0:
        for section in tqdm(sections):
            section.init_image(source, pixel_size)
            init_section_features(section, **params)
        sections_npoints = [min(len(section.points), len(section.points_alt)) for section in sections]
        mean_npoints = np.median(sections_npoints)
        for sectioni, npoints in enumerate(sections_npoints):
            if abs(npoints - mean_npoints) / mean_npoints > 0.5:
                logging.warning(f'Section {sectioni} #keypoints: {npoints}')

        show_histogram(sections_npoints, range=None, title='Mag sections #keypoints',
                       show=show_stats, out_filename=out_filename)
