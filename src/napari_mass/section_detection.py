import numpy as np
from tqdm import tqdm

from napari_mass.Section import Section
from napari_mass.image.util import *
from napari_mass.util import *


def create_detection_image(source, params):
    pixel_size = split_value_unit_list(params['pixel_size'])
    image = get_max_image_at_pixelsize(source, pixel_size)
    contour_detection_image = norm_image_variance(grayscale_image(image))
    threshold = np.quantile(contour_detection_image, 0.95)
    if threshold is not None:
        if threshold == 1:
            contour_detection_image = float2int_image(contour_detection_image >= threshold)
        else:
            contour_detection_image = float2int_image(contour_detection_image > threshold)
    else:
        threshold, contour_detection_image = \
            cv.threshold(float2int_image(contour_detection_image), 0, 255, cv.THRESH_OTSU)
    real_pixel_size = source.get_pixel_size_micrometer()[:2] * get_image_size(source) / get_image_size(contour_detection_image)
    return contour_detection_image, real_pixel_size


def detect_magsections(contour_detection_image, params, pixel_size_um=None, outfolder=None, show_stats=False):
    magsections = []
    if pixel_size_um is None:
        pixel_size = split_value_unit_list(params['pixel_size'])
        pixel_size_um = get_value_units_micrometer(pixel_size)
    detect_shape = params['shape']

    print('MagSec detection')
    contours = get_contours(contour_detection_image)
    contours = [squeeze_contour(contour) for contour in contours]   # remove empty middle dimension

    min_area = 1
    areas = [get_area(contour) for contour in contours if get_area(contour) > min_area]
    min_area = np.mean(areas) / 2
    areas2 = [get_area(contour) for contour in contours if get_area(contour) > min_area]
    mean_area = np.median(areas2)
    shapes = [get_lengths(contour) for contour in contours if get_area(contour) > min_area]
    min_shape = np.median(shapes, 0) / 2

    if outfolder is not None:
        show_histogram(areas2, range=(0, np.quantile(areas2, 0.99)), title='Mag sections area',
                       show=show_stats, out_filename=join_path(outfolder, f'distribution_magsection_area.png'))

    contours2 = []
    shapes0 = []
    for contour in contours:
        area = get_area(contour)
        if area > min_area:
            # TODO: potential fine-tuning
            # basic shape filter - can be removed, or can be expanded using range and shape ratio
            shape = get_lengths(contour)
            if np.all(shape > min_shape):
                contours2.append(contour)
            shapes0.append(shape)
    mean_shape = np.median(shapes0, 0)
    min_distance = mean_shape[1]

    images = []
    for contour in tqdm(contours2):
        area = get_area(contour)
        nsections = int(round(area / mean_area))
        if nsections == 1:
            contour1 = contour - np.min(contour, 0)
            #center, size, angle = get_rotated_rect(contour1)
            center, size, angle = get_shape_stats(contour1)
            image0 = get_contour_mask(contour1, shape=np.flip(np.max(contour1, 0)).astype(int))
            image = rotate_image(image0, angle, center)
            images.append(image)

    template_image = np.zeros(shape=np.flip(mean_shape).astype(int), dtype=np.float32)
    for image in images:
        template_image += reshape_image(image, mean_shape)
    template_image /= len(images)

    for contour in tqdm(contours2):
        area = get_area(contour)
        nsections = int(round(area / mean_area))
        if nsections > 1:
            magsections.extend(divide_section(contour_detection_image, contour, pixel_size_um, nsections,
                                              detect_shape=detect_shape, min_distance=min_distance))
        else:
            magsections.append(approx_section(contour, pixel_size_um,
                                              detect_shape=detect_shape))

    lengths = [section.lengths for section in magsections]
    skewnesses = [section.skewness for section in magsections]
    signal_ratios = [section.calc_signal_ratio(contour_detection_image, pixel_size_um) for section in magsections]

    confidences = []
    for section in magsections:
        generation_confidence = section.confidence
        lengths_confidence = calc_confidence(section.lengths, lengths)
        shape_confidence = calc_confidence(section.skewness, skewnesses) if np.mean(skewnesses) > 0.1 else 1
        signal_confidence = (1 + calc_confidence(section.signal_ratio, signal_ratios)) / 2      # reduce weight
        confidence = np.prod([generation_confidence, lengths_confidence, shape_confidence, signal_confidence])
        section.confidence = confidence
        confidences.append(confidence)

    return magsections, confidences


def divide_section(contour_detection_image, contour, pixel_size_um, nsections, min_distance=1, detect_shape='rect'):
    sections = []
    masks = create_divided_image_masks(contour, contour_detection_image.shape, nsections, min_distance)
    for mask in masks:
        masked_image = (mask * contour_detection_image).astype(np.uint8)
        contours = get_contours(masked_image)
        if len(contours) > 0:
            areas = [cv.contourArea(contour1) for contour1 in contours]
            best_contouri = np.argmax(areas)
            best_area = areas[best_contouri]
            if best_area > 0:
                best_contour = squeeze_contour(contours[best_contouri])
                sections.append(approx_section(best_contour, pixel_size_um, detect_shape=detect_shape))
    confidence = 1 - abs(len(sections) - nsections) / nsections
    for section in sections:
        section.confidence = confidence
    return sections


def approx_section(contour, pixel_size_um, detect_shape='rect'):
    if isinstance(detect_shape, int):
        approx_contour = get_approx_shape_corners(contour, detect_corners=detect_shape)
    else:
        approx_contour = get_approx_shape(contour, detect_shape=detect_shape)
    approx_contour_um = approx_contour * pixel_size_um
    return Section(approx_contour_um)


def detect_nearest_edges(polygon0, source, scaledown=0.25, margin=0.1):
    polygon = polygon0.copy()
    # careful (re)assignments to keep all points/lines pointers linked
    lines = get_polygon_lines(polygon)
    # min detection line length 80% of shortest annotation line
    min_line_len = np.min(get_line_lengths(lines)) * 0.8 * scaledown
    # line gap determined experimentally
    max_line_gap = min_line_len / 20
    polygon2 = scale_shape(polygon, 1 + margin)
    image = get_image_crop_polygon(source, polygon2)
    polygon_min = np.min(polygon2, 0)
    w, h = get_image_size(image) * scaledown
    image = image_resize(image, (int(w), int(h)))

    contour_detection_image0 = uint8_image(norm_image_minmax(detect_edges(image)))
    contour_detection_image = cv.dilate(contour_detection_image0, np.ones((3, 3)))
    # optimal HoughLinesP threshold value (# of intersections) unclear
    lines0 = cv.HoughLinesP(contour_detection_image, 1, np.pi / 180, 40,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines0 is not None:
        lines2o = [[line[0][:2], line[0][2:]] for line in lines0]
        lines2 = np.array(lines2o) / scaledown + polygon_min
        for line in lines:
            # look for match in detected lines
            best_offset = None
            for line2 in lines2:
                offset = get_line_similarity(line, line2, margin)
                if (offset is not None and
                        (best_offset is None
                         or np.mean(np.linalg.norm(offset, axis=1)) < np.mean(np.linalg.norm(best_offset, axis=1)))):
                    best_offset = offset
            if best_offset is not None:
                line[0] += best_offset[0]
                line[1] += best_offset[1]
    return polygon


def get_line_similarity(line, line2, margin):
    offset = None
    vector = line[1] - line[0]
    vector2 = line2[1] - line2[0]
    angle = get_vector_angle(vector)
    angle2 = get_vector_angle(vector2)
    angle_dif = abs(norm_angle_180(get_angle_dif(angle, angle2)))
    if angle_dif < 2:
        dists = np.abs(np.cross(vector, line2 - line[0]) / np.linalg.norm(vector))
        mean_dist = np.mean(dists)
        factors = np.dot(line2 - line[0], vector) / np.linalg.norm(vector) ** 2
        points_on_line = (-margin < factors[0] < 1 + margin and -margin < factors[1] < 1 + margin)
        max_dist = margin * np.mean([np.linalg.norm(vector), np.linalg.norm(vector2)])
        if points_on_line and mean_dist < max_dist:
            dists2 = np.abs(np.cross(vector2, line - line2[0]) / np.linalg.norm(vector2))
            offset0 = np.cross(vector2, line[0] - line2[0]) / vector2
            offset1 = offset0 / np.linalg.norm(offset0) * dists2[0]
            offset0 = np.cross(vector2, line[1] - line2[0]) / vector2
            offset2 = offset0 / np.linalg.norm(offset0) * dists2[1]
            # invert y
            offset1[0] = -offset1[0]
            offset2[0] = -offset2[0]
            offset = np.array([offset1, offset2])
    return offset
