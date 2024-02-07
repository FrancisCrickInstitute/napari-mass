import ast
import cv2 as cv
import colorsys
import math
import numpy as np
import os
import re
from scipy.stats import skew


def get_moments(data, offset=(0, 0)):
    moments = cv.moments((np.array(data) + offset).astype(np.float32))    # doesn't work for float64!
    return moments


def squeeze_contour(contour):
    if contour.ndim == 3:
        return contour[:, 0, :]


def get_area(data):
    return cv.contourArea(data)


def area2radius(area):
    # A = pi * r^2
    # r = sqrt(A / pi)
    return math.sqrt(area / math.pi)


def get_moments_center(moments, offset=(0, 0)):
    return np.array([moments['m10'], moments['m01']]) / moments['m00'] + np.array(offset)


def get_center(data, offset=(0, 0)):
    moments = get_moments(data, offset=offset)
    if moments['m00'] != 0:
        center = get_moments_center(moments)
    else:
        center = np.mean(data, 0).flatten()  # close approximation
    return center.astype(np.float32)


def get_lengths(data):
    moments = get_moments(data)
    if moments['m00'] != 0:
        lengths = get_moments_lengths(moments)
    elif len(data) > 1:
        lengths = (math.dist(np.max(data, 0), np.min(data, 0)), 0)
    else:
        lengths = (0, 0)
    return lengths


def get_moments_lengths(moments):
    # https://stackoverflow.com/questions/66309123/find-enclosing-rectangle-of-image-object-using-second-moments
    mu11 = moments['mu11'] / moments['m00']
    mu20 = moments['mu20'] / moments['m00']
    mu02 = moments['mu02'] / moments['m00']
    mu = [[mu02, mu11], [mu11, mu20]]
    l0, _ = np.linalg.eig(mu)
    l = np.array([x for x in l0 if x > 0])
    lengths = np.sqrt(12 * l)
    return tuple(sorted(lengths, reverse=True))


def get_skewness(data, angle):
    # rotate for simplified 1d skewness calculation
    rotated_data = apply_transform(data - np.mean(data, 0), create_transform(angle=angle))
    return skew(rotated_data[:, 1])


def get_angle(data):
    moments = cv.moments(data)
    mu11 = moments['mu11']
    mu20 = moments['mu20']
    mu02 = moments['mu02']
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    if data.shape[-1] != 2:
        # convert image to coordinates
        data = np.flip(np.transpose(np.where(data)), axis=1)
    # correct orientation using skewness check
    if get_skewness(data, np.rad2deg(theta)) > 0:
        theta += math.pi
        if theta > math.pi:
            theta -= 2 * math.pi
    return theta


def get_norm_rotation_angle_deg(data):
    angle = np.rad2deg(get_angle(data))
    return norm_rotation_angle(angle)


def norm_rotation_angle(angle):
    return -(angle + 90)


def norm_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def norm_angle_180(angle):
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def radius_to_area_range(size_range, slice_thickness, source_pixel_size):
    r_min = size_range[0] / 2 * 0.75  # diameter -> radius with error margin
    if slice_thickness is not None:
        min_detection_size_um = np.sqrt(r_min ** 2 - (r_min - slice_thickness) ** 2) * 2  # radius -> diameter
    else:
        min_detection_size_um = r_min * 2
    max_detection_size_um = size_range[1] * 1.5  # +50% margin
    min_detection_size, max_detection_size = np.array((min_detection_size_um, max_detection_size_um)) / source_pixel_size[:2]
    min_area, max_area = np.pi * (np.array((min_detection_size, max_detection_size)) / 2) ** 2
    return min_area, max_area


def scale_shape(data, scale):
    new_data = []
    center = get_center(data)
    for point in data:
        new_point = center + (point - center) * scale
        new_data.append(new_point)
    return np.array(new_data)


def get_polygon_lines(data):
    data2 = []
    n = len(data)
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        data2.append([data[i], data[j]])
    return data2


def get_line_lengths(lines):
    return [math.dist(line[0], line[1]) for line in lines]


def create_gaussian_kernel(dimension_x, dimension_y, sigma_x, sigma_y):
    x = cv.getGaussianKernel(dimension_x, sigma_x)
    y = cv.getGaussianKernel(dimension_y, sigma_y)
    kernel = x.dot(y.T)
    return kernel


def create_gaussian_log_kernel(dimension_x, dimension_y, sigma):
    # not working correctly
    # https://stackoverflow.com/questions/22050199/python-implementation-of-the-laplacian-of-gaussian-edge-detection
    kernel = np.zeros((dimension_y, dimension_x), dtype=np.float32)
    for y in range(dimension_y):
        for x in range(dimension_x):
            x1 = x - dimension_x / 2
            y1 = y - dimension_y / 2
            kernel[y, x] = (1 / (math.pi * sigma ** 4)) * (1 - (x1 ** 2 + y1 ** 2) / (sigma ** 2)) * (
                pow(math.e, -(x1 ** 2 + y1 ** 2) / 2 * sigma ** 2))
    return kernel


def get_camera_matrix(ref_size=(1, 1)):
    return np.array([[1, 0, (ref_size[0] - 1) / 2], [0, 1, (ref_size[1] - 1) / 2], [0, 0, 1]])  # virtual camera matrix


def calc_peak(data):
    hist, hist_range = np.histogram(data, bins=100)
    hist[0] = 0
    hist[-1] = 0
    peak_index = np.argmax(hist)
    if peak_index > 0:
        peak = np.mean([hist_range[peak_index], hist_range[peak_index + 1]])
    else:
        peak = np.median(data)
    return peak


def translate(data, new_center):
    return data - np.mean(data, 0) + new_center


def create_transform(center=(0, 0), angle=0, scale=1, translate=(0, 0), create3x3=False):
    transform = cv.getRotationMatrix2D(center, angle, scale)
    transform[:, 2] += translate
    if create3x3:
        transform = np.vstack([transform, [0, 0, 1]])
    return transform


def combine_transforms(transforms):
    combined_transform = None
    for transform in transforms:
        if len(transform) < 3:
            transform = np.vstack([transform, [0, 0, 1]])
        if combined_transform is None:
            combined_transform = transform
        else:
            combined_transform = np.dot(transform, combined_transform)
    return combined_transform


def apply_transform(points, transform):
    return np.dot([list(point) + [1] for point in points], transform.T)[:, :2]


def get_transform_angle(transform):
    angle = norm_angle(np.rad2deg(np.arctan2(transform[0][1], transform[0][0])))
    return angle


def get_transform_angle2(transform, ref_size=(1, 1)):
    # https://amroamroamro.github.io/mexopencv/matlab/cv.decomposeHomographyMat.html
    # https://learnopencv.com/rotation-matrix-to-euler-angles/
    if len(transform) == 2:
        angle = np.rad2deg(np.arctan2(transform[0][1], transform[0][0]))
    else:
        k = get_camera_matrix(ref_size)
        n, Rs, Ts, Ns = cv.decomposeHomographyMat(transform, k)
        R = Rs[0]
        angle = np.rad2deg(-cv.Rodrigues(R)[0][-1].squeeze())
    return angle


def get_transform_offset(transform):
    #translation = transform[:, 2]
    translation = transform.dot([0, 0, 1])[:2]
    return translation


def get_transform_offset2(transform, ref_size=(1, 1)):
    if len(transform) == 2:
        translation = transform[:, 2]
    else:
        k = get_camera_matrix(ref_size)
        n, Rs, Ts, Ns = cv.decomposeHomographyMat(transform, k)
        T = Ts[0]
        translation = (T[:2]).squeeze()
    return translation


def get_transform_pre_offset(transform):
    # pre-rotation offset
    translation = get_transform_offset(transform)
    transform2 = transform.copy()
    transform2[:2, 2] = 0
    if len(transform2) < 3:
        transform2 = np.vstack([transform2, [0, 0, 1]])
    pre_translation = np.dot(list(translation) + [1], np.linalg.inv(transform2).T)[:2]
    return pre_translation


def get_transform_center(transform):
    # from opencv:
    # t0 = (1-alpha) * cx - beta * cy
    # t1 = beta * cx + (1-alpha) * cy
    # where
    # alpha = cos(angle) * scale
    # beta = sin(angle) * scale
    # isolate cx and cy:
    t0, t1 = transform[:2, 2]
    scale = 1
    angle = np.arctan2(transform[0][1], transform[0][0])
    alpha = np.cos(angle) * scale
    beta = np.sin(angle) * scale
    cx = (t1 + t0 * (1 - alpha) / beta) / (beta + (1 - alpha) ** 2 / beta)
    cy = ((1 - alpha) * cx - t0) / beta
    return cx, cy


def get_rotated_rect(data, offset=(0, 0)):
    center = get_center(data, offset)
    size = get_lengths(data)
    angle = np.rad2deg(get_angle(data))
    return center, size, angle


def get_vector_angle(vector):
    return np.rad2deg(np.arctan2(vector[1], vector[0]))


def get_max_edge_angle(data):
    best_vector = []
    maxl = 0
    last_pos = data[-1]
    for pos in data:
        vector = pos - last_pos
        l = np.linalg.norm(vector)
        if l > maxl:
            maxl = l
            best_vector = vector
        last_pos = pos
    return get_vector_angle(best_vector)


def rotate_transform_180(transform):
    transform180 = transform.copy()
    transform180[0:2, 0:2] = -transform180[0:2, 0:2]
    return transform180


def get_angle_dif(angle1, angle2):
    # return angle between -180 and 180
    angle = norm_angle((angle2 - angle1) % 360)
    return angle


def calc_confidence(value, ref_values):
    confidence = 1 - np.mean(np.abs(value - np.mean(ref_values, 0)) / np.mean(ref_values, 0))
    #confidence = 1 - np.mean(np.abs(value - np.mean(ref_values, 0)) / (3 * np.std(ref_values, 0)))
    return confidence


def check_round_significants(a: float, significant_digits: int) -> float:
    rounded = round_significants(a, significant_digits)
    if a != 0:
        dif = 1 - rounded / a
    else:
        dif = rounded - a
    if abs(dif) < 10 ** -significant_digits:
        return rounded
    return a


def round_significants(a: float, significant_digits: int) -> float:
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a


def create_color_table(n):
    colors = []
    h = 0
    l = 0.5
    for i in range(n):
        colors.append(normalize_lightness(colorsys.hsv_to_rgb(h, 1, 1), l))
        h = math.fmod(h + 251 / 360, 1)
        l -= 0.22
        if l < 0.2:
            l += 0.6
    return colors


def normalize_lightness(color, level):
    r, g, b = color
    level0 = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    if level0 != 0:
        f = level / level0
        if f > 1:
            r = 1 - (1 - r) / f
            g = 1 - (1 - g) / f
            b = 1 - (1 - b) / f
        else:
            r *= f
            g *= f
            b *= f
    return r, g, b


def color_float_to_cv(rgb):
    return int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)


def ensure_even(x):
    return x + np.mod(x, 2)


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def serialise(lst, symbol=','):
    return symbol.join(lst)


def deserialise(s, symbol=','):
    return list(filter(None, map(str.strip, s.split(symbol))))


def get_numpy_type(s):
    types = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'float32': np.float32, 'float64': np.float64}
    return types[s]


def join_path(base_path, filename):
    if base_path is not None and base_path != '':
        return os.path.join(base_path, filename)
    else:
        return filename


def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    return os.path.splitext(filebase)[0]


def get_dict_permissive(dct, name):
    for key, value in dct.items():
        if key in name or name in key:
            return value
    return None


def get_dict(dct, label, default=None):
    value = dct
    for sublabel in label.split('.'):
        value = value.get(sublabel)
    if isinstance(value, dict):
        value = value.get('value')
    if value is None:
        value = default
    return value


def tags_to_dict(tags):
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict


def desc_to_dict(desc: str) -> dict:
    desc_dict = {}
    if desc.startswith('{'):
        try:
            metadata = ast.literal_eval(desc)
            return metadata
        except:
            pass
    for item in re.split(r'[\r\n\t|]', desc):
        item_sep = '='
        if ':' in item:
            item_sep = ':'
        if item_sep in item:
            items = item.split(item_sep)
            key = items[0]
            value = items[1]
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                try:
                    value = bool(value)
                except:
                    pass
            desc_dict[key] = value
    return desc_dict


def print_dict(d, compact=False, indent=0):
    s = ''
    for key, value in d.items():
        if not isinstance(value, list):
            if not compact: s += '\t' * indent
            s += str(key)
            s += ':' if compact else '\n'
        if isinstance(value, dict):
            s += print_dict(value, indent=indent + 1)
        elif isinstance(value, list):
            for v in value:
                s += print_dict(v)
        else:
            if not compact: s += '\t' * (indent + 1)
            s += str(value)
        s += ' ' if compact else '\n'
    return s


def stringdict_to_dict(string_dict):
    metadata = {}
    if isinstance(string_dict, dict):
        dict_list = string_dict.items()
    else:
        dict_list = string_dict
    for key, value in dict_list:
        keys = key.split('|')
        add_dict_tree(metadata, keys, value)
    return metadata


def add_dict_tree(metadata, keys, value):
    key = keys[0]
    if len(keys) > 1:
        if key not in metadata:
            metadata[key] = {}
        add_dict_tree(metadata[key], keys[1:], value)
    else:
        metadata[key] = value


def dict_keys_to_string(data0):
    if isinstance(data0, dict):
        data = {}
        for key, value in data0.items():
            if isinstance(key, int):
                key = str(key)
            data[key] = value
    else:
        data = data0
    return data


def dict_keys_to_int(data0):
    if isinstance(data0, dict):
        data = {}
        for key, value in data0.items():
            if key.isnumeric():
                key = int(key)
            data[key] = value
    else:
        data = data0
    return data


def split_num_text(text):
    num_texts = []
    block = ''
    is_num0 = None
    if text is None:
        return None

    for c in text:
        is_num = (c.isnumeric() or c == '.')
        if is_num0 is not None and is_num != is_num0:
            num_texts.append(block)
            block = ''
        block += c
        is_num0 = is_num
    if block != '':
        num_texts.append(block)

    num_texts2 = []
    for block in num_texts:
        block = block.strip()
        try:
            block = float(block)
        except:
            pass
        if block not in [' ', ',', '|']:
            num_texts2.append(block)
    return num_texts2


def split_value_unit_list(text):
    value_units = []
    if text is None:
        return None

    items = split_num_text(text)
    if isinstance(items[-1], str):
        def_unit = items[-1]
    else:
        def_unit = ''

    i = 0
    while i < len(items):
        value = items[i]
        if i + 1 < len(items):
            unit = items[i + 1]
        else:
            unit = ''
        if not isinstance(value, str):
            if isinstance(unit, str):
                i += 1
            else:
                unit = def_unit
            value_units.append((value, unit))
        i += 1
    return value_units


def get_value_units_micrometer(value_units0: list) -> list:
    conversions = {'nm': 1e-3, 'µm': 1, 'um': 1, 'micrometer': 1, 'mm': 1e3, 'cm': 1e4, 'm': 1e6}
    if value_units0 is None:
        return None

    values_um = []
    for value_unit in value_units0:
        if not (isinstance(value_unit, int) or isinstance(value_unit, float)):
            value_um = value_unit[0] * conversions.get(value_unit[1], 1)
        else:
            value_um = value_unit
        values_um.append(value_um)
    return values_um


def print_hbytes(bytes):
    exps = ['', 'K', 'M', 'G', 'T']
    div = 1024
    exp = 0

    while bytes > div:
        bytes /= div
        exp += 1
    return f'{bytes:.1f}{exps[exp]}B'


def pretty_num_list(items):
    return ' '.join(f'{x:.4f}' for x in items)


def validate_out_folder(root, path):
    if path is not None:
        if root is not None:
            fullpath = join_path(root, path)
        else:
            fullpath = path
    else:
        fullpath = root
    if fullpath is not None:
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
    return fullpath


def norm_order(order):
    if len(order) >= 2 and order[0] > order[-1]:
        return list(reversed(order))
    else:
        return order


def check_order(score_matrix, dist_matrix, order, min_match_score=None):
    scores = []
    distances = []
    ngood = 0

    if isinstance(order, str):
        sep = ',' if ',' in order else None
        order = [int(x.strip()) for x in order.split(sep)]

    n_unique = len(set(order))
    n = len(order)
    ntotal = n - 1
    if n != n_unique:
        print(f'Order length {n} != {n_unique} unique elements')
    if ntotal > 0:
        index0 = order[0]
        for index in order[1:]:
            score = score_matrix[index0, index]
            distance = dist_matrix[index0, index]
            scores.append(score)
            distances.append(distance)
            if min_match_score is not None and score > min_match_score:
                ngood += 1
            index0 = index
    else:
        ntotal = 1

    results = 'Order: ' + ' '.join(map(str, order)) + \
              '\n          ' + ''.join([f'{o: <7}' for o in order]) + \
              '\nScores:    ' + pretty_num_list(scores) + \
              '\nDistances: ' + pretty_num_list(distances)
    results += (f'\nScore: {np.mean(scores):.4f} Distance: {np.mean(distances):.4f}'
                f' #Good: {ngood}/{ntotal} ({ngood / ntotal * 100:.1f}%)')

    return scores, distances, ngood, order, results


def convert_rational_value(value) -> float:
    if value is not None and isinstance(value, tuple):
        value = value[0] / value[1]
    return value
