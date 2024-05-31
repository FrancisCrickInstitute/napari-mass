import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import skimage
from probreg import cpd
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KDTree

from napari_mass.image.util import *
from napari_mass.util import *


def do_section_alignment(source_section, target_section, method, min_match_rate=0.5, pixel_size=1, **params):
    transform, metrics = get_section_alignment_metrics(source_section, target_section, method, **params)
    if metrics['match_rate'] > min_match_rate and is_affine_transform(transform):
        # adjust section
        transform_coarse = create_transform(angle=-source_section.angle, create3x3=True)
        transform_combined = combine_transforms([transform_coarse, transform])
        center = source_section.center - get_transform_pre_offset(transform_combined) * pixel_size
        angle = -get_transform_angle(transform_combined)
        source_section.center = center
        source_section.angle = angle
    return transform, metrics


def get_section_alignment_metrics(source_section, target_section, method, **params):
    method = method.lower()
    if 'sparse_flow' in method:
        results = align_sections_sparse_flow(source_section, target_section, **params)
    elif 'flow' in method:
        results = align_sections_flow(source_section, target_section, **params)
    elif 'cpd' in method:
        results = align_sections_cpd(source_section, target_section, **params)
    else:
        results = align_sections_features(source_section, target_section, **params)
    return results


def align_sections_features(source_section, target_section, lowe_ratio=None, distance_factor=1, **params):
    # image feature descriptors
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance]) * distance_factor
    ransac_threshold = nn_distance

    transform1, metrics1 = \
        align_points_features(source_section.size_points, target_section.size_points,
                              source_section.descriptors, target_section.descriptors,
                              source_section.bin_image, target_section.bin_image,
                              ransac_threshold, nn_distance, lowe_ratio)

    transform2, metrics2 = \
        align_points_features(source_section.size_points, target_section.size_points_alt,
                              source_section.descriptors, target_section.descriptors_alt,
                              source_section.bin_image, target_section.bin_image_alt,
                              ransac_threshold, nn_distance, lowe_ratio)

    if metrics2['match_rate'] > metrics1['match_rate']:
        transform2 = rotate_transform_180(transform2)
        return transform2, metrics2
    else:
        return transform1, metrics1


def align_points_features(source_size_points, target_size_points,
                          source_descriptors, target_descriptors,
                          source_image, target_image,
                          ransac_threshold, nn_distance, lowe_ratio):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    cvmatches = matcher.match(source_descriptors, target_descriptors)
    source_points = [size_point[0] for size_point in source_size_points]
    target_points = [size_point[0] for size_point in target_size_points]
    matched_source_points = []
    matched_target_points = []
    for cvmatch in cvmatches:
        matched_source_points.append(source_points[cvmatch.queryIdx])
        matched_target_points.append(target_points[cvmatch.trainIdx])
    # homography needs minimum 4 points
    transform = None
    if len(matched_source_points) >= 4:
        transform, mask = cv.findHomography(np.asarray(matched_source_points), np.asarray(matched_target_points),
                                            method=cv.USAC_MAGSAC, ransacReprojThreshold=ransac_threshold)
    if transform is not None:
        # apply transform to points to calculate distance
        transformed_source_points = apply_transform(source_points, transform)
        transformed_source_size_points = \
            [(point0, size_point[1]) for point0, size_point
             in zip(transformed_source_points, source_size_points)]
        image_size = np.flip(source_image.shape[:2])
        image_transform = combine_transforms(
            [create_transform(translate=-image_size / 2), transform, create_transform(translate=image_size / 2)])
        transformed_source_image = cv.warpPerspective(source_image, image_transform, image_size)
        metrics = get_match_metrics(transformed_source_size_points, target_size_points,
                                    transformed_source_image, target_image,
                                    nn_distance, lowe_ratio)
    else:
        transform = create_transform()
        nmatches = len(matched_source_points)
        metrics = {}
        metrics['nmatches'] = nmatches
        metrics['match_rate'] = nmatches / len(cvmatches) if nmatches > 0 else 0
        metrics['distance'] = np.inf
        metrics['norm_distance'] = 1
    return transform, metrics


def align_sections_cpd(source_section, target_section, distance_factor=1,
                       lowe_ratio=None, w=0.00001, max_iter=100, tol=0.1):
    # CPD: use_cuda doesn't seem to work (well) - might be a cupy (version) issue?
    result_cpd = cpd.registration_cpd(source_section.points, target_section.points, w=w, maxiter=max_iter, tol=tol)
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance]) * distance_factor
    transformation = result_cpd.transformation
    # derived from: transform() = scale * dot(points, rot.T) + t
    transform = np.hstack([transformation.scale * transformation.rot, np.atleast_2d(transformation.t).T])
    transformed_source_points = transformation.transform(source_section.points)
    transformed_source_size_points = [(point0, size_point[1])
                                      for point0, size_point
                                      in zip(transformed_source_points, source_section.size_points)]
    image_size = np.flip(source_section.bin_image.shape[:2])
    image_transform = combine_transforms(
        [create_transform(translate=-image_size / 2), transform, create_transform(translate=image_size / 2)])
    transformed_source_image = cv.warpPerspective(source_section.bin_image, image_transform, image_size)
    metrics = get_match_metrics(transformed_source_size_points, target_section.size_points,
                                transformed_source_image, target_section.bin_image,
                                nn_distance, lowe_ratio)
    return transform, metrics


def align_sections_flow(source_section, target_section, lowe_ratio=None, **params):
    # usually called with: section (new), prev_section (reference)
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance])
    source_image = grayscale_image(source_section.bin_image)
    target_image = grayscale_image(target_section.bin_image)
    max_size = np.flip(np.max([source_image.shape[:2], target_image.shape[:2]], 0))
    source_image = reshape_image(source_image, max_size)
    target_image = reshape_image(target_image, max_size)
    #v, u = skimage.registration.optical_flow_ilk(target_image, source_image, radius=nn_distance)
    v, u = skimage.registration.optical_flow_tvl1(target_image, source_image)

    # Inverse coordinate map, which transforms coordinates in the output images
    # into their corresponding coordinates in the input image.
    inverse_map = calculate_flow_map((v, u))
    # TODO: inter/extrapolate NAN values in matrix, or interpolate using closest 4 non-nan values
    #flow_map = calculate_inverse_flow_map(inverse_map)

    vu = cv.calcOpticalFlowFarneback(target_image, source_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vu_inv = cv.calcOpticalFlowFarneback(source_image, target_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # alternative approach: redo registration with swapped source/target
    vi, ui = skimage.registration.optical_flow_tvl1(source_image, target_image)
    flow_map = calculate_flow_map((vi, ui))

    #flow_map1 = invert_displacement_vector_field(inverse_map, np.ones((1, 1)))

    source_image_warped = (skimage.transform.warp(source_image, inverse_map, mode='edge', preserve_range=True)
                           .astype(source_image.dtype))

    # testing: show magnitude and vector field
    nvec = 20  # Number of vectors to be displayed along each image dimension
    h, w = source_image.shape[:2]
    step = max(h // nvec, w // nvec)

    y, x = np.mgrid[:h:step, :w:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    norm = np.sqrt(u ** 2 + v ** 2)
    plt.imshow(norm)
    plt.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
    plt.savefig('vectors.tiff')
    plt.show()

    # get transform from registration
    transform = flow_map
    # apply transform to source points
    center = w / 2, h / 2
    transformed_source_points = [get_flow_map_position(point + center, flow_map) - center
                                 for point in source_section.points]

    #source_points = np.array(source_section.points).astype(np.float32)
    #tran_source_points = np.zeros_like(source_points)
    #points_reg = cv.calcOpticalFlowPyrLK(source_image, target_image, source_points, tran_source_points)

    transformed_source_size_points = \
        [(point0, size_point[1]) for point0, size_point
         in zip(transformed_source_points, source_section.size_points)]

    metrics = get_match_metrics(transformed_source_size_points, target_section.size_points,
                                source_image_warped, target_image,
                                nn_distance, lowe_ratio)
    return transform, metrics


def align_sections_sparse_flow(source_section, target_section, lowe_ratio=None, **params):
    # usually called with: section (new), prev_section (reference)
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance])
    source_image = grayscale_image(source_section.bin_image)
    target_image = grayscale_image(target_section.bin_image)
    max_size = np.flip(np.max([source_image.shape[:2], target_image.shape[:2]], 0))
    source_image = reshape_image(source_image, max_size)
    target_image = reshape_image(target_image, max_size)

    vi, ui = skimage.registration.optical_flow_tvl1(source_image, target_image)
    flow_map = calculate_flow_map((vi, ui))

    # apply transform to source points
    h, w = source_image.shape[:2]
    center = w / 2, h / 2
    transformed_source_points = [get_flow_map_position(point + center, flow_map) - center
                                 for point in source_section.points]

    transformed_source_size_points = \
        [(point0, size_point[1]) for point0, size_point
         in zip(transformed_source_points, source_section.size_points)]

    transform = (source_section.points, transformed_source_points)
    tree = KDTree(source_section.points, leaf_size=2)

    # use sparse map to convert image pixels
    source_image_warped = np.zeros_like(source_image)
    # TODO: check for accidental x/y swapping
    positions = np.transpose(np.where(source_image > 0))
    values = source_image[positions]
    positions_warped = np.array([get_sparse_flow_position(point, transform, tree) for point in positions - center]) + center
    source_image_warped[positions_warped.astype(int).tolist()] = values

    metrics = get_match_metrics(transformed_source_size_points, target_section.size_points,
                                source_image_warped, target_image,
                                nn_distance, lowe_ratio)
    return transform, metrics


def get_match_metrics(size_points0, size_points1, image0, image1, nn_distance, lowe_ratio=None):
    metrics = {}
    source_size_points, target_size_points = size_points0, size_points1
    # greedy assignment
    npoints0, npoints1 = len(size_points0), len(size_points1)
    npoints = min(npoints0, npoints1)
    if npoints0 == 0 or npoints1 == 0:
        return {}

    threshold = nn_distance / 4
    swapped = (npoints0 > npoints1)
    if swapped:
        size_points0, size_points1 = size_points1, size_points0
    sizes0 = [size for point, size in size_points0]
    points0 = [point for point, size in size_points0]
    points1 = [point for point, size in size_points1]
    distance_matrix = euclidean_distances(points0, points1)

    matches = []
    distances0 = []
    for rowi, row in enumerate(distance_matrix):
        sorted_indices = np.argsort(row)
        index0 = sorted_indices[0]
        distance0 = row[index0]
        matches.append((rowi, sorted_indices))
        distances0.append(distance0)
    sorted_matches = np.argsort(distances0)

    done = []
    point_matches = []
    nmatches = 0
    tot_weight = 0
    weight = 0
    matching_distances = []
    for sorted_match in sorted_matches:
        i, match = matches[sorted_match]
        tot_weight += sizes0[i]
        for ji, j in enumerate(match):
            if j not in done:
                # found best, available match
                distance0 = distance_matrix[i, j]
                distance1 = distance_matrix[i, match[ji + 1]] if ji + 1 < len(match) else np.inf
                matching_distances.append(distance0)    # use all distances to also weigh in the non-matches
                if distance0 < threshold and (lowe_ratio is None or distance0 < lowe_ratio * distance1):
                    done.append(j)
                    if swapped:
                        match = j, i
                    else:
                        match = i, j
                    point_matches.append(match)
                    nmatches += 1
                    weight += sizes0[i]
                break

    # alternative match rate for flow: image intersection / union
    if image0 is not None and image1 is not None:
        max_size = np.flip(np.max([image0.shape[:2], image1.shape[:2]], 0))
        image0 = reshape_image(image0, max_size)
        image1 = reshape_image(image1, max_size)
        image_match_rate = np.sum((image0 != 0) & (image1 != 0)) / np.count_nonzero(image1)
    else:
        image_match_rate = None

    matched_points, matched_source_points, matched_target_points = [], [], []
    for point_match in point_matches:
        source_point, target_point = source_size_points[point_match[0]][0], target_size_points[point_match[1]][0]
        matched_points.append((source_point, target_point))
        matched_source_points.append(source_point)
        matched_target_points.append(target_point)

    metrics['nmatches'] = nmatches
    metrics['match_rate'] = nmatches / npoints if npoints > 0 else 0
    metrics['size_match_rate'] = weight / tot_weight
    metrics['image_match_rate'] = image_match_rate
    distance = np.mean(matching_distances) if nmatches > 0 else np.inf
    metrics['distance'] = float(distance)
    metrics['norm_distance'] = float(distance / nn_distance)
    metrics['matches'] = point_matches
    metrics['matched_points'] = matched_points
    metrics['matched_source_points'] = matched_source_points
    metrics['matched_target_points'] = matched_target_points

    metrics['overlay_image'] = draw_image_points_overlay(image0, image1,
                                                         matched_source_points, matched_target_points,
                                                         draw_size=3)
    return metrics


def print_metrics(metrics):
    return (
        f"#matches: {metrics['nmatches']} "
        f"match rate: {metrics['match_rate']:.3f} "
        f"size match rate: {metrics['size_match_rate']:.3f} "
        f"image match rate: {metrics['image_match_rate']:.3f} "
        f"distance: {metrics['distance']:.1f} "
        f"norm distance: {metrics['norm_distance']:.1f} "
    )
