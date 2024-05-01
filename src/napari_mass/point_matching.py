import numpy as np
import cv2 as cv
import skimage
from probreg import cpd
from sklearn.metrics import euclidean_distances

from napari_mass.image.util import *
from napari_mass.util import *


def align_sections_metrics(source_section, target_section, matching_methods, min_match_rate=0.5, **params):
    results = None
    for method in matching_methods:
        results0 = align_sections(source_section, target_section, method=method, **params)
        if results is None or results0[1]['match_rate'] > results[1]['match_rate']:
            results = results0
        if results[1]['match_rate'] >= min_match_rate:
            break
    if results is not None:
        metrics = results[1]
        metrics['matched_points'] = [(source_section.points[s], target_section.points[t])
                                     for t, s in metrics['matches']]
    return results


def get_section_alignment(source_section, target_section, matching_methods, min_match_rate=0.5, pixel_size=1, **params):
    h_coarse = create_transform(angle=-source_section.angle, create3x3=True)
    h_align, metrics = align_sections_metrics(source_section, target_section, matching_methods, min_match_rate, **params)
    if h_align.ndim == 2:
        h_full = combine_transforms([h_coarse, h_align])
    else:
        h_full = h_coarse
    center = source_section.center - get_transform_pre_offset(h_full) * pixel_size
    angle = -get_transform_angle(h_full)
    return center, angle, metrics


def get_features(image, keypoints):
    #feature_model = cv.ORB_create()
    feature_model = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64)
    _, descriptors = feature_model.compute(float2int_image(image), keypoints)
    return descriptors


def align_sections(source_section, target_section, method='features', **params):
    if 'flow' in method.lower():
        return align_sections_flow(source_section, target_section, **params)
    elif 'cpd' in method.lower():
        return align_sections_cpd(source_section, target_section, **params)
    else:
        return align_sections_features(source_section, target_section, **params)


def align_sections_features(source_section, target_section, lowe_ratio=None, distance_factor=1, **params):
    # image feature descriptors
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance]) * distance_factor
    ransac_threshold = nn_distance

    transform1, metrics1 = \
        align_points(source_section.points, source_section.descriptors,
                     target_section.points, target_section.descriptors,
                     ransac_threshold, nn_distance, lowe_ratio)

    transform2, metrics2 = \
        align_points(source_section.points, source_section.descriptors,
                     target_section.points_alt, target_section.descriptors_alt,
                     ransac_threshold, nn_distance, lowe_ratio)

    if metrics2['match_rate'] > metrics1['match_rate']:
        transform2 = rotate_transform_180(transform2)
        return transform2, metrics2
    else:
        return transform1, metrics1


def align_points(source_points, source_descriptors, target_points, target_descriptors,
                 ransac_threshold, nn_distance, lowe_ratio):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    cvmatches = matcher.match(source_descriptors, target_descriptors)
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
        metrics = get_match_metrics(target_points, transformed_source_points, nn_distance, lowe_ratio)
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
    transformed_source_points0 = transformation.transform(source_section.points)
    transformed_source_points = [(point0, size_point[1])
                                 for point0, size_point in zip(transformed_source_points0, source_section.size_points)]
    metrics = get_match_metrics(target_section.size_points, transformed_source_points, nn_distance, lowe_ratio)
    # transform() = scale * dot(points, rot.T) + t
    transform = np.hstack([transformation.scale * transformation.rot, np.atleast_2d(transformation.t).T])
    return transform, metrics


def align_sections_flow(source_section, target_section, lowe_ratio=None, **params):
    # usually called with: section (new), prev_section (reference)
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance])
    source_image = grayscale_image(source_section.image)
    target_image = grayscale_image(target_section.image)
    #v, u = skimage.registration.optical_flow_ilk(target_image, source_image, radius=nn_distance)
    v, u = skimage.registration.optical_flow_tvl1(target_image, source_image)

    # Inverse coordinate map, which transforms coordinates in the output images
    # into their corresponding coordinates in the input image.
    inverse_map = calculate_flow_map((v, u))
    # TODO: inter/extrapolate NAN values in matrix, or interpolate using closest 4 non-nan values
    flow_map = calculate_inverse_flow_map(inverse_map)

    # alternative approach: redo registration with swapped source/target
    vi, ui = skimage.registration.optical_flow_tvl1(source_image, target_image)
    flow_map1 = calculate_flow_map((vi, ui))

    # debug: show original overlay
    overlay_image = np.atleast_3d(target_image) * [1, 0, 0] + np.atleast_3d(source_image) * [0, 0, 1]
    show_image(overlay_image)

    # show transformed overlay
    source_image_warped = skimage.transform.warp(source_image, inverse_map, mode='edge', preserve_range=True).astype(source_image.dtype)
    overlay_image = np.atleast_3d(target_image) * [1, 0, 0] + np.atleast_3d(source_image_warped) * [0, 0, 1]
    show_image(overlay_image)

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
    plt.tight_layout()
    plt.show()

    # get transform from registration
    transform = inverse_map
    # apply transform to source points
    center = w / 2, h / 2
    transformed_source_points = [get_flow_map_position(point + center, flow_map1) - center
                                 for point in source_section.points]
    transformed_source_size_points = \
        [(point0, size_point[1]) for point0, size_point
         in zip(transformed_source_points, source_section.size_points)]

    # test point transform
    show_image(draw_image_points_overlay(source_image, source_image_warped,
                                         source_section.points, transformed_source_points))

    metrics = get_match_metrics(target_section.size_points, transformed_source_size_points,
                                nn_distance, lowe_ratio)

    matched_source_points = [transformed_source_points[s] for t, s in metrics['matches']]
    matched_target_points = [target_section.points[t] for t, s in metrics['matches']]
    show_image(draw_image_points_overlay(target_image, source_image_warped,
                                         matched_target_points, matched_source_points))

    # alternative match rate for flow: image intersection / union
    target_image_bin = (target_image != 0)
    source_image_warped_bin = (source_image_warped != 0)
    ntotal = np.sum(target_image_bin)
    match_rate = np.sum(source_image_warped_bin & target_image_bin) / ntotal
    metrics['match_rate2'] = match_rate

    return transform, metrics


def get_match_metrics(points0, points1, nn_distance, lowe_ratio=None):
    results = {}
    # greedy assignment
    npoints0, npoints1 = len(points0), len(points1)
    npoints = min(npoints0, npoints1)
    if npoints0 == 0 or npoints1 == 0:
        return 0, np.inf, 0, np.inf, np.array([])
    has_size_points = not np.isscalar(points0[0][0])

    threshold = nn_distance / 4
    swapped = (npoints0 > npoints1)
    if swapped:
        points0, points1 = points1, points0
    if has_size_points:
        sizes0 = [size for point, size in points0]
        sizes1 = [size for point, size in points1]
        points0 = [point for point, size in points0]
        points1 = [point for point, size in points1]
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
        if has_size_points:
            tot_weight += sizes0[i]
        for ji, j in enumerate(match):
            if j not in done:
                # found best, available match
                distance0 = distance_matrix[i, j]
                distance1 = distance_matrix[i, match[ji + 1]] if ji + 1 < len(match) else np.inf
                matching_distances.append(distance0)    # use all distances to weigh in non-matches
                if distance0 < threshold and (lowe_ratio is None or distance0 < lowe_ratio * distance1):
                    done.append(j)
                    if swapped:
                        match = j, i
                    else:
                        match = i, j
                    point_matches.append(match)
                    nmatches += 1
                    if has_size_points:
                        weight += sizes0[i]
                break

    if has_size_points:
        results['size_match_rate'] = weight / tot_weight
    results['nmatches'] = nmatches
    results['match_rate'] = nmatches / npoints if npoints > 0 else 0
    distance = np.mean(matching_distances) if nmatches > 0 else np.inf
    results['distance'] = float(distance)
    results['norm_distance'] = float(distance / nn_distance)
    results['matches'] = np.array(point_matches)
    return results


# for testing:


def match_sections(section1, section2):
    matches1, transform1, match_rate1 = \
        match_points(section1.points, section1.descriptors, section1.lengths,
                     section2.points, section2.descriptors, section2.lengths)

    matches2, transform2, match_rate2 = \
        match_points(section1.points, section1.descriptors, section1.lengths,
                     section2.points_alt, section2.descriptors_alt, section2.lengths)

    if match_rate2 > match_rate1:
        return matches2, transform2, match_rate2

    return matches1, transform1, match_rate1


def match_points(points1, descriptors1, lengths1, points2, descriptors2, lengths2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    ref_size = np.mean([lengths1, lengths2], 0)
    norm_size = np.linalg.norm(ref_size)
    ransac_threshold = norm_size / 100

    cvmatches = matcher.match(descriptors1, descriptors2)
    query_points = np.zeros((len(cvmatches), 2), dtype=np.float32)
    train_points = np.zeros((len(cvmatches), 2), dtype=np.float32)
    for i, cvmatch in enumerate(cvmatches):
        query_points[i, :] = points1[cvmatch.queryIdx]
        train_points[i, :] = points2[cvmatch.trainIdx]

    matches = [cvmatch.queryIdx for cvmatch in cvmatches], [cvmatch.trainIdx for cvmatch in cvmatches]

    transform, mask =\
        cv.findHomography(query_points, train_points, method=cv.USAC_MAGSAC, ransacReprojThreshold=ransac_threshold)

    match_rate = np.mean(mask)
    return matches, transform, match_rate
