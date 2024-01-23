import numpy as np
import cv2 as cv
from probreg import cpd
from sklearn.metrics import euclidean_distances

from napari_mass.image.util import *
from napari_mass.util import *


def align_sections_metrics(source_section, target_section, matching_methods,
                           lowe_ratio=None, max_iter=100, min_match_rate=0.5):
    results = None
    for method in matching_methods:
        results0 = align_sections(source_section, target_section,
                                  method=method, lowe_ratio=lowe_ratio, max_iter=max_iter)
        if results is None or results0[1]['match_rate'] > results[1]['match_rate']:
            results = results0
        if results[1]['match_rate'] >= min_match_rate:
            break
    return results


def get_section_alignment(source_section, target_section, matching_methods,
                          lowe_ratio=None, max_iter=100, min_match_rate=0.5):
    h_coarse = create_transform(angle=-source_section.angle, create3x3=True)
    h_align, metrics = align_sections_metrics(source_section, target_section, matching_methods,
                                              lowe_ratio=lowe_ratio, max_iter=max_iter, min_match_rate=min_match_rate)
    h_full = combine_transforms([h_coarse, h_align])
    center = source_section.center - get_transform_pre_offset(h_full)
    angle = -get_transform_angle(h_full)
    return center, angle, metrics


def get_features(image, keypoints):
    #feature_model = cv.ORB_create()
    feature_model = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64)
    _, descriptors = feature_model.compute(float2int_image(image), keypoints)
    return descriptors


def align_sections(source_section, target_section, method='features', lowe_ratio=None, max_iter=100):
    if method.lower().startswith('cpd'):
        return align_sections_cpd(source_section, target_section, lowe_ratio=lowe_ratio, max_iter=max_iter)
    else:
        return align_sections_features(source_section, target_section, lowe_ratio=lowe_ratio)


def align_sections_features(source_section, target_section, lowe_ratio=None):
    # image feature descriptors
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance])
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


def align_sections_cpd(source_section, target_section, lowe_ratio=None, max_iter=100):
    # CPD: use_cuda doesn't seem to work (well) - might be a cupy (version) issue?
    result_cpd = cpd.registration_cpd(source_section.points, target_section.points, w=0.00001, maxiter=max_iter, tol=0.1)
    nn_distance = np.mean([source_section.nn_distance, target_section.nn_distance])
    transformation = result_cpd.transformation
    transformed_source_points0 = transformation.transform(source_section.points)
    transformed_source_points = [(point0, size_point[1])
                                 for point0, size_point in zip(transformed_source_points0, source_section.size_points)]
    metrics = get_match_metrics(target_section.size_points, transformed_source_points, nn_distance, lowe_ratio)
    # transform() = scale * dot(points, rot.T) + t
    transform = np.hstack([transformation.scale * transformation.rot, np.atleast_2d(transformation.t).T])
    return transform, metrics


def get_match_metrics(points0, points1, nn_distance, lowe_ratio=None):
    results = {}
    # greedy assignment
    if len(points0) == 0 or len(points1) == 0:
        return 0, np.inf, 0, np.inf, np.array([])
    has_size_points = (not np.isscalar(points0[0][0]))

    threshold = nn_distance / 4
    if len(points0) > len(points1):
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
    matches0 = []
    matches1 = []
    nmatches = 0
    tot_weight = 0
    weight = 0
    matching_distances = []
    matched_points = []
    for sorted_match in sorted_matches:
        i, match = matches[sorted_match]
        if has_size_points:
            tot_weight += sizes0[i]
        for ji, j in enumerate(match):
            if j not in done:
                # found best, available match
                distance0 = distance_matrix[i, j]
                distance1 = distance_matrix[i, match[ji + 1]] if ji + 1 < len(match) else np.inf
                matching_distances.append(distance0)
                if distance0 < threshold and (lowe_ratio is None or distance0 < lowe_ratio * distance1):
                    matches0.append(i)
                    matches1.append(j)
                    done.append(j)
                    nmatches += 1
                    if has_size_points:
                        weight += sizes0[i]
                    matched_points.append(points0[i])
                break

    if has_size_points:
        results['size_match_rate'] = weight / tot_weight
    results['nmatches'] = nmatches
    results['match_rate'] = nmatches / min(len(points0), len(points1)) if nmatches > 0 else 0
    distance = np.mean(matching_distances) if nmatches > 0 else np.inf
    results['distance'] = float(distance)
    results['norm_distance'] = float(distance / nn_distance)
    results['matched_points'] = np.array(matched_points)
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
