import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_laplace, distance_transform_edt, label
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import reconstruction
from skimage.segmentation import watershed
import tifffile
from tifffile import TiffWriter, TiffFile

from napari_mass.util import *


mpl.rcParams['figure.dpi'] = 600

confidence_color_map = mpl.colors.LinearSegmentedColormap.from_list('', [(1, 0, 0, 0.5), (1, 1, 0, 0.25), (0, 1, 0, 0)])


def show_image(image, title='', cmap=None):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if cmap is None:
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.show()


def show_image_cv(image):
    cv.imshow('', image)
    cv.waitKey()


def ensure_unsigned_type(dtype: np.dtype) -> np.dtype:
    new_dtype = dtype
    if dtype.kind == 'i' or dtype.byteorder == '>' or dtype.byteorder == '<':
        new_dtype = np.dtype(f'u{dtype.itemsize}')
    return new_dtype


def ensure_unsigned_image(image0: np.ndarray) -> np.ndarray:
    dtype0 = image0.dtype
    dtype = ensure_unsigned_type(dtype0)
    if dtype != dtype0:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        image = image0.astype(dtype) + offset
    else:
        image = image0
    return image


def convert_image_sign_type(image0: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if image0.dtype.kind == dtype.kind:
        image = image0
    elif image0.dtype.kind == 'i':
        image = ensure_unsigned_image(image0)
    else:
        # conversion without overhead
        offset = 2 ** (8 * dtype.itemsize - 1)
        image = (image0 - offset).astype(dtype)
    return image


def image_reshape(image: np.ndarray, target_size: tuple) -> np.ndarray:
    tw, th = target_size
    sh, sw = image.shape[:2]
    if sw < tw or sh < th:
        dw = max(tw - sw, 0)
        dh = max(th - sh, 0)
        padding = [(0, dh), (0, dw)]
        if len(image.shape) == 3:
            padding += [(0, 0)]
        image = np.pad(image, padding, 'edge')
    if tw < sw or th < sh:
        image = image[0:th, 0:tw]
    return image


def image_set_safe(image0, image1, offset0):
    x0, y0 = offset0
    x1, y1 = 0, 0
    w0, h0 = get_image_size(image0)
    w, h = get_image_size(image1)
    if x0 < 0:
        w += x0
        x1 -= x0
        x0 = 0
    if y0 < 0:
        h += y0
        y1 -= y0
        y0 = 0
    if x0 + w > w0:
        w = w0 - x0
    if y0 + h > h0:
        h = h0 - y0
    image0[y0: y0 + h, x0: x0 + w] = image1[y1: y1 + h, x1: x1 + w]


def image_resize(image: np.ndarray, target_size0: tuple, dimension_order: str = 'yxc') -> np.ndarray:
    shape = image.shape
    x_index = dimension_order.index('x')
    y_index = dimension_order.index('y')
    c_is_at_end = dimension_order.endswith('c')
    size = shape[x_index], shape[y_index]
    if np.mean(np.divide(size, target_size0)) < 1:
        interpolation = cv.INTER_CUBIC
    else:
        interpolation = cv.INTER_AREA
    dtype0 = image.dtype
    image = ensure_unsigned_image(image)
    target_size = tuple(np.maximum(np.round(target_size0).astype(int), 1))
    if dimension_order in ['yxc', 'yx']:
        new_image = cv.resize(np.asarray(image), target_size, interpolation=interpolation)
    elif dimension_order == 'cyx':
        new_image = np.moveaxis(image, 0, -1)
        new_image = cv.resize(np.asarray(new_image), target_size, interpolation=interpolation)
        new_image = np.moveaxis(new_image, -1, 0)
    else:
        ts = image.shape[dimension_order.index('t')] if 't' in dimension_order else 1
        zs = image.shape[dimension_order.index('z')] if 'z' in dimension_order else 1
        target_shape = list(image.shape).copy()
        target_shape[x_index] = target_size[0]
        target_shape[y_index] = target_size[1]
        new_image = np.zeros(target_shape, dtype=image.dtype)
        for t in range(ts):
            for z in range(zs):
                slices = get_numpy_slicing(dimension_order, z=z, t=t)
                image1 = image[slices]
                if not c_is_at_end:
                    image1 = np.moveaxis(image1, 0, -1)
                new_image1 = np.atleast_3d(cv.resize(np.asarray(image1), target_size, interpolation=interpolation))
                if not c_is_at_end:
                    new_image1 = np.moveaxis(new_image1, -1, 0)
                new_image[slices] = new_image1
    new_image = convert_image_sign_type(new_image, dtype0)
    return new_image



def precise_resize(image: np.ndarray, scale: np.ndarray, use_max: bool = False) -> np.ndarray:
    h, w = np.ceil(image.shape[:2] * scale).astype(int)
    shape = list(image.shape).copy()
    shape[:2] = h, w
    new_image = np.zeros(shape, dtype=np.float32)
    step_size = 1 / scale
    for y in range(h):
        for x in range(w):
            y0, y1 = np.round([y * step_size[1], (y + 1) * step_size[1]]).astype(int)
            x0, x1 = np.round([x * step_size[0], (x + 1) * step_size[0]]).astype(int)
            image1 = image[y0:y1, x0:x1]
            if image1.size > 0:
                if use_max:
                    value = np.max(image1, axis=(0, 1))
                else:
                    value = np.mean(image1, axis=(0, 1))
                new_image[y, x] = value
    return new_image.astype(image.dtype)


def precise_resize_fast(image: np.ndarray, scale: np.ndarray, use_max: bool = False) -> np.ndarray:
    factor = np.round(1 / scale).astype(int)
    h, w = np.round(image.shape[:2] / factor).astype(int)
    h0, w0 = np.array([h, w]) * factor
    image = image_reshape(image, (w0, h0))
    new_image = image.reshape((h, h0 // h, -1, w0 // w)).swapaxes(1, 2).reshape(h, w, -1)
    if use_max:
        result_image = new_image.max(axis=2)
    else:
        result_image = new_image.mean(axis=2)
    return result_image


def get_thresholded_mean(image, threshold):
    return np.mean(image > threshold)


def get_tiff_pages(tiff: TiffFile) -> list:
    pages = []
    found = False
    for serie in tiff.series:
        # has series
        for level in serie.levels:
            # has levels
            level_pages = []
            for page in level.pages:
                found = True
                level_pages.append(page)
            if len(level_pages) > 0:
                pages.append(level_pages)

    if not found:
        for page in tiff.pages:
            pages.append(page)
    return pages


def cv_load(filename, grayscale=False):
    # Note: seems to auto-convert 16-bit to 8-bit images
    if grayscale:
        return cv.imread(filename, flags=cv.IMREAD_GRAYSCALE)
    else:
        return cv.imread(filename)


def save_image(filename, image, params=None):
    if len(image.shape) > 2:
        if image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)
        else:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, image, params)


def load_tiff(filename):
    image = tifffile.imread(filename)
    if image.ndim == 3:
        # move channel axis (unless RGB)
        if image.shape[0] < image.shape[2] and image.shape[0] != 3:
            image = np.moveaxis(image, 0, -1)
    return image


def save_tiff(filename, image, tile_size=None, compression=None):
    bigtiff = (image.size * image.itemsize > 2 ** 32)
    if image.ndim == 3:
        # move channel axis (unless RGB)
        if image.shape[2] < image.shape[0] and image.shape[2] != 3:
            image = np.moveaxis(image, -1, 0)
    with TiffWriter(filename, bigtiff=bigtiff) as writer:
        writer.write(image, tile=tile_size, compression=compression)


def get_image_size(source):
    if isinstance(source, np.ndarray):
        size = np.flip(source.shape[:2])
    else:
        size = source.get_size()
    return size


def get_max_image_at_pixelsize(source, pixel_size):
    if not isinstance(source, np.ndarray):
        pixel_size0 = [(size, 'um') for size in np.multiply(source.get_pixel_size_micrometer()[:2], 4)]
        image0 = source.asarray(pixel_size=pixel_size0)
        factor = np.divide(get_value_units_micrometer(pixel_size0), get_value_units_micrometer(pixel_size)[:2])
        image = precise_resize_fast(image0, factor, use_max=True)
    else:
        image = source
    return image


def flatfield_correction(image0, dark=0, bright=1, clip=True):
    # https://imagej.net/plugins/bigstitcher/flatfield-correction
    mean_bright_dark = np.mean(bright - dark, (0, 1))
    image = (image0 - dark) * mean_bright_dark / (bright - dark)
    if clip:
        image = np.clip(image, 0, 1)
    return image


def get_image_crop(source, x, y, w, h, pixel_size=None):
    x0, y0, w, h = int(x), int(y), int(w), int(h)
    x1, y1 = x0 + w, y0 + h
    if isinstance(source, np.ndarray):
        cropped = source[y0:y1, x0:x1]
    else:
        cropped = source.render(source.asarray(pixel_size=pixel_size, x0=x0, x1=x1, y0=y0, y1=y1), source.get_dimension_order())
    return cropped


def grayscale_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 4:
        return cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    elif nchannels > 1:
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        return image


def color_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 1:
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        return image


def int2float_image(image):
    if image.dtype.kind != 'f':
        maxval = 2 ** (8 * image.dtype.itemsize) - 1
        return image / np.float32(maxval)
    else:
        return image


def float2int_image(image, dtype=np.dtype(np.uint8)):
    if not (image.dtype.kind == 'i' or image.dtype.kind == 'u') and not dtype.kind == 'f':
        maxval = 2 ** (8 * dtype.itemsize) - 1
        return (image * maxval).astype(dtype)
    else:
        return image


def uint8_image(image):
    if image.dtype.kind == 'f':
        image *= 255
    elif image.dtype.itemsize != 1:
        factor = 2 ** (8 * (image.dtype.itemsize - 1))
        image //= factor
    return image.astype(np.uint8)


def norm_image_minmax(image0):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def redimension_data(data, old_order, new_order, **kwargs):
    # able to provide optional dimension values e.g. t=0, z=0
    if new_order == old_order:
        return data

    new_data = data
    order = old_order
    # remove
    for o in old_order:
        if o not in new_order:
            index = order.index(o)
            dim_value = kwargs.get(o, 0)
            new_data = np.take(new_data, indices=dim_value, axis=index)
            order = order.replace(o, '')
    # add
    for o in new_order:
        if o not in order:
            new_data = np.expand_dims(new_data, 0)
            order = o + order
    # move
    old_indices = [order.index(o) for o in new_order]
    new_indices = list(range(len(new_order)))
    new_data = np.moveaxis(new_data, old_indices, new_indices)
    return new_data


def get_numpy_slicing(dimension_order, **slicing):
    slices = []
    for axis in dimension_order:
        index = slicing.get(axis)
        index0 = slicing.get(axis + '0')
        index1 = slicing.get(axis + '1')
        if index0 is not None and index1 is not None:
            slice1 = slice(int(index0), int(index1))
        elif index is not None:
            slice1 = int(index)
        else:
            slice1 = slice(None)
        slices.append(slice1)
    return tuple(slices)


def get_image_quantile(image: np.ndarray, quantile: float, axis=None) -> float:
    value = np.quantile(image, quantile, axis=axis).astype(image.dtype)
    return value


def normalise_values(image: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return np.clip((image.astype(np.float32) - min_value) / (max_value - min_value), 0, 1)


def norm_image_variance(image0):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    normimage = np.clip((image - np.mean(image)) / np.std(image), 0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def norm_image_quantiles(image0, quantile=0.99):
    if len(image0.shape) == 3 and image0.shape[2] == 4:
        image, alpha = image0[..., :3], image0[..., 3]
    else:
        image, alpha = image0, None
    min_value = np.quantile(image, 1 - quantile)
    max_value = np.quantile(image, quantile)
    normimage = np.clip((image - min_value) / (max_value - min_value), 0, 1).astype(np.float32)
    if alpha is not None:
        normimage = np.dstack([normimage, alpha])
    return normimage


def gamma_image(image, gamma):
    dtype = image.dtype
    maxval = 1
    if dtype.kind != 'f':
        maxval = 2 ** (8 * dtype.itemsize) - 1
        image /= np.float32(maxval)
    # option: use discrete cv lookup table (LUT) for faster processing
    image = image ** (1 / gamma)
    if dtype.kind != 'f':
        image = (image * maxval).astype(dtype)
    return image


def blur_image_single(image, sigma):
    return gaussian_filter(image, sigma)


def blur_image(image, sigma):
    nchannels = image.shape[2] if image.ndim == 3 else 1
    if nchannels not in [1, 3]:
        new_image = np.zeros_like(image)
        for channeli in range(nchannels):
            new_image[..., channeli] = blur_image_single(image[..., channeli], sigma)
    else:
        new_image = blur_image_single(image, sigma)
    return new_image


def detect_edges(image):
    image8 = uint8_image(grayscale_image(image))
    level = np.std(image8) * 3
    return cv.Canny(image8, 0, level)


def detect_edges2(image):
    return sobel(grayscale_image(image))


def gaussian_laplace_image(image, sigma):
    #kernel = -create_gaussian_log_kernel(round(6 * sigma)+1, round(6 * sigma)+1, sigma)
    #output_image = cv.filter2D(image, cv.CV_32F, kernel)
    output_image = norm_image_minmax(gaussian_laplace(image, sigma))
    return output_image


def cdf_based_threshold_image(image, quantile):
    threshold = np.quantile(image, quantile)
    bin_image = (image > threshold).astype(np.float32)
    return bin_image


def distance_image(image0):
    image = 255 - float2int_image(image0)
    dist = cv.distanceTransform(image, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    return dist


def dist_watershed_image(image):
    dist = distance_image(image)
    _, init_labels = cv.connectedComponents(float2int_image(image))
    dist_labels0 = cv.watershed(color_image(dist.astype(np.uint8)), init_labels.copy())
    return dist_labels0


def watershed_image(image0):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Watershed_Algorithm_Marker_Based_Segmentation_2.php
    image = 255 - float2int_image(image0)
    markers0 = float2int_image(image0 == 0)
    _, markers = cv.connectedComponents(markers0)
    labels = cv.watershed(color_image(image), markers)
    return labels


def dist_watershed_image2(image):
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    distance = distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=float2int_image(image))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels


def clahe_image(image):
    is_float = (image.dtype.kind == 'f')
    if is_float:
        image = float2int_image(image)
    clahe = cv.createCLAHE()
    out_image = clahe.apply(image)
    if is_float:
        out_image = int2float_image(out_image)
    return out_image


def grad_mag_n_image(image):
    gradx = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3)
    grady = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3)
    gradmagn = np.sqrt(np.square(gradx) + np.square(grady))
    return gradmagn


def impose_min_image(image, minima):
    # https://stackoverflow.com/questions/70251361/imextendedmin-and-imimposemin-functions-in-python-for-watershed-seeds-from-dista
    marker = np.full(image.shape, np.inf)
    marker[minima] = 0
    mask = np.minimum((float2int_image(image) + 1), marker)
    return reconstruction(marker, mask, method='erosion')


def color_label_image(image, label_image0):
    label_image = label_image0 + 1
    n_labels = np.max(label_image + 1)
    lut = np.array(create_color_table(n_labels))
    lut[0] = (0, 0, 0)
    color_label_image = lut[label_image]
    float_image = np.atleast_3d(int2float_image(image))
    out_image = float2int_image(color_label_image * float_image)
    return out_image


def reshape_image(image, target_size):
    # target_size: single value or [x, y]
    # create black border to avoid opencv padding lines
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    center = np.flip(image.shape[:2]) / 2
    if len(image.shape) == 3 and image.shape[2] == 4:
        reshaped_image0 = cv.getRectSubPix(image[..., :3], np.int0(target_size), center)
        reshaped_image_alpha = cv.getRectSubPix(image[..., 3], np.int0(target_size), center)
        reshaped_image = np.dstack([reshaped_image0, reshaped_image_alpha])
    else:
        reshaped_image = cv.getRectSubPix(image, np.int0(target_size), center)
    return reshaped_image


def reshape_grow_image(image, target_size, center=True):
    # target_size: single value or [x, y]
    dx, dy = target_size - get_image_size(image)
    if center:
        xx, yy = divmod(dx, 2), divmod(dy, 2)
        top, bottom, left, right = yy[0], sum(yy), xx[0], sum(xx)
    else:
        top, bottom, left, right = 0, dy, 0, dx
    reshaped_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
    return reshaped_image


def rotate_image(image, angle, center):
    (h, w) = image.shape[:2]

    transform = cv.getRotationMatrix2D(center, angle, 1)
    cos = np.abs(transform[0, 0])
    sin = np.abs(transform[0, 1])

    # compute the new bounding dimensions of the image
    nW = ensure_even(int((h * sin) + (w * cos)))
    nH = ensure_even(int((h * cos) + (w * sin)))

    # adjust the rotation matrix to take into account translation
    transform[0, 2] += (nW / 2) - center[0]
    transform[1, 2] += (nH / 2) - center[1]

    # perform the actual rotation and return the image
    return cv.warpAffine(image, transform, (nW, nH))


def get_image_crop_polygon(source, polygon, pixel_size=None):
    polygon_min, polygon_max = np.min(polygon, 0), np.max(polygon, 0)
    w, h = np.ceil(polygon_max - polygon_min).astype(int)
    x, y = polygon_min
    cropped = get_image_crop(source, x, y, w, h, pixel_size=pixel_size)
    # use mask for alpha channel only
    polygon1 = polygon - (x, y)
    mask = get_contour_mask(polygon1, shape=(h, w), dtype=cropped.dtype)
    alpha_image = np.dstack([color_image(cropped), mask])
    return alpha_image


def create_divided_image_masks(contour, image_shape, nsections, min_distance):
    masks = []
    mask = get_contour_mask(contour, shape=image_shape, dtype=np.uint8, color=255)
    dist = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    local_maxs = peak_local_max(dist, labels=mask, num_peaks=nsections, min_distance=int(min_distance))
    if len(local_maxs) < nsections:
        return create_divided_image_masks_moments(contour, image_shape, nsections)

    markers = np.zeros_like(dist)
    for point in local_maxs:
        markers[tuple(point)] = 1

    #preview = color_image(mask * 127)
    #cv.polylines(preview, [np.flip(local_maxs, 1)], False, (255, 0, 0), 1, cv.LINE_AA)
    #for point in local_maxs:
    #    cv.drawMarker(preview, np.flip(point), (255, 255, 0), cv.MARKER_CROSS, 4, 1, cv.LINE_AA)
    #show_image(preview)

    markers = label(markers, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist, markers, mask=mask)
    #show_image(labels, cmap='rainbow')

    # extract masks
    for sectioni in range(nsections):
        label_mask = np.zeros_like(mask)
        label_mask[labels == (sectioni + 1)] = 1
        masks.append(label_mask)
    return masks


def create_divided_image_masks_moments(contour, image_shape, nsections):
    masks = []
    rotated_rect = cv.minAreaRect(contour)
    center0, size0, angle = rotated_rect
    angle_rad = np.deg2rad(angle)
    length_dim = np.argmax(size0)
    length_part = size0[length_dim] / nsections
    size = np.array(size0)
    size[length_dim] = length_part
    if length_dim == 1:
        angle_rad += np.pi / 2
    delta = np.array((np.cos(angle_rad), np.sin(angle_rad))) * size0[length_dim]
    start_point = np.array(center0) - delta / 2

    delta /= nsections
    for sectioni in range(nsections):
        center = start_point + delta / 2
        box = cv.boxPoints((center, size, angle))
        mask = get_contour_mask(box, shape=image_shape)
        masks.append(mask)
        start_point += delta
    return masks


def get_contours(binimage):
    #contours0 = cv.findContours(binimage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # uses 8-connectivity
    contours0 = cv.findContours(binimage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # uses 8-connectivity
    contours = contours0[0] if len(contours0) == 2 else contours0[1]
    contours = list(contours)
    # reorder top y contours first:
    contours.sort(key=lambda c: np.min(c[:, :, 1]))

    #nlabels, labels = cv.connectedComponents(binimage, connectivity=4)
    #contours = [cv.findContours(float2int_image(labels == labeli), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0][0]
    #            for labeli in range(1, nlabels)]

    return contours


def get_approx_shape(contour, detect_shape='rect'):
    if detect_shape.lower().startswith('rect'):
        rotated_rect = cv.minAreaRect(contour)
        approx_contour = cv.boxPoints(rotated_rect)
    elif detect_shape.lower().startswith('circ'):
        # TODO: untested
        center, radius = cv.minEnclosingCircle(contour)
        npoints = 16
        approx_contour = [(math.cos(2 * math.pi / npoints * x) * radius,
                           math.sin(2 * math.pi / npoints * x) * radius) for x in range(0, npoints + 1)]
    else:
        approx_contour = contour
    return approx_contour


def get_approx_shape_corners(contour, detect_corners):
    approx_contour = contour
    lb = 0.01
    ub = 0.1
    it = 0
    max_it = 10
    while len(approx_contour) != detect_corners:
        relative_epsilon = (lb + ub) / 2
        epsilon = relative_epsilon * cv.arcLength(contour, True)
        approx_contour = squeeze_contour(cv.approxPolyDP(contour, epsilon, True))
        if len(approx_contour) > detect_corners:
            lb = (lb + ub) / 2
        elif len(approx_contour) < detect_corners:
            ub = (lb + ub) / 2
        it += 1
        if it > max_it:
            break
    return approx_contour


def compare_image(image0, image1, show=False):
    dif, dif_max, dif_mean, psnr = compare_image_dist(image0, image1)
    print(f'rgb dist max: {dif_max:.1f} mean: {dif_mean:.1f} PSNR: {psnr:.1f}')
    if show:
        show_image(dif)
        show_image((dif * 10).astype(np.uint8))
    return dif


def compare_image_dist(image0, image1):
    dif = cv.absdiff(image0, image1)
    psnr = cv.PSNR(image0, image1)
    if dif.size > 1000000000:
        # split very large array
        rgb_maxs = []
        rgb_means = []
        for dif1 in np.array_split(dif, 16):
            rgb_dif = np.linalg.norm(dif1, axis=2)
            rgb_maxs.append(np.max(rgb_dif))
            rgb_means.append(np.mean(rgb_dif))
        rgb_max = np.max(rgb_maxs)
        rgb_mean = np.mean(rgb_means)
    else:
        rgb_dif = np.linalg.norm(dif, axis=2)
        rgb_max = np.max(rgb_dif)
        rgb_mean = np.mean(rgb_dif)
    return dif, rgb_max, rgb_mean, psnr


def test_detect_contour(contour0, image=None):
    min_contour = np.min(contour0, 0)
    max_contour = np.max(contour0, 0)
    size = (max_contour - min_contour) * 2
    thickness = int(np.ceil(np.mean(size) / 200))
    contour = contour0 - np.min(contour0, 0) + size * 0.25
    if image is None:
        shape = np.flip(size).astype(int)
        image = get_contour_mask(contour, shape=shape, color=0.5)
    else:
        image = reshape_image(image, size)
    back_image = color_image(float2int_image(image))
    out_image = np.zeros_like(back_image)

    center, lengths, angle = get_rotated_rect(contour)
    contour_moments = center, lengths, norm_rotation_angle(angle)
    print('contour moments:', contour_moments)
    box = cv.boxPoints((center, lengths, angle))
    cv.drawContours(out_image, [box.astype(int)], 0, color=(0, 255, 0), thickness=thickness)

    center, lengths, angle = get_rotated_rect(image)
    image_moments = center, lengths, norm_rotation_angle(angle)
    print('image moments:', image_moments)
    box = cv.boxPoints((center, lengths, angle))
    cv.drawContours(out_image, [box.astype(int)], 0, color=(255, 255, 0), thickness=thickness)

    if len(contour) > 4:
        ellipse = cv.fitEllipse(contour.squeeze())
        center, lengths, angle = ellipse
        rotated_ellipse = center, np.flip(lengths), -angle
        print('ellipse:', rotated_ellipse)

        cv.ellipse(out_image, ellipse, color=(255, 0, 0), thickness=thickness)
        #cv.ellipse(out_image, np.int0(center), np.int0(lengths) // 2, angle, 0, 360, color=(255, 0, 0))

    out_image = cv.addWeighted(back_image, 1, out_image, 0.8, gamma=0)

    return out_image


def get_contour_mask(contour, image=None, shape=None, dtype=np.float32, color=None, smooth=False):
    if image is None:
        image = np.zeros(shape, dtype=dtype)
    line_type = cv.LINE_AA if smooth else None
    if color is None:
        if image.dtype.kind == 'f':
            color = 1.0
        else:
            color = 2 ** (8 * image.dtype.itemsize) - 1
    mask = cv.drawContours(image, [np.round(contour).astype(int)], -1, color, thickness=cv.FILLED, lineType=line_type)
    return mask


def get_contour_points(binimage, min_area=1, max_area=None):
    contours = get_contours(binimage)
    area_contours = [(contour, cv.contourArea(contour)) for contour in contours]
    area_contours.sort(key=lambda contour_area: contour_area[1], reverse=True)
    area_points = [(get_center(contour), area) for contour, area in area_contours
                   if area >= min_area and (max_area is None or area <= max_area)]
    return area_points


def get_contour_points_regionprops(binimage, min_area=1, max_area=None):
    source_props = regionprops(label(binimage)[0])
    area_points0 = [(np.flip(prop.centroid), prop.area) for prop in source_props]
    #area_points0.sort(key=lambda area_points: area_points[1], reverse=True)
    area_points = [(point, area) for point, area in area_points0
                   if area >= min_area and (max_area is None or area <= max_area)]
    return area_points


def create_image_overlay(image1, image2):
    shape0 = list(np.max([image1.shape[:2], image2.shape[:2]], 0))
    shape = list(shape0) + [3]
    size = np.flip(shape0)
    image = np.zeros(shape, dtype=image1.dtype)
    image[..., 0] = grayscale_image(reshape_image(image1, size))     # red channel
    image[..., 2] = grayscale_image(reshape_image(image2, size))     # blue channel
    return image


def draw_text(image, text, position=None, font_size=1, thickness=1, color=(127, 127, 127)):
    if position is None:
        position = (0, font_size * 24)
    if len(image.shape) > 2 and image.shape[2] == 4:
        color = list(color) + [255]
    cv.putText(image, text, (int(position[0]), int(position[1])), cv.FONT_HERSHEY_SIMPLEX, int(np.ceil(font_size)),
               color, thickness, lineType=cv.LINE_AA)


def show_histogram(data, range=(0, 1), bins=100, title='', show=True, out_filename=None):
    plt.figure()
    plt.hist(data, range=range, bins=bins)
    if title != '':
        plt.title(title)
    if out_filename is not None:
        plt.savefig(out_filename)
    if show:
        plt.show()
    plt.close()


def draw_points(points):
    x, y = zip(*points)
    plt.scatter(x, y)
    plt.show()


def draw_point_sets(points1, points2, matches=None, title='', show=True):
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    # 90 degrees rotation, negative Y
    if len(points1) > 0:
        plt.scatter(points1[:, 1], -points1[:, 0], color=[1, 0, 0, 0.5], edgecolors='none')
    if len(points2) > 0:
        plt.scatter(points2[:, 1], -points2[:, 0], color=[0, 0, 1, 0.5], edgecolors='none')
    if matches is not None and len(matches) > 0:
        plt.scatter(matches[:, 1], -matches[:, 0], color=[0, 0, 0, 1], marker='.', s=1)
    if title != '':
        plt.title(title)
    if show:
        plt.show()
    return plt
