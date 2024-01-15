import numpy as np

from src.napari_mass.image.util import *


min_quantile = 0.001
max_quantile = 0.999


def init_channel_levels(source):
    level = len(source.sizes) - 1
    if level == 0:
        # no downscale resolution available; use random sampling
        factor = 2 ** 4
        nchannels = source.get_nchannels()
        shape = list(np.flip(source.get_size()) // factor)
        if nchannels > 1:
            shape += [nchannels]
        image0 = np.zeros(shape=shape, dtype=source.get_pixel_type())
        tsize = 256
        for y in range(0, shape[0], tsize):
            for x in range(0, shape[1], tsize):
                sx, sy = x * factor, y * factor
                tile = source._asarray_level(level, sx, sy, sx + tsize, sy + tsize)
                x1, y1 = min(x + tsize, shape[1]), min(y + tsize, shape[0])
                image0[y:y1, x:x1] = tile[0:y1 - y, 0:x1 - x]
    else:
        image0 = source._asarray_level(level)
    for channeli, channel in enumerate(source.channels):
        image = image0[..., channeli] if image0.ndim > 2 else image0
        channel['Min'] = get_image_quantile(image, min_quantile)
        channel['Max'] = get_image_quantile(image, max_quantile)


def normalise_values(image, channel):
    min_value = channel['Min'] if 'Min' in channel else get_image_quantile(image, min_quantile)
    max_value = channel['Max'] if 'Max' in channel else get_image_quantile(image, max_quantile)
    return np.clip((image.astype(np.float32) - min_value) / (max_value - min_value), 0, 1)


def normalise_channels(source, image, render_rgb=False):
    is_rgb = (source.get_size_xyzct()[3] == 3)
    needs_normalisation = (source.pixel_types[0].itemsize > 1)
    if source.channel is not None and image.ndim > 2:
        image = image[..., source.channel]

    if render_rgb and not is_rgb:
        new_image = np.zeros(list(image.shape[:2]) + [3], dtype=np.float32)
        tot_alpha = 0
        for channeli, channel in enumerate(source.get_channels()):
            if source.channel is None or channeli == source.channel:
                if image.ndim > 2:
                    channel_values = image[..., channeli]
                else:
                    channel_values = image
                if needs_normalisation:
                    channel_values = normalise_values(channel_values, channel)
                else:
                    channel_values = int2float_image(channel_values)
                if 'Color' in channel and channel['Color'] != '':
                    rgba = color_split(channel['Color'], nchannels=4)
                else:
                    rgba = [255, 255, 255, 255]
                color = np.divide(rgba[:3], 255, dtype=np.float32)
                alpha = np.divide(rgba[3], 255, dtype=np.float32)
                if alpha == 0:
                    alpha = 1
                new_image += np.atleast_3d(channel_values) * alpha * color
                tot_alpha += alpha
        new_image = float2int_image(new_image / tot_alpha, image.dtype)
        needs_normalisation = False
    else:
        new_image = image

    if needs_normalisation:
        maxval = 2 ** (8 * new_image.dtype.itemsize) - 1
        channeli = source.channel if source.channel is not None else 0
        new_image = (normalise_values(new_image, source.get_channels()[channeli]) * maxval).astype(image.dtype)
    return new_image


def flatfield_correction(image0, dark=0, bright=1, clip=True):
    # https://imagej.net/plugins/bigstitcher/flatfield-correction
    mean_bright_dark = np.mean(bright - dark, (0, 1))
    image = (image0 - dark) * mean_bright_dark / (bright - dark)
    if clip:
        image = np.clip(image, 0, 1)
    return image
