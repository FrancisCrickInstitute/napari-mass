import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from napari_mass.image.color_conversion import hexrgb_to_rgba
from napari_mass.image.normalisation import *
from napari_mass.OmeSource import OmeSource


class ZarrSource(OmeSource):
    """Zarr-compatible image source"""

    filename: str
    """original filename / URL"""
    levels: list
    """list of all image arrays for different sizes"""

    def __init__(self, filename: str,
                 channel: str = '',
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False):

        super().__init__()

        try:
            root = zarr.open_group(filename, mode='r')
            self.metadata = root.attrs.asdict()

            paths = []
            dimension_order = 'tczyx'
            if 'multiscales' in self.metadata:
                for scale in self.metadata.get('multiscales', []):
                    for index, dataset in enumerate(scale.get('datasets', [])):
                        paths.append(dataset.get('path', str(index)))
                    axes = scale.get('axes', [])
                    if len(axes) > 0:
                        dimension_order = ''.join([axis.get('name') for axis in axes])
            else:
                paths = root.array_keys()
            self.dimension_order = dimension_order

            self.levels = []
            for path in paths:
                data = root.get(path)
                self.levels.append(data)

                xyzct = [1, 1, 1, 1, 1]
                for i, n in enumerate(data.shape):
                    xyzct_index = 'xyzct'.index(dimension_order[i])
                    xyzct[xyzct_index] = n
                self.sizes_xyzct.append(xyzct)
                self.sizes.append((xyzct[0], xyzct[1]))
                self.pixel_types.append(data.dtype)
                self.pixel_nbits.append(data.dtype.itemsize * 8)
        except GroupNotFoundError as e:
            raise FileNotFoundError(f'Read error: {e}')

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

        self.channel = None
        if channel != '':
            if isinstance(channel, int):
                self.channel = channel
            else:
                for channeli, channel0 in enumerate(self.channels):
                    if channel.lower() in channel0.get('Name', '').lower():
                        self.channel = channeli

        init_channel_levels(self)

    def _find_metadata(self):
        pixel_size = []
        channels = []
        for scale in self.metadata.get('multiscales', []):
            axes = ''.join([axis.get('name', '') for axis in scale.get('axes', [])])
            units = [axis.get('unit', '') for axis in scale.get('axes', [])]
            scale1 = [0, 0, 0, 0, 0]
            datasets = scale.get('datasets')
            if datasets is not None:
                coordinateTransformations = datasets[0].get('coordinateTransformations')
                if coordinateTransformations is not None:
                    scale1 = coordinateTransformations[0].get('scale', scale1)
            for axis in 'xyz':
                if axis in axes:
                    index = axes.index(axis)
                    pixel_size.append((scale1[index], units[index]))
                else:
                    pixel_size.append((1, ''))
        nchannels = self.sizes_xyzct[0][3]
        for data in self.metadata.values():
            if isinstance(data, dict):
                n = len(data.get('channels', []))
                for channel0 in data.get('channels', []):
                    channel = {'Name': channel0.get('label'), 'SamplesPerPixel': nchannels // n}
                    if 'color' in channel0:
                        channel['Color'] = hexrgb_to_rgba(channel0['color'])
                    channels.append(channel)
        if len(channels) == 0:
            if nchannels == 3:
                channels = [{'Name': '', 'SamplesPerPixel': nchannels}]
            else:
                channels = [{'Name': '', 'SamplesPerPixel': 1}] * nchannels
        self.source_pixel_size = pixel_size
        self.channels = channels
        self.source_mag = 0

    def get_asarray(self, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        return normalise_channels(self, self.asarray(x0, y0, x1, y1))

    def get_asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1, render_rgb: bool = False):
        return normalise_channels(self, self._asarray_level(level, x0, y0, x1, y1), render_rgb=render_rgb)

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        size_xyzct = self.sizes_xyzct[level]
        if x1 < 0 or y1 < 0:
            x1, y1, _, _, _ = size_xyzct
        if self.dimension_order.endswith('yx'):
            image = self.levels[level][..., y0:y1, x0:x1].squeeze()
            if len(image.shape) > 2:
                image = np.moveaxis(image, 0, -1)  # move axis 0 (channel/z) to end
        else:
            image = self.levels[level][y0:y1, x0:x1].squeeze()
        return image
