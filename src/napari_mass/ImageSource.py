import numpy as np

from napari_mass.image.normalisation import *
from napari_mass.TiffSource import TiffSource


class ImageSource(TiffSource):
    def __init__(self, filename, channel='', source_pixel_size=None, target_pixel_size=None):
        super().__init__(filename, source_pixel_size=source_pixel_size, target_pixel_size=target_pixel_size)
        self.channel = None
        if channel != '':
            if isinstance(channel, int):
                self.channel = channel
            else:
                for channeli, channel0 in enumerate(self.channels):
                    if channel.lower() in channel0.get('Name', '').lower():
                        self.channel = channeli
        init_channel_levels(self)

    def get_asarray(self, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        return normalise_channels(self, self.asarray(x0, y0, x1, y1))

    def get_asarray_level(self, level: int = 0, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1, render_rgb=False) -> np.ndarray:
        return normalise_channels(self, self._asarray_level(level, x0, y0, x1, y1), render_rgb=render_rgb)

    def get_asarray_level_micrometer(self, level: int = 0, umx0: float = 0, umy0: float = 0, umx1: float = -1, umy1: float = -1) -> np.ndarray:
        (x0, y0), (x1, y1) = np.divide([(umx0, umy0), (umx1, umy1)], self.get_pixel_size_micrometer()[:2]).astype(int)
        return normalise_channels(self, self._asarray_level(level, x0, y0, x1, y1))
