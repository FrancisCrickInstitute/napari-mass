# https://pypi.org/project/tifffile/


from enum import Enum
import numpy as np
import tifffile
from tifffile import TiffFile, TiffPage

from src.napari_mass.file.FileDict import FileDict
from src.napari_mass.OmeSource import OmeSource
from src.napari_mass.TiffTileMetadata import TiffTileMetadata, sort_tiles, calc_tiles_stats
from src.napari_mass.image.normalisation import *
from src.napari_mass.image.util import *
from src.napari_mass.Stitcher import *
from src.napari_mass.util import *


class TiffTileSource(OmeSource):
    """Tiled Tiff-compatible image source"""

    filenames: list
    """original filename"""

    def __init__(self,
                 filenames: list,
                 composition_metadata: dict,
                 max_stitch_offset_um: float,
                 stitching_filename: str = '',
                 stitch_channel: str = '',
                 channel: str = '',
                 source_pixel_size: list = None,
                 target_pixel_size: list = None,
                 source_info_required: bool = False,
                 flatfield_filename: str = ''):

        super().__init__()
        self.filenames = filenames
        self.stitching = FileDict(stitching_filename)
        self.composition_metadata = composition_metadata
        if flatfield_filename is not None and flatfield_filename != '':
            self.flatfield_image = load_tiff(flatfield_filename)
        else:
            self.flatfield_image = None

        filename = filenames[0]
        tile_rects = []
        width0, height0 = 0, 0
        for tile_meta in self.composition_metadata:
            bounds = tile_meta['Bounds']
            index = bounds['StartM']
            tilex0, tiley0 = bounds['StartX'], bounds['StartY']
            tilex1, tiley1 = tilex0 + bounds['SizeX'], tiley0 + bounds['SizeY']
            width0, height0 = max(tilex1, width0), max(tiley1, height0)
            tile_filename = tile_meta['Filename']
            for filename0 in filenames:
                if tile_filename in filename0:
                    tile_filename = filename0
            tile_rects.append(TiffTileMetadata(index, tilex0, tiley0, tilex1, tiley1, tile_filename))
        self.tile_rects = sort_tiles(tile_rects)
        self.tile_stats = calc_tiles_stats(tile_rects)
        self.tile_stats_overlap = self.tile_stats[2]
        [rect.set_position(self.tile_stats[0]) for rect in self.tile_rects]

        tiff = TiffFile(filename)
        if tiff.is_ome and tiff.ome_metadata is not None:
            xml_metadata = tiff.ome_metadata
            self.metadata = tifffile.xml2dict(xml_metadata)
            if 'OME' in self.metadata:
                self.metadata = self.metadata['OME']
                self.has_ome_metadata = True
        elif tiff.is_imagej:
            self.metadata = tiff.imagej_metadata

        self.pages = get_tiff_pages(tiff)
        self.factors = []
        first_page = None
        for page0 in self.pages:
            npages = len(page0)
            if isinstance(page0, list):
                page = page0[0]
            else:
                page = page0
            if first_page is None:
                first_page = page
            shape = page.shape
            factor = np.flip(page.shape) / np.flip(first_page.shape)
            width, height = ((width0, height0) * factor).astype(int)
            if isinstance(page, TiffPage):
                depth = page.imagedepth * npages
                bitspersample = page.bitspersample
            else:
                depth = npages
                bitspersample = page.dtype.itemsize * 8
            nchannels = shape[2] if len(shape) > 2 else 1
            nt = 1
            if tiff.is_ome:
                pixels = self.metadata.get('Image', {}).get('Pixels', {})
                depth = int(pixels.get('SizeZ', depth))
                nchannels = int(pixels.get('SizeC', nchannels))
                nt = int(pixels.get('SizeT', nt))
            self.sizes.append((width, height))
            self.sizes_xyzct.append((width, height, depth, nchannels, nt))
            self.pixel_types.append(page.dtype)
            self.pixel_nbits.append(bitspersample)
            self.factors.append(factor)

        self._init_metadata(filename,
                            source_pixel_size=source_pixel_size,
                            target_pixel_size=target_pixel_size,
                            source_info_required=source_info_required)

        self.stitch_channel = self.set_channel(stitch_channel, set_channel=False)
        self.set_channel(channel)
        pixel_size_um = np.mean(self.get_pixel_size_micrometer()[:2])
        self.max_stitch_offset = max_stitch_offset_um / pixel_size_um

    def set_channel(self, channel, set_channel=True):
        new_channel = None
        if channel != '':
            if isinstance(channel, int):
                new_channel = channel
            else:
                for channeli, channel0 in enumerate(self.channels):
                    if channel.lower() in channel0.get('Name', '').lower():
                        new_channel = channeli
        if set_channel:
            self.channel = new_channel
        return new_channel

    def _find_metadata(self):
        pixel_size = []
        pixel_size_unit = ''
        channels = []
        mag = 0
        page = self.pages[0]
        if isinstance(page, list):
            page = page[0]
        # from OME metadata
        if page.is_ome:
            self._get_ome_metadate()
            return

        # from imageJ metadata
        pixel_size_z = None
        if len(pixel_size) == 0 and self.metadata is not None and 'spacing' in self.metadata:
            pixel_size_unit = self.metadata.get('unit', '')
            pixel_size_z = (self.metadata['spacing'], pixel_size_unit)
        if mag == 0 and self.metadata is not None:
            mag = self.metadata.get('Mag', 0)
        # from page TAGS
        metadata = tags_to_dict(page.tags)
        if len(pixel_size) < 2:
            if pixel_size_unit == '':
                pixel_size_unit = metadata.get('ResolutionUnit', '')
                if isinstance(pixel_size_unit, Enum):
                    pixel_size_unit = pixel_size_unit.name
                pixel_size_unit = pixel_size_unit.lower()
                if pixel_size_unit == 'none':
                    pixel_size_unit = ''
            res0 = metadata.get('XResolution')
            if res0 is not None:
                if isinstance(res0, tuple):
                    res0 = res0[0] / res0[1]
                if res0 != 0:
                    pixel_size.append((1 / res0, pixel_size_unit))
            res0 = metadata.get('YResolution')
            if res0 is not None:
                if isinstance(res0, tuple):
                    res0 = res0[0] / res0[1]
                if res0 != 0:
                    pixel_size.append((1 / res0, pixel_size_unit))
        if len(channels) == 0:
            nchannels = self.sizes_xyzct[0][3]
            photometric = str(metadata.get('PhotometricInterpretation', '')).lower().split('.')[-1]
            channels = [{'Name': photometric, 'SamplesPerPixel': nchannels}]
        if mag == 0:
            mag = metadata.get('Mag', 0)
        # from description
        if not page.is_ome:
            metadata = desc_to_dict(page.description)
            if mag == 0:
                mag = metadata.get('Mag', metadata.get('AppMag', 0))
            if len(pixel_size) < 2 and 'MPP' in metadata:
                pixel_size.append((metadata['MPP'], 'µm'))
                pixel_size.append((metadata['MPP'], 'µm'))
        if pixel_size_z is not None and len(pixel_size) == 2:
            pixel_size.append(pixel_size_z)
        self.source_pixel_size = pixel_size
        self.source_mag = mag
        self.channels = channels

    def get_asarray(self, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        return normalise_channels(self, self.asarray(x0, y0, x1, y1))

    def get_asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1, render_rgb: bool = False):
        return normalise_channels(self, self._asarray_level(level, x0, y0, x1, y1), render_rgb=render_rgb)

    def _asarray_level(self, level: int, x0: float = 0, y0: float = 0, x1: float = -1, y1: float = -1) -> np.ndarray:
        factor = self.factors[level]
        # define corresponding tiles
        if x0 == 0 and y0 == 0 and (x1 < 0 or y1 < 0 or np.allclose((x1, y1), self.sizes[0][:2], atol=4)):
            # active stitching for large array currently not supported
            do_stitching = False
            x1, y1 = self.sizes[0]
            rects = self.tile_rects
            start_rect = TiffTileMetadata(0, 0, 0, 0, 0, '')
            end_rect = TiffTileMetadata(1, x1, y1, x1, y1, '')
            w, h = int(x1 * factor[0]), int(y1 * factor[1])
        else:
            do_stitching = True
            start_rect, end_rect = None, None
            rects = []
            # see if matching single whole rect
            for rect in self.tile_rects:
                if rect.x0 == x0 and rect.y0 == y0 and rect.x1 == x1 and rect.y1 == y1:
                    start_rect = rect
                    end_rect = rect
                    rects = [rect]

            if len(rects) == 0:
                for rect in self.tile_rects:
                    if rect.inside(x0, y0) and (start_rect is None or rect < start_rect):
                        start_rect = rect
                    if rect.inside(x1, y1) and (end_rect is None or rect > end_rect):
                        end_rect = rect
                for rect in self.tile_rects:
                    if start_rect <= rect <= end_rect:
                        rects.append(rect)

            w = int((end_rect.x1 - start_rect.x0) * factor[0])
            h = int((end_rect.y1 - start_rect.y0) * factor[1])
        # create output canvas
        size_xyzct = self.sizes_xyzct[level]
        nchannels0 = size_xyzct[3]
        nchannels = nchannels0 if self.channel is None else 1
        n = size_xyzct[2] * nchannels
        dtype = self.pixel_types[level]
        shape = [h, w]
        if n > 1:
            shape += [n]
        out = np.zeros(shape, dtype)

        images = []
        for rect in rects:
            image = TiffFile(rect.filename).asarray(level=level)
            if image.ndim > 2 and image.shape[0] == nchannels0:
                image = np.moveaxis(image, 0, -1)
            if self.flatfield_image is not None:
                image = float2int_image(flatfield_correction(int2float_image(image), self.flatfield_image), dtype=dtype)
            if self.channel is not None:
                image = image[..., self.channel]
            images.append(image)

        if len(rects) > 1:
            # perform stitching using caching
            overlap = (self.tile_stats_overlap * factor).astype(int)
            stitcher = Stitcher(rects, start_rect, end_rect, images, overlap, self.max_stitch_offset, self.stitch_channel, do_stitching)
            offsets_label = '_'.join(map(str, sorted([rect.index for rect in rects])))
            if offsets_label in self.stitching:
                stitching = self.stitching[offsets_label]
                stitcher.process_stored(out, stitching['order'], stitching['offsets'])
            else:
                order, offsets, confidence = stitcher.process(out)
                self.stitching[offsets_label] = {
                    'order': order, 'offsets': offsets, 'confidence': confidence
                }
                self.stitching.save()
        elif len(rects) == 1:
            out = images[0]

        # extract section from stitched image
        target_x0 = int((x0 - start_rect.x0) * factor[0])
        target_y0 = int((y0 - start_rect.y0) * factor[1])
        dw = int((x1 - x0) * factor[0])
        dh = int((y1 - y0) * factor[1])
        if target_x0 < 0:
            target_x0 = 0
        if target_x0 + dw >= w:
            target_x0 = w - dw
        if target_y0 < 0:
            target_y0 = 0
        if target_y0 + dh >= h:
            target_y0 = h - dh
        image = out[target_y0: target_y0 + dh, target_x0: target_x0 + dw]
        if image.ndim == 3 and image.shape[-1] == 1:
            return image[..., 0]
        else:
            return image
