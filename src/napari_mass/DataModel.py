# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# https://coderedirect.com/questions/260685/opencv-c-rectangle-detection-which-has-irregular-side
# https://medium.com/temp08050309-devpblog/cv-5-canny-edge-detector-with-image-gradients-a42f07dc69c
# https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://medium.vaningelgem.be/traveling-salesman-problem-mlrose-2opt-85b765976a6e
# https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# https://stackoverflow.com/questions/43126580/match-set-of-x-y-points-to-another-set-that-is-scaled-rotated-translated-and
# https://github.com/dmishin/tsp-solver

# - get sections from fluor image
# - transform orientation of sections
# - map/order sections based on pattern in fluor image
# - expand sections to include entire ROI section
#   - may require manual annotation of single section
#   - expand new section coordinates
#   - based on mapping/alignment to find correct coordinates
# - define landmark positions


import cv2 as cv
import glob
import logging
from napari.layers.base import ActionType
from napari_ome_zarr._reader import napari_get_reader
import numpy as np
import os.path
from tifffile import xml2dict

from napari_mass.TiffSource import TiffSource
from napari_mass.file.FileDict import FileDict
from napari_mass.file.DataFile import DataFile
from napari_mass.Point import Point
from napari_mass.Section import Section, get_section_sizes, get_section_images
from napari_mass.TiffTileSource import TiffTileSource
from napari_mass.OmeZarrSource import OmeZarrSource
from napari_mass.parameters import *
from napari_mass.image.util import *
from napari_mass.util import *


class DataModel:
    LAYER_COLORS = {
        'magnet': 'limegreen',
        'sample': 'cyan',
        'roi': 'yellow',
        'focus': 'red',
        'landmark': 'blue'
    }

    SHAPE_THICKNESS = 10
    POINT_SIZE = 50

    def __init__(self, params):
        self.params = params
        self.params_init_done = False
        self.init_done = False
        self.debug = True

        self.data = DataFile()
        self.matches = FileDict()
        self.transforms = FileDict()

        self.magsections_initialised = False
        self.magnet_template_initialised = False
        self.sample_template_initialised = False

        self.colors = create_color_table(1000)
        self.sections = []
        self.section_images = []

    def set_params(self, params):
        self.params = params
        self.params_init_done = True

    def init(self):
        logging.info('Initialising output')

        params = self.params
        output_params = params['project']['output']

        base_folder = os.path.dirname(get_dict_value(params, 'project.filename'))

        self.outfolder = validate_out_folder(base_folder, get_dict_value(output_params, 'folder'))

        data_filename = join_path(self.outfolder, get_dict_value(output_params, 'datafile'))
        self.data = DataFile(data_filename)

        matching_filename = join_path(self.outfolder, 'matching.json')
        self.matches = FileDict(matching_filename)

        transforms_filename = join_path(self.outfolder, 'transforms.json')
        self.transforms = FileDict(transforms_filename)

        self.init_done = True

    def init_layers(self):
        logging.info('Initialising layers')

        self.layers = []
        self.layers.extend(self.init_image_layers())
        if self.layers:
            self.layers.extend(self.init_data_layers().values())
        return self.layers

    def init_image_layers(self):
        image_layers = []
        params = self.params
        input_params = params['input']
        output_params = params['project']['output']

        project_filename = get_dict_value(params, 'project.filename')
        if project_filename:
            base_folder = os.path.dirname(project_filename)
        else:
            base_folder = None

        self.output_pixel_size = get_value_units_micrometer(split_value_unit_list(
                                    get_dict_value(output_params, 'pixel_size', '2um')))

        # main image
        source_filenames = []
        source_filename = get_dict_value(input_params, 'source', '')
        if source_filename:
            source_filenames.append(source_filename)
        source_filename = get_dict_value(input_params, 'source2', '')
        if source_filename:
            source_filenames.append(source_filename)

        if len(source_filenames) == 0:
            logging.warning('Source not set')
            return []

        sources = []

        channel_names = []
        scales = []
        blendings = []
        contrast_limits = []
        contrast_limits_range = []
        colormaps = []
        visibles = []

        small_image = None

        image_layers = []

        for source_filename in source_filenames:
            source = get_source(base_folder, source_filename)
            source_pixel_size = source.get_pixel_size_micrometer()
            if len(source_pixel_size) == 0:
                source_pixel_size = [1, 1]
            sources.append(source)
            self.source_pixel_size = source_pixel_size
            source_rendered = source.render(source.asarray(pixel_size=self.output_pixel_size))
            if small_image is None:
                small_image = source_rendered
            elif source_rendered.shape == small_image.shape:
                dtype = small_image.dtype
                small_image = ((small_image + source_rendered) / 2).astype(dtype)
            else:
                logging.error('Image shapes do not match')

            if 'c' in source.dimension_order and not source.is_rgb:
                # channels_axis appears to be incompatible with RGB channels
                c_index = source.dimension_order.index('c')
            else:
                c_index = None

            # set image layers
            if isinstance(source, OmeZarrSource):
                path = join_path(base_folder, source_filename)
                reader = napari_get_reader(path)
                image_layers.extend(reader(path))
            else:
                data = source.get_source_dask()
                source_pixel_size = source_pixel_size[:2]
                channels = source.get_channels()
                nchannels = len(channels)

                for channeli, channel in enumerate(channels):
                    channel_name = channel.get('label')
                    if not channel_name:
                        channel_name = get_filetitle(source.source_reference)
                        if nchannels > 1:
                            channel_name += f' #{channeli}'
                    blending_mode = 'additive'
                    visible = True
                    channel_color = channel.get('color')
                    if channel_color:
                        channel_color = tuple(channel_color)
                    window = source.get_channel_window(channeli)
                    window_limit = window['min'], window['max']
                    window_range = window['start'], window['end']
                    if nchannels > 1:
                        channel_names.append(channel_name)
                        blendings.append(blending_mode)
                        contrast_limits.append(window_limit)
                        contrast_limits_range.append(window_range)
                        colormaps.append(channel.get('color'))
                        scales.append(source_pixel_size)
                        visibles.append(visible)
                    else:
                        channel_names = channel_name
                        blendings = blending_mode
                        contrast_limits = window_limit
                        contrast_limits_range = window_range
                        colormaps = channel_color
                        scales = source_pixel_size
                        visibles = visible

                source_metadata = {'name': channel_names,
                                   'blending': blendings,
                                   'scale': scales,
                                   'contrast_limits': contrast_limits,
                                   #'contrast_limits_range': contrast_limits_range,     # not supported as parameter
                                   'colormap': colormaps,
                                   'channel_axis': c_index,
                                   'visible': visibles,
                                   'metadata': source.metadata}
                image_layers.append((data, source_metadata, 'image'))

        self.source = sources[0]
        self.small_image = small_image
        return image_layers

    def get_source_contrast_windows(self):
        windows = []
        channels = self.source.get_channels()
        for channeli, channel in enumerate(channels):
            window = self.source.get_channel_window(channeli)
            windows.append(window)
        return windows

    def init_data_layers(self, top_path=[DATA_SECTIONS_KEY, '*']):
        data_layers = {}
        input_params = self.params['input']
        for layer_name in deserialise(get_dict_value(input_params, 'layers', '')):
            data_layers[layer_name] = self.init_data_layer(layer_name, top_path)
        return data_layers

    def init_data_layer(self, layer_name, top_path=[DATA_SECTIONS_KEY, '*']):
        if layer_name == 'landmarks' and DATA_TEMPLATE_KEY not in top_path:
            path = [layer_name, '*', DATA_SOURCE_KEY]
        else:
            path = top_path + [layer_name]
            if layer_name in ['rois', 'focus']:
                path += ['*']
        values0 = self.data.get_values(path)
        value_type = self.data.get_value_type(layer_name)
        values = [np.flip(value[value_type]) for value in values0]

        if value_type == 'polygon':
            layer_type = 'shapes'
        else:
            layer_type = 'points'
        layer_color = get_dict_permissive(self.LAYER_COLORS, layer_name)

        if layer_type == 'shapes':
            metadata = {'name': layer_name, 'shape_type': 'polygon',
                        'face_color': 'transparent', 'edge_color': layer_color, 'edge_width': self.SHAPE_THICKNESS}
        else:
            metadata = {'name': layer_name,
                        'face_color': layer_color, 'edge_color': 'transparent', 'size': self.POINT_SIZE}
        data_layer = values, metadata, layer_type
        return data_layer

    def get_output_scale(self):
        pixel_size = list(self.output_pixel_size)
        if len(pixel_size) < 2:
            pixel_size = pixel_size * 2
        return pixel_size[:2]

    def section_data_changed(self, action, name, indices, values):
        if name == 'landmarks':
            path = [name, '*', DATA_SOURCE_KEY]
        else:
            path = [DATA_SECTIONS_KEY, '*', name]
        return self.data_changed(action, path, indices, values)

    def template_data_changed(self, action, name, indices, values):
        path = [DATA_TEMPLATE_KEY, name]
        return self.data_changed(action, path, indices, values)

    def data_changed(self, action, path, indices, values):
        data_order_changed = False
        modified = False
        value_type = None

        if path[-1] in ['rois', 'focus']:
            path += ['*']

        for key in path:
            value_type = self.data.get_value_type(key)
            if value_type is not None:
                break

        for index in indices:
            if action == ActionType.ADDED or action == ActionType.CHANGED:
                value = np.flip(values[index])
                if value_type == 'polygon':
                    value = Section(value)
                elif value_type == 'location':
                    value = Point(value)
                if action == ActionType.ADDED:
                    modified = self.data.add_value(path, value)
                elif action == ActionType.CHANGED:
                    modified = self.data.set_value(path, index, value)
            elif action == ActionType.REMOVED:
                modified = self.data.remove_value(path, index)
                if modified:
                    data_order_changed = True
        if modified:
            self.data.save()
        return data_order_changed

    def update_section_order(self, new_order):
        new_data = {}
        for index, key in enumerate(new_order):
            new_data[index] = self.data[DATA_SECTIONS_KEY][key]
        self.data[DATA_SECTIONS_KEY] = new_data
        self.data.save()

    def get_section_images(self, section_name):
        self.section_images = []
        self.sections = []
        values = self.data.get_values(DATA_SECTIONS_KEY + '/*/' + section_name)
        value_type = self.data.get_value_type(section_name)
        if value_type == 'polygon':
            order = self.data.get_values('serial_order/order')
            if not order:
                order = range(len(values))
            self.sections = [Section(values[index]) for index in order]
            self.section_images = get_section_images(self.sections, self.source, pixel_size=self.output_pixel_size)
        return self.section_images

    def init_sample_template(self):
        sample = self.data.get_values([DATA_TEMPLATE_KEY, 'sample'])
        if not sample:
            values = self.data.get_values([DATA_SECTIONS_KEY, '*', 'sample'])
            sections = [Section(value) for value in values]
            element_size, large_size0 = get_section_sizes(sections)
            padded_size_factor = 1
            large_size = np.multiply(large_size0, padded_size_factor)
            centre, size, rotation = (np.divide(large_size, 2), element_size.astype(float), 0)
            sample = Section(cv.boxPoints((centre, size, rotation)))
            self.data.add_value([DATA_TEMPLATE_KEY, 'sample'], sample)
            self.data.save()

    def get_template_elements(self, element_name, ref_element_name='sample'):
        ref_element = self.data.get_values([DATA_TEMPLATE_KEY, ref_element_name])[0]
        elements = self.data.get_values([DATA_TEMPLATE_KEY, element_name, '*'])
        value_type = self.data.get_value_type(element_name)
        if len(elements) > 0:
            section_points = []
            section_centers = []
            ref_center = ref_element['center']
            for element in elements:
                data = element[value_type]
                if value_type == 'polygon':
                    center = element['center']
                else:
                    center = data
                    data = [data]
                new_points = np.array(data) - ref_center
                new_center = np.array(center) - ref_center
                angle = 0
                if new_center[1] > ref_center[1]:
                    angle = norm_angle(angle + 180)
                h = create_transform(angle=angle)
                section_points.append(apply_transform(new_points, h))
                section_centers.append(apply_transform([new_center], h)[0])
            self.template_section_points = section_points
            self.template_section_centers = section_centers
            return True
        else:
            self.template_section_points = []
            self.template_section_centers = []
            return False

    # def align_sections(self, layer_name, reorder=True):
    #     distance_factor = 1
    #     detection_params = self.params.get(layer_name).copy()
    #     detection_params['min_npoints'] = 10
    #     min_match_rate = 0.1
    #     order = self.data.get_values('serial_order/order')
    #     n = len(order)
    #
    #     pixel_size = self.output_pixel_size  # use reduced pixel size
    #
    #     init_sections_features(self.sections, source=self.source, pixel_size=pixel_size,
    #                            image_function=create_brightfield_detection_image,
    #                            detection_params=detection_params)
    #
    #     if reorder:
    #         for index1, index2 in np.transpose(np.triu_indices(n, 1)):
    #             section1, section2 = self.sections[index1], self.sections[index2]
    #             metrics = get_section_alignment_metrics(section2, section1, 'cpd')[-1]
    #             # TODO: store in distance/score matrix
    #
    #     prev_section = None
    #     for sectioni in order:
    #         section = self.sections[sectioni]
    #         if prev_section is not None:
    #             transform, metrics = do_section_alignment(
    #                 section, prev_section, method='cpd', min_match_rate=min_match_rate,
    #                 pixel_size=pixel_size, distance_factor=distance_factor,
    #                 w=0.001, max_iter=200, tol=0.1)
    #             # TODO: transform has scale as well which is currently ignored
    #             if metrics['match_rate'] > min_match_rate:
    #                 # 1. adjust section using fine alignment
    #                 element = self.data['sections'][sectioni][layer_name]
    #                 element['center'] = section.center
    #                 element['angle'] = section.angle
    #
    #                 # 2. re-init points
    #                 section.init_features(self.source, pixel_size,
    #                                       create_brightfield_detection_image,
    #                                       detection_params)
    #                 flow_map, metrics2 = get_section_alignment_metrics(
    #                     section, prev_section, 'flow',
    #                     pixel_size=pixel_size, distance_factor=distance_factor,
    #                     w=0.001, max_iter=200, tol=0.1)
    #                 #print(f'match rate: {metrics["match_rate"]:.3f}',
    #                 #      f'dcenter: {math.dist(section.center, center):.3f}',
    #                 #      f'dangle: {get_angle_dif(section.angle, angle):.3f}')
    #                 # 3. store position map
    #                 #matched_section_points = [match[0] for match in metrics['matched_points']]
    #                 #matched_prev_section_points = [match[1] for match in metrics['matched_points']]
    #                 # visualise
    #                 #show_image(draw_image_points_overlay(section.image, prev_section.image,
    #                 #                                     matched_section_points, matched_prev_section_points))
    #
    #         prev_section = section
    #     self.data.save()

    def propagate_elements(self, element_name, ref_element_name='sample'):
        value_type = self.data.get_value_type(element_name)
        for section in self.data[DATA_SECTIONS_KEY].values():
            ref_element = section.get(ref_element_name)
            if ref_element:
                section[element_name] = {}
                for index, (section_points, section_center) \
                        in enumerate(zip(self.template_section_points, self.template_section_centers)):
                    # transform corners of template section (relative from center), using transform of each section
                    h = create_transform(angle=ref_element['angle'], translate=ref_element['center'])
                    new_section_center = apply_transform([section_center], h)[0]
                    if value_type == 'polygon':
                        new_section_points = apply_transform(section_points, h)
                        value = {
                            'polygon': new_section_points.tolist(),
                            'center': new_section_center.tolist(),
                            'angle': ref_element['angle']
                        }
                    else:
                        value = {
                            'location': new_section_center.tolist()
                        }
                    section[element_name][index] = value
        self.data.save()

    def draw_output(self, layer_names=[], top_path=[DATA_SECTIONS_KEY, '*'],
                    serial_order_labels=False, stage_order_labels=False):
        top_element_name = top_path[0]
        if top_element_name == 'template':
            back_image = np.mean(self.section_images, axis=0).astype(self.section_images[0].dtype)
            # TODO: remove alpha channel?
        else:
            back_image = self.small_image
        back_image = color_image(back_image)
        out_image = np.zeros_like(back_image)

        for layer_name in deserialise(get_dict_value(self.params['input'], 'layers', '')):
            if not layer_names or layer_name in layer_names:
                path = top_path + [layer_name]
                if layer_name in ['rois', 'focus']:
                    path += ['*']
                values = self.data.get_values(path)
                value_type = self.data.get_value_type(layer_name)
                for index, value in enumerate(values):
                    element = None
                    if value_type == 'polygon':
                        element = Section(value)
                    elif value_type == 'location':
                        element = Point(value)
                    if element:
                        self.draw_section_element(out_image, element, index,
                                                  serial_order_labels, stage_order_labels)

        out_image = cv.addWeighted(back_image, 1, out_image, 0.9, gamma=0)
        filename = f'{top_element_name}'
        if serial_order_labels:
            filename += '_ordered'
        elif stage_order_labels:
            filename += '_stage_ordered'
        filename += '.tiff'
        save_image(join_path(self.outfolder, filename), out_image)

    def draw_section_element(self, out_image, element, index, serial_order_labels=False, stage_order_labels=False):
        draw_size = 8
        if hasattr(element, 'confidence') and element.confidence < 1:
            color0 = np.array(confidence_color_map(element.confidence))
            color = color0[:3] * color0[3] * 255  # pre-calc alpha for black background
            get_contour_mask(element.polygon / self.output_pixel_size, image=out_image, color=color, smooth=True)

        color = color_float_to_cv(self.get_label_color(index))
        if serial_order_labels and len(self.order) > 1:
            label = self.order.index(index)
            label_color = (255, 255, 0)
        elif stage_order_labels and len(self.stage_order) > 1:
            label = self.stage_order.index(index)
            label_color = (255, 0, 255)
        else:
            label = index
            label_color = (192, 192, 192)

        element.draw(out_image, self.output_pixel_size, label,
                     color=color, label_color=label_color, draw_size=draw_size)

    def get_label_color(self, label):
        return self.colors[label % len(self.colors)]


def get_source(base_folder, input, input_filename=None):
    source = None
    channel = ''
    source_pixel_size = None
    pixel_size = None
    if input is None:
        return None
    elif isinstance(input, dict):
        filename0 = input.get('filename', input_filename)
        channel = input.get('channel', '')
        source_pixel_size = split_value_unit_list(input.get('source_pixel_size'))
        pixel_size = split_value_unit_list(input.get('pixel_size'))
    else:
        filename0 = input
    filename = join_path(base_folder, filename0)
    if os.path.exists(filename):
        ext = os.path.splitext(filename)[1]
        if 'zar' in ext:
            source = OmeZarrSource(filename, source_pixel_size=source_pixel_size, target_pixel_size=pixel_size)
        elif 'tif' in ext:
            source = TiffSource(filename, source_pixel_size=source_pixel_size, target_pixel_size=pixel_size)
        else:
            source = cv_load(filename)
            if channel.isnumeric():
                source = source[..., channel]
        s = f'Image source: {filename}'
        if channel != '':
            s += f' channel: {channel}'
        logging.info(s)
    else:
        logging.warning(f'File path does not exist: {filename}')
    return source
