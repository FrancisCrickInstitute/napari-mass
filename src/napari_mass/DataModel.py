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


from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2 as cv
import glob
import logging
from napari.layers.base import ActionType
from napari_ome_zarr._reader import transform
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os.path
from sklearn.metrics import euclidean_distances
from tifffile import xml2dict
from tsp_solver.greedy import solve_tsp
from tqdm import tqdm

from napari_mass.TiffSource import TiffSource
from napari_mass.file.FileDict import FileDict
from napari_mass.file.csv_file import csv_write
from napari_mass.file.DataFile import DataFile
from napari_mass.ExternalTspSolver import ExternalTspSolver
from napari_mass.point_matching import get_match_metrics, get_section_alignment, align_sections_metrics
from napari_mass.Point import Point
from napari_mass.section_detection import create_detection_image, detect_magsections, detect_nearest_edges
from napari_mass.Section import Section, init_section_features, get_section_sizes, get_section_images
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

    SHAPE_THICKNESS = 20
    POINT_SIZE = 50

    def __init__(self, params):
        self.params = params
        self.params_init_done = False
        self.init_done = False
        self.debug = True

        log_filename = join_path(validate_out_folder(None, 'log'), 'mass.log')
        # replace millisecond comma with dot:
        log_format = '%(asctime)s.%(msecs)03d %(levelname)s: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=datefmt,
                            handlers=[logging.StreamHandler(), logging.FileHandler(log_filename, encoding='utf-8')],
                            encoding='utf-8', force=True)
        logging.info('Microscopy Array Section Setup logging started')

        self.data = DataFile()
        self.matches = FileDict()
        self.transforms = FileDict()

        self.sections = {}
        self.order = []
        self.order_confidences = []
        self.stage_order = []
        self.landmarks = {}
        self.magsections_initialised = False
        self.magnet_template_initialised = False
        self.sample_template_initialised = False
        self.magsections_finalised = False
        self.order_finalised = False
        self.sample_sections_finalised = False
        self.landmarks_finalised = False

        self.colors = create_color_table(1000)

    def set_params(self, params):
        self.params = params
        self.params_init_done = True

    def init(self):
        logging.info('Initialising output')

        params = self.params
        output_params = params['project']['output']

        base_folder = os.path.dirname(get_dict(params, 'project.filename'))

        self.outfolder = validate_out_folder(base_folder, get_dict(output_params, 'folder'))

        data_filename = join_path(self.outfolder, get_dict(output_params, 'datafile'))
        self.data = DataFile(data_filename, load=False)

        matching_filename = join_path(self.outfolder, 'matching.json')
        self.matches = FileDict(matching_filename, load=False)

        transforms_filename = join_path(self.outfolder, 'transforms.json')
        self.transforms = FileDict(transforms_filename, load=False)

        self.init_done = True

    def init_layers(self):
        logging.info('Initialising layers')

        self.load_data()

        self.layers = []
        self.layers.extend(self.init_image_layers())
        if self.layers:
            self.layers.extend(self.init_data_layers().values())
        return self.layers

    def init_image_layers(self):
        params = self.params
        input_params = params['input']
        output_params = params['project']['output']

        project_filename = get_dict(params, 'project.filename')
        if project_filename:
            base_folder = os.path.dirname(project_filename)
        else:
            base_folder = None

        self.output_pixel_size = get_value_units_micrometer(split_value_unit_list(
                                    get_dict(output_params, 'pixel_size', '2um')))

        # main image
        input_filename = get_dict(input_params, 'source.filename')
        if input_filename is None or input_filename == '':
            logging.warning('Source not set')
            return []

        self.source = get_source(base_folder, input_filename)
        source = self.source
        self.source_pixel_size = source.get_pixel_size_micrometer()

        #self.small_image = source.render(self.source.asarray(pixel_size=self.output_pixel_size))

        # hq fluor image
        hq_params = input_params.get('fluor_hq')
        filename = get_dict(hq_params, 'filename')
        if filename:
            tile_path = join_path(base_folder, filename)
            channel = get_dict(hq_params, 'channel', '')
            tile_filenames = glob.glob(tile_path)
            flatfield_filename = join_path(base_folder, get_dict(hq_params, 'flatfield_filename'))
            stitching_filename = join_path(self.outfolder, 'stitching.json')
            composition_metadata_xml = open(join_path(base_folder, get_dict(hq_params, 'metadata')), encoding='utf8').read()
            composition_metadata = xml2dict(composition_metadata_xml)['ExportDocument']['Image']
            self.fluor_hq_source = TiffTileSource(tile_filenames, composition_metadata,
                                                  max_stitch_offset_um=get_dict(hq_params, 'max_stitch_offset_um'),
                                                  stitching_filename=stitching_filename,
                                                  channel=channel,
                                                  flatfield_filename=flatfield_filename)
            s = f'HQ Fluor source: {tile_path}'
            if channel != '':
                s += f' channel: {channel}'
            logging.info(s)
        else:
            self.fluor_hq_source = source

        # set image layers
        if isinstance(source, OmeZarrSource):
            reader = Reader(parse_url(join_path(base_folder, input_filename)))
            image_layers = transform(reader())()
        else:
            image_layers = []
            source_pixel_size = source.get_pixel_size_micrometer()[:2]
            data = source.get_source_dask()
            channels = source.get_channels()
            nchannels = len(channels)
            channel_names = []
            scales = []
            blendings = []
            contrast_limits = []
            contrast_limits_range = []
            colormaps = []
            for channeli, channel in enumerate(channels):
                channel_name = channel.get('label')
                if not channel_name:
                    channel_name = get_filetitle(source.source_reference)
                    if nchannels > 1:
                        channel_name += f' #{channeli}'
                blending_mode = 'additive'
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
                else:
                    channel_names = channel_name
                    blendings = blending_mode
                    contrast_limits = window_limit
                    contrast_limits_range = window_range
                    colormaps = channel_color
                    scales = source_pixel_size

            if 'c' in source.dimension_order:
                c_index = source.dimension_order.index('c')
            else:
                c_index = None
            source_metadata = {'name': channel_names,
                               'blending': blendings,
                               'scale': scales,
                               'contrast_limits': contrast_limits,
                               #'contrast_limits_range': contrast_limits_range,     # not supported as parameter
                               'colormap': colormaps,
                               'channel_axis': c_index}
            image_layers.append((data, source_metadata, 'image'))
        return image_layers

    def get_source_contrast_windows(self):
        windows = []
        channels = self.source.get_channels()
        for channeli, channel in enumerate(channels):
            window = self.source.get_channel_window(channeli)
            windows.append(window)
        return windows

    def init_data_layers(self, top_path=DATA_SECTIONS_KEY + '/*'):
        data_layers = {}
        input_params = self.params['input']
        for layer_name in deserialise(get_dict(input_params, 'layers', '')):
            path = top_path + '/' + layer_name
            if layer_name == 'rois':
                path += '/*'
            values0 = self.data.get_values(path)
            value_type = self.data.get_value_type(layer_name)
            values = [np.flip(value[value_type]) for value in values0]
            data_layers[layer_name] = self.init_data_layer(layer_name, values, value_type)
        return data_layers

    def init_data_layer(self, layer_name, values, value_type):
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

    def section_data_changed(self, name, action, indices, values):
        if name == 'landmarks':
            top_level = name
            name = DATA_SOURCE_KEY
        else:
            top_level = DATA_SECTIONS_KEY + '/*'
        self.data_changed(name, action, indices, values, top_level)

    def template_data_changed(self, name, action, indices, values):
        self.data_changed(name, action, indices, values, DATA_TEMPLATE_KEY)

    def data_changed(self, name, action, indices, values, top_level):
        modified = False
        path = top_level + '/' + name
        if name == 'rois':
            path += '/*'
        value_type = self.data.get_value_type(name)
        for index in indices:
            if action == ActionType.ADDED or action == ActionType.CHANGED:
                value = np.flip(values[index])
                if value_type == 'polygon':
                    value = Section(value).to_dict()
                elif value_type == 'location':
                    value = Point(value).to_dict()
                if action == ActionType.ADDED:
                    modified = self.data.add_value(path, value)
                elif action == ActionType.CHANGED:
                    modified = self.data.set_value(path, index, value)
            elif action == ActionType.REMOVED:
                modified = self.data.remove_value(path, index)
        if modified:
            self.data.save()

    def update_section_order(self, new_order):
        new_data = {}
        for index, key in enumerate(new_order):
            new_data[index] = self.data[DATA_SECTIONS_KEY][key]
        self.data[DATA_SECTIONS_KEY] = new_data
        self.data.save()

    def get_section_images(self, section_name):
        images = []
        values = self.data.get_values(DATA_SECTIONS_KEY + '/*/' + section_name)
        value_type = self.data.get_value_type(section_name)
        if value_type == 'polygon':
            order = self.data.get_values('serial_order/order')
            if not order:
                order = range(len(values))
            sections = [Section(values[index]) for index in order]
            images = get_section_images(sections, self.source, pixel_size=self.output_pixel_size)
        return images

    def extract_rois(self):
        self.sample_overlay_image, self.sample_template = self.create_section_overlay_image('sample')
        sample = self.sample_template

        values = self.data.get_values(DATA_TEMPLATE_KEY + '/rois')
        if len(values) > 0:
            section_points = []
            section_centers = []
            for value in values.values():
                roi = Section(value)
                if roi is not None:
                    roi.center -= sample.center
                    roi.polygon -= sample.center
                    angle = -90
                    if roi.center[1] > sample.center[1]:
                        angle = norm_angle(angle + 180)
                    h = create_transform(angle=angle)
                    section_points.append(apply_transform(roi.polygon, h))
                    section_centers.append(apply_transform([roi.center], h)[0])
            self.template_section_points = section_points
            self.template_section_centers = section_centers
        else:
            logging.warning('No ROI changes')

    def propagate_rois(self):
        indices = set()
        for section in self.sections.values():
            if 'rois' in section:
                for index in section['rois']:
                    indices.add(index)
        if len(indices) == 0:
            roi_index = 0
        else:
            roi_index = max(indices) + 1

        for section in self.sections.values():
            sample_section = section['sample']
            for index, (section_points, section_center) in enumerate(zip(self.template_section_points, self.template_section_centers)):
                # transform corners of template section (relative from magnet center), using transform of each section
                h = create_transform(angle=sample_section.angle, translate=sample_section.center)
                new_section_points = apply_transform(section_points, h)
                new_section_center = apply_transform([section_center], h)[0]
                value = {'polygon': new_section_points, 'center': new_section_center, 'angle': sample_section.angle,
                         'confidence': 1}
                if 'rois' not in section:
                    section['rois'] = {}
                section['rois'][roi_index + index] = Section(value)
        self.sample_sections_finalised = True








    def magsection_changed(self, section_name, indices):
        if (section_name == 'magnets' or section_name == 'sections') and 'confidence' in self.shape_editor.layer_names:
            for index in indices:
                if self.update_confidence_image('magnet', index):
                    self.shape_editor.main_viewer.layers[self.shape_editor.layer_names.index('confidence')].refresh()

    def create_confidence_image(self, section_name):
        # reduced resolution, using alpha channel
        shape = list(np.divide(self.small_image.shape[:2], 4).astype(int)) + [4]
        self.confidence_image = np.zeros(shape, dtype=np.float32)
        for section0 in self.sections:
            section = section0[section_name]
            if section.confidence < 1:
                color = list(confidence_color_map(section.confidence))
                get_contour_mask(section.polygon / self.output_pixel_size, image=self.confidence_image, color=color, smooth=True)

    def update_confidence_image(self, section_name, index):
        if index < len(self.sections):
            section = self.sections[index][section_name]
            if section.confidence < 1:
                get_contour_mask(section.polygon / self.output_pixel_size, image=self.confidence_image, color=(0, 0, 0, 0), smooth=True)
                return True
        return False

    def order_changed(self, datafile, section_name, index):
        # TODO: modify self.order with index(es) / value(s)?
        datafile[section_name] = self.order
        datafile.save()

    def process(self):
        confidences_all = []

        logging.info(f'* Workflow: {self.workflow}')
        if not self.clear:
            self.load_data()
        for step0 in self.workflow:
            logging.info(f'* Workflow step: {step0}')
            step = step0.lower()
            confidences = []
            if step.startswith('mag'):
                confidences = self.process_magnet_sections()
            elif step.startswith('sample'):
                confidences = self.process_sample_sections()
            elif step.startswith('roi'):
                self.process_rois()
            elif step.startswith('landmark'):
                self.process_landmarks()
            confidences_all.append(np.mean(confidences))

        if self.visualisation:
            self.draw_sections(['magnet', 'sample', 'rois', 'landmarks'])
        confidence = np.mean(confidences_all)
        logging.info(f'Overall confidence: {confidence:.4f}')
        logging.shutdown()

    def process_magnet_sections(self):
        self.matching_methods = ensure_list(get_dict(self.params, 'matching.method', 'features'))

        confidences = []
        detection_params = self.params['mag_detection']
        min_match_rate = detection_params['min_match_rate']
        # mag section detection
        if self.automation and not self.magsections_finalised:
            contour_detection_image, real_pixel_size = create_detection_image(self.fluor_source, detection_params)
            magsections, detect_confidences = detect_magsections(contour_detection_image, detection_params,
                                                                 real_pixel_size,
                                                                 outfolder=self.outfolder,
                                                                 show_stats=self.interactive)
            self.sections = {index: {'magnet': value} for index, value in enumerate(magsections)}
            confidences.append(self.conf_stats(detect_confidences, 'mag_sec_detection'))
            self.check_init_magsections()
            self.magsections_finalised = True
            self.save_sections()

        # ordering & alignment
        if self.visualisation:
            self.draw_sections('magnet')
            self.draw_sections_compose('magnet')
        self.calculate_magsections_matching()
        if self.automation and not self.order_finalised:
            confidences.append(self.conf_stats(self.order_magsections(), 'ordering', threshold=min_match_rate))
            self.save_order()
            confidences.append(self.conf_stats(self.align_magsections(), 'alignment', threshold=min_match_rate))
            self.save_sections()
        confidences.append(self.conf_stats(self.validate_order(), 'order_validation', threshold=min_match_rate))
        if self.visualisation:
            self.draw_order()
            self.draw_sections('magnet', serial_order_labels=True)
        return confidences

    def process_sample_sections(self):
        confidences = []
        # sample section detection & curation
        if 'magnets' in self.workflow:
            if not self.sample_sections_finalised:
                if self.automation:
                    confidences.append(self.conf_stats(self.detect_sample_section(), 'sample_section_detection'))
                self.propagate_sample_sections()
                self.save_sections()
        else:
            # no magnets
            if not (self.sample_sections_finalised and self.order_finalised):
                confidences.append(self.conf_stats(self.detect_sample_sections(), 'sample_sections_detection'))

        self.define_stage_order()

        if self.visualisation:
            section_names = ['sample']
            self.draw_sections(section_names)
            self.draw_sections(section_names, serial_order_labels=True)
            self.draw_sections(section_names, stage_order_labels=True)
        if self.visualisation:
            self.draw_sample_reconstruct()
        return confidences

    def process_rois(self):
        if self.visualisation:
            self.draw_sections(['rois'])

    def process_landmarks(self):
        # landmarks
        if self.visualisation:
            self.draw_sections(['landmarks'])

        if self.visualisation:
            section_names = ['magnet', 'sample', 'landmarks']
            self.draw_sections(section_names)
            self.draw_sections(section_names, serial_order_labels=True)

    # # # mag sec detection

    def align_magsections(self):
        # adjust mag section based on alignment
        # - coarse alignment (-> angle/center)
        # - adjust alignment between 2 sections
        # - convert coarse angle to transformation, combine with second transformation, extract angle/center
        # - adjust angle/center of section 2
        # - recalculate images and points of section 2

        self.check_init_magsections(get_features=True)
        detection_params = self.params['mag_detection']
        lowe_ratio = detection_params['lowe_ratio']
        max_iter = detection_params['max_iter']
        min_match_rate = detection_params['min_match_rate']

        logging.info('Aligning ordered MagSec sections')
        prev_section = None
        for sectioni in tqdm(self.order):
            section = self.sections[sectioni]['magnet']
            if prev_section is not None:
                center, angle, metrics = \
                    get_section_alignment(section, prev_section, self.matching_methods,
                                          lowe_ratio=lowe_ratio, max_iter=max_iter, min_match_rate=min_match_rate)
                if metrics['match_rate'] > min_match_rate:
                    # adjust section angle / center
                    section.center = center
                    section.angle = angle
                section.init_features(self.fluor_hq_source, detection_params=detection_params)
            prev_section = section
        results = self.check_order(self.order)
        confidences = results[0]
        self.order_finalised = True
        return confidences

    def curate_magsections(self):
        section_name = 'magnet'
        self.create_confidence_image('magnet')
        self.shape_editor.init('Curate magnet sections', self.data, [section_name], section_name,
                               self.output_pixel_size,
                               sources=self.back_images + [self.confidence_image],
                               source_names=self.back_images_names + ['confidence'],
                               source_blending_modes=self.back_images_blending + ['translucent'],
                               annotation_widget=True,
                               on_change_function=self.magsection_changed)
        self.shape_editor.show(block=True)
        # store values
        if self.shape_editor.changed:
            for index, value in self.shape_editor.get_values(section_name).items():
                section = Section(value)
                if index not in self.sections:
                    self.sections[index] = {}
                self.sections[index][section_name] = section
                self.data.set_sections_value(section_name, index, section)
            self.data.save()

    def flip_magsection_angles(self):
        for section in self.sections.values():
            magsection = section('magnet')
            magsection.angle = norm_angle(magsection.angle + 180)
        self.save_sections()

    def draw_sections(self, section_names, serial_order_labels=False, stage_order_labels=False):
        back_image = color_image(self.small_image)
        out_image = np.zeros_like(back_image)

        # add confidence to back_image
        for index, section in self.sections.items():
            for section_name in section_names:
                if section_name in section:
                    section_element = section[section_name]
                    if isinstance(section_element, dict):
                        for subindex, subsection in section_element.items():
                            self.draw_section_element(out_image, subsection, index,
                                                      serial_order_labels, stage_order_labels)
                    else:
                        self.draw_section_element(out_image, section_element, index,
                                                  serial_order_labels, stage_order_labels)

        if 'landmarks' in section_names:
            for index, landmark in self.landmarks.items():
                self.draw_section_element(out_image, landmark, index)

        out_image = cv.addWeighted(back_image, 1, out_image, 0.9, gamma=0)
        filename = f'annotated_{"_".join(section_names)}'
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

    def draw_sections_compose(self, section_name):
        source = self.small_image
        sizes = [section[section_name].get_size() / self.output_pixel_size for section in self.sections.values()]
        size = int(np.max(sizes))
        margin_size = int(size * 1.1)
        section_images = []
        for element in self.sections.values():
            section = element[section_name]
            section_image = float2int_image(section.extract_rotated_image_raw(source, self.output_pixel_size, (size, size)))
            section_image[..., 3][section_image[..., 3] == 0] = 127     # show some background
            section_images.append(section_image)

        n = len(self.sections)
        nx = int(np.ceil(math.sqrt(n)))
        ny = int(np.ceil(n / nx))
        compose_image = np.zeros((ny * margin_size, nx * margin_size, 4), dtype=np.uint8)
        for position, ((index, section), section_image) in enumerate(zip(self.sections.items(), section_images)):
            x, y = position % nx, position // nx
            x0, y0 = x * margin_size, y * margin_size
            compose_image[y0:y0 + size, x0:x0 + size] = section_image
            draw_text(compose_image, str(index), (x0, y0 + margin_size))

        filename = 'annotated_magsections_compose.tiff'
        save_image(join_path(self.outfolder, filename), compose_image)

    def save_sections(self):
        sections = self.sections_to_dict(self.sections)
        self.data.set_section('sections', sections)
        self.data.save()

    def sections_to_dict(self, objects):
        elements = {}
        for label, value in objects.items():
            if hasattr(value, 'to_dict'):
                value = value.to_dict()
            elif isinstance(value, dict):
                value = self.sections_to_dict(value)
            elements[label] = value
        return elements

    def load_objects(self, elements):
        objects = {}
        for label, value in elements.items():
            if isinstance(value, dict):
                if 'polygon' in value:
                    value = Section(value)
                elif 'location' in value:
                    value = Point(value)
                else:
                    value = self.load_objects(value)
            objects[label] = value
        return objects

    def load_data(self):
        self.data.load()
        self.sections = self.load_objects(self.data.get_section(DATA_SECTIONS_KEY))
        self.matches.load()
        self.transforms.load()
        self.order = self.data.get_section('serial_order').get('order', [])
        self.order_confidences = self.data.get_section('serial_order').get(DATA_CONFIDENCE_SUBKEY, [])
        self.stage_order = self.data.get_section('stage_order').get('order', [])
        self.landmarks = {index: Point(value[DATA_SOURCE_KEY])
                          for index, value in self.data.get_section('landmarks').items()}

    def check_init_magsections(self, get_features=False):
        if not self.magsections_initialised:
            logging.info('MagSec image preparation')
            out_filename = join_path(self.outfolder, 'distribution_keypoints.png') if self.visualisation else None
            detection_params = self.params['mag_detection']
            sections = [section['magnet'] for section in self.sections.values()]
            if get_features:
                init_section_features(sections, self.fluor_hq_source,
                                      detection_params=detection_params,
                                      show_stats=self.visualisation, out_filename=out_filename)
            self.magsections_size = np.linalg.norm(np.median([section.lengths for section in sections], 0))
            if get_features:
                self.magsections_initialised = True

    # # # mag sec ordering

    def calculate_magsections_matching(self):
        self.check_init_magsections(get_features=True)
        use_external_matching = False
        if use_external_matching:
            if not self.order_finalised:
                for sectioni, section in self.sections.items():
                    csv_write(join_path(self.outfolder, f'magsection_points_{sectioni:03}.txt'), section.points)
        else:
            self.calculate_distance_matrix()

    def calculate_distance_matrix(self):
        logging.info('MagSec distance matrix')
        n = len(self.sections)
        self.norm_score_matrix = np.zeros([n, n])
        self.norm_distance_matrix = np.zeros([n, n])
        self.score_matrix = np.zeros([n, n])
        self.distance_matrix = np.zeros([n, n])
        self.transform_matrix = np.empty([n, n], dtype=object)
        self.matches_new = 0
        indices = np.transpose(np.triu_indices(n))
        with tqdm(total=len(indices)) as progress_bar:
            #logging.info(f'#cores: {os.cpu_count()}')
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.calculate_distance_matrix_element, indices1) for indices1 in indices]
                for _ in as_completed(futures):
                    progress_bar.update()
        if self.matches_new > 0:
            self.save_matches()
        for i in range(n):
            self.norm_distance_matrix[i, i] = np.Infinity
            self.distance_matrix[i, i] = np.Infinity
        csv_write(join_path(self.outfolder, 'scores.csv'), self.norm_score_matrix)
        csv_write(join_path(self.outfolder, 'distances.csv'), self.norm_distance_matrix)

    def calculate_distance_matrix_element(self, indices):
        i, j = indices
        if i != j:
            index = f'{i}_{j}'
            metrics = self.matches.get(index)
            transform = self.transforms.get(index)
            if metrics is None:
                detection_params = self.params['mag_detection']
                lowe_ratio = detection_params['lowe_ratio']
                max_iter = detection_params['max_iter']
                min_match_rate = detection_params['min_match_rate']
                try:
                    source_section = self.sections[i]['magnet']
                    target_section = self.sections[j]['magnet']
                    dtransform, metrics =\
                        align_sections_metrics(source_section, target_section, self.matching_methods,
                                               lowe_ratio=lowe_ratio, max_iter=max_iter, min_match_rate=min_match_rate)
                    # calculate full section to section transform
                    transform = combine_transforms([
                        create_transform(translate=-source_section.center),
                        create_transform(angle=-source_section.angle),
                        dtransform,
                        create_transform(angle=target_section.angle),
                        create_transform(translate=target_section.center)
                    ])
                except Exception as e:
                    logging.exception('Calculating distance matrix element')
                    raise e
                metrics.pop('matched_points', None)
                self.matches[index] = metrics
                self.transforms[index] = transform.tolist()
                self.matches_new += 1
            self.assign_matrices(i, j, metrics, transform)

    def assign_matrices(self, i, j, metrics, transform):
        self.score_matrix[i, j] = metrics['nmatches']
        self.score_matrix[j, i] = metrics['nmatches']
        self.norm_score_matrix[i, j] = metrics['match_rate']
        self.norm_score_matrix[j, i] = metrics['match_rate']
        self.distance_matrix[i, j] = metrics['distance']
        self.distance_matrix[j, i] = metrics['distance']
        self.norm_distance_matrix[i, j] = metrics['norm_distance']
        self.norm_distance_matrix[j, i] = metrics['norm_distance']
        self.transform_matrix[i, j] = transform
        self.transform_matrix[j, i] = np.linalg.inv(transform)

    def order_magsections(self):
        logging.info('MagSec ordering')

        norm_score_matrix_1 = 1 - self.norm_score_matrix
        score_matrix_1_int = (np.max(self.score_matrix) - self.score_matrix).astype(int)
        logging.info('tsp_solver.greedy')
        self.order = norm_order(solve_tsp(norm_score_matrix_1, optim_steps=10))
        confidences, _, _, _, _ = self.check_order(self.order)

        tsp_params = self.params.get('tsp')
        if tsp_params is not None:
            tsp_solver = ExternalTspSolver(tsp_params)
            logging.info('concorde')
            self.check_order(norm_order(tsp_solver.solve_tsp_concorde(score_matrix_1_int)))
            logging.info('LKH')
            self.check_order(norm_order(tsp_solver.solve_tsp_lkh(score_matrix_1_int)))

        self.order_confidences = confidences
        return confidences

    def validate_order(self):
        detection_params = self.params['mag_detection']
        min_match_rate = detection_params['min_match_rate']
        lowe_ratio = detection_params['lowe_ratio']
        self.check_init_magsections(get_features=True)
        logging.info('Validating order')
        ngood = 0
        scores = []
        distances = []
        prev_section = None
        for sectioni in self.order:
            section = self.sections[sectioni]['magnet']
            if prev_section is not None:
                nn_distance = np.mean([prev_section.nn_distance, section.nn_distance])
                metrics = get_match_metrics(prev_section.size_points, section.size_points, nn_distance, lowe_ratio=lowe_ratio)
                match_rate = metrics['match_rate']
                scores.append(match_rate)
                distances.append(metrics['norm_distance'])
                if match_rate > min_match_rate:
                    ngood += 1
            prev_section = section

        logging.info('Order scores')
        self.check_order(self.order)

        logging.info('Align scores')
        ntotal = len(self.order) - 1
        results = 'Order: ' + ' '.join(map(str, self.order)) + \
                  '\n          ' + ''.join([f'{o: <7}' for o in self.order]) + \
                  '\nScores:    ' + pretty_num_list(scores) + \
                  '\nDistances: ' + pretty_num_list(distances)
        results += (f'\nScore: {np.mean(scores):.4f} Distance: {np.mean(distances):.4f}'
                    f' #Good: {ngood}/{ntotal} ({ngood / ntotal * 100:.1f}%)')
        logging.info(results)

        for sectioni, row in enumerate(self.norm_score_matrix):
            max_section_score = np.max(row)
            if max_section_score < min_match_rate:
                logging.info(f'Section {sectioni} max match rate: {max_section_score:.4f}')
        return scores

    def define_stage_order(self):
        stage_order = None
        nsections = len(self.sections)
        if nsections > 1:
            points = [section['sample'].center for section in self.sections.values()]
            distance_matrix = euclidean_distances(points)
            stage_order = norm_order(solve_tsp(distance_matrix, optim_steps=10))
            logging.info(f'Total natural distance {self.get_total_distance(range(nsections), distance_matrix):.0f} um')
            logging.info(f'Total order distance {self.get_total_distance(self.order, distance_matrix):.0f} um')
            logging.info(f'Total stage distance {self.get_total_distance(self.stage_order, distance_matrix):.0f} um')
        if stage_order is not None and stage_order != self.stage_order:
            self.stage_order = stage_order
            self.save_stage_order()

    def get_total_distance(self, order, distance_matrix):
        results = check_order(distance_matrix, distance_matrix, order)
        return np.sum(results[1])

    def check_order(self, order):
        detection_params = self.params['mag_detection']
        min_match_rate = detection_params['min_match_rate']
        results = check_order(self.norm_score_matrix, self.norm_distance_matrix, order, min_match_rate)
        logging.info(results[-1])
        return results

    def curate_ordered_magsections(self):
        images = get_section_images(self.sections, 'magnet', self.fluor_hq_source)
        self.order_editor.init(self.data, 'serial_order', images)
        self.order_editor.show('Section ordering', block=True)

    def draw_order(self):
        images = get_section_images(self.sections, 'magnet', self.fluor_hq_source)
        logging.info('Order image output')
        prev_index = None
        prev_image = None
        size = get_image_size(images[0])
        fps = 2
        vidwriter = cv.VideoWriter(join_path(self.outfolder, 'magsections.mp4'), -1, fps, size, isColor=False)
        for order_index, section_index in enumerate(tqdm(self.order)):
            image = grayscale_image(images[section_index])
            save_image(join_path(self.outfolder, f'magsection_{section_index:03}.tiff'), image)
            if order_index > 0:
                score = self.order_confidences[order_index - 1]
                overlay_image = create_image_overlay(prev_image, image)
                draw_text(overlay_image, f'{score:.4f}')
                save_image(join_path(self.outfolder, f'overlay_{prev_index:03}-{section_index:03}.tiff'), overlay_image)
            annotated_image = image.copy()
            draw_text(annotated_image, f'{section_index:>3}', thickness=2, color=(192, 192, 0))
            vidwriter.write(annotated_image)
            prev_index = section_index
            prev_image = image
        vidwriter.release()

    def save_matches(self):
        self.matches.save(sort=True)
        self.transforms.save(sort=True)

    def save_order(self):
        self.data.set_section('serial_order/order', self.order)
        self.data.set_section('serial_order/confidences', self.order_confidences)
        self.data.save()

    def save_stage_order(self):
        self.data.set_section('stage_order/order', self.stage_order)
        self.data.save()

    # # # sample section detection

    def create_section_overlay_image(self, template_name, padded_size_factor=1):
        logging.info('Section overlay')
        source = self.bf_source
        pixel_size = source.get_pixel_size_micrometer()[:2]
        element_size, large_size0 = get_section_sizes(self.sections, template_name, pixel_size)
        # flip for easier curation
        large_size = np.flip(large_size0) * padded_size_factor
        # sample image will be rotated
        # convert pixels back to um
        section_template = Section(cv.boxPoints((np.divide(large_size, 2), np.flip(element_size).astype(float), 0)) * pixel_size)
        shape = list(np.flip(large_size)) + [3]
        sum_image = np.zeros(shape)
        pixel_size = source.get_pixel_size_micrometer()
        for section in tqdm(self.sections.values()):
            sum_image += section[template_name].extract_rotated_image_raw(source, pixel_size, np.flip(large_size), use_alpha=False)
        overlay_image0 = sum_image / len(self.sections)
        overlay_image = norm_image_minmax(overlay_image0)
        save_image(join_path(self.outfolder, f'{template_name}_section_overlay.tiff'), float2int_image(overlay_image))
        if self.visualisation:
            show_image(overlay_image, title=f'{template_name} section overlay')
        return overlay_image, section_template

    def autodetect_sample_section(self):
        confidences = []
        # TODO: use line detection instead
        #detection_image = norm_image_minmax(image_resize(self.section_overlay_image, np.divide(size, 10)))
        #edges = cv.Canny(float2int_image(cv.GaussianBlur(detection_image, (5, 5), 0)), 0, 80)
        #edges = cv.Sobel(src=detection_image, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
        #edges = cv.Laplacian(src=detection_image, ddepth=cv.CV_32F, ksize=11)
        return confidences

    def detect_sample_sections(self):
        # TODO: explore if feasible to auto-detect sample sections from scratch (w/o magnets)?
        pixel_size = get_value_units_micrometer(split_value_unit_list(self.params['sample_detection']['pixel_size']))
        sections = []
        confidences = []
        image = int2float_image(self.bf_source.asarray(pixel_size=pixel_size))
        peak_val = calc_peak(image)
        image1 = abs(image - peak_val)
        contour_detection_image = cv.Canny(uint8_image(image1), 80, 100)
        return confidences

        for index, contour in enumerate(get_contours(contour_detection_image)):
            section = Section(contour)
            self.sections[index]['sample'] = section
        return confidences

    def detect_sample_section(self):
        if not self.magnet_template_initialised:
            self.magnet_overlay_image, self.magnet_template = self.create_section_overlay_image('magnet', padded_size_factor=3)
            self.magnet_template_initialised = True
        confidences = self.autodetect_sample_section()
        logging.info('Sample section detection')
        # TODO: use automatic detection based on edge detection
        return confidences

    def curate_sample_section(self):
        if not self.magnet_template_initialised:
            self.magnet_overlay_image, self.magnet_template = self.create_section_overlay_image('magnet', padded_size_factor=3)
            self.magnet_template_initialised = True
        magnet = self.magnet_template
        # TODO: add existing sample (using sample from section 0)
        section_data = DataFile()
        section_data.set_sections_value('magnet', 0, magnet.to_dict())
        self.shape_editor.init('Curate a sample section as template', section_data, ['magnet', 'sample'], 'sample',
                               self.output_pixel_size,[float2int_image(self.magnet_overlay_image)])
        self.shape_editor.show(block=True)
        values = self.shape_editor.get_values('sample')
        sample = Section(values[0])
        if sample is not None:
            sample.center -= magnet.center
            sample.polygon -= magnet.center
            angle = -90
            if sample.center[1] > magnet.center[1]:
                # adjust (invert) all magnet angles so vector points towards sample section
                self.flip_magsection_angles()
                angle = norm_angle(angle + 180)
            h = create_transform(angle=angle)
            self.template_section_points = apply_transform(sample.polygon, h)
            self.template_section_center = apply_transform([sample.center], h)[0]
        else:
            raise ValueError('Sample section is not defined')

    def propagate_sample_sections(self):
        for section in self.sections.values():
            magsection = section['magnet']
            # transform corners of template section (relative from magnet center), using transform of each section
            h = create_transform(angle=magsection.angle, translate=magsection.center)
            section_points = apply_transform(self.template_section_points, h)
            section_center = apply_transform([self.template_section_center], h)[0]
            value = {'polygon': section_points, 'center': section_center, 'angle': magsection.angle, 'confidence': 0.5}
            section['sample'] = Section(value)
        self.sample_sections_finalised = True

    def curate_sample_sections(self):
        section_name = 'sample'
        self.shape_editor.init('Curate sample sections', self.data, [section_name], section_name,
                               self.output_pixel_size,
                               sources=self.back_images, source_names=self.back_images_names,
                               source_blending_modes=self.back_images_blending, order=self.order, annotation_widget=True,
                               on_paste_function=self.section_pasting)
        self.shape_editor.show(block=True)
        # store values
        if self.shape_editor.changed:
            for index, section in self.sections.items():
                section.pop(section_name, None)
                self.data.remove_value(section_name, index)
            self.sections = {index: value for index, value in self.sections.items() if value != {}}
            for index, value in self.shape_editor.get_values(section_name).items():
                section = Section(value)
                if index not in self.sections:
                    self.sections[index] = {}
                self.sections[index][section_name] = section
                self.data.set_sections_value(section_name, index, section)
            self.order = self.shape_editor.order
            self.save_order()
            self.data.save()

    def section_pasting(self, data):
        detection_params = self.params['sample_detection']
        snap_scaledown = detection_params.get('snap_scaledown')
        snap_margin = detection_params.get('snap_margin')
        return detect_nearest_edges(data, self.source, snap_scaledown, snap_margin)

    def curate_sample_alignment(self):
        images = get_section_images(self.sections, 'sample', self.fluor_hq_source)
        if len(images) > 0:
            self.order_editor.init(self.data, 'serial_order', images)
            self.order_editor.show('Sample alignment', block=True)

    def draw_sample_reconstruct(self):
        images = get_section_images(self.sections, 'sample', self.fluor_hq_source)
        logging.info('Order image output')
        size = get_image_size(next(iter(images.values())))
        fps = 5
        vidwriter = cv.VideoWriter(join_path(self.outfolder, 'sample.mp4'), -1, fps, size, isColor=False)
        for section_index in tqdm(self.order):
            image = grayscale_image(images[section_index])
            save_image(join_path(self.outfolder, f'sample_{section_index:03}.tiff'), image)
            annotated_image = image.copy()
            draw_text(annotated_image, f'{section_index:>3}', thickness=2, color=(192, 192, 0))
            vidwriter.write(annotated_image)
        vidwriter.release()

    def curate_landmarks(self):
        section_name = 'landmarks'
        layer_sections = []
        if len(self.sections) > 0:
            layer_sections.append('magnet')
            layer_sections.append('sample')
        layer_sections.append(section_name)
        self.shape_editor.init('Set landmarks', self.data, layer_sections, section_name, self.output_pixel_size,
                               sources=self.back_images, source_names=self.back_images_names,
                               source_blending_modes=self.back_images_blending)
        self.shape_editor.show(block=True)
        # store values
        if self.shape_editor.changed:
            landmarks = {}
            for index, value in self.shape_editor.get_values(section_name).items():
                landmark = Point(value)
                self.landmarks[index] = landmark
                landmarks[index] = landmark.to_dict()
                self.data.set_sections_value(DATA_SOURCE_KEY, index, landmark.to_dict(), top_level='landmarks')
            self.data.save()
        self.landmarks_finalised = True

    def conf_stats(self, confidences, label='', threshold=0.5):
        if len(confidences) > 0:
            confidence = np.mean(confidences)
            logging.info(f'confidence: {confidence:.4f} +- {np.std(confidences):.4f}')
            logging.info(f'N>={threshold}: {np.sum([conf >= threshold for conf in confidences]) / len(confidences):.4f}')
            if self.visualisation:
                out_filename = join_path(self.outfolder, f'confidence_{label}.png')
            else:
                out_filename = None
            show_histogram(confidences, title='Confidence ' + label.replace('_', ' '),
                           show=self.visualisation, out_filename=out_filename)
        else:
            confidence = 0
        return confidence

    # # # rois

    def curate_roi(self):
        if not self.sample_template_initialised:
            self.sample_overlay_image, self.sample_template = self.create_section_overlay_image('sample')
            self.sample_template_initialised = True
        sample = self.sample_template
        section_data = DataFile()
        section_data.set_sections_value('sample', 0, sample.to_dict())
        # TODO: add existing ROIs (using rois from section 0)
        self.shape_editor.init('Curate a ROI as template', section_data, ['sample', 'rois'], 'rois',
                               self.source_pixel_size, [float2int_image(self.sample_overlay_image)])
        self.shape_editor.show(block=True)
        values = self.shape_editor.get_values('rois')
        if len(values) > 0:
            section_points = []
            section_centers = []
            for value in values.values():
                roi = Section(value)
                if roi is not None:
                    roi.center -= sample.center
                    roi.polygon -= sample.center
                    angle = -90
                    if roi.center[1] > sample.center[1]:
                        angle = norm_angle(angle + 180)
                    h = create_transform(angle=angle)
                    section_points.append(apply_transform(roi.polygon, h))
                    section_centers.append(apply_transform([roi.center], h)[0])
            self.template_section_points = section_points
            self.template_section_centers = section_centers
        else:
            logging.warning('No ROI changes')

    def propagate_rois0(self):
        indices = set()
        for section in self.sections.values():
            if 'rois' in section:
                for index in section['rois']:
                    indices.add(index)
        if len(indices) == 0:
            roi_index = 0
        else:
            roi_index = max(indices) + 1

        for section in self.sections.values():
            sample_section = section['sample']
            for index, (section_points, section_center) in enumerate(zip(self.template_section_points, self.template_section_centers)):
                # transform corners of template section (relative from magnet center), using transform of each section
                h = create_transform(angle=sample_section.angle, translate=sample_section.center)
                new_section_points = apply_transform(section_points, h)
                new_section_center = apply_transform([section_center], h)[0]
                value = {'polygon': new_section_points, 'center': new_section_center, 'angle': sample_section.angle,
                         'confidence': 1}
                if 'rois' not in section:
                    section['rois'] = {}
                section['rois'][roi_index + index] = Section(value)
        self.sample_sections_finalised = True

    # # # misc

    def get_label_color(self, label):
        return self.colors[label % len(self.colors)]


def get_source(base_folder, input, input_filename=None):
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
    return source
