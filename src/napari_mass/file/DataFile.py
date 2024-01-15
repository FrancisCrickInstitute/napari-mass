from datetime import datetime
import numpy as np

from src.napari_mass.file.FileDict import FileDict
from src.napari_mass.parameters import *
from src.napari_mass.util import get_dict_permissive


class DataFile(FileDict):
    # coordinates: [x, y]
    # angles: angle of rotation; regular vector domain

    value_types = {
        'magnet': 'polygon',
        'sample': 'polygon',
        'roi': 'polygon',
        'focus': 'polygon',
        'landmark': 'location',
        'serial_order': 'list',
        'scan_order': 'list',
    }

    def __init__(self, filename=None, load=True):
        super().__init__(filename, load)
        if self == {}:
            self[CREATED_KEY] = str(datetime.now())
            self[SOURCE_KEY] = NAME
            self[SECTIONS_KEY] = {}

    def import_magc(self, magc_filename):
        filter_keys = ['number', 'compression', 'area', 'template']
        magc = MagcFile(magc_filename)
        magc.load()
        old_dict = magc.to_dict_implicit(search_plural=True, filter_keys=filter_keys, reshape_coordinates=True)
        self.convert_old(old_dict)

    def import_old(self, old_filename):
        old_dict = DataFile(old_filename, load=True)
        self.dims = old_dict.get('magnets', {}).get('dims')
        self.convert_old(old_dict)

    def convert_old(self, old_dict):
        sections = self[SECTIONS_KEY]
        for key, contents in old_dict.items():

            if key == 'sections':
                key = 'sample'
            if key == 'magnets':
                key = 'magnet'
            if key == 'roi':
                key = 'rois'
            if key == 'serialorder':
                key = 'serial_order'
            if key == 'tsporder':
                key = 'stage_order'

            if key == 'landmarks':
                self[key] = {index: {'source': value} for index, value in contents.items()}
            elif key in ['serial_order', 'stage_order']:
                self[key] = contents[next(iter(contents))]
            else:
                for index, value in contents.items():
                    if index not in sections:
                        sections[index] = {}
                    if key == 'magnet' and 'location' in value:
                        value = {'polygon': [value['location']]}
                    elif key == 'focus' and 'polygon' in value:
                        value = {index: {'location': point} for index, point in enumerate(value['polygon'])}
                    elif key == 'rois' and isinstance(next(iter(value)), str):
                        value = {0: value}
                    sections[index][key] = value

    def get_section(self, section_name, default_value={}):
        return self.get(section_name, default_value)

    def get_values(self, element_name):
        values = {}
        value_type = get_dict_permissive(self.value_types, element_name)

        if element_name in self:
            top_level = element_name
            element_name = None
        else:
            top_level = SECTIONS_KEY

        for index, element in self.get(top_level, {}).items():
            if element_name is None or element_name in element:
                if element_name is not None:
                    value = element[element_name]
                else:
                    value = element
                if 'source' in value:
                    value = value['source']
                if isinstance(value, dict):
                    if 'polygon' in value:
                        value = value['polygon']
                    elif 'location' in value:
                        value = value['location']
                values[index] = value
        return values, value_type

    def set_section(self, section_name, values):
        paths = []
        current_dict = self
        last_dict = current_dict
        last_path = section_name
        for path in section_name.split('/'):
            if path not in current_dict:
                current_dict[path] = {}
            last_dict = current_dict
            last_path = path
            current_dict = current_dict[path]
            paths.append(path)
        last_dict[last_path] = values

    def set_sections_value(self, section_name, index, value, top_level=SECTIONS_KEY):
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if top_level not in self:
            self[top_level] = {}
        if index not in self[top_level]:
            self[top_level][index] = {}
        self[top_level][index][section_name] = value

    def remove_value(self, section_name, index):
        self[SECTIONS_KEY][index].pop(section_name, None)
        if self[SECTIONS_KEY][index] == {}:
            self[SECTIONS_KEY].pop(index, None)


if __name__ == '__main__':
    data = DataFile('../SBEMimage/array/example/wafer_example1.json', load=False)
    data.import_magc('../SBEMimage/array/example/wafer_example1.magc')
    #data.import_old('../annotated_old.yml')
    data.save()
