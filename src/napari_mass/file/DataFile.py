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
            self[SOURCE_KEY] = NAME + ' ' + VERSION
            self[SECTIONS_KEY] = {}

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

    def set_sections_value(self, section_name, key, value, top_level=SECTIONS_KEY):
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if top_level not in self:
            self[top_level] = {}
        if key < 0:
            # added: get best new index
            key = 0
            while key in self[top_level] and section_name in self[top_level][key]:
                key += 1
        if key not in self[top_level]:
            self[top_level][key] = {}
        self[top_level][key][section_name] = value

    def remove_value(self, section_name, key, top_level=SECTIONS_KEY):
        value = self[top_level][key].pop(section_name, None)
        if value is None:
            print('DataFile warning: tried removing non-existent item')
        if self[top_level][key] == {}:
            # no section contents -> remove index
            self[top_level].pop(key, None)
            # collapse empty section
            if len(self[top_level]) > 0:
                max_key = max(self[top_level].keys())
                for key in range(key + 1, max_key + 1):
                    if key in self[top_level]:
                        self[top_level][key - 1] = self[top_level].pop(key)
