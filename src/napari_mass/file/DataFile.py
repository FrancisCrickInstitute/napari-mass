from datetime import datetime
import numpy as np

from napari_mass.file.FileDict import FileDict
from napari_mass.parameters import *
from napari_mass.util import *


class DataFile(FileDict):
    # coordinates: [x, y]
    # angles: angle of rotation; regular vector domain

    value_types = {
        'magnet': 'polygon',
        'sample': 'polygon',
        'roi': 'polygon',
        'focus': 'location',
        'landmark': 'location',
        'serial_order': 'list',
        'scan_order': 'list',
    }

    def __init__(self, filename=None, load=True):
        super().__init__(filename, load)
        if self == {}:
            self[DATA_SOURCE_KEY] = NAME + ' ' + VERSION
            self[DATA_CREATED_KEY] = str(datetime.now())
            self[DATA_TEMPLATE_KEY] = {}
            self[DATA_SECTIONS_KEY] = {}

    def get_section_keys(self, element_name=None, top_level=DATA_SECTIONS_KEY):
        if top_level in self:
            return [key for key, value in self[top_level].items()
                    if (element_name is not None and element_name in value) or len(value) > 0]
        return []

    def get_section(self, name, default_value={}):
        return self.get(name, default_value)

    def get_value_type(self, name):
        return get_dict_permissive(self.value_types, name)

    def get_values(self, keys, dct=None):
        if dct is None:
            dct = self
        if '/' in keys:
            keys = deserialise(keys, '/')

        for keyi, key in enumerate(keys):
            if key == '*':
                values = []
                for subdct in dct.values():
                    values1 = self.get_values(keys[keyi + 1:], subdct)
                    if values1:
                        values.extend(values1)
                return values
            elif key.isnumeric():
                key = int(key)
            dct = dct.get(key, {})
        if dct:
            return ensure_list(dct)
        else:
            return []

    def set_value(self, keys, index, value, setting=False, dct=None):
        if dct is None:
            dct = self
        if not isinstance(keys, list):
            keys = deserialise(keys, '/')

        final_keyi = len(keys) - 1
        final_index_keyi = -1
        for keyi, key in enumerate(keys):
            if key == '*':
                final_index_keyi = keyi

        for keyi, key in enumerate(keys[:-1]):
            is_final_index = (keyi == final_index_keyi - 1)
            is_final_key = (keyi == final_keyi - 1)
            if is_final_index:
                setting = True
            if setting and is_final_key and index == 0:
                dct[key] = value
            elif key == '*':
                for subdct in dct.values():
                    index = self.set_value(keys[keyi + 1:], index, value, setting, subdct)

            dct = dct.get(key, {})
        if isinstance(dct, dict):
            index -= 1
        else:
            index -= len(dct)
        return index

    def get_values1(self, element_name, top_level=DATA_SECTIONS_KEY):
        values = {}
        value_type = self.get_value_type(element_name)

        if element_name in self:
            top_level = element_name
            element_name = None

        for index, element in self.get(top_level, {}).items():
            if element_name is None or element_name in element:
                if element_name is not None:
                    value = element[element_name]
                else:
                    value = element
                if DATA_SOURCE_KEY in value:
                    value = value[DATA_SOURCE_KEY]
                values[index] = value
        return values, value_type

    def set_section(self, key, values):
        paths = []
        current_dict = self
        last_dict = current_dict
        last_path = key
        for path in key.split('/'):
            if path not in current_dict:
                current_dict[path] = {}
            last_dict = current_dict
            last_path = path
            current_dict = current_dict[path]
            paths.append(path)
        last_dict[last_path] = values

    def set_sections_value(self, section_name, indices, value, top_level=DATA_SECTIONS_KEY):
        index = indices.split('/')
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if top_level not in self:
            self[top_level] = {}
        if index not in self[top_level]:
            self[top_level][index] = {}
        self[top_level][index][section_name] = value

    def remove_value(self, section_name, key, top_level=DATA_SECTIONS_KEY):
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
