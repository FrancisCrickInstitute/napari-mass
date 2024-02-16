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
        if not isinstance(keys, list):
            keys = deserialise(keys, '/')

        for keyi, key in enumerate(keys):
            if key == '*':
                values = []
                for subdct in dct.values():
                    values1 = self.get_values(keys[keyi + 1:], subdct)
                    if values1:
                        values.extend(values1)
                return values
            elif isinstance(key, str) and key.isnumeric():
                key = int(key)
            dct = dct.get(key, {})
        if dct:
            return ensure_list(dct)
        else:
            return []

    def set_value(self, keys, index, value):
        if not isinstance(keys, list):
            keys = deserialise(keys, '/')
        if hasattr(value, 'to_dict'):
            value = value.to_dict()

        final_key = keys[-1]
        is_index = (final_key == '*')
        for item in self.get_values(keys[:-1]):
            if is_index:
                n = len(item)
                final_key = index
            else:
                n = 1
            if index < n:
                if value:
                    # set
                    item[final_key] = value
                    return True
                else:
                    # remove
                    old_value = item.pop(final_key, None)
                    return old_value is not None
            index -= n
        return False

    def add_value(self, keys, value, dct=None):
        if dct is None:
            dct = self
        if not isinstance(keys, list):
            keys = deserialise(keys, '/')
        if hasattr(value, 'to_dict'):
            value = value.to_dict()

        for keyi, key in enumerate(keys):
            is_final_key = (keyi == len(keys) - 1)
            if key == '*':
                index = 0
                while index in dct:
                    if not is_final_key and not self.exists(keys[keyi + 1:], dct.get(index, {})):
                        break
                    index += 1
                key = index
            if is_final_key:
                dct[key] = value
                return True
            if key not in dct:
                dct[key] = {}
            dct = dct[key]
        return False

    def remove_value(self, keys, index):
        return self.set_value(keys, index, None)

    def exists(self, keys, dct=None):
        if dct is None:
            dct = self
        if not isinstance(keys, list):
            keys = deserialise(keys, '/')

        for key in keys:
            if key in dct:
                dct = dct[key]
            else:
                return False
        return True

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
