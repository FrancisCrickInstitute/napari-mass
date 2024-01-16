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
