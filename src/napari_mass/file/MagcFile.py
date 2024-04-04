import configparser
import os
import numpy as np

from napari_mass.util import *


class MagcFile(dict):
    # coordinates: [x, y]

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.value_types = {
            'sections': 'polygon',
            'rois': 'polygon',
            'focus': 'polygon',
            'landmarks': 'location',
            'magnets': 'polygon',
            'serial_order': 'serial_order',
            'stage_order': 'stage_order',
        }

    def load(self):
        config = configparser.ConfigParser()
        if os.path.exists(self.filename):
            config.read_file(open(self.filename))
            #dct = {section_name: dict(config[section_name]) for section_name in config.sections()}     # simple
            for key in config.sections():
                if not key.startswith('end_'):
                    contents = get_config_subitems(config[key])
                    key_parts = key.split('.')
                    key0 = key_parts[0]
                    if key0 not in self and key0 + 's' in self:
                        key_parts[0] = key0 + 's'
                    add_dict_tree(self, key_parts, contents)
            return True
        return False

    def to_dict_implicit(self, search_plural=False, filter_keys=[], reshape_coordinates=False):
        dct = {}
        config = configparser.ConfigParser()
        if os.path.exists(self.filename):
            config.read_file(open(self.filename))
            for key in config.sections():
                if not key.startswith('end_'):
                    contents = get_config_subitems(config[key])
                    # filter keys
                    for filter_key in filter_keys:
                        contents.pop(filter_key, None)
                    if reshape_coordinates and 'polygon' in contents:
                        contents['polygon'] = np.reshape(contents['polygon'], (-1, 2)).tolist()
                    key_parts = key.split('.')
                    key0 = key_parts[0]
                    # search for singular/plural matches
                    if search_plural and key0 not in dct and key0 + 's' in dct:
                        key_parts[0] = key0 + 's'
                    add_dict_tree_indexed(dct, key_parts, contents)
        return dct

    def from_datafile_dict(self, dct):
        new_dct = {}
        for key, value in dct.items():
            if key == 'sections':
                n = len(value)
                for index, subsection in value.items():
                    for subkey, subvalue in subsection.items():
                        if subkey == 'sample':
                            subkey = 'sections'
                        elif subkey == 'magnet':
                            subkey = 'magnets'
                        if subkey in self.value_types:
                            if subkey not in new_dct:
                                new_dct[subkey] = {'number': n}
                            if len(subvalue) == 1:
                                subvalue = subvalue[list(iter(subvalue))[0]]
                            if 'polygon' in subvalue:
                                subvalue['polygon'] = str(subvalue['polygon']).replace('[', '').replace(']', '')
                            new_dct[subkey][f'{index:04}'] = subvalue
            elif key in self.value_types:
                new_dct[key] = value
        self.update(new_dct)
        return new_dct

    def get_value(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            return value.get(key, value)

    def get_values(self, key):
        values = []
        dims = (1, 1)
        if key in self:
            section = self[key]
            n = section['number']
            dims = section.get('dims', (1, 1))
            for i in range(n):
                topsection = section[f'{i:04}']
                if next(iter(topsection)).isnumeric():
                    subsections = list(iter(topsection.values()))
                else:
                    subsections = [topsection]
                for subsection in subsections:
                    values.append(subsection)
        return values, dims

    def set_values(self, key, values, dims):
        section_s = {'number': len(values), 'dims': list(dims)}
        section = {}
        value_type = self.value_types[key]
        for i, value0 in enumerate(values):
            value = value0.astype(np.float32)
            value_dict = {value_type: value.flatten()}
            if value_type == 'polygon':
                value_dict |= {
                    'center': get_center(value),
                    'area': get_area(value),
                    'angle': get_norm_rotation_angle_deg(value),
                }
            section[f'{i:04}'] = value_dict
        self[key] = section_s
        if key.endswith('s'):
            key_single = key[:-1]
        else:
            key_single = key
        self[key_single] = section

    def set_value(self, key, value):
        self[key] = {key: value}

    def save(self):
        config = configparser.ConfigParser()
        for key in self:
            section = self[key]
            for subkey in section:
                value = convert_to_string(section[subkey])
                if isinstance(subkey, int) or str.isnumeric(subkey):
                    merged_key = key + '.' + str(subkey)
                    config[merged_key] = value
                else:
                    if key not in config:
                        config[key] = {}
                    config[key][subkey] = value

        with open(self.filename, 'w') as file:
            config.write(file)


def get_config_subitems(items):
    dct = {}
    for key in items:
        value = items[key]
        if ',' in value:
            value = np.fromstring(value, sep=',')
        dct[key] = auto_convert(value)
    return dct


def auto_convert(value, to_int=None):
    if isinstance(value, (list, np.ndarray)):
        to_int = True
        for val in value:
            if isinstance(val, str):
                if '.' in val:
                    to_int = False
            else:
                if val != int(val):
                    to_int = False
        return [auto_convert_value(val, to_int=to_int) for val in value]
    else:
        return auto_convert_value(value, to_int=to_int)


def auto_convert_value(value0, to_int=None):
    value = value0
    try:
        if to_int is None and isinstance(value0, str):
            to_int = ('.' not in value0)
        value = float(value0)
        if to_int:
            value = int(value)
    except ValueError:
        pass
    return value


def convert_to_string(value0):
    if isinstance(value0, dict):
        value = {}
        for key in value0:
            value[key] = convert_to_string(value0[key])
    elif isinstance(value0, (list, np.ndarray)):
        value = ','.join(map(str, value0))
    else:
        value = str(value0)
    return value


def add_dict_tree_indexed(metadata, keys, value):
    key = keys[0]
    if len(keys) > 1:
        if not keys[0].isnumeric() and keys[1].isnumeric():
            metadata = metadata[key]
        else:
            if int(key) not in metadata:
                metadata[int(key)] = {}
            metadata = metadata[int(key)]
        add_dict_tree_indexed(metadata, keys[1:], value)
    elif key.isnumeric():
        metadata[int(key)] = value
    else:
        metadata[key] = value


if __name__ == '__main__':
    filename = 'E:/Personal/Crick/VP/magc/annotation_test/wafer_example1.magc'
    outfilename = 'E:/Personal/Crick/VP/magc/annotation_test/wafer_example2.magc'
    magc_file = MagcFile(filename)
    magc_file.load()
    magc_file.filename = outfilename
    magc_file.save()
