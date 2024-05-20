import json
import os.path
import yaml

from napari_mass.util import *


class FileDict(dict):
    def __init__(self, filename=None, load=True):
        super().__init__()
        self.filename = filename
        if load:
            self.load()

    def load(self):
        if self.filename is not None and os.path.exists(self.filename):
            ext = os.path.splitext(self.filename)[-1].lower()
            with open(self.filename, 'r') as infile:
                if ext == '.json':
                    dct = dict_keys_to_int(json.load(infile, object_hook=json_keys_to_int))
                else:
                    dct = yaml.load(infile, Loader=yaml.Loader)
                self.update(dct)
                return True
        return False

    def save(self, sort=False):
        if self.filename is not None and self.filename != '':
            ext = os.path.splitext(self.filename)[-1]
            dct = dict(self)
            if sort:
                dct = sorted(dct.items(), key=lambda x: tuple(map(int, x[0].split('_'))))
            with open(self.filename, 'w') as outfile:
                if ext == '.json':
                    json.dump(dict_keys_to_string(dct), outfile, indent=4)
                else:
                    yaml.dump(dct, outfile, default_flow_style=None, sort_keys=False)


def json_keys_to_int(data0):
    if isinstance(data0, dict):
        data = {}
        for key, value in data0.items():
            if key.isnumeric():
                key = int(key)
            data[key] = value
    else:
        data = data0
    return data
