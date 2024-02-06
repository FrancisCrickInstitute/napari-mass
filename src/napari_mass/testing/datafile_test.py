from napari_mass.file.DataFile import DataFile
from napari_mass.parameters import *


def modify_datafile(filename):
    data = DataFile(filename)
    samples = data.get_values([DATA_SECTIONS_KEY, '*', 'sample'])
    rois = data.get_values([DATA_SECTIONS_KEY, '*', 'rois', '*'])
    order = data.get_values('serial_order/order')

    data.set_value([DATA_SECTIONS_KEY, '*', 'sample'], 5, {'test'})
    data.set_value([DATA_SECTIONS_KEY, '*', 'rois', '*'], 5, {'test'})
    data.remove_value([DATA_SECTIONS_KEY, '*', 'rois', '*'], 2)
    pass


if __name__ == '__main__':
    filename = 'D:/slides/EM04613/mass/mass.json'
    modify_datafile(filename)
