from napari_mass.file.DataFile import DataFile
from napari_mass.parameters import *


def modify_datafile(filename):
    data = DataFile(filename)
    samples = data.get_values([DATA_SECTIONS_KEY, '*', 'sample'])
    rois = data.get_values([DATA_SECTIONS_KEY, '*', 'rois', '*'])
    focus = data.get_values([DATA_SECTIONS_KEY, '*', 'focus', '*'])
    order = data.get_values('serial_order/order')

    template_rois = data.get_values([DATA_TEMPLATE_KEY, 'rois', '*'])
    template_focus = data.get_values([DATA_TEMPLATE_KEY, 'focus', '*'])

    data.set_value([DATA_SECTIONS_KEY, '*', 'sample'], 5, {'test'})
    data.set_value([DATA_SECTIONS_KEY, '*', 'rois', '*'], 5, {'test'})
    data.remove_value([DATA_SECTIONS_KEY, '*', 'rois', '*'], 2)
    data.remove_value([DATA_SECTIONS_KEY, '*', 'sample'], 7)
    data.add_value([DATA_SECTIONS_KEY, '*', 'sample'], {'new'})

    data.add_value([DATA_TEMPLATE_KEY, 'sample'], {'new'})
    data.add_value([DATA_TEMPLATE_KEY, 'sample'], {'new2'})
    template_sample = data.get_values([DATA_TEMPLATE_KEY, 'sample'])

    data.remove_value([DATA_TEMPLATE_KEY, 'rois', '*'], 0)

    data.remove_value([DATA_SECTIONS_KEY, '*'], 2)

    pass


if __name__ == '__main__':
    filename = 'D:/slides/EM04676_02/mass/data.mass.json'
    modify_datafile(filename)
