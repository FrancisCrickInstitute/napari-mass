from napari_mass.Section import Section, get_section_sizes, init_section_features
from napari_mass.TiffSource import TiffSource
from napari_mass.file.DataFile import DataFile
from napari_mass.image.util import *
from napari_mass.parameters import *
from napari_mass.point_matching import align_sections_metrics


if __name__ == '__main__':
    folder = 'D:/slides/EM04613/'
    source_filename = folder + 'EM04613_04_20x_WF_Reflection-02-Stitching-01.ome.tif'
    source = TiffSource(source_filename)
    target_pixel_size = [2]
    data_filename = folder + 'mass/data.mass.json'
    data = DataFile(data_filename)
    sample_data = data.get_values([DATA_SECTIONS_KEY, '*', 'sample'])[:2]
    sections = [Section(sample) for sample in sample_data]
    target_size = get_section_sizes(sections, target_pixel_size)[1]
    init_section_features(sections, source=source, pixel_size=target_pixel_size, target_size=target_size,
                          image_function=create_brightfield_detection_image, show_stats=False)
    transform, metrics = align_sections_metrics(sections[1], sections[0], matching_methods=['flow'])
    print('match_rate', metrics['match_rate'])
