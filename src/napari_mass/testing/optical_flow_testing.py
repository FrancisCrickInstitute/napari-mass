from features import init_section_features, init_sections_features
from napari_mass.Section import Section
from napari_mass.TiffSource import TiffSource
from napari_mass.file.DataFile import DataFile
from napari_mass.image.util import *
from napari_mass.parameters import *
from napari_mass.point_matching import do_section_alignment, print_metrics


if __name__ == '__main__':
    folder = 'D:/slides/EM04613/'
    source_filename = folder + 'EM04613_04_20x_WF_Reflection-02-Stitching-01.ome.tif'
    source = TiffSource(source_filename)
    target_pixel_size = [1]
    data_filename = folder + 'mass/data.mass.json'
    data = DataFile(data_filename)
    sample_data = data.get_values([DATA_SECTIONS_KEY, '*', 'sample'])[:2]
    min_match_rate = 0.1

    sections = [Section(sample) for sample in sample_data][:2]
    source_section, target_section = sections[1], sections[0]
    init_sections_features(sections, source=source, pixel_size=target_pixel_size,
                           image_function=create_brightfield_detection_image, show_stats=False)

    # course CPD alignment
    method = 'cpd'
    transform, metrics = do_section_alignment(source_section, target_section, method=method, min_match_rate=min_match_rate,
                                              distance_factor=1, w=0.001, max_iter=200, tol=0.1)
    print(print_metrics(metrics))

    aligned_image = metrics['overlay_image']
    save_image(f'aligned_{method}.tiff', aligned_image)
    show_image(aligned_image)

    matched_source_points = [source_section.points[s] for s, t in metrics['matches']]
    matched_target_points = [target_section.points[t] for s, t in metrics['matches']]
    matched_image = draw_image_points_overlay(target_section.bin_image, source_section.bin_image,
                                              matched_target_points, matched_source_points, draw_size=3)
    save_image(f'matched_{method}.tiff', matched_image)
    show_image(matched_image)
    # update features
    init_section_features(source_section, image_function=create_brightfield_detection_image)

    # fine optical flow alignment
    method = 'flow'
    transform, metrics = do_section_alignment(source_section, target_section, method=method, min_match_rate=min_match_rate)
    print(print_metrics(metrics))

    aligned_image = metrics['overlay_image']
    save_image(f'aligned_{method}.tiff', aligned_image)
    show_image(aligned_image)

    matched_source_points = [source_section.points[s] for s, t in metrics['matches']]
    matched_target_points = [target_section.points[t] for s, t in metrics['matches']]
    matched_image = draw_image_points_overlay(target_section.bin_image, source_section.bin_image,
                                              matched_target_points, matched_source_points, draw_size=3)
    save_image(f'matched_{method}.tiff', matched_image)
    show_image(matched_image)
