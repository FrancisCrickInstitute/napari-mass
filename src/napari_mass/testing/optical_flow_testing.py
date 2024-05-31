from features import init_section_features, init_sections_features, get_image_features
from napari_mass.Section import Section, get_section_sizes
from napari_mass.TiffSource import TiffSource
from napari_mass.file.DataFile import DataFile
from napari_mass.image.util import *
from napari_mass.parameters import *
from napari_mass.point_matching import do_section_alignment, print_metrics, align_points_features


def show_stats(metrics, method, source_section, target_section, output=True):
    print(print_metrics(metrics))

    matched_source_points = [source_section.points[s] for s, t in metrics['matches']]
    matched_target_points = [target_section.points[t] for s, t in metrics['matches']]
    matched_image = draw_image_points_overlay(target_section.bin_image, source_section.bin_image,
                                              matched_target_points, matched_source_points, draw_size=3)
    if output:
        save_image(f'matched_{method}.tiff', matched_image)
        show_image(matched_image)

        aligned_image = metrics['overlay_image']
        save_image(f'aligned_{method}.tiff', aligned_image)
        show_image(aligned_image)


def get_overall_sections(sections, target_pixel_size):
    images = []
    size = get_section_sizes(sections, target_pixel_size)[1]
    for section in sections:
        images.append(reshape_image(section.bin_image, size))
    image = np.mean(images, axis=0)

    # get points from overall image
    points0, size_points0, keypoints0, descriptors0 = get_image_features(image)

    for section in sections:
        points, size_points, keypoints, descriptors = get_image_features(section.bin_image)
        align_points_features(size_points, size_points0,
                              descriptors, descriptors0, None, None)
    return image


if __name__ == '__main__':
    folder = 'D:/slides/EM04613/'
    source_filename = folder + 'EM04613_04_20x_WF_Reflection-02-Stitching-01.ome.tif'
    source = TiffSource(source_filename)
    target_pixel_size = [1]
    data_filename = folder + 'mass/data.mass.json'
    data = DataFile(data_filename)
    sample_data = data.get_values([DATA_SECTIONS_KEY, '*', 'sample'])
    min_match_rate = 0.1
    size_range = None
    output = False

    sections = [Section(sample) for sample in sample_data]

    source_section, target_section = sections[1], sections[0]
    init_sections_features(sections, source=source, pixel_size=target_pixel_size,
                           image_function=create_brightfield_detection_image, size_range=size_range)

    # fine align (features/CPD) all sections
    # extract overall image
    # use this to flow align 'normal'? no...

    # TODO: new alignment: use matched points as map / inverse map

    #image = get_overall_sections(sections, target_pixel_size)
    #show_image(image)

    # course CPD alignment
    method = 'cpd'
    transform, metrics = do_section_alignment(source_section, target_section, method=method,
                                              min_match_rate=min_match_rate,
                                              distance_factor=1, w=0.001, max_iter=200, tol=0.1)
    show_stats(metrics, method, source_section, target_section, output)

    # update features
    init_section_features(source_section, image_function=create_brightfield_detection_image, size_range=size_range)

    # fine optical flow alignment
    method = 'sparse_flow'
    transform, metrics = do_section_alignment(source_section, target_section, method=method,
                                              min_match_rate=min_match_rate)
    show_stats(metrics, method, source_section, target_section, output)
