from napari_mass.TiffSource import TiffSource


def open_as_daskarray(filename):
    source = TiffSource(filename)
    data = source.as_daskarray()
    print(data)


if __name__ == '__main__':
    filename = 'D:/slides/EM04613/EM04613_04_20x_WF_Reflection-02-Stitching-01.ome.tif'
    open_as_daskarray(filename)
