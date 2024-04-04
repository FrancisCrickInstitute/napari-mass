import os

from napari_mass.file.DataFile import DataFile
from napari_mass.file.MagcFile import MagcFile


def datafile_to_magc(infilename, outfilename):
    datafile = DataFile(infilename)
    magc = MagcFile(outfilename)
    magc.from_datafile_dict(datafile)
    magc.save()


if __name__ == '__main__':
    infilename = 'D:/slides/EM04613/mass/data_single_roi.mass.json'
    outfilename = os.path.splitext(infilename)[0] + '.magc'
    datafile_to_magc(infilename, outfilename)
