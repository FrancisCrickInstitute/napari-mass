import numpy as np


class TiffTileMetadata:
    def __init__(self, index, x0, y0, x1, y1, filename):
        self.index = index
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.filename = filename

    def set_position(self, size):
        self.position = np.round(np.divide((self.x0, self.y0), size)).astype(int)

    def inside(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def __lt__(self, other):
        # self < other
        return (self.x0 < other.x0 and self.y0 <= other.y0) or (self.x0 <= other.x0 and self.y0 < other.y0)

    def __le__(self, other):
        # self < other
        return self < other or self == other

    def __eq__(self, other):
        return self.x0 == other.x0 and self.y0 == other.y0

    def __hash__(self):
        return self.index

    def __str__(self):
        return f'{self.index}: {self.position}'

    def __repr__(self):
        return str(self)


def calc_tiles_stats(tiles):
    xs, ys = set(), set()
    sizes = []
    for tile in tiles:
        xs.add(tile.x0)
        ys.add(tile.y0)
        sizes.append((tile.x1 - tile.x0, tile.y1 - tile.y0))
    size = np.mean(sizes, 0)
    dx = np.diff(sorted(xs)) if len(xs) > 1 else [size[0]]
    dy = np.diff(sorted(ys)) if len(ys) > 1 else [size[1]]
    offset = np.mean(dx), np.mean(dy)
    overlap = size - offset
    return offset, size, overlap


def sort_tiles(tiles):
    tiles_sorted = []
    xs = set()
    for tile in tiles:
        xs.add(tile.x0)
    for x in sorted(xs):
        xtiles = [tile for tile in tiles if tile.x0 == x]
        tiles_sorted.extend(sorted(xtiles))
    return tiles_sorted
