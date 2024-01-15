import numpy as np

from src.napari_mass.image.util import *


class StitchTile:
    def __init__(self, rect, offset0, image, overlap, stitch_channel=None):
        self.rect = rect
        self.offset0 = offset0
        self.x0, self.y0 = rect.x0, rect.y0
        self.position = rect.position
        self.image0 = image
        self.image = int2float_image(image)
        if stitch_channel is not None:
            self.image = self.image[..., stitch_channel]
        self.overlap = overlap
        self.coverlap = (0, 0, 0, 0)

    def get_image_bg_level(self):
        return np.mean(self.image)

    def calc_overlap_score_auto(self, image_threshold):
        self.signal, self.nsignal = self.calc_overlap_score(self.overlap, image_threshold)

    def calc_overlap_score(self, overlap, image_threshold):
        mean_threshold = 0.005
        ox0, oy0, ox1, oy1 = overlap
        signal = 0
        nsignal = 0
        if ox0 > 0:
            signal0 = get_thresholded_mean(self.image[:, :ox0], image_threshold)
            signal += signal0
            if signal0 > mean_threshold:
                nsignal += 1
        if oy0 > 0:
            signal0 = get_thresholded_mean(self.image[:oy0, :], image_threshold)
            signal += signal0
            if signal0 > mean_threshold:
                nsignal += 1
        if ox1 > 0:
            signal0 = get_thresholded_mean(self.image[:, -ox1:], image_threshold)
            signal += signal0
            if signal0 > mean_threshold:
                nsignal += 1
        if oy1 > 0:
            signal0 = get_thresholded_mean(self.image[-oy1:, :], image_threshold)
            signal += signal0
            if signal0 > mean_threshold:
                nsignal += 1
        return signal, nsignal

    def __lt__(self, other):
        # self < other
        return self.nsignal < other.nsignal or (self.nsignal == other.nsignal and self.signal < other.signal)

    def __str__(self):
        return str(self.rect) + f' - {self.coverlap}'

    def __repr__(self):
        return str(self)


class Stitcher:
    def __init__(self, rects, start_rect, end_rect, images, base_overlaps, max_offset, stitch_channel=None, active_stitching=True):
        self.stitch_channel = stitch_channel
        self.active_stitching = active_stitching
        self.tiles = []
        self.max_offset = max_offset
        for rect, image in zip(rects, images):
            offset0 = np.array((rect.x0, rect.y0)) - (start_rect.x0, start_rect.y0)
            ox0 = (rect.x0 > start_rect.x0) * base_overlaps[0]
            ox1 = (rect.x0 < end_rect.x0) * base_overlaps[0]
            oy0 = (rect.y0 > start_rect.y0) * base_overlaps[1]
            oy1 = (rect.y0 < end_rect.y0) * base_overlaps[1]
            overlap = ox0, oy0, ox1, oy1
            self.tiles.append(StitchTile(rect, offset0, image, overlap, stitch_channel))

    def process(self, out_image):
        shape = out_image.shape
        if self.stitch_channel is not None:
            shape = shape[:2]
        out_image0 = np.zeros(shape, dtype=np.float32)
        self.sort()
        offsets = self.find_offsets(out_image0)
        order = [tile.rect.index for tile in self.tiles]
        self.final_stitch(out_image, offsets)
        return order, offsets, self.confidence

    def process_stored(self, out_image, order, offsets):
        tiles2 = []
        for index in order:
            for tile in self.tiles:
                if tile.rect.index == index:
                    tiles2.append(tile)
        self.tiles = tiles2
        self.final_stitch(out_image, offsets)

    def sort(self):
        # sort based on: sum total overlap * #overlaps
        # then iterate checking min dist to existing tile in solution == 1
        tiles = self.tiles
        bg_threshold = np.mean([tile.get_image_bg_level() for tile in tiles])
        [tile.calc_overlap_score_auto(bg_threshold) for tile in tiles]
        tiles.sort(reverse=True)
        tiles2 = tiles[:1]
        scores = [0]
        noverlaps = [0]
        # tile needs to connect - finding maximum connection
        while len(tiles2) < len(tiles):
            best_tile = None
            best_score = None
            best_coverlap = (0, 0, 0, 0)
            best_noverlap = 0
            for tile in tiles:
                if tile not in tiles2 and self.is_connecting_tile(tile, tiles2):
                    score, coverlap, noverlap = self.calc_connection(tile, tiles2, bg_threshold)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_tile = tile
                        best_coverlap = coverlap
                        best_noverlap = noverlap
            best_tile.coverlap = best_coverlap
            tiles2.append(best_tile)
            scores.append(best_score)
            noverlaps.append(best_noverlap)
        self.tiles = tiles2
        self.stitch_scores = scores
        self.stitch_overlaps = noverlaps

    def is_connecting_tile(self, tile, tiles2):
        for tile2 in tiles2:
            if np.linalg.norm(tile2.position - tile.position) == 1:
                return True
        return False

    def calc_connection(self, tile, tiles2, image_threshold):
        score = 0
        coverlap = (0, 0, 0, 0)
        noverlap = 0
        for tile2 in tiles2:
            if np.linalg.norm(tile2.position - tile.position) == 1:
                dx, dy = tile2.position - tile.position
                overlap = np.multiply(tile.overlap, (dx < 0, dy < 0, dx > 0, dy > 0))
                signal, nsignal = tile.calc_overlap_score(overlap, image_threshold)
                score += signal
                coverlap += overlap
                noverlap += nsignal
        return score, coverlap, noverlap

    def find_offsets(self, out_image):
        # stitch using stitch image, store offsets
        offsets = []
        doffsets = []
        confidences = []
        first = True
        for tile, stitch_overlap in zip(self.tiles, self.stitch_overlaps):
            offset0 = tile.offset0
            if self.active_stitching and not first and stitch_overlap > 0:
                offset0 = np.round(offset0 + np.mean(doffsets, 0)).astype(int)
                offset, confidence = \
                    stitch_tile_fast_mask(out_image, tile.image, offset0, tile.coverlap, self.max_offset)
                confidences.append(confidence)
            else:
                offset = offset0
            offsets.append(offset)
            doffset = np.array(offset) - offset0
            doffsets.append(doffset)
            image_set_safe(out_image, tile.image, offset)
            first = False
        self.confidence = float(np.mean(confidences)) if len(confidences) > 0 else 1
        offsets = np.round(offsets - np.mean(doffsets, 0)).astype(int).tolist()
        return offsets

    def final_stitch(self, out_image, offsets):
        # perform final stitch using mean offset + offsetX
        for tile, offset in zip(self.tiles, offsets):
            image_set_safe(out_image, tile.image0, offset)


def stitch_tile_fourier(image, tile, offset0, overlap0):
    # uses subsection of (potentially large) image, w/o loss of accuracy
    # potential: only use overlay part(s) to improve speed?
    margin = 100
    x0, y0 = offset0
    overlap = (x0 > 0, y0 > 0) * overlap0.astype(int)
    if not np.allclose(overlap, 0):
        w0, h0 = get_image_size(image)
        w, h = get_image_size(tile)

        x1 = min(x0 + w + margin, w0)
        y1 = min(y0 + h + margin, h0)
        x0 = x0 - margin
        y0 = y0 - margin

        left = margin
        if x0 < 0:
            left += x0
            x0 = 0
        top = margin
        if y0 < 0:
            top += y0
            y0 = 0
        right = (x1 - x0) - w - left
        bottom = (y1 - y0) - h - top

        image0 = int2float_image(image[y0:y1, x0:x1])
        image1 = cv.copyMakeBorder(int2float_image(tile), top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

        # Use mask:
        #mask = np.zeros(tile.shape, dtype=np.uint8)
        #if overlap[0] != 0:
        #    mask[:, :overlap[0]] = 1
        #if overlap[1] != 0:
        #    mask[:overlap[1], :] = 1
        #mask = cv.copyMakeBorder(mask, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
        #shifts = phase_cross_correlation(image0, image1, reference_mask=mask, overlap_ratio=0.8, return_error=False)

        shifts = phase_cross_correlation(image0, image1, return_error=False)

        offset0 += shifts.astype(int)[0:2]
    return offset0


def stitch_tile_fast_mask(image, tile, offset0, overlap, max_offset):
    # opencv cross correlation using matchTemplate
    # TODO: if using this for stitching large image, take small region from larger image? x/y drift may be problematic
    image_size = get_image_size(image)
    margin = int(np.ceil(max_offset))
    offset1 = offset0 + (margin, margin)
    padded_size = image_size + (margin * 2, margin * 2)
    image0 = reshape_grow_image(int2float_image(image), padded_size)
    image1 = int2float_image(tile)
    mask = create_overlap_mask(overlap, image1.shape[:2])

    result0 = cv.matchTemplate(image0, image1, cv.TM_CCOEFF_NORMED, mask)
    result = np.clip(result0, 0, 1)

    y, x = np.indices(result.shape, sparse=True)
    dist_factor = max_offset
    dist_map = dist_factor / (dist_factor + np.sqrt((x - offset1[0]) ** 2 + (y - offset1[1]) ** 2))
    result2 = (result * dist_map)[:-1, :-1]     # remove edge pixels; sometimes giving false maxima

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result2)
    offset = np.array(max_loc) - (margin, margin)
    doffset = np.array(offset) - offset0
    signal = 0.01 * max_val / np.mean(result)
    confidence = np.clip(signal, 0, 1)
    if np.linalg.norm(doffset) > max_offset or confidence < 0.01:
        print(f'Warning: stitching confidence: {confidence:.3f} distance: {doffset}')
        offset = offset0
    return offset, confidence


def create_overlap_mask(overlap, shape, dtype=np.float32):
    mask = np.zeros(shape, dtype=dtype)
    ox0, oy0, ox1, oy1 = overlap
    if ox0 != 0:
        mask[:, 0: ox0] = 1
    if oy0 != 0:
        mask[0: oy0, :] = 1
    if ox1 > 0:
        mask[:, -ox1:] = 1
    if oy1 > 0:
        mask[-oy1:, :] = 1
    return mask
