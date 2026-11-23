"""Re-measure a single historical CT landmark from the two public acquisitions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import product
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from scipy import ndimage as nd, signal

from map_points import NAMES, Registration


@lru_cache(maxsize=48)
def chunk(url, key, shape):
    path = url + '/'.join(map(str, key))
    try:
        with urlopen(path, timeout=60) as response:
            data = response.read()
    except HTTPError as exc:
        if exc.code != 404:
            raise
        return np.zeros(shape, 'u1')
    return np.frombuffer(data, 'u1').reshape(shape)


def sample(url, coords, meta):
    size = np.array(meta['chunks'])
    if meta['compressor'] is not None or meta['dtype'] != '|u1' or meta['order'] != 'C':
        raise ValueError('This recheck expects the documented uncompressed uint8 volume')
    flat = coords.reshape(3, -1)
    lo = np.floor(flat.min(1)).astype(int) - 1
    hi = np.ceil(flat.max(1)).astype(int) + 2
    if (lo < 0).any() or (hi > meta['shape']).any():
        raise ValueError('Cropped box crosses scan bounds')
    box = np.zeros(hi - lo, 'u1')
    ranges = [range(a // c, (b - 1) // c + 1) for a,b,c in zip(lo, hi, size)]
    def fill(key):
        begin = np.asarray(key) * size
        start, end = np.maximum(begin, lo), np.minimum(begin + size, hi)
        dst = tuple(slice(a,b) for a,b in zip(start - lo, end - lo))
        src = tuple(slice(a,b) for a,b in zip(start - begin, end - begin))
        box[dst] = chunk(url, key, tuple(size))[src]
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(fill, product(*ranges)))
    return nd.map_coordinates(box.astype('f4'), coords - lo[:, None, None, None], order=1, prefilter=False)


def normalized_ct(data):
    high = data - nd.gaussian_filter(data, 2)
    return high / np.sqrt(nd.gaussian_filter(high**2, 3) + 9)


def relative_peak(template, search):
    ref = normalized_ct(template)[3:-3,3:-3,3:-3].copy()
    ref -= ref.mean()
    src = normalized_ct(search)
    ones = np.ones_like(ref)
    sums = signal.fftconvolve(src, ones, mode='valid')
    sums2 = signal.fftconvolve(src**2, ones, mode='valid')
    den = np.sqrt(np.maximum(sums2 - sums**2 / ref.size, 1e-4)) * np.linalg.norm(ref)
    scores = signal.fftconvolve(src, ref[::-1,::-1,::-1], mode='valid') / den
    peak = np.array(np.unravel_index(np.argmax(scores), scores.shape))
    if (peak <= 1).any() or (peak >= np.array(scores.shape)-2).any():
        raise ValueError('The strongest correlation landed at the search boundary')
    refined = peak.astype(float)
    middle = scores[tuple(peak)]
    for axis in np.eye(3, dtype=int):
        left, right = scores[tuple(peak-axis)], scores[tuple(peak+axis)]
        curvature = left - 2*middle + right
        if curvature < 0:
            refined += axis * np.clip(0.5 * (left-right) / curvature, -0.5, 0.5)
    return refined-(np.array(scores.shape)-1)/2, float(middle)


def run(scroll, index):
    reg = Registration(scroll)
    records = json.loads((reg.folder / 'local-match.json').read_text())['points']
    spec = json.loads((reg.folder / 'coordinates.json').read_text())
    anchor = records[index]
    center = np.array(anchor['low_zyx'])
    shift = np.zeros(3)
    meta = json.loads((reg.folder.parent.parent / 'scan-metadata.json').read_text())
    listed, rescan = [spec[k]['zarr'] for k in ('listed','rescan')]
    stages = []
    def grid(c, r, spacing):
        return c[:, None, None, None] + spacing * (np.indices((2*r+1,)*3, dtype='f4')-r)
    for level in (2, 1, 0):
        step = 2**level
        radius = 22 if level == 0 else 19
        lookup = (grid(center, radius, step) + .5) / step - .5
        low = sample(listed + f'{level}/', lookup, meta[listed][f'{level}/.zarray'])
        trial = grid(center + shift, radius+10, step)
        native = np.einsum('ij,j...->i...', reg.matrix, trial) + reg.translation[:,None,None,None]
        lookup = (native + .5)/(4*step) - .5
        high = sample(rescan + f'{level+2}/', lookup, meta[rescan][f'{level+2}/.zarray'])
        delta, score = relative_peak(low, high)
        shift += step * delta
        stages.append({'listed_pyramid_level': level, 'peak_correlation': score})
    difference = float(np.linalg.norm(shift - anchor['offset_low_zyx']))
    print(json.dumps({'scroll':scroll, 'anchor':index, 'remeasured_shift_zyx':shift.tolist(),
                      'stages':stages, 'difference_from_archive_listed_voxels':difference}, indent=2))
    if difference > .03:
        raise RuntimeError('The fresh measurement differs from the historical match by more than 0.03 listed voxels')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scroll', choices=NAMES)
    ap.add_argument('index', type=int, choices=range(9))
    args = ap.parse_args()
    run(args.scroll, args.index)
