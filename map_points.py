"""Map native voxel centers between the two acquisitions of a Herculaneum scroll."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ('PHerc0846A', 'PHerc1203')


def vector_in(value, shape, name):
    v = np.asarray(value, dtype=float)
    if v.shape != (len(shape),) or not np.isfinite(v).all():
        raise ValueError(f'{name} requires {len(shape)} finite numbers')
    if (v < 0).any() or (v > np.asarray(shape) - 1).any():
        raise ValueError(f'{name} is outside its voxel-center bounds {shape}')
    return v


class Registration:
    def __init__(self, scroll):
        if scroll not in NAMES:
            raise ValueError(f'Unknown scroll {scroll}')
        self.folder = ROOT / 'calibrations' / scroll
        spec = json.loads((self.folder / 'coordinates.json').read_text())
        self.listed_shape = spec['listed']['shape_zyx']
        self.rescan_shape = spec['rescan']['shape_zyx']
        anchors = json.loads((self.folder / 'local-match.json').read_text())
        self.matrix = np.array(anchors['high_from_low_matrix'])
        self.translation = np.array(anchors['high_from_low_offset'])
        self.width = anchors['surface_width']
        fit = anchors['points'][:5]
        self.fit = np.linalg.lstsq([self.design(a['yx']) for a in fit],
                                  [a['offset_low_zyx'] for a in fit], rcond=None)[0]

    def design(self, patch_yx):
        y, x = vector_in(patch_yx, (self.width, self.width), 'patch row and column')
        return np.array([1.0, (y - 384) / 224, (x - 384) / 224])

    def offset(self, patch_yx):
        return np.zeros(3) if patch_yx is None else self.design(patch_yx) @ self.fit

    def forward(self, listed_zyx, patch_yx=None):
        p = vector_in(listed_zyx, self.listed_shape, 'listed scan')
        q = self.matrix @ (p + self.offset(patch_yx)) + self.translation
        return vector_in(q, self.rescan_shape, 'partial rescan')

    def inverse(self, rescan_zyx, patch_yx=None):
        q = vector_in(rescan_zyx, self.rescan_shape, 'partial rescan')
        p = np.linalg.solve(self.matrix, q - self.translation) - self.offset(patch_yx)
        return vector_in(p, self.listed_shape, 'listed scan')

    def surface_point(self, patch_yx):
        y, x = vector_in(patch_yx, (self.width, self.width), 'patch row and column')
        y0, x0 = int(y), int(x)
        dy, dx = y - y0, x - x0
        y1, x1 = min(y0 + 1, self.width - 1), min(x0 + 1, self.width - 1)
        with np.load(self.folder / 'listed-sheet.npz', allow_pickle=False) as data:
            xyz, valid = data['zyx'], data['valid']
            position = np.zeros(3)
            for row, wy in ((y0, 1 - dy), (y1, dy)):
                for col, wx in ((x0, 1 - dx), (x1, dx)):
                    if not wx * wy:
                        continue
                    if not valid[row, col]:
                        raise ValueError('The upstream sheet has no valid sample here')
                    position += (wx * wy) * xyz[:, row, col].astype(float)
        return position, self.forward(position, (y, x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('forward', 'inverse', 'surface'):
        cmd = sub.add_parser(name)
        cmd.add_argument('scroll', choices=NAMES)
        if name == 'surface':
            cmd.add_argument('yx', nargs=2, type=float, metavar='PIXEL', help='row then column')
        else:
            cmd.add_argument('zyx', nargs=3, type=float, metavar='VOXEL', help='z then y then x')
            cmd.add_argument('--patch-yx', nargs=2, type=float, metavar=('ROW', 'COLUMN'),
                             help='apply the residual fit for this pilot patch at the stated row and column')
    args = parser.parse_args()
    reg = Registration(args.scroll)
    try:
        if args.mode == 'surface':
            uv = args.yx
            low, high = reg.surface_point(uv)
        else:
            uv = args.patch_yx
            if args.mode == 'forward':
                low, high = np.array(args.zyx), reg.forward(args.zyx, uv)
            else:
                low, high = reg.inverse(args.zyx, uv), np.array(args.zyx)
    except ValueError as error:
        parser.exit(2, str(error) + '\n')
    out = {'scroll': args.scroll, 'listed_zyx': low.tolist(), 'rescan_zyx': high.tolist(),
           'mapping': 'coarse affine' if uv is None else 'local sheet correction', 'patch_yx': uv}
    if uv is not None:
        out['inside_anchor_rectangle'] = all(160 <= c <= 608 for c in uv)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
