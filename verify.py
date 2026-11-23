"""Replay held-out scores and verify the CLI's paired coordinate system."""
import json
from pathlib import Path
import unittest

import numpy as np

from map_points import NAMES, Registration


class CoordinateTests(unittest.TestCase):
    def test_independent_anchor_agreement(self):
        for scroll in NAMES:
            reg = Registration(scroll)
            anchors = json.loads((reg.folder / 'local-match.json').read_text())['points']
            comparison = json.loads((reg.folder / 'comparison.json').read_text())
            measured = []
            for idx, anchor in enumerate(anchors):
                uv, match = anchor['yx'], np.array(anchor['low_zyx'])
                low, high = reg.surface_point(uv)
                np.testing.assert_allclose(low, match, atol=1e-6, rtol=0)
                original = reg.inverse(high, uv)
                np.testing.assert_allclose(original, low, atol=1e-8, rtol=0)
                if idx >= 5:
                    anchored_high = reg.matrix @ (low + anchor['offset_low_zyx']) + reg.translation
                    displacement = np.linalg.solve(reg.matrix, high - anchored_high)
                    measured.append(float(np.linalg.norm(displacement)))
            np.testing.assert_allclose(measured, comparison['heldout_anchor_errors_listed_voxels'], atol=1e-8)
            self.assertLess(max(measured) * 9.362, 6)
            print(scroll, 'withheld errors in micrometers', np.round(np.array(measured)*9.362, 3))

    def test_roundtrips_fractional_points(self):
        for scroll in NAMES:
            reg = Registration(scroll)
            for uv in [(384.25, 384.75), (160, 608), (200, 392.5)]:
                low, high = reg.surface_point(uv)
                np.testing.assert_allclose(reg.inverse(high, uv), low, atol=1e-8, rtol=0)
                np.testing.assert_allclose(reg.inverse(reg.forward(low)), low, atol=1e-8, rtol=0)

    def test_rejects_missing_field(self):
        for scroll in NAMES:
            reg = Registration(scroll)
            with self.assertRaisesRegex(ValueError, 'partial rescan'):
                reg.forward([0, 0, 0])
            with self.assertRaises(ValueError):
                reg.inverse([float('nan'), 0, 0])
            with self.assertRaises(ValueError):
                reg.surface_point([0, 768])


if __name__ == '__main__':
    unittest.main(verbosity=2)
