from __future__ import annotations

import math

import numpy as np
import pytest

from labelme._automation._geometry import compute_circle_from_mask


def test_compute_circle_from_mask_returns_none_when_empty() -> None:
    assert compute_circle_from_mask(mask=np.zeros((10, 10), dtype=bool)) is None


def test_compute_circle_from_mask_centroid_and_area_equivalent_radius() -> None:
    mask = np.zeros((11, 11), dtype=bool)
    mask[0:3, 0:3] = True

    circle = compute_circle_from_mask(mask=mask)

    assert circle is not None
    assert circle.cx == pytest.approx(1)
    assert circle.cy == pytest.approx(1)
    assert circle.radius == pytest.approx(math.sqrt(9 / math.pi))
