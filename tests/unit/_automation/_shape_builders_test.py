from __future__ import annotations

import math

import numpy as np
import pytest

from labelme._automation import Detection
from labelme._automation import shapes_from_detections


def test_shapes_from_detections_circle_with_mask_uses_centroid_and_area() -> None:
    [shape] = shapes_from_detections(
        detections=[
            Detection(
                bbox=(10, 20, 30, 50),
                mask=np.ones((31, 21), dtype=bool),
            )
        ],
        shape_type="circle",
    )

    assert shape.shape_type == "circle"
    expected_cx = 10 + 10
    expected_cy = 20 + 15
    expected_radius = math.sqrt(21 * 31 / math.pi)
    assert shape.points[0].x() == pytest.approx(expected_cx)
    assert shape.points[0].y() == pytest.approx(expected_cy)
    assert shape.points[1].x() == pytest.approx(expected_cx + expected_radius)
    assert shape.points[1].y() == pytest.approx(expected_cy)


def test_shapes_from_detections_circle_without_mask_falls_back_to_inscribed() -> None:
    [shape] = shapes_from_detections(
        detections=[Detection(bbox=(0, 0, 10, 20))],
        shape_type="circle",
    )

    assert shape.shape_type == "circle"
    assert shape.points[0].x() == pytest.approx(5)
    assert shape.points[0].y() == pytest.approx(10)
    assert shape.points[1].x() == pytest.approx(10)
    assert shape.points[1].y() == pytest.approx(10)
