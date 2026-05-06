from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PyQt5.QtCore import QPointF

from labelme.shape import Shape

from ._geometry import compute_polygon_from_mask


@dataclass
class Detection:
    bbox: tuple[float, float, float, float] | None = None
    mask: NDArray[np.bool_] | None = None
    label: str | None = None
    description: str | None = None


def _build_shape(
    shape_type: Literal["rectangle", "polygon", "mask"],
    points: list[QPointF],
    *,
    mask: NDArray[np.bool_] | None = None,
    label: str | None = None,
    description: str | None = None,
) -> Shape:
    shape = Shape(
        label=label,
        shape_type=shape_type,
        mask=mask,
        description=description,
    )
    shape.points = points
    shape.point_labels = [1] * len(points)
    shape.close()
    return shape


def _shape_from_detection(
    detection: Detection,
    shape_type: Literal["rectangle", "polygon", "mask"],
) -> Shape | None:
    if shape_type == "rectangle":
        if detection.bbox is None:
            return None
        xmin, ymin, xmax, ymax = detection.bbox
        return _build_shape(
            shape_type="rectangle",
            points=[QPointF(xmin, ymin), QPointF(xmax, ymax)],
            label=detection.label,
            description=detection.description,
        )
    if shape_type == "polygon":
        if detection.mask is None:
            return None
        polygon = compute_polygon_from_mask(mask=detection.mask)
        if detection.bbox is not None:
            polygon = polygon + np.array(
                [detection.bbox[0], detection.bbox[1]], dtype=np.float32
            )
        if len(polygon) < 2:
            return None
        return _build_shape(
            shape_type="polygon",
            points=[QPointF(p[0], p[1]) for p in polygon],
            label=detection.label,
            description=detection.description,
        )
    if shape_type == "mask":
        if detection.bbox is None or detection.mask is None:
            return None
        xmin = int(detection.bbox[0])
        ymin = int(detection.bbox[1])
        xmax = int(detection.bbox[2])
        ymax = int(detection.bbox[3])
        return _build_shape(
            shape_type="mask",
            points=[QPointF(xmin, ymin), QPointF(xmax, ymax)],
            mask=detection.mask,
            label=detection.label,
            description=detection.description,
        )
    raise ValueError(f"Unsupported shape_type: {shape_type!r}")


def shapes_from_detections(
    detections: list[Detection],
    shape_type: Literal["rectangle", "polygon", "mask"],
) -> list[Shape]:
    shapes: list[Shape] = []
    for detection in detections:
        shape = _shape_from_detection(detection=detection, shape_type=shape_type)
        if shape is not None:
            shapes.append(shape)
    return shapes
