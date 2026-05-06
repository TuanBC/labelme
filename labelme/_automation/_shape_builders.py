from __future__ import annotations

import json
from typing import Literal

import numpy as np
import osam.types
from loguru import logger
from numpy.typing import NDArray
from PyQt5.QtCore import QPointF

from labelme.shape import Shape

from ._geometry import compute_polygon_from_mask


def shape_from_annotation(
    annotation: osam.types.Annotation,
    output_format: Literal["polygon", "mask"],
) -> Shape | None:
    if annotation.mask is None:
        return None

    mask: np.ndarray = annotation.mask

    if output_format == "mask":
        if annotation.bounding_box is None:
            return None
        bb = annotation.bounding_box
        shape = Shape()
        shape.refine(
            shape_type="mask",
            points=[QPointF(bb.xmin, bb.ymin), QPointF(bb.xmax, bb.ymax)],
            point_labels=[1, 1],
            mask=mask,
        )
        shape.close()
        return shape
    elif output_format == "polygon":
        points = compute_polygon_from_mask(mask=mask)
        if len(points) < 2:
            return None
        if annotation.bounding_box is not None:
            bb = annotation.bounding_box
            points = points + np.array([bb.xmin, bb.ymin], dtype=np.float32)
        shape = Shape()
        shape.refine(
            shape_type="polygon",
            points=[QPointF(point[0], point[1]) for point in points],
            point_labels=[1] * len(points),
        )
        shape.close()
        return shape
    raise ValueError(f"Unsupported output_format: {output_format!r}")


def shapes_from_ai_response(
    response: osam.types.GenerateResponse,
    output_format: Literal["polygon", "mask"],
) -> list[Shape]:
    if output_format not in ["polygon", "mask"]:
        raise ValueError(
            f"output_format must be 'polygon' or 'mask', not {output_format}"
        )

    if not response.annotations:
        logger.warning("No annotations returned")
        return []

    annotations = sorted(
        response.annotations,
        key=lambda a: a.score if a.score is not None else 0,
        reverse=True,
    )

    shapes: list[Shape] = []
    for annotation in annotations:
        shape = shape_from_annotation(
            annotation=annotation, output_format=output_format
        )
        if shape is not None:
            shapes.append(shape)
    return shapes


def shapes_from_bboxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    masks: list[NDArray[np.bool_]] | None,
    shape_type: Literal["rectangle", "polygon", "mask"],
) -> list[Shape]:
    shapes: list[Shape] = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        text: str = texts[label]
        xmin, ymin, xmax, ymax = box

        points: list[list[float]] = []
        mask: NDArray[np.bool_] | None = None
        if shape_type == "rectangle":
            points = [[xmin, ymin], [xmax, ymax]]
        elif shape_type == "polygon":
            if masks is None:
                points = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                    [xmin, ymin],
                ]
            else:
                polygon = compute_polygon_from_mask(mask=masks[i])
                points = (polygon + np.array([xmin, ymin], dtype=np.float32)).tolist()
        elif shape_type == "mask":
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            points = [[xmin, ymin], [xmax, ymax]]
            if masks is None:
                mask = np.zeros((ymax - ymin, xmax - xmin), dtype=bool)
            else:
                mask = masks[i]
        else:
            raise ValueError(f"Unsupported shape_type: {shape_type!r}")

        shape = Shape(
            label=text,
            shape_type=shape_type,
            mask=mask,
            description=json.dumps(dict(score=score.item(), text=text)),
        )
        for point in points:
            shape.add_point(QPointF(point[0], point[1]))
        shapes.append(shape)
    return shapes
