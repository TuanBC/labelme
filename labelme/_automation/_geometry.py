from __future__ import annotations

from typing import NamedTuple

import imgviz
import numpy as np
import skimage
from loguru import logger
from numpy.typing import NDArray


class Circle(NamedTuple):
    cx: float
    cy: float
    radius: float


def compute_circle_from_mask(mask: NDArray[np.bool_]) -> Circle | None:
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    # Area-equivalent radius: matches the mask's pixel area, not its extent.
    # For elongated or sparse masks the resulting circle may be smaller than
    # the tightest enclosing one.
    return Circle(
        cx=float(xs.mean()),
        cy=float(ys.mean()),
        radius=float(np.sqrt(mask.sum() / np.pi)),
    )


def _get_contour_length(contour: NDArray[np.float32]) -> float:
    contour_start: NDArray[np.float32] = contour
    contour_end: NDArray[np.float32] = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


def compute_polygon_from_mask(mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    contours: NDArray[np.float32] = skimage.measure.find_contours(
        np.pad(mask, pad_width=1)
    )
    if len(contours) == 0:
        logger.warning("No contour found, so returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    contour: NDArray[np.float32] = max(contours, key=_get_contour_length)
    POLYGON_APPROX_TOLERANCE: float = 0.004
    polygon: NDArray[np.float32] = skimage.measure.approximate_polygon(
        coords=contour,
        tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
    )
    polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
    polygon = polygon[:-1]  # drop last point that is duplicate of first point

    if 0:
        import PIL.Image

        image_pil = PIL.Image.fromarray(imgviz.gray2rgb(imgviz.bool2ubyte(mask)))
        imgviz.draw.line_(image_pil, yx=polygon, fill=(0, 255, 0))
        for point in polygon:
            imgviz.draw.circle_(image_pil, center=point, diameter=10, fill=(0, 255, 0))
        imgviz.io.imsave("contour.jpg", np.asarray(image_pil))

    return polygon[:, ::-1]  # yx -> xy
