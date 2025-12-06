from enum import Enum
from typing import Sequence, NamedTuple, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


CONT_COLORS = {
    'bg': (223, 223, 223),
    'green': (193, 223, 129),
    'orange': (236, 193, 157),
    'blue': (142, 202, 219),
    'pink': (218, 115, 162),
    'after_contour': (196, 196, 196),
}
_opposite_color = {
    'green': 'pink',
    'orange': 'blue',
}
OPPOSITE_COLORS_BY_NAME = dict(**_opposite_color, **{v: k for k, v in _opposite_color.items()})


class Mode(str, Enum):
    pos = 'pos'
    neg = 'neg'
    baseline = 'baseline'
    double_opp = 'double_opp'
    double_same = 'double_same'
    posneg = 'posneg'
    negpos = 'negpos'


def create_contour_inputs(h, w, color, bg_color, mode, inner_color, contour_color, base_radius, wobble_ampl,
                          n_wobbles, thickness, bg_color2=None, opp_color=None, show=False) -> Tuple[np.ndarray, dict]:
    """
    Prepare the blobby contours inputs.

    Args:
        h: image height.
        w: image width.
        color: chromatic contour color.
        bg_color: background color.
        mode: mode.
        inner_color: color inside the chromatic contour.
        contour_color: after-contour color.
        base_radius: base radius for the blob shape.
        wobble_ampl: amplitude for the shape wobbliness.
        n_wobbles: the number of wobbles.
        thickness: contour thickness.
        bg_color2: background color for the after frame.
        opp_color: opponent color for the double-contour mode.
        show: show the generated images.

    Returns:
        A batch of images (2 or 3) and an info dict with the used spec.
    """
    if not isinstance(bg_color, Sequence):
        bg_color = (bg_color,) * 3
    eq_color = (int(np.mean(color)),) * 3
    bg_color = eq_color if bg_color == 'eq' else bg_color
    bg_color2 = eq_color if bg_color2 == 'eq' else bg_color if bg_color2 is None else bg_color2
    inner_color = eq_color if inner_color == 'eq' else bg_color if inner_color is None else inner_color

    contours = _configure_contours(color, bg_color, mode, None, None, opp_color, contour_color)
    imgs = _create_blob(h=h, w=w, contours=contours, bg_color=bg_color, inner_color=inner_color, bg_color2=bg_color2,
                        base_radius=base_radius, wobble_ampl=wobble_ampl, n_wobbles=n_wobbles, thickness=thickness, show=show)
    info = dict(contours=contours, h=h, w=w, bg_color=bg_color, inner_color=inner_color, bg_color2=bg_color2,
                base_radius=base_radius, wobble_ampl=wobble_ampl, n_wobbles=n_wobbles, thickness=thickness)
    return imgs, info


def create_rect_contours(h, w, color, bg_color, mode, inner_color, color_width, contour_width, bg_pad, opp_color,
                         contour_color, show=False) -> Tuple[np.ndarray, dict]:
    """ A rectangular version of the contours. """
    contours = _configure_contours(color, bg_color, mode, color_width, contour_width, opp_color, contour_color)
    imgs = _chromatic_contours((h, w), contours=contours, bg_color=bg_color, bg_pad=bg_pad,
                               inner_color=inner_color, show=show)
    info = dict(contours=contours, h=h, w=w, bg_color=bg_color, inner_color=inner_color, bg_pad=bg_pad)
    return imgs, info


def _configure_contours(color, bg_color, mode, color_width, contour_width, opp_color, contour_color):
    if mode in [Mode.pos, Mode.neg, Mode.baseline]:
        if mode == Mode.baseline:
            contour_color = bg_color
        contours = _single_color_contours(color, contour_color=contour_color, color_width=color_width,
                                          contour_width=contour_width, contour_inside=mode == 'pos')
    elif mode in [Mode.double_same, Mode.double_opp]:
        color_in = color if mode == Mode.double_same else opp_color
        assert color_in is not None
        contours = _double_color_contours(color_out=color, color_in=color_in, contour_color=contour_color,
                                          color_out_width=color_width, contour_width=contour_width,
                                          color_in_width=color_width)
    elif mode in [Mode.posneg, Mode.negpos]:
        contours = _sequential_contours(color, contour_color=contour_color, color_width=color_width,
                                        contour_width=contour_width, inner_first=mode == Mode.posneg)
    else:
        raise ValueError(mode)
    return contours


class Contour(NamedTuple):
    color: tuple
    width: int
    stage: int


def _single_color_contours(color, contour_color, color_width, contour_width, contour_inside):
    contours = [
        Contour(color=color, width=color_width, stage=0),
        Contour(color=contour_color, width=contour_width, stage=1),
    ]
    if contour_inside:
        return contours
    return contours[::-1]


def _double_color_contours(color_out=(200, 100, 150), color_in=(100, 200, 150), contour_color=(100, 100, 100),
                           color_out_width=2, contour_width=2, color_in_width=2):
    return [
        Contour(color=color_out, width=color_out_width, stage=0),
        Contour(color=contour_color, width=contour_width, stage=1),
        Contour(color=color_in, width=color_in_width, stage=0),
    ]


def _sequential_contours(color, contour_color, color_width=2, contour_width=2, inner_first=True):
    contour_stages = [2, 1] if inner_first else [1, 2]
    return [
        Contour(color=contour_color, width=contour_width, stage=contour_stages[0]),
        Contour(color=color, width=color_width, stage=0),
        Contour(color=contour_color, width=contour_width, stage=contour_stages[1]),

    ]


def _chromatic_contours(img_size, contours: List[Contour], bg_color, bg_pad, inner_color, show=True):
    n_imgs = len({c.stage for c in contours})
    imgs = [Image.new("RGB", img_size, color=tuple(bg_color)) for _ in range(n_imgs)]
    draws = [ImageDraw.Draw(img) for img in imgs]
    d = bg_pad
    # contours are ordered from outside inside
    for c in contours:
        draws[c.stage].rectangle([d, d, img_size[0]-1-d, img_size[1]-1-d],
                                 outline=c.color, width=c.width, fill=inner_color if c.stage==0 else bg_color)
        d += c.width
    imgs = [np.array(img) for img in imgs]
    if show:
        plt.imshow(np.concatenate(imgs, axis=1))
        plt.show()
    return np.stack(imgs)


def _create_blob(h, w, contours, bg_color, inner_color, base_radius, wobble_ampl, n_wobbles, thickness=5, bg_color2=None, show=False):
    bg_color2 = bg_color2 or bg_color
    mask = create_blob_mask(h=h, w=w, base_radius=base_radius, wobble_amp=wobble_ampl, num_wobbles=n_wobbles)

    # Dilate to get non-overlapping rings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1))
    dilate1 = cv2.dilate(mask, kernel)  # mask + ring1
    dilate2 = cv2.dilate(dilate1, kernel)  # mask + ring1 + ring2

    rings = [cv2.subtract(dilate2, dilate1), cv2.subtract(dilate1, mask)]  # contour outside shape
    if len(contours) == 3:
        dilate3 = cv2.dilate(dilate2, kernel)
        rings.insert(0, cv2.subtract(dilate3, dilate2))  # next adjacent contour

    n_imgs = len({c.stage for c in contours})
    imgs = [np.full((h, w, 3), bg_color if i==0 else bg_color2, dtype=np.uint8) for i, _ in enumerate(range(n_imgs))]
    for i, c in enumerate(contours):
        if c.stage == 0:
            imgs[c.stage][mask == 255] = inner_color
            for r in range(i+1, len(rings)):
                imgs[c.stage][rings[r] == 255] = inner_color
        imgs[c.stage][rings[i] == 255] = c.color
    if show:
        for img in imgs:
            plt.imshow(img)
            plt.show()
    return np.stack(imgs)


def create_blob_mask(h, w, base_radius, wobble_amp, num_wobbles, num_points=359, rotate=0):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    r = base_radius + wobble_amp * np.sin(num_wobbles * (angles+rotate))
    x = (w//2 + r * np.cos(angles)).astype(np.int32)
    y = (h//2 + r * np.sin(angles)).astype(np.int32)
    pts = np.stack([x, y], axis=-1).reshape(-1, 1, 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=255)
    return mask
