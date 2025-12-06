import cv2
import matplotlib.pyplot as plt
import numpy as np

from caifi.stimuli.contours import create_blob_mask


STAR_COLORS = {
    'red': (221, 156, 157),
    'green': (109, 184, 181),
    'center': (175, 175, 175),
    'bg': (200, 200, 200)
}


def create_round_stars(color1, color2, inner_color, bg_color, contour_color, contour_bg_color=None, size=50,
                       base_radius=15, wobble_ampl=5, contour_width=2, inner_contour=False, show=False):
    """
    Create stars inputs.

    Args:
        color1: first star color.
        color2: second star color.
        inner_color: color of the overlapping region.
        bg_color: background color.
        contour_color: test contour color.
        contour_bg_color: background color of the test contour.
        size: image size.
        base_radius: star radius.
        wobble_ampl: star arms amplitude.
        contour_width: contour width.
        inner_contour: whether to add an additional test contour inside.
        show: whether to show the inputs.

    Returns:
        A list of [chromatic image, contour of star1, contour of star2] and an info dict with the spec.
    """
    contour_bg_color = contour_bg_color or bg_color
    info = dict(color1=color1, color2=color2, inner_color=inner_color, bg_color=bg_color, contour_color=contour_color,
                contour_bg_color=contour_bg_color, size=size, base_radius=base_radius, wobble_ampl=wobble_ampl,
                contour_width=contour_width)

    def star_masks(rot):
        mask = create_blob_mask(h=size, w=size, base_radius=base_radius, wobble_amp=wobble_ampl, num_wobbles=4, rotate=rot)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * contour_width + 1, 2 * contour_width + 1))
        dilate = cv2.dilate(mask, kernel)  # mask + ring1
        ring = cv2.subtract(dilate, mask)  # contour just outside shape
        return mask.astype(bool), ring.astype(bool)

    mask1, ring1 = star_masks(np.pi / 8)
    mask2, ring2 = star_masks(3 * np.pi / 8)

    overlap_mask = mask1 & mask2
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1

    img = np.ones((*mask1.shape, 3), dtype=np.uint8)*bg_color
    img[only1, :] = color1
    img[only2, :] = color2
    img[overlap_mask, :] = inner_color

    contour1 = np.ones_like(img) * contour_bg_color
    contour1[ring1] = contour_color

    contour2 = np.ones_like(img) * contour_bg_color
    contour2[ring2] = contour_color

    if inner_contour:
        inner_size = size//10
        for contour in [contour1, contour2]:
            contour[size//2-inner_size: size//2+inner_size,
                    size//2-inner_size: size//2+inner_size, :] = contour_color
            contour[size//2-inner_size+contour_width: size//2+inner_size-contour_width,
                    size//2-inner_size+contour_width: size//2+inner_size-contour_width, :] = contour_bg_color
    if show:
        plt.imshow(np.concatenate([img, contour1, contour2], axis=1))
        plt.show()
    return [img, contour1, contour2], info
