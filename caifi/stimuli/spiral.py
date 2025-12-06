from pathlib import Path

import PIL
import numpy as np
from matplotlib import pyplot as plt


def load_spiral(color=(255, 0, 0), bg_color=255, cont_color=0, cont_bg=255, show=False):
    """
    Load the spiral inputs and adjust the colors.

    Args:
        color: spiral color.
        bg_color: background color.
        cont_color: test contour color.
        cont_bg: background color for the test contour.
        show: show the images.

    Returns:
        A list of [chromatic image, full contour, outer contour, inner contour] and an info dict with the spec.
    """
    root = Path(__file__).parents[2] / 'imgs' / 'spiral'
    color_img = root / 'redspiral_50x50.png'
    cont_full = root / 'cont_full_50x50.png'
    cont_out = root / 'cont_outer_50x50.png'
    cont_in = root / 'cont_inner_50x50.png'

    info = dict(color=color, bg_color=bg_color, cont_color=cont_color, cont_bg=cont_bg)

    img = np.array(PIL.Image.open(color_img).convert('RGB'))
    color_mask = np.all(img==(255, 0, 0), axis=-1)
    img[color_mask] = color
    img[~color_mask] = bg_color

    def update_cont(cont_img):
        cont = np.array(PIL.Image.open(cont_img).convert('RGB'))
        cont[cont == 0] = cont_color
        cont[cont == 255] = cont_bg
        return cont

    cont_full, cont_out, cont_in = [update_cont(cont_img) for cont_img in [cont_full, cont_out, cont_in]]
    if show:
        plt.imshow(np.concatenate([img, cont_full, cont_out, cont_in], axis=1))
        plt.show()
    return [img, cont_full, cont_out, cont_in], info
