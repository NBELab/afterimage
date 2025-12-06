import pickle
from pathlib import Path

import numpy as np


class RgbOppConv:
    """ RGB <-> Opponent space converter. """
    def __init__(self, mat: np.ndarray):
        self.rgb2opp_mat = mat
        self.opp2rgb_mat = np.linalg.inv(self.rgb2opp_mat)

    def rgb_to_opp(self, img):
        return img @ self.rgb2opp_mat.T

    def opp_to_rgb(self, img):
        return img @ self.opp2rgb_mat.T


def normalize_to_range(so, do, so_mask=False, clip_val=None):
    if do.max() - do.min() < 0.01:
        return do
    if so.max() - so.min() < 0.01:
        return np.clip(do, -clip_val, clip_val) if clip_val else do
    norm_do = (do - do.min()) / (do.max() - do.min()) * (so.max() - so.min()) + so.min()
    if so_mask:
        mask = np.abs(so) < 0.01
        norm_do[mask] = do[mask]
    return norm_do


def load_inputs_outputs(results_path_or_obj, steps, so_mask, clip_opp=True, norm=True):
    results = results_path_or_obj
    if isinstance(results_path_or_obj, (str, Path)):
        with open(results_path_or_obj / 'results.pickle', 'rb') as f:
            results = pickle.load(f)
    sample_rate = results['sample_rate']
    steps = [s // sample_rate for s in steps]
    rg = results['rg_out'][steps]
    by = results['by_out'][steps]
    inp = results['inputs'][steps]
    conv = RgbOppConv(results['rgb2opp'])
    rg_so, by_so, lum_so = np.moveaxis(conv.rgb_to_opp(inp), -1, 0)
    lum = results['lum_out'][steps]
    lum_clip_val = sum(conv.rgb2opp_mat[2]) if clip_opp else None
    lum_norm = [normalize_to_range(lum_so_, lum_, so_mask=so_mask, clip_val=lum_clip_val) for lum_so_, lum_ in zip(lum_so, lum)]
    if norm:
        rg_clip_val = conv.rgb2opp_mat[0, 0] if clip_opp else None
        by_clip_val = conv.rgb2opp_mat[1, -1] if clip_opp else None
        rg_norm = [normalize_to_range(rg_so_, rg_, so_mask=so_mask, clip_val=rg_clip_val) for rg_so_, rg_ in zip(rg_so, rg)]
        by_norm = [normalize_to_range(by_so_, by_, so_mask=so_mask, clip_val=by_clip_val) for by_so_, by_ in zip(by_so, by)]
        out = conv.opp_to_rgb(np.stack([rg_norm, by_norm, lum_norm], axis=-1))
    else:
        out = conv.opp_to_rgb(np.stack([rg, by, lum_norm], axis=-1))
    return inp, out, conv
