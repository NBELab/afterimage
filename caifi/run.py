import dataclasses
import functools
import json
import pickle
from typing import Literal, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from nengo import Network, Node, Connection, Probe, Simulator
from nengo.processes import PresentInput
import nengo_ocl

from caifi.modules import mask_module, pos_relu_ens, modulated_edge_triggered_diffusion, adaptive_double_opponency, conv_filter2d
from caifi.util import load_inputs_outputs, RgbOppConv


@dataclasses.dataclass
class Cfg:
    # Laplacian multiplicative factors for edge detection:
    krg: float = 5
    kby: float = 5
    klum: float = 2

    # Time constants for adaptive high pass filters
    rg_hp_tau: float = 1
    by_hp_tau: float = 1
    lum_hp_tau: float = None

    # diffusion parameters
    diffusion_tau: float = 0.01  # recurrent connection time constant
    diffusion_kin: float = 0.25  # diffusion coefficient for input
    diffusion_kr: float = 2  # diffusion coefficient for recurrent connection
    diffusion_n_neurons: int = 200
    diffusion_neuron_type: Literal['lif', 'relu', 'direct'] = 'lif'

    so_inhibit_edge: bool = True  # whether to enable inhibition by SO of edge amplification
    ampl_factor: Optional[float] = 10.  # edge amplification factor
    mask_thresh: float = 0.1    # input threshold for mask

    seed: int = 42
    extra: dict = dataclasses.field(default_factory=dict)


M = np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0],
             [-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
             [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]])


def build_model(opp_imgs: np.ndarray, cfg: Cfg, t_present: float, probe: bool) -> Tuple[Network, Tuple[Probe], Dict]:
    """
    Build the SNN model.

    Args:
        opp_imgs: a batch of input images in opponent space (bhwc).
        cfg: config object.
        t_present: time in seconds to present each image.
        probe: whether to probe intermediate outputs for debug.

    Returns:
        Network object, a tuple of output probes for rg, by and lum channels, and debug probes dict.
    """
    h, w = opp_imgs.shape[1:-1]
    size = h*w
    # pad inputs for convolution (nengo Convolution only supports 'same' with zero padding).
    opp_padded, unpad_indices, (h, w) = _pad_imgs(opp_imgs)
    rg_so_padded, by_so_padded, lum_padded = np.moveaxis(opp_padded, -1, 0)
    probes = {}
    with Network(seed=cfg.seed) as net:
        rg_so_padded = Node(PresentInput(rg_so_padded, presentation_time=t_present))
        by_so_padded = Node(PresentInput(by_so_padded, presentation_time=t_present))
        lum_padded = Node(PresentInput(lum_padded, presentation_time=t_present))

        rg_do_ada, probes['rg_do_ada'] = adaptive_double_opponency(rg_so_padded, h, w, k=cfg.krg, tau=cfg.rg_hp_tau,
                                                                   probe=probe)
        by_do_ada, probes['by_do_ada'] = adaptive_double_opponency(by_so_padded, h, w, k=cfg.kby, tau=cfg.by_hp_tau,
                                                                   probe=probe)
        lum_do_ada, probes['lum_do_ada'] = adaptive_double_opponency(lum_padded, h, w, k=cfg.klum, tau=cfg.lum_hp_tau,
                                                                     probe=probe)

        ctrl = None
        if cfg.ampl_factor:
            lum_edge_mask, probes['lum_edge_mask'] = mask_module(lum_do_ada, size, thresh=cfg.mask_thresh, probe=probe)
            if cfg.so_inhibit_edge:
                # identify pixels with chromatic single-opponent signal
                rg_so_mask, _ = mask_module(rg_so_padded[unpad_indices], size, thresh=cfg.mask_thresh)
                by_so_mask, _ = mask_module(by_so_padded[unpad_indices], size, thresh=cfg.mask_thresh)
                so_mask = pos_relu_ens(size)
                Connection(rg_so_mask.output, so_mask.input)
                Connection(by_so_mask.output, so_mask.input)
                # dilate the so mask by 1 pixel using 3x3 box filter (2x2 doesn't play nice) and send a strong
                # inhibitory signal to luminance edge mask.
                Connection(so_mask.output, lum_edge_mask.input,
                           transform=conv_filter2d(-100 * np.ones((3, 3)), h, w, padding='same'))

            ctrl = pos_relu_ens(size)
            Connection(lum_edge_mask.output, ctrl.input)
            ctrl.add_output('ampl', function=lambda d: cfg.ampl_factor * d)
            if probe:
                probes['edge_mask_ctrl'] = Probe(ctrl.output, synapse=0.1)
                probes['edge_mask_ctrl_ampl'] = Probe(ctrl.ampl, synapse=0.1)
            ctrl = ctrl.ampl

        diffusion = functools.partial(modulated_edge_triggered_diffusion, h=h, w=w, tau=cfg.diffusion_tau,
                                      dk_in=cfg.diffusion_kin, dk_r=cfg.diffusion_kr, dnn=cfg.diffusion_n_neurons,
                                      diffusion_neuron_type=cfg.diffusion_neuron_type, probe=probe)
        rg_do_diffusion_output, p_rg_out, probes['rg_do_diffusion_in'] = diffusion(rg_do_ada, ctrl)
        by_do_diffusion_output, p_by_out, probes['by_do_diffusion_in'] = diffusion(by_do_ada, ctrl)
        lum_do_diffusion_output, p_lum_out, _ = diffusion(lum_do_ada, inp_ctrl=None)
        probes = {k: v for k, v in probes.items() if v is not None}
        return net, (p_rg_out, p_by_out, p_lum_out), probes


def run(cfg: Cfg, imgs: np.ndarray, t_present: float, rgb2opp_mat=M, ocl=False, probe=False, show=True,
        save_dir=None, exp_name=None, override_dir=False, save_sample_rate=10):
    """
    Build and compile the network, and run the simulation.

    Args:
        cfg: config object.
        imgs: a batch of input RGB images.
        t_present: time in seconds to present each image.
        rgb2opp_mat: transformation matrix from rgb to opponent space.
        ocl: whether to use OpenCL acceleration.
        probe: whether to probe intermediate outputs for debug.
        show: whether to show predicted percepts.
        save_dir: dir path to save results.
        exp_name: experiment name for results subdir.
        override_dir: whether to allow overriding an existing results dir.
        save_sample_rate: temporal sampling rate for outputs to save.
    """
    assert (show or save_dir) and (exp_name or not save_dir)
    print('Running experiment:', exp_name)
    if show:
        plt.imshow(np.concatenate(list(imgs), axis=1))
        plt.show()

    opp_imgs = RgbOppConv(rgb2opp_mat).rgb_to_opp(imgs)

    exp_dir = None
    if save_dir:
        exp_dir = Path(save_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=override_dir)

    net, (p_rg_out, p_by_out, p_lum_out), probes = build_model(opp_imgs, cfg, t_present, probe)

    # run the simulation
    sim_cls = nengo_ocl.Simulator if ocl else Simulator
    t_run = len(imgs) * t_present
    with sim_cls(net) as sim:
        sim.run(t_run+sim.dt)

    # save / visualize results
    _save_and_vis(sim, imgs, p_rg_out, p_by_out, p_lum_out, t_present, cfg, rgb2opp_mat,
                  show, exp_dir, save_sample_rate, probes)


def _pad_imgs(imgs: np.ndarray):
    """ Pad input images. """
    assert len(imgs.shape) == 4 and imgs.shape[-1] == 3
    h, w = imgs.shape[1:3]

    opp_padded = np.pad(imgs, ([0], [1], [1], [0]), mode='edge')

    pad_mask = np.ones((h + 2, w + 2))
    pad_mask[1:-1, 1:-1] = 0
    # 1d indices to retrieve the original region from flattened padded image
    unpad_indices = np.where(pad_mask.flatten() == 0)[0].tolist()
    return opp_padded, unpad_indices, (h, w)


def _save_and_vis(sim, imgs, p_rg_out, p_by_out, p_lum_out, t_present, cfg, rgb2opp_mat,
                  show, save_dir, save_sample_rate, probes, ncols=10):
    dt = sim.dt
    h, w = imgs[0].shape[:2]
    sim_steps = sim.data[p_rg_out].shape[0]

    def concat_inputs(inputs):
        return np.concatenate([np.repeat(inp[None, ...], repeats=int(t_present / dt + 1e-6)+1, axis=0) for inp in inputs])

    data = {
        'inputs': concat_inputs(imgs)[::save_sample_rate],
        'rg_out': sim.data[p_rg_out].reshape((-1, h, w))[::save_sample_rate],
        'by_out': sim.data[p_by_out].reshape((-1, h, w))[::save_sample_rate],
        'lum_out': sim.data[p_lum_out].reshape((-1, h, w))[::save_sample_rate],
        'rgb2opp': rgb2opp_mat,
        'sample_rate': save_sample_rate
    }
    for k, p in probes.items():
        data[k] = sim.data[p].reshape((-1, h, w))[::save_sample_rate]

    steps = np.linspace(0, sim_steps-1, ncols+1, endpoint=True).astype(int).tolist()[1:]
    vis_data = {}
    vis_data['inputs'], out_raw, _ = load_inputs_outputs(data, steps, so_mask=False, clip_opp=False, norm=False)
    _, vis_data['out_norm_masked'], _ = load_inputs_outputs(data, steps, so_mask=True, clip_opp=True, norm=True)
    _, vis_data['out_norm_unmasked'], _ = load_inputs_outputs(data, steps, so_mask=False, clip_opp=True, norm=True)
    for k, p in probes.items():
        vis_data[k] = sim.data[p].reshape((-1, h, w))[steps]
    if probes:
        vis_data['out_raw'] = out_raw
        vis_data['rg_out'] = sim.data[p_rg_out].reshape((-1, h, w))[steps]
        vis_data['by_out'] = sim.data[p_by_out].reshape((-1, h, w))[steps]
        vis_data['lum_out'] = sim.data[p_lum_out].reshape((-1, h, w))[steps]

    nrows = len(vis_data)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    for r, (k, d) in enumerate(vis_data.items()):
        axs[r, 0].set_title(k)
        for c, s in enumerate(steps):
            axs[r, c].axis('off')
            axs[r, c].imshow(d[c])
    for c, s in enumerate(steps):
        axs[0, c].set_title(f'{s*dt:.2f}s')
    if show:
        plt.show()

    if save_dir:
        with open(save_dir / f'results.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open(save_dir / 'cfg.json', 'w') as f:
            json.dump(dataclasses.asdict(cfg), f, indent=4)
        fig.savefig(str(save_dir / 'vis.png'))
        print(f'saved results at  {save_dir}')

    return data
