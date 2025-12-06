from typing import Optional, Literal, Tuple

import nengo
import numpy as np
from nengo import Connection, Probe
from nengo.base import NengoObject
from nengo.dists import Uniform
from nengo.networks import EnsembleArray
from nengo.neurons import NeuronType
from nengo.params import NumberParam


class HighPass(nengo.LinearFilter):
    """ High pass synapse. """
    tau = NumberParam("tau", low=0)

    def __init__(self, tau, **kwargs):
        super().__init__([tau, 0], [tau, 1], **kwargs)
        self.tau = tau


def conv_filter2d(kernel: np.ndarray, h: int, w: int, padding: Literal['valid', 'same']):
    """ Convolution with arbitrary 2d filter transform. """
    return nengo.Convolution(n_filters=1, input_shape=(h, w, 1), kernel_size=kernel.shape,
                             strides=(1, 1), padding=padding, channels_last=True, init=kernel[:, :, None, None])


def relu_ens(size, thresh=0., max_rate=200, probe=False) -> Tuple[EnsembleArray, Optional[Probe]]:
    """ Spiking rectified linear ensemble with two neurons, tuned to positive and negative values respectively."""
    ens = EnsembleArray(2, size, 1,
                        neuron_type=nengo.SpikingRectifiedLinear(),
                        encoders=[[-1], [1]], intercepts=[thresh, thresh], max_rates=Uniform(max_rate, max_rate))
    p = Probe(ens.output, synapse=0.1) if probe else None
    return ens, p


def pos_relu_ens(size, intercept=0., max_rate=200) -> EnsembleArray:
    """ Spiking rectified linear ensemble responsive to positive values. """
    return EnsembleArray(1, size, 1, neuron_type=nengo.SpikingRectifiedLinear(), encoders=[[1]],
                         intercepts=Uniform(intercept, intercept), max_rates=Uniform(max_rate, max_rate))


def mask_module(inp, size: int, thresh: float, probe=False) -> Tuple[EnsembleArray, Optional[Probe]]:
    """
    Attach magnitude mask module, computing max(abs(x), thresh).

    Args:
        inp: input object.
        size: input size (h*w).
        thresh: minimal absolute value to pass through as mask.
        probe: whether to probe the output.

    Returns:
        Module's output and an optional probe object.
    """
    pos, neg = _demux_pos_neg(inp, size)
    mask = pos_relu_ens(size=size, intercept=thresh)
    _connection(pos, mask.input)
    _connection(neg, mask.input)
    p_mask = Probe(mask.output, synapse=0.1) if probe else None
    return mask, p_mask


def adaptive_double_opponency(inp: NengoObject, h: int, w: int, k: float,
                              tau: Optional[float], probe=False) -> Tuple[EnsembleArray, Optional[Probe]]:
    """
    Attach a module for edge detection (through Laplacian filter) with adaptation (high-pass synapse).

    Args:
        inp: input object.
        h: original height.
        w: original width.
        k: laplacian multiplicative factor.
        tau: adaptation time constant.
        probe: whether to probe the output.

    Returns:
        Module's output and an optional probe object.
    """
    do, _ = _laplacian_module(inp, h + 2, w + 2, k=-k, padding='valid')
    do_ada, p_do_ada = relu_ens(h * w, probe=probe)
    synapse = HighPass(tau) if tau else nengo.params.Default
    _connection(do, do_ada.input, synapse=synapse)
    return do_ada, p_do_ada


def modulated_edge_triggered_diffusion(inp_edge: NengoObject, inp_ctrl: Optional[NengoObject], h: int, w: int,
                                       tau: float, dk_r: float, dk_in: float, dnn: int, diffusion_neuron_type: str,
                                       probe: bool) -> Tuple[EnsembleArray, Probe, Optional[Probe]]:
    """
    Attach recurrent diffusion module on top of 'inp_edge', or a modulation of it: if 'inp_ctrl' is passed,
    it is connected to each of the demuxed 'inp_edge' pos and neg channels, which are then muxed back.

    Args:
        inp_edge: input edge.
        inp_ctrl: optional object to modulate.
        h: original height.
        w: original width.
        tau: time const for recurrent connection.
        dk_r: recurrent connection diffusion coefficient for diffusion module.
        dk_in: input connection diffusion coefficient for diffusion module.
        dnn: number of neurons for diffusion ensemble.
        probe: whether to probe the input to diffusion module.
        diffusion_neuron_type: neuron type for diffusion ensemble.

    Returns:
        Module's output, outputs probe and an optional probe for diffusion module input.
    """
    size = h * w
    if inp_ctrl:
        do_pos, do_neg = _demux_pos_neg(inp_edge.output, size)
        _connection(inp_ctrl, do_pos.input)
        _connection(inp_ctrl, do_neg.input)
        diffusion_inp = _mux_pos_neg(do_pos.output, do_neg.output, size).output
    else:
        diffusion_inp = inp_edge.output
    p_diffusion_inp = Probe(diffusion_inp, synapse=0.1) if probe else None
    neuron_type = {'lif': nengo.LIF, 'relu': nengo.SpikingRectifiedLinear, 'direct': nengo.Direct}[
        diffusion_neuron_type]()
    do_diffusion_ens = _diffusion_module(diffusion_inp, h, w, k_in=dk_in, k_r=dk_r, tau=tau, n_neurons=dnn,
                                            neuron_type=neuron_type)
    p_out = Probe(do_diffusion_ens.output, synapse=0.05)
    return do_diffusion_ens, p_out, p_diffusion_inp


def _connection(pre, post, *args, **kwargs):
    """ Create connection. """
    if isinstance(pre, EnsembleArray):
        pre = pre.output
    return Connection(pre, post, *args, **kwargs)


def _laplacian_module(inp, h: int, w: int, k: float,
                      padding: Literal['same', 'valid'], probe=False) -> Tuple[EnsembleArray, Optional[Probe]]:
    """ """
    assert padding in ['same', 'valid']
    out_size = h * w if padding == 'same' else (h - 2) * (w - 2)
    laplacian_ens, _ = relu_ens(size=out_size)
    _connection(inp, laplacian_ens.input, transform=_laplacian_transform(h=h, w=w, k=k, padding=padding))
    p_laplacian = Probe(laplacian_ens.output, synapse=0.1) if probe else None
    return laplacian_ens, p_laplacian


def _diffusion_module(inp, h: int, w: int, k_in: float, k_r: float, tau: float,
                      n_neurons: int, neuron_type: NeuronType) -> EnsembleArray:
    """ Diffusion recurrent module. Implements the diffusion equation dx/dt = k_r*L(x) + k_in*L(inp). """
    size = h * w
    diffusion_ens = EnsembleArray(n_neurons, size, neuron_type=neuron_type)
    _connection(inp, diffusion_ens.input, transform=k_in, synapse=tau)
    _connection(diffusion_ens, diffusion_ens.input, transform=_laplacian_transform(h, w, k_r, 'same'), synapse=tau)
    _connection(diffusion_ens, diffusion_ens.input, transform=1, synapse=tau)
    return diffusion_ens


def _demux_pos_neg(inp, size: int):
    """ Sends the input to two ensembles that respond to positive and to negative inputs respectively.
        Both pass the input through with positive sign. """
    pos = pos_relu_ens(size)
    _connection(inp, pos.input, transform=1)
    neg = pos_relu_ens(size)
    _connection(inp, neg.input, transform=-1)
    return pos, neg


def _mux_pos_neg(inp_pos, inp_neg, size: int) -> EnsembleArray:
    """ Restore split input. """
    ens, _ = relu_ens(size)
    _connection(inp_pos, ens.input)
    _connection(inp_neg, ens.input, transform=-1)
    return ens


def _laplacian_transform(h: int, w: int, k: float, padding: Literal['valid', 'same']):
    """ Convolution with Laplacian filter transform. """
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    return conv_filter2d(k*laplacian, h, w, padding=padding)
