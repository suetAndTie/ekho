from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from keras.utils import np_utils
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
from hparams import hparams


torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, initial_value=None,
            fast=False, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    model.eval()
    if fast:
        model.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
            assert c.ndim == 2
        Tc = c.shape[0]
        upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not hparams.upsample_conditional_features:
            c = np.repeat(c, upsample_factor, axis=0)

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

    if initial_value is None:
        if is_mulaw_quantize(hparams.input_type):
            initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
        else:
            initial_value = 0.0

    if is_mulaw_quantize(hparams.input_type):
        assert initial_value >= 0 and initial_value < hparams.quantize_channels
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    initial_input = initial_input.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat
