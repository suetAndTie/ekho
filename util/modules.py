# coding: utf-8

'''
modules.py
based on https://github.com/r9y9/deepvoice3_pytorch/blob/master/deepvoice3_pytorch/modules.py
'''

import torch
from torch import nn
import math
import numpy as np
from torch.nn import functional as F
import util.conv as conv


def position_encoding_init(n_position, d_pos_vec, position_rate=1.0,
                           sinusoidal=True):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc = torch.from_numpy(position_enc).float()
    if sinusoidal:
        position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    return position_enc


def sinusoidal_encode(x, w):
    y = w * x
    y[1:, 0::2] = torch.sin(y[1:, 0::2].clone())
    y[1:, 1::2] = torch.cos(y[1:, 1::2].clone())
    return y


class SinusoidalEncoding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim,
                 *args, **kwargs):
        super(SinusoidalEncoding, self).__init__(num_embeddings, embedding_dim,
                                                 padding_idx=0,
                                                 *args, **kwargs)
        self.weight.data = position_encoding_init(num_embeddings, embedding_dim,
                                                  position_rate=1.0,
                                                  sinusoidal=False)

    def forward(self, x, w=1.0):
        isscaler = np.isscalar(w)
        assert self.padding_idx is not None

        if isscaler or w.size(0) == 1:
            weight = sinusoidal_encode(self.weight, w)
            return F.embedding(
                x, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            # TODO: cannot simply apply for batch
            # better to implement efficient function
            pe = []
            for batch_idx, we in enumerate(w):
                weight = sinusoidal_encode(self.weight, we)
                pe.append(F.embedding(
                    x[batch_idx], weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse))
            pe = torch.stack(pe)
            return pe


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0, **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTranspose1d(in_channels, out_channels, kernel_size, dropout=0,
                    std_mul=1.0, **kwargs):
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, n_speakers, speaker_embed_dim,
                 in_channels, out_channels, kernel_size,
                 dropout, padding=None, dilation=1, causal=False, residual=False,
                 *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        self.residual = residual
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size,
                           dropout=dropout, padding=padding, dilation=dilation,
                           *args, **kwargs)
        if n_speakers > 1:
            self.speaker_proj = Linear(speaker_embed_dim, out_channels)
        else:
            self.speaker_proj = None

    def forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, False)

    def incremental_forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, True)

    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if self.speaker_proj is not None:
            softsign = F.softsign(self.speaker_proj(speaker_embed))
            # Since conv layer assumes BCT, we need to transpose
            softsign = softsign if is_incremental else softsign.transpose(1, 2)
            a = a + softsign
        x = a * torch.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x

    def clear_buffer(self):
        self.conv.clear_buffer()


def Conv1d1x1(in_channels, out_channels, bias=True, weight_normalization=True):
    """1-by-1 convolution layer
    """
    if weight_normalization:
        assert bias
        return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                      dilation=1, bias=bias, std_mul=1.0)
    else:
        return conv.Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, bias=bias)


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


def ConvTranspose2d(in_channels, out_channels, kernel_size,
                    weight_normalization=True, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    m.bias.data.zero_()
    if weight_normalization:
        return nn.utils.weight_norm(m)
    else:
        return m


class HighwayConv1d(nn.Module):
    """Weight normzlized Conv1d + Highway network (support incremental forward)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=None,
                 dilation=1, causal=False, dropout=0, std_mul=None, glu=False):
        super(HighwayConv1d, self).__init__()
        if std_mul is None:
            std_mul = 4.0 if glu else 1.0
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.dropout = dropout
        self.glu = glu

        self.conv = Conv1d(in_channels, 2 * out_channels,
                           kernel_size=kernel_size, padding=padding,
                           dilation=dilation, dropout=dropout,
                           std_mul=std_mul)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, is_incremental):
        """Forward
        Args:
            x: (B, in_channels, T)
        returns:
            (B, out_channels, T)
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        if self.glu:
            x = F.glu(x, dim=splitdim)
            return (x + residual) * math.sqrt(0.5)
        else:
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            T = torch.sigmoid(b)
            return (T * a + (1 - T) * residual)

    def clear_buffer(self):
        self.conv.clear_buffer()


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit
    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None,
                 cin_channels=-1, gin_channels=-1,
                 dropout=1 - 0.95, padding=None, dilation=1, causal=True,
                 bias=True, weight_normalization=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        if weight_normalization:
            assert bias
            self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                               padding=padding, dilation=dilation,
                               bias=bias, std_mul=1.0, *args, **kwargs)
        else:
            self.conv = conv.Conv1d(residual_channels, gate_channels, kernel_size,
                                    padding=padding, dilation=dilation,
                                    bias=bias, *args, **kwargs)

        # local conditioning
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels,
                                      bias=bias,
                                      weight_normalization=weight_normalization)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=bias,
                                      weight_normalization=weight_normalization)
        else:
            self.conv1x1g = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias,
                                     weight_normalization=weight_normalization)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias,
                                      weight_normalization=weight_normalization)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not
        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb

        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip,
                  self.conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()
