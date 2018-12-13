import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from helper_fn import Conv1d, ConvTranspose1d, Embedding, Linear, GradMultiply
from helper_fn import get_mask_from_lengths, SinusoidalEncoding, Conv1dGLU


def expand_speaker_embed(inputs_btc, speaker_embed=None, tdim=1):
    if speaker_embed is None:
        return None
    # expand speaker embedding for all time steps
    # (B, N) -> (B, T, N)
    ss = speaker_embed.size()
    speaker_embed_btc = speaker_embed.unsqueeze(1).expand(
        ss[0], inputs_btc.size(tdim), ss[-1])
    return speaker_embed_btc


class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_speakers, speaker_embed_dim,
                 padding_idx=None, embedding_weight_std=0.1,
                 convolutions=((64, 5, .1),) * 7,
                 max_positions=512, dropout=0.1, apply_grad_scaling=False):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.apply_grad_scaling = apply_grad_scaling

        # Text input embeddings
        self.embed_tokens = Embedding(
            n_vocab, embed_dim, padding_idx, embedding_weight_std)

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_fc1 = Linear(speaker_embed_dim, embed_dim, dropout=dropout)
            self.speaker_fc2 = Linear(speaker_embed_dim, embed_dim, dropout=dropout)
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = embed_dim
        self.convolutions = nn.ModuleList()
        std_mul = 1.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, embed_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())
        x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
            x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))

        input_embedding = x

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        keys = x.transpose(1, 2)

        if speaker_embed_btc is not None:
            keys = keys + F.softsign(self.speaker_fc2(speaker_embed_btc))

        # scale gradients (this only affects backward, not forward)
        if self.apply_grad_scaling and self.num_attention_layers is not None:
            keys = GradMultiply.apply(keys, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, dropout=0.1,
                 window_ahead=3, window_backward=1,
                 key_projection=True, value_projection=True):
        super(AttentionLayer, self).__init__()
        self.query_projection = Linear(conv_channels, embed_dim)
        if key_projection:
            self.key_projection = Linear(embed_dim, embed_dim)
            # According to the DeepVoice3 paper, intiailize weights to same values
            # TODO: Does this really work well? not sure..
            if conv_channels == embed_dim:
                self.key_projection.weight.data = self.query_projection.weight.data.clone()
        else:
            self.key_projection = None
        if value_projection:
            self.value_projection = Linear(embed_dim, embed_dim)
        else:
            self.value_projection = None

        self.out_projection = Linear(embed_dim, conv_channels)
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        keys, values = encoder_out
        residual = query
        if self.value_projection is not None:
            values = self.value_projection(values)
        # TODO: yes, this is inefficient
        if self.key_projection is not None:
            keys = self.key_projection(keys.transpose(1, 2)).transpose(1, 2)

        # attention
        x = self.query_projection(query)
        x = torch.bmm(x, keys)

        mask_value = -float("inf")
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)

        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, embed_dim, n_speakers, speaker_embed_dim,
                 in_dim=80, r=5,
                 max_positions=512, padding_idx=None,
                 preattention=((128, 5, 1),) * 4,
                 convolutions=((128, 5, 1),) * 4,
                 attention=True, dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=True,
                 value_projection=True,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        in_channels = in_dim * r
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions, convolutions[0][0])
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions, embed_dim)
        # Used for compute multiplier for positional encodings
        if n_speakers > 1:
            self.speaker_proj1 = Linear(speaker_embed_dim, 1, dropout=dropout)
            self.speaker_proj2 = Linear(speaker_embed_dim, 1, dropout=dropout)
        else:
            self.speaker_proj1, self.speaker_proj2 = None, None

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        in_channels = in_dim * r
        std_mul = 1.0
        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=False))
            self.attention.append(
                AttentionLayer(out_channels, embed_dim,
                               dropout=dropout,
                               window_ahead=window_ahead,
                               window_backward=window_backward,
                               key_projection=key_projection,
                               value_projection=value_projection)
                if attention[i] else None)
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.last_conv = Conv1d(in_channels, in_dim * r, kernel_size=1,
                                padding=0, dilation=1, std_mul=std_mul,
                                dropout=dropout)

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_dim * r, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    def forward(self, x, speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        for f in self.convolutions:
            # Case for upsampling
            if speaker_embed_btc is not None and speaker_embed_btc.size(1) != x.size(-1):
                speaker_embed_btc = expand_speaker_embed(x, speaker_embed, tdim=-1)
                speaker_embed_btc = F.dropout(
                    speaker_embed_btc, p=self.dropout, training=self.training)
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        return torch.sigmoid(x)
