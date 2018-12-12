'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''
import numpy as np
import torch
from config import config
import util.audio as aud
import util.wavenet_util as wutil

def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """Create batch"""
    r = config.outputs_per_step
    downsample_step = config.downsample_step

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)
    target_lengths = [len(x[1]) for x in batch]
    max_target_len = max(target_lengths)

    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    if max_target_len % downsample_step != 0:
        max_target_len += downsample_step - max_target_len % downsample_step
        assert max_target_len % downsample_step == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    max_target_len += b_pad * downsample_step

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)

    b = np.array([_pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    if config.use_wavenet:
        # (B, T)
        if wutil.is_mulaw_quantize(config.input_type):
            padding_value = aud.mulaw_quantize(0, mu=config.quantize_channels)
            y_batch = np.array([_pad(x[0], max_target_len, constant_values=padding_value)
                                for x in batch], dtype=np.int)
        else:
            y_batch = np.array([_pad(x[0], max_target_len) for x in batch], dtype=np.float32)
        assert len(y_batch.shape) == 2

        # Add extra axis
        if wutil.is_mulaw_quantize(config.input_type):
            y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
        else:
            y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        c = np.array([_pad_2d(x[2], max_target_len, b_pad=b_pad) for x in batch],
                     dtype=np.float32)
        y_batch = torch.FloatTensor(c)

    # text positions
    text_positions = np.array([_pad(np.arange(1, len(x[0]) + 1), max_input_len)
                               for x in batch], dtype=np.int)
    text_positions = torch.LongTensor(text_positions)

    max_decoder_target_len = max_target_len // r // downsample_step

    # frame positions
    s, e = 1, max_decoder_target_len + 1
    # if b_pad > 0:
    #    s, e = s - 1, e - 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
        len(batch), max_decoder_target_len)

    # done flags
    done = np.array([_pad(np.zeros(len(x[1]) // r // downsample_step - 1),
                          max_decoder_target_len, constant_values=1)
                     for x in batch])
    done = torch.FloatTensor(done).unsqueeze(-1)

    speaker_ids = torch.LongTensor([b[3] for b in batch])

    return x_batch, input_lengths, mel_batch, y_batch, \
        (text_positions, frame_positions), done, target_lengths, speaker_ids
