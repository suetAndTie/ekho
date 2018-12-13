'''
Adapted from
https://github.com/r9y9/deepvoice3_pytorch/blob/master/audio.py
https://github.com/Kyubyong/deepvoice3/blob/master/prepro.py
https://github.com/Jonathan-LeRoux/lws/blob/master/python/lws.pyx
'''


import torch
from torch import nn

from basemodels import MultiSpeakerTTSModel, AttentionSeq2Seq


def ekho(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
               downsample_step=1,
               n_speakers=1, speaker_embed_dim=16, padding_idx=0,
               dropout=(1 - 0.95), kernel_size=5,
               encoder_channels=128,
               decoder_channels=256,
               converter_channels=256,
               query_position_rate=1.0,
               key_position_rate=1.29,
               ):
    from builder import wavenet

    time_upsampling = max(downsample_step // r, 1)

    # Seq2seq
    h = encoder_channels  # hidden dim (channels)
    k = kernel_size   # kernel size
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        embedding_weight_std=embedding_weight_std,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3)],
    )

    h = decoder_channel
    seq2seq = AttentionSeq2Seq(encoder, decoder)

    # Post net
    if use_decoder_state_for_postnet_input:
        in_dim = h // r
    else:
        in_dim = mel_dim
    h = converter_channels
    converter = Converter(
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        time_upsampling=time_upsampling,
        convolutions=[(h, k, 1), (h, k, 3), (2 * h, k, 1), (2 * h, k, 3)],
    )
