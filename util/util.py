'''
util.py
the utility functions of the project
'''

import os
import sys
from warnings import warn
from datetime import datetime
from pathlib import Path
from importlib import import_module
import torch
import torch.nn as nn
import frontend
import util.audio as audio
from config import config
import numpy as np
import matplotlib
matplotlib.use('Agg') # To use on linux server without $DISPLAY
from matplotlib import cm
import matplotlib.pyplot as plt

fs = config.sample_rate

def create_model(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
               downsample_step=1,
               n_speakers=1, speaker_embed_dim=16, padding_idx=0,
               dropout=(1 - 0.95), kernel_size=5,
               encoder_channels=128,
               decoder_channels=256,
               converter_channels=256,
               query_position_rate=1.0,
               key_position_rate=1.29,
               use_memory_mask=False,
               trainable_positional_encodings=False,
               force_monotonic_attention=True,
               use_decoder_state_for_postnet_input=True,
               max_positions=512,
               embedding_weight_std=0.1,
               speaker_embedding_weight_std=0.01,
               freeze_embedding=False,
               window_ahead=3,
               window_backward=1,
               key_projection=False,
               value_projection=False,
               ):
    """Build model
    """
    from model import Encoder, Decoder, Converter, AttentionSeq2Seq, MultiSpeakerTTSModel

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

    h = decoder_channels
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        preattention=[(h, k, 1), (h, k, 3)],
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask,
        window_ahead=window_ahead,
        window_backward=window_backward,
        key_projection=key_projection,
        value_projection=value_projection,
    )

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

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
        speaker_embedding_weight_std=speaker_embedding_weight_std,
        freeze_embedding=freeze_embedding)

    return model




def build_model():
    # path = Path(model_path)
    # experiment_dir = path.parent
    # sys.path.append(str(experiment_dir.resolve()))
    # module = import_module(str(path.stem))

    _frontend = getattr(frontend, config.frontend)

    model = create_model(
        n_speakers=config.n_speakers,
        speaker_embed_dim=config.speaker_embed_dim,
        n_vocab=_frontend.n_vocab,
        embed_dim=config.text_embed_dim,
        mel_dim=config.num_mels,
        linear_dim=config.fft_size // 2 + 1,
        r=config.outputs_per_step,
        downsample_step=config.downsample_step,
        padding_idx=config.padding_idx,
        dropout=config.dropout,
        kernel_size=config.kernel_size,
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        converter_channels=config.converter_channels,
        use_memory_mask=config.use_memory_mask,
        trainable_positional_encodings=config.trainable_positional_encodings,
        force_monotonic_attention=config.force_monotonic_attention,
        use_decoder_state_for_postnet_input=config.use_decoder_state_for_postnet_input,
        max_positions=config.max_positions,
        speaker_embedding_weight_std=config.speaker_embedding_weight_std,
        freeze_embedding=config.freeze_embedding,
        window_ahead=config.window_ahead,
        window_backward=config.window_backward,
        key_projection=config.key_projection,
        value_projection=config.value_projection,
    )
    return model

# def build_config(config_path):
#     path = Path(config_path)
#     experiment_dir = path.parent
#     sys.path.append(str(experiment_dir.resolve()))
#     module = import_module(str(path.stem))
#
#     config = module.Config()
#     return config


def eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker):
    # harded coded
    texts = [
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "Generative adversarial network or variational auto-encoder.",
        "Please call Stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]
    import synthesis
    _frontend = getattr(frontend, config.frontend)
    synthesis._frontend = _frontend

    eval_output_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare model for evaluation
    model_eval = build_model().to(device)
    model_eval.load_state_dict(model.state_dict())

    # hard coded
    speaker_ids = [0, 1, 10] if ismultispeaker else [None]
    for speaker_id in speaker_ids:
        speaker_str = "multispeaker{}".format(speaker_id) if speaker_id is not None else "single"

        for idx, text in enumerate(texts):
            signal, alignment, _, mel = synthesis.tts(
                model_eval, text, p=0, speaker_id=speaker_id, fast=True)
            signal /= np.max(np.abs(signal))

            # Alignment
            path = os.path.join(eval_output_dir, "step{:09d}_text{}_{}_alignment.png".format(
                global_step, idx, speaker_str))
            save_alignment(path, alignment, global_step)
            tag = "eval_averaged_alignment_{}_{}".format(idx, speaker_str)
            # writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

            # Mel
            # writer.add_image("(Eval) Predicted mel spectrogram text{}_{}".format(idx, speaker_str),
                             # prepare_spec_image(mel), global_step)

            # Audio
            path = os.path.join(eval_output_dir, "step{:09d}_text{}_{}_predicted.wav".format(
                global_step, idx, speaker_str))
            audio.save_wav(signal, path)

            try:
                writer.add_audio("(Eval) Predicted audio signal {}_{}".format(idx, speaker_str),
                                 signal, global_step, sample_rate=fs)
            except Exception as e:
                warn(str(e))
                pass


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


def spec_loss(y_hat, y, mask, masked_loss_weight, binary_divergence_weight, priority_bin=None, priority_w=0):
    masked_l1 = MaskedL1Loss()
    l1 = nn.L1Loss()

    w = masked_loss_weight

    # L1 loss
    if w > 0:
        assert mask is not None
        l1_loss = w * masked_l1(y_hat, y, mask=mask) + (1 - w) * l1(y_hat, y)
    else:
        assert mask is None
        l1_loss = l1(y_hat, y)

    # Priority L1 loss
    if priority_bin is not None and priority_w > 0:
        if w > 0:
            priority_loss = w * masked_l1(
                y_hat[:, :, :priority_bin], y[:, :, :priority_bin], mask=mask) \
                + (1 - w) * l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
        else:
            priority_loss = l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
        l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

    # Binary divergence loss
    if binary_divergence_weight <= 0:
        binary_div = y.data.new(1).zero_()
    else:
        y_hat_logits = logit(y_hat)
        z = -y * y_hat_logits + torch.log1p(torch.exp(y_hat_logits))
        if w > 0:
            binary_div = w * masked_mean(z, mask) + (1 - w) * z.mean()
        else:
            binary_div = z.mean()

    return l1_loss, binary_div


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch,
                    train_seq2seq, train_postnet):
    if train_seq2seq and train_postnet:
        suffix = ""
        m = model
    elif train_seq2seq:
        suffix = "_seq2seq"
        m = model.seq2seq
    elif train_postnet:
        suffix = "_postnet"
        m = model.postnet

    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_step{:09d}{}.pth".format(step, suffix))
    optimizer_state = optimizer.state_dict() if config.save_optimizer_state else None
    torch.save({
        "state_dict": m.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


def save_alignment(path, attn, global_step):
    plot_alignment(attn.T, path, info="{}, {}, step={}".format(
        config.builder, time_string(), global_step))


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def prepare_spec_image(spectrogram):
    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram.T) * 255)


def save_states(global_step, writer, mel_outputs, linear_outputs, attn, mel, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    # Multi-hop attention
    if attn is not None and attn.dim() == 4:
        for i, alignment in enumerate(attn):
            alignment = alignment[idx].cpu().data.numpy()
            tag = "alignment_layer{}".format(i + 1)
            # writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

            # save files as well for now
            alignment_dir = os.path.join(checkpoint_dir, "alignment_layer{}".format(i + 1))
            os.makedirs(alignment_dir, exist_ok=True)
            path = os.path.join(alignment_dir, "step{:09d}_layer_{}_alignment.png".format(
                global_step, i + 1))
            save_alignment(path, alignment, global_step)

        # Save averaged alignment
        alignment_dir = os.path.join(checkpoint_dir, "alignment_ave")
        os.makedirs(alignment_dir, exist_ok=True)
        path = os.path.join(alignment_dir, "step{:09d}_alignment.png".format(global_step))
        alignment = attn.mean(0)[idx].cpu().data.numpy()
        save_alignment(path, alignment, global_step)

        tag = "averaged_alignment"
        # writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1).T) * 255), global_step)

    # Predicted mel spectrogram
    if mel_outputs is not None:
        mel_output = mel_outputs[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        # writer.add_image("Predicted mel spectrogram", mel_output, global_step)

    # Predicted spectrogram
    if linear_outputs is not None:
        linear_output = linear_outputs[idx].cpu().data.numpy()
        spectrogram = prepare_spec_image(audio._denormalize(linear_output))
        # writer.add_image("Predicted linear spectrogram", spectrogram, global_step)

        # Predicted audio signal
        signal = audio.inv_spectrogram(linear_output.T)
        signal /= np.max(np.abs(signal))
        path = os.path.join(checkpoint_dir, "step{:09d}_predicted.wav".format(
            global_step))
        try:
            writer.add_audio("Predicted audio signal", signal, global_step, sample_rate=fs)
        except Exception as e:
            warn(str(e))
            pass
        audio.save_wav(signal, path)

    # Target mel spectrogram
    if mel_outputs is not None:
        mel_output = mel[idx].cpu().data.numpy()
        mel_output = prepare_spec_image(audio._denormalize(mel_output))
        # writer.add_image("Target mel spectrogram", mel_output, global_step)

    # Target spectrogram
    if linear_outputs is not None:
        linear_output = y[idx].cpu().data.numpy()
        spectrogram = prepare_spec_image(audio._denormalize(linear_output))
        # writer.add_image("Target linear spectrogram", spectrogram, global_step)
