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
import util.wavenet_util as wavenet_util
import util.mixture as mix
import librosa

use_cuda = torch.cuda.is_available()
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
    if config.use_wavenet:
        converter = Converter(
            in_channels=in_dim,
            out_channels=config.out_channels,
            # out_channels=linear_dim,
            layers=config.layers,
            stacks=config.stacks,
            residual_channels=config.residual_channels,
            gate_channels=config.gate_channels,
            skip_out_channels=config.skip_out_channels,
            cin_channels=mel_dim,
            gin_channels=speaker_embed_dim,
            weight_normalization=config.weight_normalization,
            n_speakers=n_speakers,
            dropout=config.dropout,
            kernel_size=config.kernel_size,
            upsample_conditional_features=config.upsample_conditional_features,
            upsample_scales=config.upsample_scales,
            freq_axis_kernel_size=config.freq_axis_kernel_size,
            scalar_input=wavenet_util.is_scalar_input(config.input_type),
            legacy=config.legacy
        )
    else:
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

def clone_as_averaged_model(device, model, ema):
    assert ema is not None
    averaged_model = build_model().to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model

def save_waveplot(path, y_hat, y_target):
    sr = config.sample_rate

    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y_target, sr=sr)
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(y_hat, sr=sr)
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

def wavenet_eval_model(global_step, writer, device, model, y, c, g, input_lengths, eval_dir, ema=None):
    if ema is not None:
        print("Using averaged model for evaluation")
        model = clone_as_averaged_model(device, model, ema)
        model.make_generation_fast_()

    model.eval()
    idx = np.random.randint(0, len(y))
    length = input_lengths[idx].data.cpu().item()

    # (T,)
    y_target = y[idx].view(-1).data.cpu().numpy()[:length]

    if c is not None:
        if config.upsample_conditional_features:
            c = c[idx, :, :length // audio.get_hop_size()].unsqueeze(0)
        else:
            c = c[idx, :, :length].unsqueeze(0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))
    if g is not None:
        # TODO: test
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if wavenet_util.is_mulaw_quantize(config.input_type):
        initial_value = audio.mulaw_quantize(0, config.quantize_channels)
    elif wavenet_util.is_mulaw(config.input_type):
        initial_value = audio.mulaw(0.0, config.quantize_channels)
    else:
        initial_value = 0.0
    print("Intial value:", initial_value)

    # (C,)
    if wavenet_util.is_mulaw_quantize(config.input_type):
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=config.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, config.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
    initial_input = initial_input.to(device)

    # Run the model in fast eval mode
    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
            log_scale_min=config.log_scale_min)

    if wavenet_util.is_mulaw_quantize(config.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = audio.inv_mulaw_quantize(y_hat, config.quantize_channels)
        y_target = audio.inv_mulaw_quantize(y_target, config.quantize_channels)
    elif wavenet_util.is_mulaw(config.input_type):
        y_hat = audio.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), config.quantize_channels)
        y_target = audio.inv_mulaw(y_target, config.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=config.sample_rate)
    path = os.path.join(eval_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=config.sample_rate)

    # save figure
    path = os.path.join(eval_dir, "step{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_hat, y_target)


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


def load_checkpoint(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


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

    if config.use_wavenet:
        # idx = np.random.randint(0, len(linear_outputs))
        # length = input_lengths[idx]
        idx = min(1, len(input_lengths) - 1)
        input_length = input_lengths[idx]
        length = input_length

        # (B, C, T)
        if linear_outputs.dim() == 4:
            linear_outputs = linear_outputs.squeeze(-1)

        if wavenet_util.is_mulaw_quantize(config.input_type):
            # (B, T)
            linear_outputs = F.softmax(linear_outputs, dim=1).max(1)[1]

            # (T,)
            linear_outputs = linear_outputs[idx].data.cpu().long().numpy()
            y = y[idx].view(-1).data.cpu().long().numpy()

            linear_outputs = audio.inv_mulaw_quantize(linear_outputs, config.quantize_channels)
            y = audio.inv_mulaw_quantize(y, config.quantize_channels)
        else:
            # (B, T)
            linear_outputs = mix.sample_from_discretized_mix_logistic(
                linear_outputs, log_scale_min=config.log_scale_min)
            # (T,)
            linear_outputs = linear_outputs[idx].view(-1).data.cpu().numpy()
            y = y[idx].view(-1).data.cpu().numpy()

            if wavenet_util.is_mulaw(config.input_type):
                linear_outputs = audio.inv_mulaw(linear_outputs, config.quantize_channels)
                y = audio.inv_mulaw(y, config.quantize_channels)

        # Mask by length
        # linear_outputs[length:] = 0
        # y[length:] = 0

        print('yhat', linear_outputs.shape, linear_outputs)
        print('y', y.shape, y)
        # Save audio
        audio_dir = os.path.join(checkpoint_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        path = os.path.join(audio_dir, "step{:09d}_predicted.wav".format(global_step))
        librosa.output.write_wav(path, linear_outputs, sr=config.sample_rate)
        path = os.path.join(audio_dir, "step{:09d}_target.wav".format(global_step))
        librosa.output.write_wav(path, y, sr=config.sample_rate)

        return

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
