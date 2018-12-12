'''
train.py
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''

import os
import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

import eval
import frontend
from config import config
from loss import spec_loss
import util.util as ut
import util.lrschedule as lrschedule
from util.attention import guided_attentions
import util.mixture as mix
import util.wavenet_util as wavenet_util

from dataset.datasource import TextDataSource, MelSpecDataSource, LinearSpecDataSource
from dataset.dataset import FileDataset, PyTorchDataset
from data_loader.collate import collate_fn
from data_loader.sampler import PartialyRandomizedSimilarTimeLengthSampler

from tqdm import tqdm
from tensorboardX import SummaryWriter


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
_frontend = None  # to be set later


def train(device, model, data_loader, optimizer, writer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=1000, nepochs=None,
          clip_thresh=1.0,
          train_seq2seq=True, train_postnet=True):
    global _frontend
    _frontend = getattr(frontend, config.frontend)
    linear_dim = model.linear_dim
    r = config.outputs_per_step
    downsample_step = config.downsample_step
    current_lr = init_lr

    # WAVENET CRITERION
    if config.use_wavenet:
        if wavenet_util.is_mulaw_quantize(config.input_type):
            wavenet_criterion = mix.MaskedCrossEntropyLoss()
        else:
            wavenet_criterion = mix.DiscretizedMixturelogisticLoss()

    binary_criterion = nn.BCELoss()

    assert train_seq2seq or train_postnet

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, input_lengths, mel, y, positions, done, target_lengths,
                   speaker_ids) \
                in tqdm(enumerate(data_loader)):
            model.train()
            ismultispeaker = speaker_ids is not None
            # Learning rate schedule
            if config.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, config.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **config.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Downsample mel spectrogram
            if downsample_step > 1:
                mel = mel[:, 0::downsample_step, :].contiguous()


            # get mask
            if config.use_wavenet:
                mask = ut.sequence_mask(input_lengths, max_len=y.size(1)).unsqueeze(-1)
                mask = mask[:, 1:, :]
                mask = mask.to(device)

            # Lengths
            input_lengths = input_lengths.long().numpy()
            decoder_lengths = target_lengths.long().numpy() // r // downsample_step

            max_seq_len = max(input_lengths.max(), decoder_lengths.max())
            if max_seq_len >= config.max_positions:
                raise RuntimeError(
                    """max_seq_len ({}) >= max_positions ({})
Input text or decoder target length exceeded the maximum length.
Please set a larger value for ``max_position`` in hyper parameters.""".format(
                        max_seq_len, config.max_positions))

            # Transform data to CUDA device
            if train_seq2seq:
                x = x.to(device)
                text_positions = text_positions.to(device)
                frame_positions = frame_positions.to(device)
            if train_postnet:
                y = y.to(device)
            mel, done = mel.to(device), done.to(device)
            target_lengths = target_lengths.to(device)
            speaker_ids = speaker_ids.to(device) if ismultispeaker else None

            # Create mask if we use masked loss
            if config.masked_loss_weight > 0:
                # decoder output domain mask
                decoder_target_mask = ut.sequence_mask(
                    target_lengths / (r * downsample_step),
                    max_len=mel.size(1)).unsqueeze(-1)
                if downsample_step > 1:
                    # spectrogram-domain mask
                    target_mask = ut.sequence_mask(
                        target_lengths, max_len=y.size(1)).unsqueeze(-1)
                else:
                    target_mask = decoder_target_mask
                # shift mask
                decoder_target_mask = decoder_target_mask[:, r:, :]
                target_mask = target_mask[:, r:, :]
            else:
                decoder_target_mask, target_mask = None, None

            # Apply model
            if train_seq2seq and train_postnet:
                mel_outputs, linear_outputs, attn, done_hat = model(
                    x, mel, speaker_ids=speaker_ids,
                    text_positions=text_positions, frame_positions=frame_positions,
                    input_lengths=input_lengths)
            elif train_seq2seq:
                assert speaker_ids is None
                mel_outputs, attn, done_hat, _ = model.seq2seq(
                    x, mel,
                    text_positions=text_positions, frame_positions=frame_positions,
                    input_lengths=input_lengths)
                # reshape
                mel_outputs = mel_outputs.view(len(mel), -1, mel.size(-1))
                linear_outputs = None
            elif train_postnet:
                assert speaker_ids is None
                linear_outputs = model.postnet(mel)
                mel_outputs, attn, done_hat = None, None, None

            # Losses
            w = config.binary_divergence_weight

            # mel:
            if train_seq2seq:
                mel_l1_loss, mel_binary_div = spec_loss(
                    mel_outputs[:, :-r, :], mel[:, r:, :], decoder_target_mask)
                mel_loss = (1 - w) * mel_l1_loss + w * mel_binary_div

            # done:
            if train_seq2seq:
                done_loss = binary_criterion(done_hat, done)

            # linear:
            if train_postnet:
                if config.use_wavenet:
                    if wavenet_util.is_mulaw_quantize(config.input_type):
                        linear_outputs = linear_outputs.unsqueeze(-1)
                        linear_loss = wavenet_criterion(linear_outputs[:, :, :-r, :], y[:, r:, :], mask=mask)
                    else:
                        linear_loss = wavenet_criterion(linear_outputs[:, :, :-r], y[:, r:, :], mask=mask)
                else:
                    n_priority_freq = int(config.priority_freq / (config.sample_rate * 0.5) * linear_dim)
                    linear_l1_loss, linear_binary_div = spec_loss(
                        linear_outputs[:, :-r, :], y[:, r:, :], target_mask,
                        priority_bin=n_priority_freq,
                        priority_w=config.priority_freq_weight)
                    linear_loss = (1 - w) * linear_l1_loss + w * linear_binary_div

            # Combine losses
            if train_seq2seq and train_postnet:
                loss = mel_loss + linear_loss + done_loss
            elif train_seq2seq:
                loss = mel_loss + done_loss
            elif train_postnet:
                loss = linear_loss

            # attention
            if train_seq2seq and config.use_guided_attention:
                soft_mask = guided_attentions(input_lengths, decoder_lengths,
                                              attn.size(-2),
                                              g=config.guided_attention_sigma)
                soft_mask = torch.from_numpy(soft_mask).to(device)
                attn_loss = (attn * soft_mask).mean()
                loss += attn_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                ut.save_states(
                    global_step, writer, mel_outputs, linear_outputs, attn,
                    mel, y, input_lengths, checkpoint_dir)
                ut.save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch,
                    train_seq2seq, train_postnet)

            if global_step > 0 and global_step % config.eval_interval == 0:
                ut.eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker)

            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), clip_thresh)
            optimizer.step()

            # Logs
            writer.add_scalar("loss", float(loss.item()), global_step)
            if train_seq2seq:
                writer.add_scalar("done_loss", float(done_loss.item()), global_step)
                writer.add_scalar("mel loss", float(mel_loss.item()), global_step)
                writer.add_scalar("mel_l1_loss", float(mel_l1_loss.item()), global_step)
                writer.add_scalar("mel_binary_div_loss", float(mel_binary_div.item()), global_step)
            if train_postnet:
                writer.add_scalar("linear_loss", float(linear_loss.item()), global_step)
                if not config.use_wavenet:
                    writer.add_scalar("linear_l1_loss", float(linear_l1_loss.item()), global_step)
                    writer.add_scalar("linear_binary_div_loss", float(
                        linear_binary_div.item()), global_step)
            if train_seq2seq and config.use_guided_attention:
                writer.add_scalar("attn_loss", float(attn_loss.item()), global_step)
            if clip_thresh > 0:
                writer.add_scalar("gradient norm", grad_norm, global_step)
            writer.add_scalar("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(data_loader))
        writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed', help="Directory containing the dataset")
    parser.add_argument('--experiment_dir', default='experiment', help="Directory containing model and configs")
    parser.add_argument('--resume', default=None,
                        help="Optional, path containing weights to reload for training")  # 'best' or 'train'
    args = parser.parse_args()
    # model_path = os.path.join(args.experiment_dir, 'model.py')
    # config_path = os.path.join(args.experiment_dir, 'config.py')
    # if not os.path.isfile(model_path):
    #     raise IOError("No model.py found at {}".format(model_path))
    # if not os.path.isfile(config_path):
    #     raise IOError("No config.py found at {}".format(config_path))

    _frontend = getattr(frontend, config.frontend)

    # Load dataset
    text = FileDataset(TextDataSource(args.data_dir))
    mel = FileDataset(MelSpecDataSource(args.data_dir))
    linear = FileDataset(LinearSpecDataSource(args.data_dir))
    dataset = PyTorchDataset(text, mel, linear)

    # Make sampler
    frame_lengths = mel.file_data_source.frame_lengths
    sampler = PartialyRandomizedSimilarTimeLengthSampler(
        frame_lengths, batch_size=config.batch_size
    )

    # Make DataLoader
    data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    pin_memory=config.pin_memory
                )

    # Make Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ut.build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Make writer
    writer = SummaryWriter(log_dir=config.log_event_path)

    ##### OPTIONAL LOAD #####
    if args.resume is not None:
        checkpoint = util.load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        if config.save_optimizer_state:
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_epoch = checkpoint["global_epoch"]
        global_step = checkpoint["global_step"]

    try:
        train(device, model, data_loader, optimizer, writer,
              init_lr=config.initial_learning_rate,
              checkpoint_dir=config.checkpoint_dir,
              checkpoint_interval=config.checkpoint_interval,
              nepochs=config.nepochs,
              clip_thresh=config.clip_thresh,
              train_seq2seq=config.train_seq2seq, train_postnet=config.train_postnet)
    except KeyboardInterrupt:
        ut.save_checkpoint(
            model, optimizer, global_step, config.checkpoint_dir, global_epoch,
            config.train_seq2seq, config.train_postnet)
