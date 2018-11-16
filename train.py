import os
import argparse
import torch
import torch.optim
import torch.utils.data
from dataset.datasource import TextDataSource, MelSpecDataSource, LinearSpecDataSource
from dataset.dataset import FileDataset, PyTorchDataset
from data_loader.collate import collate_fn
import util.util as ut
from config import config


def train(device, model, data_loader, optimizer, writer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0,
          train_seq2seq=True, train_postnet=True):
    linear_dim = model.linear_dim
    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step
    current_lr = init_lr

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
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Downsample mel spectrogram
            if downsample_step > 1:
                mel = mel[:, 0::downsample_step, :].contiguous()

            # Lengths
            input_lengths = input_lengths.long().numpy()
            decoder_lengths = target_lengths.long().numpy() // r // downsample_step

            max_seq_len = max(input_lengths.max(), decoder_lengths.max())
            if max_seq_len >= hparams.max_positions:
                raise RuntimeError(
                    """max_seq_len ({}) >= max_posision ({})
Input text or decoder targget length exceeded the maximum length.
Please set a larger value for ``max_position`` in hyper parameters.""".format(
                        max_seq_len, hparams.max_positions))

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
            if hparams.masked_loss_weight > 0:
                # decoder output domain mask
                decoder_target_mask = sequence_mask(
                    target_lengths / (r * downsample_step),
                    max_len=mel.size(1)).unsqueeze(-1)
                if downsample_step > 1:
                    # spectrogram-domain mask
                    target_mask = sequence_mask(
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
            w = hparams.binary_divergence_weight

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
                n_priority_freq = int(hparams.priority_freq / (fs * 0.5) * linear_dim)
                linear_l1_loss, linear_binary_div = spec_loss(
                    linear_outputs[:, :-r, :], y[:, r:, :], target_mask,
                    priority_bin=n_priority_freq,
                    priority_w=hparams.priority_freq_weight)
                linear_loss = (1 - w) * linear_l1_loss + w * linear_binary_div

            # Combine losses
            if train_seq2seq and train_postnet:
                loss = mel_loss + linear_loss + done_loss
            elif train_seq2seq:
                loss = mel_loss + done_loss
            elif train_postnet:
                loss = linear_loss

            # attention
            if train_seq2seq and hparams.use_guided_attention:
                soft_mask = guided_attentions(input_lengths, decoder_lengths,
                                              attn.size(-2),
                                              g=hparams.guided_attention_sigma)
                soft_mask = torch.from_numpy(soft_mask).to(device)
                attn_loss = (attn * soft_mask).mean()
                loss += attn_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, writer, mel_outputs, linear_outputs, attn,
                    mel, y, input_lengths, checkpoint_dir)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch,
                    train_seq2seq, train_postnet)

            if global_step > 0 and global_step % hparams.eval_interval == 0:
                eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker)

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
                writer.add_scalar("linear_l1_loss", float(linear_l1_loss.item()), global_step)
                writer.add_scalar("linear_binary_div_loss", float(
                    linear_binary_div.item()), global_step)
            if train_seq2seq and hparams.use_guided_attention:
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
    parser.add_argument('--experiment_dir', default='experiment/example', help="Directory containing model and configs")
    parser.add_argument('--resume', default=None,
                        help="Optional, name of the file in --experiment_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()
    model_path = os.path.join(args.experiment_dir, 'model.py')
    config_path = os.path.join(args.experiment_dir, 'config.py')
    if not os.path.isfile(model_path):
        raise IOError("No model.py found at {}".format(model_path))
    if not os.path.isfile(config_path):
        raise IOError("No config.py found at {}".format(config_path))

    # Load dataset
    text = FileDataset(TextDataSource(args.data_dir))
    mel = FileDataset(MelSpecDataSource(args.data_dir))
    linear = FileDataset(LinearSpecDataSource(args.data_dir))
    dataset = PyTorchDataset(text, mel, linear)

    # TODO: Sampler??

    # Make DataLoader
    data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=config.pin_memory
                )
    # TODO: Trainer??

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ut.build_model(model_path).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    try:
        pass
        # TODO: train
    except KeyboardInterrupt:
        pass
        # TODO: save checkpoint
