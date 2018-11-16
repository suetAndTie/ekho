import sys
from pathlib import Path
from importlib import import_module
import torch
import torch.nn as nn

def build_model(model_path):
    path = Path(model_path)
    experiment_dir = path.parent
    sys.path.append(str(experiment_dir.resolve()))
    module = import_module(str(path.stem))

    model = module.Model()
    return model

def build_config(config_path):
    path = Path(config_path)
    experiment_dir = path.parent
    sys.path.append(str(experiment_dir.resolve()))
    module = import_module(str(path.stem))

    config = module.Config()
    return config


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
