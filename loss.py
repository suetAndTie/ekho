'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''

import torch
import torch.nn as nn
from config import config
import util.util as ut

def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def masked_mean(y, mask):
    # (B, T, D)
    mask_ = mask.expand_as(y)
    return (y * mask_).sum() / mask_.sum()




class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction="sum")

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = ut.sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()


def spec_loss(y_hat, y, mask, priority_bin=None, priority_w=0):
    masked_l1 = MaskedL1Loss()
    l1 = nn.L1Loss()

    w = config.masked_loss_weight

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
    if config.binary_divergence_weight <= 0:
        binary_div = y.data.new(1).zero_()
    else:
        y_hat_logits = logit(y_hat)
        z = -y * y_hat_logits + torch.log1p(torch.exp(y_hat_logits))
        if w > 0:
            binary_div = w * masked_mean(z, mask) + (1 - w) * z.mean()
        else:
            binary_div = z.mean()

    return l1_loss, binary_div
