'''
syntheis.py
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/synthesis.py
'''

import os
import argparse
import numpy as np
import torch
import util.audio as audio
import util.util as ut


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None


def tts(model, text, p=0, speaker_id=None, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.
    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
    speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


if __name__ == '__main__':
    # TODO FIX!!!!!!
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='data', help="Path to file with lines containing input")
    parser.add_argument('--output', default='model/test', help="Directory containing model and configs")
    parser.add_argument('--resume', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'

    args = parser.parse_args()
    # model_path = os.path.join(args.experiment_dir, 'model.py')
    # config_path = os.path.join(args.experiment_dir, 'config.py')
    # assert os.path.isfile(model_path), "No model.py found at {}".format(model_path)
    # assert os.path.isfile(config_path), "No config.py found at {}".format(config_path)

    model = ut.build_model().to(device)
    checkpoint = ut.load_checkpoint(args.resume)
    model.load_state_dict(checkpoint["state_dict"])

    raise NotImplementedError('Need to finish main of synthesis.py')
