import torch
import numpy as np
import librosa
import librosa.display
import IPython
from IPython.display import Audio
# need this for English text processing frontend
import nltk

def load_multi_speaker_model():
  checkpoint_path = "https://drive.google.com/open?id=1MQP_JSUFcKOvVRz75ty56IHozJZfGij5"
  if not exists(checkpoint_path):
    !curl -O -L "https://drive.google.com/open?id=1MQP_JSUFcKOvVRz75ty56IHozJZfGij5"

import hparams
import json

# Load parameters from preset
with open(preset_single) as f:
  hparams.hparams.parse_json(f.read())

# Inject frontend text processor
import synthesis
import train
import frontend
synthesis._frontend = getattr(frontend, "en")
train._frontend =  getattr(frontend, "en")

# alises
fs = hparams.hparams.sample_rate
hop_length = hparams.hparams.hop_size

from train import build_model
from train import restore_parts, load_checkpoint

model_1 = build_model()
model_1 = load_checkpoint(checkpoint_path_single, model_1, None, True)

load_multi_speaker_model()

import hparams
import json

# Newly added params. Need to inject dummy values
for dummy, v in [("fmin", 0), ("fmax", 0), ("rescaling", False),
                 ("rescaling_max", 0.999),
                 ("allow_clipping_in_normalization", False)]:
  if hparams.hparams.get(dummy) is None:
    hparams.hparams.add_hparam(dummy, v)

# Load parameters from preset
with open(preset_multi) as f:
  hparams.hparams.parse_json(f.read())

from train import build_model
from train import restore_parts, load_checkpoint
checkpoint_path_multi = "https://drive.google.com/open?id=1MQP_JSUFcKOvVRz75ty56IHozJZfGij5"

model_2 = build_model()
model_2 = load_checkpoint(checkpoint_path_multi, model_2, None, True)



def tts(model, text, p=0, speaker_id=None, fast=True, figures=True):
  from synthesis import tts as _tts
  waveform, alignment, spectrogram, mel = _tts(model, text, p, speaker_id, fast)
  if figures:
      visualize(alignment, spectrogram)
  IPython.display.display(Audio(waveform, rate=fs))

def visualize(alignment, spectrogram):
  label_fontsize = 12
  figure(figsize=(12,12))

  subplot(2,1,1)
  imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
  xlabel("Decoder timestamp", fontsize=label_fontsize)
  ylabel("Encoder timestamp", fontsize=label_fontsize)
  colorbar()

  subplot(2,1,2)
  librosa.display.specshow(spectrogram.T, sr=fs,
                           hop_length=hop_length, x_axis="time", y_axis="linear")
  xlabel("Time", fontsize=label_fontsize)
  ylabel("Hz", fontsize=label_fontsize)
  tight_layout()
  colorbar()

texts = [
    "Samir and Trevor are Seniors at Stanford.",
    "Trevor is from Kansas, while Samir is from Washington.",
    "Samir likes to play chess sometimes and hang out with friends other times.",
    "You're mom is a deep generative model. hahaha.",
    "I know I said top five but I'm top two and I'm not two cause I got one you wish you had one",
    "nani",
    "boku wa nihon jing tu chu goku jing no hahu",
    "Anthony knows everything",
    "A I is the new electricity."
]

for idx, text in enumerate(texts):
  print(idx, text)
  tts(model_1, text, figures=False)

text = "Fred lu is a beast."
for speaker_id in range(4):
  print(speaker_id)
  tts(model_2, text, speaker_id=speaker_id, figures=False)
