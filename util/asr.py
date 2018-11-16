'''
https://github.com/r9y9/deepvoice3_pytorch/blob/master/ljspeech.py
'''

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from . import audio
import librosa
from config import config
import util.audio as audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.
      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar
      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(out_dir, 'text.csv'), encoding='utf-8') as f:
        for line in f:
            path, speaker_id, text = line.split(',')
            text = text.strip() # remove newline
            # if len(text) < config.min_text:
            #     continue
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, path, speaker_id, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, flac_path, speaker_id, text):
    '''Preprocesses a single utterance audio/text pair.
    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.
    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      flac_path: Path to the audio file containing the speech input
      speaker_id: id of speaker
      text: The text spoken in the input audio file
    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text, speaker_id) tuple to write to data.csv
    '''
    sr = config.sample_rate
    data = audio.load_flac(flac_path)

    # Trim silence
    data, _ = librosa.effects.trim(data, top_db=15)

    if config.rescaling:
        data = data / np.abs(data).max() * config.rescaling_max

    # Compute the linear-scale spectrogram from the data:
    spectrogram = audio.spectrogram(data).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the data:
    mel_spectrogram = audio.melspectrogram(data).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'asr-spec-%05d.npy' % index
    mel_filename = 'asr-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
