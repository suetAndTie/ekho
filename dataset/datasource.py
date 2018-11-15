'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''

import os
import numpy as np
from config import config

######### TEMPLATE ##########
class FileDataSource(object):
    """File data source interface.
    Users are expected to implement custum data source for your own data.
    All file data sources must implement this interface.
    """

    def collect_files(self):
        """Collect data source files
        Returns:
            List or tuple of list: List of files, or tuple of list if you need
            multiple files to collect features.
        """
        raise NotImplementedError

    def collect_features(self, *args):
        """Collect features given path(s).
        Args:
            args: File path or tuple of file paths
        Returns:
            2darray: ``T x D`` features represented by 2d array.
        """
        raise NotImplementedError


########## DATA SOURCE ##########
class TextDataSource(FileDataSource):
    def __init__(self, data_root, speaker_id=None):
        self.data_root = data_root
        self.speaker_ids = None
        self.multi_speaker = False
        # If not None, filter by speaker_id
        self.speaker_id = speaker_id

    def collect_files(self):
        csv_file = os.path.join(self.data_root, "text.csv")

        df = np.genfromtxt(csv_file, delimiter=',') # [files, texts]
        files = df[:,0]

        return files

    def collect_features(self, *args):
        # TODO: START HERE!!!!!!!!!!!!!!!!
        if self.multi_speaker:
            text, speaker_id = args
        else:
            text = args[0]
        global _frontend
        if _frontend is None:
            _frontend = getattr(frontend, 'en')
        seq = _frontend.text_to_sequence(text, p=config.replace_pronunciation_prob)

        if self.multi_speaker:
            return np.asarray(seq, dtype=np.int32), int(speaker_id)
        else:
            return np.asarray(seq, dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None):
        self.data_root = data_root
        self.col = col
        self.frame_lengths = []
        self.speaker_id = speaker_id

    def collect_files(self):
        csv_file = os.path.join(self.data_root, 'data.csv')

        df = np.genfromtxt(csv_file, delimiter=',') # (spectrogram_filename, mel_filename, n_frames, text)
        files = df[:,self.col]

        return files

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(MelSpecDataSource, self).__init__(data_root, 1, speaker_id)



class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(LinearSpecDataSource, self).__init__(data_root, 0, speaker_id)
