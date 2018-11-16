'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''

import os
import numpy as np
from config import config
import frontend

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
        self._frontend = None

    def collect_files(self):
        csv_file = os.path.join(self.data_root, "text.csv")

        df = np.genfromtxt(csv_file, delimiter=',', dtype=str) # [files, speaker_id, texts]
        texts = df[:,2]
        self.multi_speaker = df.shape[1] == 3

        if self.multi_speaker:
            speaker_ids = df[:,1].astype(int)
            # filter by speaker_id, use multi_speaker dataset as single speaker
            if self.speaker_id is not None:
                indices = speaker_ids == self.speaker_id
                texts = texts[indices]
                self.multi_speaker = False
                return texts

            return texts, speaker_ids

        return texts

    def collect_features(self, *args):
        if self.multi_speaker:
            text, speaker_id = args
        else:
            text = args[0]
        if self._frontend is None:
            self._frontend = getattr(frontend, config.frontend)
        seq = self._frontend.text_to_sequence(text, p=config.replace_pronunciation_prob)

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

        # (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
        df = np.genfromtxt(csv_file, delimiter=',', dtype=str)
        self.multi_speaker = df.shape[1] == 5
        self.frame_lengths = df[:, 2].astype(int)
        paths = df[:,self.col]
        for i in range(len(paths)):
            paths[i] = os.path.join(self.data_root, paths[i])

        if self.multi_speaker and self.speaker_id is not None:
            speaker_ids = df[:, 4]
            indices = np.array(speaker_ids) == self.speaker_id
            paths = paths[indices]
            self.frame_lengths = self.frame_lengths[indices]

        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(MelSpecDataSource, self).__init__(data_root, 1, speaker_id)



class LinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, speaker_id=None):
        super(LinearSpecDataSource, self).__init__(data_root, 0, speaker_id)
