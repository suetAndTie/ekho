'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/datasets/__init__.py
'''

import os
import numpy as np

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


class Dataset(object):
    """Dataset represents a fixed-sized set of features composed of multiple
    utterances.
    """

    def __getitem__(self, idx):
        """Get access to the dataset.
        Args:
            idx : index
        Returns:
            features
        """
        raise NotImplementedError

    def __len__(self):
        """Length of the dataset
        Returns:
            int: length of dataset. Can be number of utterances or number of
            total frames depends on implementation.
        """
        raise NotImplementedError

########## DATA SOURCE ##########

class TextDataSource(FileDataSource):
    def __init__(self, data_root, config, speaker_id=None):
        self.data_root = data_root
        self.speaker_ids = None
        self.multi_speaker = False
        # If not None, filter by speaker_id
        self.speaker_id = speaker_id
        self.config = config

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
        seq = _frontend.text_to_sequence(text, p=self.config.replace_pronunciation_prob)

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

    def collect_files(self, file_name):
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



'''
TODO BELOW
WHAT IS PREPROCESSING?
'''




########## DATASET ##########

class FileSourceDataset(Dataset):
    """FileSourceDataset
    Most basic dataset implementation. It supports utterance-wise iteration and
    has utility (:obj:`asarray` method) to convert dataset to an three
    dimentional :obj:`numpy.ndarray`.
    Speech features have typically different number of time resolusion,
    so we cannot simply represent dataset as an
    array. To address the issue, the dataset class represents set
    of features as ``N x T^max x D`` array by padding zeros where ``N`` is the
    number of utterances, ``T^max`` is maximum number of frame lenghs and ``D``
    is the dimention of features, respectively.
    While this dataset loads features on-demand while indexing, if you are
    dealing with relatively small dataset, it might be useful to convert it to
    an array, and then do whatever with numpy/scipy functionalities.
    Attributes:
        file_data_source (FileDataSource): Data source to specify 1) where to
            find data to be loaded and 2) how to collect features from them.
        collected_files (ndarray): Collected files are stored.
    Args:
        file_data_source (FileDataSource): File data source.
    Examples:
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> for (x, y) in zip(X, Y):
        ...     print(x.shape, y.shape)
        ...
        (578, 425) (578, 187)
        (675, 425) (675, 187)
        (606, 425) (606, 187)
        >>> X.asarray(1000).shape
        (3, 1000, 425)
        >>> Y.asarray(1000).shape
        (3, 1000, 187)
    """

    def __init__(self,
                 file_data_source):
        self.file_data_source = file_data_source
        collected_files = self.file_data_source.collect_files()

        # Multiple files
        if isinstance(collected_files, tuple):
            collected_files = np.asarray(collected_files).T
            lengths = np.array([len(files) for files in collected_files])
            if not (lengths == lengths[0]).all():
                raise RuntimeError(
                    """Mismatch of number of collected files {}.
You must collect same number of files when you collect multiple pair of files.""".format(
                        tuple(lengths)))
        else:
            collected_files = np.atleast_2d(collected_files).T
        if len(collected_files) == 0:
            warn("No files are collected. You might have specified wrong data source.")

        self.collected_files = collected_files

    def __collect_features(self, paths):
        try:
            return self.file_data_source.collect_features(*paths)
        except TypeError as e:
            warn("TypeError while iterating dataset.\n" +
                 "Likely there's mismatch in number of pair of collected files and " +
                 "expected number of arguments of `collect_features`.\n" +
                 "Number of argments: {}\n".format(len(paths)) +
                 "Arguments: {}".format(*paths))
            raise e

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            current, stop, step = idx.indices(len(self))
            return [self[i] for i in range(current, stop, step)]

        paths = self.collected_files[idx]
        return self.__collect_features(paths)

    def __len__(self):
        return len(self.collected_files)

    def asarray(self, padded_length=None, dtype=np.float32,
                padded_length_guess=1000, verbose=0):
        """Convert dataset to numpy array.
        This try to load entire dataset into a single 3d numpy array.
        Args:
            padded_length (int): Number of maximum time frames to be expected.
              If None, it is set to actual maximum time length.
            dtype (numpy.dtype): Numpy dtype.
            padded_length_guess: (int): Initial guess of max time length of
              padded dataset array. Used if ``padded_length`` is None.
        Returns:
            3d-array: Array of shape ``N x T^max x D`` if ``padded_length`` is
            None, otherwise ``N x padded_length x D``.
        """
        collected_files = self.collected_files
        if padded_length is not None:
            T = padded_length
        else:
            T = padded_length_guess  # initial guess

        D = self[0].shape[-1]
        N = len(self)
        X = np.zeros((N, T, D), dtype=dtype)
        lengths = np.zeros(N, dtype=np.int)

        if verbose > 0:
            def custom_range(x):
                return tqdm(range(x))
        else:
            custom_range = range

        for idx in custom_range(len(collected_files)):
            paths = collected_files[idx]
            x = self.__collect_features(paths)
            lengths[idx] = len(x)
            if len(x) > T:
                if padded_length is not None:
                    raise RuntimeError(
                        """Num frames {} exceeded: {}.
Try larger value for padded_length, or set to None""".format(len(x), T))
                warn("Reallocating array because num frames {} exceeded current guess {}.\n".format(
                    len(x), T) +
                    "To avoid memory re-allocations, try large `padded_length_guess` " +
                    "or set `padded_length` explicitly.")
                n = len(x) - T
                # Padd zeros to end of time axis
                X = np.pad(X, [(0, 0), (0, n), (0, 0)],
                           mode="constant", constant_values=0)
                T = X.shape[1]
            X[idx][:len(x), :] = x
            lengths[idx] = len(x)

        if padded_length is None:
            max_len = np.max(lengths)
            X = X[:, :max_len, :]
        return X
