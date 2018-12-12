'''
Adapted from
https://github.com/r9y9/deepvoice3_pytorch/blob/master/audio.py
https://github.com/Kyubyong/deepvoice3/blob/master/prepro.py
https://github.com/Jonathan-LeRoux/lws/blob/master/python/lws.pyx
'''

import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from config import config
import lws



def load_flac(path):
    return librosa.core.load(path, sr=config.sample_rate)[0]


def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, config.sample_rate, wav.astype(np.int16))


def preemphasis(y, coef=0.97):
    """Pre-emphasis
    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.
    Returns:
        array: Output filtered signal.
    """

    b = np.array([1., -coef], y.dtype)
    a = np.array([1.], y.dtype)
    return signal.lfilter(b, a, y)


def inv_preemphasis(x, coef=0.97):
    """Inverse operation of pre-emphasis
    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.
    Returns:
        array: Output filtered signal.
    """
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)


def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - config.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + config.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** config.power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - config.ref_level_db
    if not config.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - config.min_level_db >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(config.fft_size, config.hop_size, mode="speech")



# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if config.fmax is not None:
        assert config.fmax <= config.sample_rate // 2
    return librosa.filters.mel(config.sample_rate, config.fft_size,
                               fmin=config.fmin, fmax=config.fmax,
                               n_mels=config.num_mels)


def _amp_to_db(x):
    min_level = np.exp(config.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - config.min_level_db) / -config.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -config.min_level_db) + config.min_level_db


# PREPROCESSING
# https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()



def mulaw(x, mu=256):
    """Mu-Law companding
    Method described in paper [1]_.
    .. math::
        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])
    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).
    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> y = P.mulaw_quantize(x)
        >>> print(y.min(), y.max(), y.dtype)
        15 246 int64
    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)
