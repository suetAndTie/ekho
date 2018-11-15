'''
https://github.com/r9y9/deepvoice3_pytorch/blob/master/deepvoice3_pytorch/frontend/en/__init__.py
'''


# coding: utf-8
from .frontend.text.symbols import symbols

import nltk
from random import random

n_vocab = len(symbols)

_arpabet = nltk.corpus.cmudict.dict()


def _maybe_get_arpabet(word, p):
    try:
        phonemes = _arpabet[word][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        return word

    return '{%s}' % phonemes if random() < p else word


def mix_pronunciation(text, p):
    text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    return text


def text_to_sequence(text, p=0.0):
    if p >= 0:
        text = mix_pronunciation(text, p)
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text


from .frontend.text import sequence_to_text
