'''
wavenet_util.py
Based on https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/util.py
'''

def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw"


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"


def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)
