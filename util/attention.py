'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py
'''
import numpy as np
from numba import jit


@jit(nopython=True)
def guided_attention(N, max_N, T, max_T, g):
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T)**2 / (2 * g * g))
    return W


def guided_attentions(input_lengths, target_lengths, max_target_len, g=0.2):
    B = len(input_lengths)
    max_input_len = input_lengths.max()
    W = np.zeros((B, max_target_len, max_input_len), dtype=np.float32)
    for b in range(B):
        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_lengths[b], max_target_len, g).T
    return W
