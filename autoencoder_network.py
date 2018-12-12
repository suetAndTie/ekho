from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf

class Encoder(object):
    def __init__(self, training=True, scope="encoder", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("text_embedding"):
                embedding = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, Tx, e)

            with tf.variable_scope("encoder_prenet"):
                tensor = fc_block(embedding, hp.enc_channels, training=training) # (N, Tx, c)

            with tf.variable_scope("encoder_conv"):
                for i in range(hp.enc_layers):
                    outputs = conv_block(tensor,
                                        size=hp.enc_filter_size,
                                        rate=2**i,
                                        training=training,
                                        scope="encoder_conv_{}".format(i)) # (N, Tx, c)
                    tensor = (outputs + tensor) * tf.sqrt(0.5)

            with tf.variable_scope("encoder_postnet"):
                keys = fc_block(tensor, hp.embed_size, training=training) # (N, Tx, e)
                vals = tf.sqrt(0.5) * (keys + embedding) # (N, Tx, e)

        return keys, vals

class Decoder(object):
    def __init__(self, inputs, keys, vals, prev_max_attentions_li=None, training=True, scope="decoder", reuse=None):
        pass

        
