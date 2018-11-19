'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/hparams.py
'''

class BaseConfig(object):
    def __init__(self, **kwargs):
        # TRAINING
        self.lr = 1e-4
        self.batch_size = 32 # total over all gpus
        self.epochs = 100
        self.gpus = 1
        self.num_workers = 2
        self.lr_schedule = "noam_learning_rate_decay"
        self.lr_schedule_kwargs = {}


        # SAVE
        self.checkpoint_interval=10000
        self.eval_interval=10000
        self.save_optimizer_state=True,


        # TEXT
        self.frontend = 'en'
        self.replace_pronunciation_prob = 0.5
        self.min_text = 20



        # PREPROCESSING
        self.num_mels = 80
        self.fmin = 125
        self.fmax = 7600
        self.fft_size = 1024 # fft points (samples)
        self.hop_size = 256
        self.sample_rate = 16000 # original 22050, ASR 16000
        self.preemphasis = 0.97
        self.min_level_db = -100
        self.ref_level_db = 20
        # rescale waveform
        # x is input waveform, y is rescaled waveform
        # y = x / np.abs(x).max() * rescaling_max
        self.rescaling = False
        self.rescaling_max = 0.999
        # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
        # happen depends on min_level_db and ref_level_db, causing clipping noise.
        # If False, assertion is added to ensure no clipping happens.
        self.allow_clipping_in_normalization = True


        # MODEL
        self.outputs_per_step = 1
        self.downsample_step = 4
        self.pin_memory = True
        self.max_positions = 512
		self.n_speakers = 1
		self.speaker_embed_dim = 16
		self.text_embed_dim = 128
		self.outputs_per_step = 1
		self.downsample_step = 4
		self.padding_idx = 0
		self.dropout = 1 - 0.95
		self.kernel_size = 3
		self.encoder_channels = 256
		self.decoder_channels = 256
		self.converter_channels = 256 # Note: large comuptation cost
		self.use_memory_mask = True
		self.trainable_positional_encodings = False
		self.use_decoder_state_for_postnet_input = True
		self.max_positions = 512
		self.speaker_embedding_weight_std = 0.01
		self.freeze_embedding = False
		self.key_projection = False
		self.value_projection = False


		# ATTENTION
		# this can be list for multple layers of attention
    	# e.g., [True, False, False, False, True]
    	self.force_monotonic_attention = True
		# Attention constraint for incremental decoding
		self.window_ahead = 3
		# Attention constraint for incremental decoding
		self.window_backward = 1


		# AUDIO
		self.num_mels = 80
		self.fft_size 1024



        # LOSS
        self.masked_loss_weight = 0.5 # (1 - w) * loss + w * masked_loss
        self.binary_divergence_weight = 0.1 # set 0 to disable
        self.priority_freq = 3000 # heuristic: priotrize [0 ~ priotiry_freq] for linear loss
        self.priority_freq_weight = 0.0
        self.use_guided_attention = True
        self.guided_attention_sigma = 0.2



    def set(self, k, v):
        setattr(self, k, v)

    def __str__(self):
        return str(self.__dict__)

if __name__ == '__main__':
    config = BaseConfig()
    print(config)
