'''
Based on
https://github.com/r9y9/deepvoice3_pytorch/blob/master/hparams.py
'''

class BaseConfig(object):
	def __init__(self, **kwargs):
		self.lr = 1e-4
		self.batch_size = 32 # total over all gpus
		self.epochs = 100
		self.gpus = 1
		self.num_workers = 1
		self.data_shape = (1024, 1024, 3)


		# Text type
		self.frontend = 'en'

		self.replace_pronunciation_prob = 0.5
		self.min_text = 20



		# PREPROCESSING
		self.num_mels = 80
		self.fmin = 125
		self.fmax = 7600
		self.fft_size = 1024 # fft points (samples)
		self.hop_size = 256
		self.sample_rate = 22050
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



	def set(self, k, v):
		setattr(self, k, v)

	def __str__(self):
		return str(self.__dict__)

if __name__ == '__main__':
	config = BaseConfig()
	print(config)
