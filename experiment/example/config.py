from base import BaseConfig

class Config(BaseConfig):
	def __init__(self, **kwargs):
		self.set('lr', 1e-4)
		self.set('batch_size', 32) # total over all gpus
		self.set('epochs', 100)
		self.set('gpus', 1)
		self.set('data_shape', (1024, 1024, 3))

		self.num_mels = 80
		self.fmin = 125
		self.fmax = 7600
		self.fft_size = 1024
		self.hop_size = 256
		self.sample_rate = 22050
		self.preemphasis = 0.97
		self.min_level_db = -100
		self.ref_level_db = 20

		self.outputs_per_step = 1
		self.downsample_step = 4
		self.pin_memory = True
		self.max_positions = 512




	def set(self, k, v):
		setattr(self, k, v)

	def __str__(self):
		return str(self.__dict__)
