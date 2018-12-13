

class BaseConfig(object):
	def __init__(self, **kwargs):
		self.set('lr', 1e-4)
		self.set('batch_size', 32) # total over all gpus
		self.set('epochs', 100)
		self.set('gpus', 1)
		self.set('data_shape', (1024, 1024, 3))


	def set(self, k, v):
		setattr(self, k, v)

	def __str__(self):
		return str(self.__dict__)

if __name__ == '__main__':
	config = BaseConfig()
	print(config)
