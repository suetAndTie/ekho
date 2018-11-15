from base.base_config import BaseConfig

class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)

        # ADD ANYTHING EXTRA HERE

config = Config()
