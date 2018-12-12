import numpy as np
from base.base_config import BaseConfig

class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)

        # ADD ANYTHING EXTRA HERE

        # WAVENET
        self.use_wavenet =  True
        self.input_type = 'raw'
        # This should equal to `quantize_channels` if mu-law quantize enabled
        # otherwise num_mixture * 3 (pi, mean, log_scale)
        self.out_channels = 10 * 3
        self.layers = 24
        self.stacks = 4
        self.residual_channels = 512
        self.gate_channels = 512 # split into 2 gropus internally for gated activation
        self.skip_out_channels = 256
        # If True, apply weight normalization as same as DeepVoice3
        self.weight_normalization = True
        self.kernel_size = 3
        # If True, use transposed convolutions to upsample conditional features,
        # otherwise repeat features to adjust time resolution
        self.upsample_conditional_features = False
        # should np.prod(upsample_scales) == hop_size
        self.upsample_scales = [4, 4, 4, 4]
        # Freq axis kernel size for upsampling network
        self.freq_axis_kernel_size = 3
        # Use legacy code or not. Default is True since we already provided a model
        # based on the legacy code that can generate high-quality audio.
        # Ref: https://github.com/r9y9/wavenet_vocoder/pull/73
        self.legacy = True

        # WAVENET LOSS FUNCTION
        self.quantize_channels = 65536,  # 65536 or 256
        #  Mixture of logistic distributions:
        self.log_scale_min = float(np.log(1e-14))

config = Config()
