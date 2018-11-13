import torch.nn as nn
import torchvision
from base import BaseModel


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x
