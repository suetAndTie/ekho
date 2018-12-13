import torch.nn as nn
import torchvision


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        # TODO: FILL
        self.net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = BaseModel()
    print(net)
