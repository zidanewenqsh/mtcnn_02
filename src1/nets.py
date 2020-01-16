import torch
from torch import nn, Tensor
import numpy as np
import time
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mark = int(1000*time.time())
    def paraminit(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)
    def forward(self, *input:Tensor) -> Tensor:
        raise NotImplementedError

class PNet(Net):
    def __init__(self, name="pnet"):
        super(PNet, self).__init__()
        self.name = name
        self.size = 12
        self.layer_1 = nn.Sequential(
            # 12
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 10
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 5
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 1
        )
        self.layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, input:Tensor)->tuple:
        layer1 = self.layer_1(input)
        conf = self.layer_2_1(layer1)
        offset = self.layer_2_2(layer1)
        return conf, offset

class RNet(Net):
    def __init__(self, name="rnet"):
        super(RNet, self).__init__()
        self.name = name
        self.size = 24
        self.layer_1 = nn.Sequential(
            # 24
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 22
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 11
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 9
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 4
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.PReLU()
            # 3
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=64 * 3 * 3, out_features=128, bias=True),
            nn.PReLU()
        )
        self.layer_3_1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.layer_3_2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=4, bias=True)
        )

    def forward(self, input:Tensor)->tuple:
        layer1 = self.layer_1(input)
        layer1_ = torch.reshape(layer1, shape=(-1, 64 * 3 * 3))
        layer2 = self.layer_2(layer1_)
        conf = self.layer_3_1(layer2)
        offset = self.layer_3_2(layer2)
        return conf, offset

class ONet(Net):
    def __init__(self, name="onet"):
        super(ONet, self).__init__()
        self.name = name
        self.size = 48
        self.layer_1 = nn.Sequential(
            # 48
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 46
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 23
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 21
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 10
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # 8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.PReLU(),
            # 3
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=256, bias=True),
            nn.PReLU()
        )
        self.layer_3_1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.layer_3_2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=4, bias=True)
        )


    def forward(self, input:Tensor)->tuple:
        layer1 = self.layer_1(input)
        layer1_ = torch.reshape(layer1, shape=(-1, 128 * 3 * 3))
        layer2 = self.layer_2(layer1_)
        conf = self.layer_3_1(layer2)
        offset = self.layer_3_2(layer2)
        return conf, offset

if __name__ == '__main__':
    net = Net()
    # torch.save(net, "pnet.org")

