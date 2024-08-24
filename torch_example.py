from torch import nn
import torch

class Red(nn.Sequential):
    def __init__(self) -> None:
        super().__init__([
            nn.Linear(1, 2, bias = True),
            nn.ReLU(),
            nn.Linear(2, 4, bias = True),
            nn.ReLU(),
            nn.Linear(4, 2, bias = True),
            nn.Sigmoid(),
        ])

if __name__ == '__main__':
    model = Red()
    model(torch.randn(1,1))

