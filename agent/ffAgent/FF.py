import torch as T
import torch.nn as nn
import torch.nn.functional as F

class FF(nn.Module):
    """Generic feedforward network with configurable input/output dims."""
    def __init__(self, inDim:int, outDim:int,
                 hiddenDim:int=128) -> None:
        super(FF, self).__init__()
        self.layer1 = nn.Linear(inDim, hiddenDim)
        self.layer2 = nn.Linear(hiddenDim, hiddenDim)
        self.layer3 = nn.Linear(hiddenDim, outDim)

    def forward(self, x:T.Tensor) -> T.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
