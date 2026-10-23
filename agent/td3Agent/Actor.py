import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Deterministic policy. Layer names match FF so BC weights from
    ffAgent can be loaded via state_dict (strict=False)."""
    def __init__(self, obsDims:list, actDim:int,
                 hiddenDim:int=128) -> None:
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(*obsDims, hiddenDim)
        self.layer2 = nn.Linear(hiddenDim, hiddenDim)
        self.layer3 = nn.Linear(hiddenDim, actDim)

    def forward(self, observation:T.Tensor) -> T.Tensor:
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        return T.tanh(self.layer3(x))
