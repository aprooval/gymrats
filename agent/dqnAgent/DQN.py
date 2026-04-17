import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, obsDims:list, numActions:int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(*obsDims, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, numActions)

    def forward(self, observation:T.Tensor) -> T.Tensor:
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
