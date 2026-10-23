import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Twin Q-networks in one module. Each returns Q(s,a) for a shared
    (state, action) input."""
    def __init__(self, obsDims:list, actDim:int,
                 hiddenDim:int=128) -> None:
        super(Critic, self).__init__()
        inDim = obsDims[0] + actDim

        self.q1Layer1 = nn.Linear(inDim, hiddenDim)
        self.q1Layer2 = nn.Linear(hiddenDim, hiddenDim)
        self.q1Layer3 = nn.Linear(hiddenDim, 1)

        self.q2Layer1 = nn.Linear(inDim, hiddenDim)
        self.q2Layer2 = nn.Linear(hiddenDim, hiddenDim)
        self.q2Layer3 = nn.Linear(hiddenDim, 1)

    def forward(self, state:T.Tensor, action:T.Tensor) -> tuple:
        sa = T.cat([state, action], dim=-1)

        q1 = F.relu(self.q1Layer1(sa))
        q1 = F.relu(self.q1Layer2(q1))
        q1 = self.q1Layer3(q1)

        q2 = F.relu(self.q2Layer1(sa))
        q2 = F.relu(self.q2Layer2(q2))
        q2 = self.q2Layer3(q2)
        return q1, q2

    def q1(self, state:T.Tensor, action:T.Tensor) -> T.Tensor:
        sa = T.cat([state, action], dim=-1)
        q1 = F.relu(self.q1Layer1(sa))
        q1 = F.relu(self.q1Layer2(q1))
        return self.q1Layer3(q1)
