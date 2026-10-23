import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """State-value network V(s) for PPO. Training-only, so it runs wider
    than the actor. Takes state alone (no action, unlike the TD3 critic)."""
    def __init__(self, obsDim:int, hiddenDim:int=256) -> None:
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(obsDim, hiddenDim)
        self.layer2 = nn.Linear(hiddenDim, hiddenDim)
        self.layer3 = nn.Linear(hiddenDim, 1)

    def forward(self, observation:T.Tensor) -> T.Tensor:
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
