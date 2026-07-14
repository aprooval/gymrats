import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Gaussian policy for PPO. The mean trunk uses layer1/2/3 names so a
    BC'd FF (ffAgent) can be warm-started via state_dict (strict=False).
    logStd is a state-independent learnable parameter. The mean is left
    un-squashed (matches FF exactly); actions are clipped/scaled to bounds
    at the agent boundary, and log-probs are plain Gaussian."""
    def __init__(self, obsDim:int, actDim:int, hiddenDim:int=128,
                 logStdInit:float=-0.5) -> None:
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(obsDim, hiddenDim)
        self.layer2 = nn.Linear(hiddenDim, hiddenDim)
        self.layer3 = nn.Linear(hiddenDim, actDim)
        self.logStd = nn.Parameter(T.full((actDim,), float(logStdInit)))

    def forward(self, observation:T.Tensor) -> tuple:
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        mean = self.layer3(x)
        std  = T.exp(self.logStd).expand_as(mean)
        return mean, std

    def distribution(self, observation:T.Tensor) -> T.distributions.Normal:
        mean, std = self.forward(observation)
        return T.distributions.Normal(mean, std)
