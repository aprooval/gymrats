from agent.ffAgent.FF import FF

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class Agent:
    """Generic feedforward agent — wraps FF network with optimizer and
    predict/train interface. Problem-specific wiring lives in players."""
    def __init__(self, inDim:int, outDim:int,
                 hiddenDim:int=128, lr:float=1e-3):
        self._device = T.device("mps" if T.backends.mps.is_available()
                                    else "cpu")
        self.net = FF(inDim, outDim, hiddenDim).to(self._device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self._criterion = nn.MSELoss()

    def predict(self, observation:np.ndarray) -> np.ndarray:
        """Forward pass, returns output as numpy."""
        state = T.tensor(observation, dtype=T.float32).to(self._device)
        with T.no_grad():
            out = self.net(state)
        return out.cpu().numpy()

    def trainStep(self, inputs:np.ndarray, targets:np.ndarray) -> float:
        """Single gradient step. Returns MSE loss value."""
        self.optimizer.zero_grad()
        x = T.tensor(inputs, dtype=T.float32).to(self._device)
        y = T.tensor(targets, dtype=T.float32).to(self._device)

        pred = self.net(x)
        loss = self._criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path:str) -> None:
        T.save(self.net.state_dict(), path)

    def load(self, path:str) -> None:
        self.net.load_state_dict(T.load(path, map_location=self._device))
