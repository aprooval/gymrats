from agent.mdnAgent.MDN import MDN

import numpy as np
import torch as T
import torch.optim as optim

class Agent:
    """Generic MDN agent — wraps MDN network with optimizer and
    predict/train interface. Problem-specific wiring lives in players."""
    def __init__(self, inDim:int, outDim:int, K:int=2,
                 hiddenDim:int=128, lr:float=1e-3):
        self._device = T.device("mps" if T.backends.mps.is_available()
                                    else "cpu")
        self.net = MDN(inDim, outDim, K, hiddenDim).to(self._device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def predict(self, observation:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #forward pass, returns (means, vars, weights)
        state = T.tensor(observation, dtype=T.float32).to(self._device)
        with T.no_grad():
            means, logVars, logits = self.net(state)
        weights = T.softmax(logits, dim=-1)
        return (means.cpu().numpy(),
                T.exp(logVars).cpu().numpy(),
                weights.cpu().numpy())

    def trainStep(self, inputs:np.ndarray, targets:np.ndarray) -> float:
        """Single gradient step. Returns NLL loss value."""
        self.optimizer.zero_grad()
        x = T.tensor(inputs, dtype=T.float32).to(self._device)
        y = T.tensor(targets, dtype=T.float32).to(self._device)

        means, logVars, logits = self.net(x)
        loss = MDN.nllLoss(y, means, logVars, logits)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path:str) -> None:
        T.save(self.net.state_dict(), path)

    def load(self, path:str) -> None:
        self.net.load_state_dict(T.load(path, map_location=self._device))
