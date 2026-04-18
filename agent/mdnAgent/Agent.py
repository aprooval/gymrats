from agent.mdnAgent.MDN import MDN
from agent.Memory import Memory

import numpy as np
import torch as T
import torch.optim as optim

class Agent:
    def __init__(self, inDim:int, outDim:int, K:int=2,
                 hiddenDim:int=128, lr:float=1e-3, **hyperparams):
        self._device = T.device("mps" if T.backends.mps.is_available()
                                    else "cpu")
        self.net = MDN(inDim, outDim, K, hiddenDim).to(self._device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        capacity        = hyperparams.get("MEM_CAPACITY", 100000)
        self.batchSize  = hyperparams.get("BATCH_SIZE", 128)
        self.inputMem   = Memory(capacity, np.float32, inDim)
        self.targetMem  = Memory(capacity, np.float32, outDim)
        self._stage: list[np.ndarray] = []

    #behavior cloning
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

    #online learning
    def memorize(self, input:np.ndarray) -> None:
        self._stage.append(input)

    def commitEpisode(self, target:np.ndarray) -> None:
        """Label all staged inputs with target and push to replay. Clears stage."""
        for inp in self._stage:
            self.inputMem.push(inp)
            self.targetMem.push(target)
        self._stage = []

    def learn(self) -> float | None:
        """One gradient step from replay. Returns NLL loss, or None if buffer not full."""
        if self.inputMem.counter <= self.batchSize:
            return None
        seed    = np.random.randint(0, 9999999)
        inputs  = self.inputMem.sample(self.batchSize, seed)
        targets = self.targetMem.sample(self.batchSize, seed)
        return self.trainStep(inputs, targets)

    #shared, behavior cloning and online learning
    def predict(self, observation:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #forward pass, returns (means, vars, weights)
        state = T.tensor(observation, dtype=T.float32).to(self._device)
        with T.no_grad():
            means, logVars, logits = self.net(state)
        weights = T.softmax(logits, dim=-1)
        return (means.cpu().numpy(),
                T.exp(logVars).cpu().numpy(),
                weights.cpu().numpy())

    def save(self, path:str) -> None:
        T.save(self.net.state_dict(), path)

    def load(self, path:str) -> None:
        self.net.load_state_dict(T.load(path, map_location=self._device))
