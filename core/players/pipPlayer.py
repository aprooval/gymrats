"""PIP Player — wires MDN agent as a Predicted Intercept Point estimator.

Input:  nav state (posNed[3], velNed[3], mach[1]) + target posNed[3] = 10
Output: K=2 mixture components, each (mean[3], var[3], weight[1]) = 14 total
"""
from agent.mdnAgent.Agent import Agent as MdnAgent

import numpy as np

INPUT_DIM  = 10   # posNed(3) + velNed(3) + mach(1) + targetPosNed(3)
OUTPUT_DIM = 3    # PIP is a 3D position
K          = 2    # mixture components

class PipPlayer:
    def __init__(self, lr:float=1e-3, hiddenDim:int=128):
        self.agent = MdnAgent(INPUT_DIM, OUTPUT_DIM, K=K,
                              hiddenDim=hiddenDim, lr=lr)

    def buildInput(self, posNed:np.ndarray, velNed:np.ndarray,
                   mach:float, targetPosNed:np.ndarray) -> np.ndarray:
        return np.concatenate([posNed, velNed, [mach], targetPosNed])

    def predict(self, posNed:np.ndarray, velNed:np.ndarray,
                mach:float, targetPosNed:np.ndarray):
        """Returns (means, vars, weights) for the PIP mixture."""
        x = self.buildInput(posNed, velNed, mach, targetPosNed)
        return self.agent.predict(x)

    def trainStep(self, inputs:np.ndarray, targets:np.ndarray) -> float:
        """inputs: (batch, 10), targets: (batch, 3) PIP positions."""
        return self.agent.trainStep(inputs, targets)

    def flatOutput(self, means:np.ndarray, vars:np.ndarray,
                   weights:np.ndarray) -> np.ndarray:
        """Flatten MDN output to a fixed-size vector for guidance net input.
        Returns (K*7,) = (14,): [mean0, var0, w0, mean1, var1, w1]"""
        parts = []
        for k in range(K):
            parts.extend([means[k], vars[k], [weights[k]]])
        return np.concatenate(parts)

    def save(self, path:str) -> None:
        self.agent.save(path)

    def load(self, path:str) -> None:
        self.agent.load(path)
