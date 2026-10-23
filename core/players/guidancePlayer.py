"""Guidance Player — wires FF agent as a guidance law.

Input:  nav state (posNed[3], velNed[3], mach[1]) + PIP MDN output (K*7=14) = 24
Output: accCmd (3D acceleration command in NED)
"""
from agent.ffAgent.Agent import Agent as FfAgent

import numpy as np

NAV_DIM    = 10   # posNed(3) + velNed(3) + mach(1) + targetPosNed(3)
MDN_DIM    = 14   # K=2 * (mean[3] + var[3] + weight[1])
INPUT_DIM  = NAV_DIM + MDN_DIM  # 24
OUTPUT_DIM = 3    # accCmd NED

class GuidancePlayer:
    def __init__(self, lr:float=1e-3, hiddenDim:int=128):
        self.agent = FfAgent(INPUT_DIM, OUTPUT_DIM,
                             hiddenDim=hiddenDim, lr=lr)

    def buildInput(self, posNed:np.ndarray, velNed:np.ndarray,
                   mach:float, targetPosNed:np.ndarray,
                   mdnFlat:np.ndarray) -> np.ndarray:
        nav = np.concatenate([posNed, velNed, [mach], targetPosNed])
        return np.concatenate([nav, mdnFlat])

    def predict(self, posNed:np.ndarray, velNed:np.ndarray,
                mach:float, targetPosNed:np.ndarray,
                mdnFlat:np.ndarray) -> np.ndarray:
        """Returns accCmd (3,) in NED."""
        x = self.buildInput(posNed, velNed, mach, targetPosNed, mdnFlat)
        return self.agent.predict(x)

    def trainStep(self, inputs:np.ndarray, targets:np.ndarray) -> float:
        """inputs: (batch, 24), targets: (batch, 3) accCmd."""
        return self.agent.trainStep(inputs, targets)

    def save(self, path:str) -> None:
        self.agent.save(path)

    def load(self, path:str) -> None:
        self.agent.load(path)
