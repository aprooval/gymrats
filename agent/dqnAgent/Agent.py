from agent.dqnAgent.DQN import DQN as dqn
from agent.Memory import Memory as mem

import math
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, obsDims:list, actDims:list, actSpace:np.ndarray, 
                 **hyperparams:dict):
        #member variables
        self._steps  = 0
        self._device = T.device("mps" if T.backends.mps.is_available()
                                   else "cpu")
        self.actSpace = actSpace

        #kwargs
        if "MEM_CAPACITY" in hyperparams.keys():
            self.stateMem     = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32, *obsDims)
            self.nextStateMem = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32, *obsDims)
            self.actionMem    = mem(hyperparams["MEM_CAPACITY"],
                                    np.int64, *actDims)
            self.rewardMem    = mem(hyperparams["MEM_CAPACITY"],
                                    np.float32)
            self.doneMem      = mem(hyperparams["MEM_CAPACITY"],np.bool_)
        else:
            self.stateMem     = mem(100000, np.float32, *obsDims)
            self.nextStateMem = mem(100000, np.float32, *obsDims)
            self.actionMem    = mem(100000, np.int64,   *actDims) 
            self.rewardMem    = mem(100000, np.float32) 
            self.doneMem      = mem(100000, np.bool_)
        if "LR" in hyperparams.keys():
            self.lr = hyperparams["LR"]
        else:
            self.lr = 1e-4
        if "TAU" in hyperparams.keys():
            self.tau = hyperparams["TAU"]
        else:
            self.tau = 0.005
        if "GAMMA" in hyperparams.keys():
            self.gamma = hyperparams["GAMMA"]
        else:
            self.gamma = 0.99
        if "EPS_END" in hyperparams.keys():
            self.epsEnd = hyperparams["EPS_END"]
        else:
            self.epsEnd = 0.05
        if "EPS_START" in hyperparams.keys():
            self.epsStart = hyperparams["EPS_START"]
        else:
            self.epsStart = 0.9
        if "EPS_DECAY" in hyperparams.keys():
            self.epsDecay = hyperparams["EPS_DECAY"]
        else:
            self.epsDecay = 1000
        if "BATCH_SIZE" in hyperparams.keys():
            self.batchSize = hyperparams["BATCH_SIZE"]
        else:
            self.batchSize = 128

        #networks
        self.policyNet = dqn(obsDims, actSpace.size).to(self._device)
        self.targetNet = dqn(obsDims, actSpace.size).to(self._device)
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.optimizer = optim.AdamW(self.policyNet.parameters(),
                                     lr = self.lr, amsgrad=True)

    def act(self, observation:np.ndarray) -> np.int64:
        self.epsilon = self.epsEnd + (self.epsStart - self.epsEnd)*\
            math.exp(-1. * self._steps / self.epsDecay)
        self._steps += 1

        if random.random() > self.epsilon:
            state  = T.tensor(observation).to(self._device)
            with T.no_grad():
                action = self.policyNet.forward(state).argmax().item()
            action = np.int64(action)
        else:
            action = np.random.choice(self.actSpace.size)
            action = np.int64(action)
        return action
    
    def memorize(self, state:np.ndarray, action:np.ndarray, 
                 nextState:np.ndarray, reward:np.float32, 
                 done:np.bool_) -> None:
        self.stateMem.push(state)
        self.nextStateMem.push(nextState)
        self.actionMem.push(action)
        self.rewardMem.push(reward)
        self.doneMem.push(done)
    
    def learn(self) -> None:
        if self.stateMem.counter <= self.batchSize:
            return
        
        self.optimizer.zero_grad()

        seed           = np.random.randint(0,9999999)
        actBatch       = self.actionMem.sample(self.batchSize, seed)
        stateBatch     = self.stateMem.sample(self.batchSize, seed)
        nextStateBatch = self.nextStateMem.sample(self.batchSize,seed)
        rewardBatch    = self.rewardMem.sample(self.batchSize, seed)
        doneBatch      = self.doneMem.sample(self.batchSize, seed)

        actBatch       = T.Tensor(actBatch).long().to(self._device)
        stateBatch     = T.Tensor(stateBatch).to(self._device)
        nextStateBatch = T.Tensor(nextStateBatch).to(self._device)
        rewardBatch    = T.Tensor(rewardBatch).to(self._device)
        doneBatch      = T.Tensor(doneBatch).bool().to(self._device)

        value = self.policyNet.forward(stateBatch).gather(1,actBatch)
        with T.no_grad():
            nextValue = self.targetNet.forward(nextStateBatch).max(1)[0]
        nextValue[doneBatch] = 0.0
        reward_ = nextValue*self.gamma + rewardBatch

        criterion = nn.SmoothL1Loss()
        loss = criterion(value.squeeze(), reward_)#.to(self.device)
        loss.backward()
        nn.utils.clip_grad_value_(self.policyNet.parameters(),100)
        self.optimizer.step()

        policyNetDict = self.policyNet.state_dict()
        targetNetDict = self.targetNet.state_dict()
        for key in policyNetDict:
            targetNetDict[key] = policyNetDict[key]*self.tau +\
                targetNetDict[key]*(1-self.tau)
        self.targetNet.load_state_dict(targetNetDict)
