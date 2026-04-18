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
        self.lr           = hyperparams.get("LR", 1e-4)
        self.tau          = hyperparams.get("TAU", 0.005)
        self.gamma        = hyperparams.get("GAMMA", 0.99)
        self.epsEnd       = hyperparams.get("EPS_END", 0.05)
        self.epsStart     = hyperparams.get("EPS_START", 0.9)
        self.epsDecay     = hyperparams.get("EPS_DECAY", 1000)
        self.batchSize    = hyperparams.get("BATCH_SIZE", 128)
        capacity          = hyperparams.get("MEM_CAPACITY", 100000)
        self.stateMem     = mem(capacity, np.float32, *obsDims)
        self.nextStateMem = mem(capacity, np.float32, *obsDims)
        self.actionMem    = mem(capacity, np.int64,   *actDims)
        self.rewardMem    = mem(capacity, np.float32)
        self.doneMem      = mem(capacity, np.bool_)

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
