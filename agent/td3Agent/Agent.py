from agent.td3Agent.Actor  import Actor  as actor
from agent.td3Agent.Critic import Critic as critic
from agent.Memory          import Memory as mem

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class Agent:
    """TD3 agent. Continuous action space, bounded by actLow/actHigh.
    Actor outputs tanh in [-1,1] and is scaled to bounds at act time."""
    def __init__(self, obsDims:list, actDim:int,
                 actLow:np.ndarray, actHigh:np.ndarray,
                 **hyperparams:dict):
        self._steps    = 0
        self._updates  = 0
        self._device   = T.device("mps" if T.backends.mps.is_available()
                                       else "cpu")

        self.actLow   = T.tensor(actLow,  dtype=T.float32).to(self._device)
        self.actHigh  = T.tensor(actHigh, dtype=T.float32).to(self._device)
        self.actScale = (self.actHigh - self.actLow) / 2.0
        self.actBias  = (self.actHigh + self.actLow) / 2.0

        #kwargs
        capacity          = hyperparams.get("MEM_CAPACITY", 1000000)
        self.lr           = hyperparams.get("LR", 3e-4)
        self.tau          = hyperparams.get("TAU", 0.005)
        self.gamma        = hyperparams.get("GAMMA", 0.99)
        self.batchSize    = hyperparams.get("BATCH_SIZE", 256)
        self.policyNoise  = hyperparams.get("POLICY_NOISE", 0.2)
        self.noiseClip    = hyperparams.get("NOISE_CLIP", 0.5)
        self.policyFreq   = hyperparams.get("POLICY_FREQ", 2)
        self.explNoise    = hyperparams.get("EXPL_NOISE", 0.1)

        #replay buffer — action stored in tanh space [-1,1]
        self.stateMem     = mem(capacity, np.float32, *obsDims)
        self.nextStateMem = mem(capacity, np.float32, *obsDims)
        self.actionMem    = mem(capacity, np.float32, actDim)
        self.rewardMem    = mem(capacity, np.float32)
        self.doneMem      = mem(capacity, np.bool_)

        #networks
        self.actor        = actor(obsDims, actDim).to(self._device)
        self.actorTarget  = actor(obsDims, actDim).to(self._device)
        self.actorTarget.load_state_dict(self.actor.state_dict())
        self.actorOpt     = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic       = critic(obsDims, actDim).to(self._device)
        self.criticTarget = critic(obsDims, actDim).to(self._device)
        self.criticTarget.load_state_dict(self.critic.state_dict())
        self.criticOpt    = optim.Adam(self.critic.parameters(), lr=self.lr)

    def _scale(self, tanhAct:T.Tensor) -> T.Tensor:
        return tanhAct * self.actScale + self.actBias

    def act(self, observation:np.ndarray,
            explore:bool=True) -> np.ndarray:
        #returns action in environment (scaled) space
        state = T.tensor(observation, dtype=T.float32).to(self._device)
        with T.no_grad():
            tanhAct = self.actor(state)
            if explore:
                noise = T.randn_like(tanhAct) * self.explNoise
                tanhAct = (tanhAct + noise).clamp(-1.0, 1.0)
            action = self._scale(tanhAct)
        self._steps += 1
        return action.cpu().numpy()

    def memorize(self, state:np.ndarray, action:np.ndarray,
                 nextState:np.ndarray, reward:np.float32,
                 done:np.bool_) -> None:
        """Action must be in tanh space [-1,1]. Callers that produced it
        via act() should divide by actScale and subtract actBias, or
        store the pre-scale tanh output alongside."""
        self.stateMem.push(state)
        self.nextStateMem.push(nextState)
        self.actionMem.push(action)
        self.rewardMem.push(reward)
        self.doneMem.push(done)

    def learn(self) -> None:
        if self.stateMem.counter <= self.batchSize:
            return

        seed           = np.random.randint(0, 9999999)
        stateBatch     = self.stateMem.sample(self.batchSize, seed)
        nextStateBatch = self.nextStateMem.sample(self.batchSize, seed)
        actBatch       = self.actionMem.sample(self.batchSize, seed)
        rewardBatch    = self.rewardMem.sample(self.batchSize, seed)
        doneBatch      = self.doneMem.sample(self.batchSize, seed)

        stateBatch     = T.tensor(stateBatch).to(self._device)
        nextStateBatch = T.tensor(nextStateBatch).to(self._device)
        actBatch       = T.tensor(actBatch).to(self._device)
        rewardBatch    = T.tensor(rewardBatch).to(self._device).unsqueeze(1)
        doneBatch      = T.tensor(doneBatch).to(self._device).unsqueeze(1)

        #target action smoothing + clipped double-Q
        with T.no_grad():
            noise = (T.randn_like(actBatch) * self.policyNoise)\
                      .clamp(-self.noiseClip, self.noiseClip)
            nextTanh = (self.actorTarget(nextStateBatch) + noise)\
                         .clamp(-1.0, 1.0)
            q1Next, q2Next = self.criticTarget(nextStateBatch, nextTanh)
            qNext  = T.min(q1Next, q2Next)
            qNext[doneBatch] = 0.0
            qTarget = rewardBatch + self.gamma * qNext

        #critic update
        q1, q2 = self.critic(stateBatch, actBatch)
        criticLoss = nn.functional.mse_loss(q1, qTarget) \
                   + nn.functional.mse_loss(q2, qTarget)
        self.criticOpt.zero_grad()
        criticLoss.backward()
        self.criticOpt.step()

        self._updates += 1

        #delayed actor + target updates
        if self._updates % self.policyFreq == 0:
            actorLoss = -self.critic.q1(stateBatch,
                                        self.actor(stateBatch)).mean()
            self.actorOpt.zero_grad()
            actorLoss.backward()
            self.actorOpt.step()

            self._polyak(self.actor,  self.actorTarget)
            self._polyak(self.critic, self.criticTarget)

    def _polyak(self, source:nn.Module, target:nn.Module) -> None:
        srcDict = source.state_dict()
        tgtDict = target.state_dict()
        for key in srcDict:
            tgtDict[key] = srcDict[key] * self.tau\
                         + tgtDict[key] * (1 - self.tau)
        target.load_state_dict(tgtDict)

    def warmStartActor(self, ffStatePath:str) -> None:
        """Load BC'd FF weights into actor and actor target. FF layer
        names (layer1/2/3) match Actor, so strict=False handles any
        extra keys."""
        stateDict = T.load(ffStatePath, map_location=self._device)
        self.actor.load_state_dict(stateDict, strict=False)
        self.actorTarget.load_state_dict(self.actor.state_dict())

    def save(self, path:str) -> None:
        T.save({
            "actor":        self.actor.state_dict(),
            "actorTarget":  self.actorTarget.state_dict(),
            "critic":       self.critic.state_dict(),
            "criticTarget": self.criticTarget.state_dict(),
        }, path)

    def load(self, path:str) -> None:
        ckpt = T.load(path, map_location=self._device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actorTarget.load_state_dict(ckpt["actorTarget"])
        self.critic.load_state_dict(ckpt["critic"])
        self.criticTarget.load_state_dict(ckpt["criticTarget"])
