import numpy as np

class RolloutBuffer:
    """On-policy trajectory store for PPO with GAE(lambda). Holds a fixed
    (numSteps x numEnvs) block of transitions, then computes advantages and
    returns and yields flattened minibatches. Actions/logprobs are stored in
    the policy's own (normalized) space, matching what the agent sampled."""
    def __init__(self, numSteps:int, numEnvs:int, obsDim:int, actDim:int,
                 gamma:float=0.99, lam:float=0.95):
        self.numSteps = numSteps
        self.numEnvs  = numEnvs
        self.gamma    = gamma
        self.lam      = lam

        self.obs      = np.zeros((numSteps, numEnvs, obsDim), dtype=np.float32)
        self.actions  = np.zeros((numSteps, numEnvs, actDim), dtype=np.float32)
        self.logProbs = np.zeros((numSteps, numEnvs), dtype=np.float32)
        self.rewards  = np.zeros((numSteps, numEnvs), dtype=np.float32)
        self.dones    = np.zeros((numSteps, numEnvs), dtype=np.float32)
        self.values   = np.zeros((numSteps, numEnvs), dtype=np.float32)

        self.advantages = np.zeros((numSteps, numEnvs), dtype=np.float32)
        self.returns    = np.zeros((numSteps, numEnvs), dtype=np.float32)
        self._ptr = 0

    def reset(self) -> None:
        self._ptr = 0

    def add(self, obs:np.ndarray, action:np.ndarray, logProb:np.ndarray,
            reward:np.ndarray, done:np.ndarray, value:np.ndarray) -> None:
        """Store one time slice across all envs (each arg leads with numEnvs)."""
        i = self._ptr
        self.obs[i]      = obs
        self.actions[i]  = action
        self.logProbs[i] = logProb
        self.rewards[i]  = reward
        self.dones[i]    = done
        self.values[i]   = value
        self._ptr += 1

    def computeGae(self, lastValue:np.ndarray) -> None:
        """Backward GAE sweep. Convention: dones[t] is the terminal flag
        *resulting from* transition t, so it masks the bootstrap for step t.
        lastValue = V(obs after the final stored transition), used only when
        that transition is non-terminal. Each arg leads with numEnvs."""
        adv = np.zeros(self.numEnvs, dtype=np.float32)
        for t in reversed(range(self.numSteps)):
            nextNonTerminal = 1.0 - self.dones[t]
            if t == self.numSteps - 1:
                nextValue = lastValue
            else:
                nextValue = self.values[t + 1]
            delta = (self.rewards[t] + self.gamma * nextValue * nextNonTerminal
                     - self.values[t])
            adv = delta + self.gamma * self.lam * nextNonTerminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def iterMinibatches(self, batchSize:int, seed:int):
        """Yield flattened (obs, act, oldLogProb, advantage, return) batches."""
        n = self.numSteps * self.numEnvs
        obs   = self.obs.reshape(n, -1)
        act   = self.actions.reshape(n, -1)
        logp  = self.logProbs.reshape(n)
        adv   = self.advantages.reshape(n)
        ret   = self.returns.reshape(n)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        for start in range(0, n, batchSize):
            b = idx[start:start + batchSize]
            yield obs[b], act[b], logp[b], adv[b], ret[b]
