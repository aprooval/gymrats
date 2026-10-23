from agent.ppoAgent.Actor        import Actor  as actor
from agent.ppoAgent.Critic       import Critic as critic
from agent.ppoAgent.RolloutBuffer import RolloutBuffer as buffer
from agent.ffAgent.FF            import FF as ff

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class Agent:
    """PPO agent, continuous actions bounded by actLow/actHigh. The policy is
    Gaussian in a normalized [-1,1]-ish space; actions are clipped and scaled
    to bounds at the env boundary. Supports a decayable behavior-cloning
    auxiliary loss (action MSE to a frozen BC'd FF) and BC warm-start of the
    actor mean trunk."""
    def __init__(self, obsDim:int, actDim:int,
                 actLow:np.ndarray, actHigh:np.ndarray,
                 numSteps:int, numEnvs:int, **hyperparams:dict):
        self._device = T.device("mps" if T.backends.mps.is_available()
                                     else "cpu")
        self.obsDim = obsDim
        self.actDim = actDim

        self.actLow   = T.tensor(actLow,  dtype=T.float32).to(self._device)
        self.actHigh  = T.tensor(actHigh, dtype=T.float32).to(self._device)
        self.actScale = (self.actHigh - self.actLow) / 2.0
        self.actBias  = (self.actHigh + self.actLow) / 2.0

        #kwargs
        self.lr          = hyperparams.get("LR", 3e-4)
        self.gamma       = hyperparams.get("GAMMA", 0.99)
        self.lam         = hyperparams.get("LAM", 0.95)
        self.clip        = hyperparams.get("CLIP", 0.2)
        self.epochs      = hyperparams.get("EPOCHS", 10)
        self.miniBatch   = hyperparams.get("MINIBATCH", 4096)
        self.entCoef     = hyperparams.get("ENT_COEF", 0.01)
        self.vfCoef      = hyperparams.get("VF_COEF", 0.5)
        self.maxGradNorm = hyperparams.get("MAX_GRAD_NORM", 0.5)
        self.bcCoef      = hyperparams.get("BC_COEF", 0.0)
        actorHidden      = hyperparams.get("ACTOR_HIDDEN", 128)
        criticHidden     = hyperparams.get("CRITIC_HIDDEN", 256)

        #networks
        self.actor  = actor(obsDim, actDim, actorHidden).to(self._device)
        self.critic = critic(obsDim, criticHidden).to(self._device)
        self.opt    = optim.Adam(list(self.actor.parameters())
                                 + list(self.critic.parameters()), lr=self.lr)

        #on-policy storage
        self.buffer = buffer(numSteps, numEnvs, obsDim, actDim,
                             gamma=self.gamma, lam=self.lam)

        #optional frozen BC reference for the auxiliary loss
        self._bcNet    = None
        self._bcHidden = actorHidden

        # running return normalizer — critic predicts (ret - mean) / std;
        # act() un-normalizes before returning value for GAE
        self._retMean = 0.0
        self._retStd  = 1.0

    def _scale(self, tanhAct:T.Tensor) -> T.Tensor:
        return tanhAct * self.actScale + self.actBias

    def act(self, observation:np.ndarray) -> tuple:
        """Sample actions for a batch of envs. Returns (physical action for the
        env, normalized sampled action, log-prob, value) — all numpy."""
        state = T.tensor(observation, dtype=T.float32).to(self._device)
        with T.no_grad():
            dist    = self.actor.distribution(state)
            sample  = dist.sample()
            logProb = dist.log_prob(sample).sum(-1)
            rawVal  = self.critic(state).squeeze(-1)
            value   = rawVal * self._retStd + self._retMean  # un-normalize for GAE
            envAct  = self._scale(sample.clamp(-1.0, 1.0))
        return (envAct.cpu().numpy(), sample.cpu().numpy(),
                logProb.cpu().numpy(), value.cpu().numpy())

    def evaluate(self, observation:np.ndarray) -> np.ndarray:
        """Deterministic (mean) physical action for eval/deploy."""
        state = T.tensor(observation, dtype=T.float32).to(self._device)
        with T.no_grad():
            mean, _ = self.actor(state)
            envAct  = self._scale(mean.clamp(-1.0, 1.0))
        return envAct.cpu().numpy()

    def remember(self, obs:np.ndarray, action:np.ndarray, logProb:np.ndarray,
                 reward:np.ndarray, done:np.ndarray, value:np.ndarray) -> None:
        """Store one time slice. action/logProb are what act() returned."""
        self.buffer.add(obs, action, logProb, reward, done, value)

    def learn(self, lastObs:np.ndarray) -> dict:
        #bootstrap value for the step after the rollout, finish GAE
        state = T.tensor(lastObs, dtype=T.float32).to(self._device)
        with T.no_grad():
            rawLast   = self.critic(state).squeeze(-1)
            lastValue = (rawLast * self._retStd + self._retMean).cpu().numpy()
        self.buffer.computeGae(lastValue)

        #global advantage normalization
        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # update running return normalizer from this rollout's returns
        retAll = self.buffer.returns.flatten()
        self._retMean = float(retAll.mean())
        self._retStd  = max(float(retAll.std()), 1e-8)

        stats = {"policyLoss": 0.0, "valueLoss": 0.0,
                 "entropy": 0.0, "bcLoss": 0.0, "n": 0}
        for _ in range(self.epochs):
            seed = np.random.randint(0, 9999999)
            for obsB, actB, logpB, advB, retB in \
                    self.buffer.iterMinibatches(self.miniBatch, seed):
                obsB  = T.tensor(obsB).to(self._device)
                actB  = T.tensor(actB).to(self._device)
                logpB = T.tensor(logpB).to(self._device)
                advB  = T.tensor(advB).to(self._device)
                retB  = T.tensor(retB).to(self._device)

                dist    = self.actor.distribution(obsB)
                newLogp = dist.log_prob(actB).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = T.exp(newLogp - logpB)
                surr1 = ratio * advB
                surr2 = T.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advB
                policyLoss = -T.min(surr1, surr2).mean()

                retB_n    = (retB - self._retMean) / self._retStd
                value     = self.critic(obsB).squeeze(-1)
                valueLoss = nn.functional.mse_loss(value, retB_n)

                loss = (policyLoss + self.vfCoef * valueLoss
                        - self.entCoef * entropy)

                bcLoss = T.tensor(0.0, device=self._device)
                if self.bcCoef > 0.0 and self._bcNet is not None:
                    mean, _ = self.actor(obsB)
                    with T.no_grad():
                        bcTarget = self._bcNet(obsB)
                    bcLoss = nn.functional.mse_loss(mean, bcTarget)
                    loss = loss + self.bcCoef * bcLoss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()), self.maxGradNorm)
                self.opt.step()

                stats["policyLoss"] += policyLoss.item()
                stats["valueLoss"]  += valueLoss.item()
                stats["entropy"]    += entropy.item()
                stats["bcLoss"]     += bcLoss.item()
                stats["n"]          += 1

        self.buffer.reset()
        n = max(stats.pop("n"), 1)
        return {k: v / n for k, v in stats.items()}

    def preTrainCritic(self, vecEnv, nRollouts:int, epochs:int=3) -> None:
        """Collect nRollouts rollout blocks with the current actor (inference only),
        compute GAE returns, then fit only the critic via MSE on those returns.
        Warms up the value function before PPO to prevent critic divergence."""
        allObs, allRet = [], []
        for r in range(nRollouts):
            obs = vecEnv.obs()
            for _ in range(self.buffer.numSteps):
                envAct, normAct, logp, val = self.act(obs)
                rew, done, _ = vecEnv.step(envAct)
                self.buffer.add(obs, normAct, logp, rew, done, val)
                obs = vecEnv.obs()
            state = T.tensor(obs, dtype=T.float32).to(self._device)
            with T.no_grad():
                rawLast = self.critic(state).squeeze(-1)
                lastVal = (rawLast * self._retStd + self._retMean).cpu().numpy()
            self.buffer.computeGae(lastVal)
            allObs.append(self.buffer.obs.reshape(-1, self.obsDim).copy())
            allRet.append(self.buffer.returns.flatten().copy())
            self.buffer.reset()
            if (r + 1) % max(nRollouts // 5, 1) == 0:
                print(f"[criticPreTrain] collected {r+1}/{nRollouts} rollouts")

        obsAll = np.concatenate(allObs, axis=0).astype(np.float32)
        retAll = np.concatenate(allRet, axis=0).astype(np.float32)
        self._retMean = float(retAll.mean())
        self._retStd  = max(float(retAll.std()), 1e-8)
        retNorm = (retAll - self._retMean) / self._retStd
        print(f"[criticPreTrain] {len(obsAll)} transitions  "
              f"retMean={self._retMean:.1f}  retStd={self._retStd:.1f}")

        criticOpt = optim.Adam(self.critic.parameters(), lr=self.lr)
        n = len(obsAll)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            tot, nb = 0.0, 0
            for s in range(0, n, self.miniBatch):
                b = idx[s:s + self.miniBatch]
                obsB = T.tensor(obsAll[b]).to(self._device)
                retB = T.tensor(retNorm[b]).to(self._device)
                criticOpt.zero_grad()
                loss = nn.functional.mse_loss(self.critic(obsB).squeeze(-1), retB)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.maxGradNorm)
                criticOpt.step()
                tot += loss.item(); nb += 1
            print(f"[criticPreTrain] epoch {ep}  valueLoss {tot/max(nb,1):.4f}")

    def setBcCoef(self, coef:float) -> None:
        """Trainer decays this toward 0 over training."""
        self.bcCoef = coef

    def setBcReference(self, ffPath:str) -> None:
        """Load a frozen BC'd FF as the auxiliary-loss target policy."""
        net = ff(self.obsDim, self.actDim, self._bcHidden).to(self._device)
        net.load_state_dict(T.load(ffPath, map_location=self._device))
        for p in net.parameters():
            p.requires_grad = False
        net.eval()
        self._bcNet = net

    def warmStartActor(self, ffPath:str) -> None:
        """Init the actor mean trunk from BC'd FF weights (layer1/2/3 match);
        logStd and the critic are left fresh."""
        stateDict = T.load(ffPath, map_location=self._device)
        self.actor.load_state_dict(stateDict, strict=False)

    def save(self, path:str) -> None:
        T.save({"actor":   self.actor.state_dict(),
                "critic":  self.critic.state_dict(),
                "retMean": self._retMean,
                "retStd":  self._retStd}, path)

    def load(self, path:str) -> None:
        ckpt = T.load(path, map_location=self._device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        # older checkpoints predate the return normalizer
        self._retMean = ckpt.get("retMean", 0.0)
        self._retStd  = ckpt.get("retStd", 1.0)
