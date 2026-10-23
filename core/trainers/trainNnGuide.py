"""trainNnGuide.py — BC warm-start + PPO fine-tune of the pyfads missile
guidance policy, then export to the pure-numpy format ofs/guidance/nnGuide.py
expects.

Pipeline:
  0. (prereq) run pyfads offline/gym/collectDemos.py to produce
     offline/data/{demoDataset.npz, normStats.npz}
  1. BC: train an FF (60->4) to imitate ccKill's normalized action, save bc.pth
  2. PPO: vectorized MissileEnv rollouts, actor warm-started from BC + a decayed
     BC-auxiliary loss, clipped-surrogate updates
  3. export actor weights -> nnGuideWeights.npz (l1/l2/l3), reusing normStats.npz

Run from anywhere; paths are resolved to the two repos below.
"""
import sys
import os
import argparse
import numpy as np
import torch as T

os.environ.setdefault("MPLBACKEND", "Agg")

PYFADS  = "/Users/apoorvachaubal/Documents/source/pyfads"
GYMRATS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, GYMRATS)
sys.path.insert(0, PYFADS)

from agent.ppoAgent.Agent import Agent   as PpoAgent
from agent.ffAgent.Agent  import Agent   as FfAgent
from offline.gym.missileEnv import MissileEnv

DATA_DIR = os.path.join(PYFADS, "offline", "data")
NET_DIR  = os.path.join(GYMRATS, "networks")


# ----------------------------------------------------------------------
# data + normalization
# ----------------------------------------------------------------------
def loadData(datasetPath=None):
    dsPath = datasetPath or os.path.join(DATA_DIR, "demoDataset.npz")
    demo  = np.load(dsPath)
    # normStats always come from the original demo distribution (train/deploy parity)
    stats = np.load(os.path.join(DATA_DIR, "normStats.npz"))
    mean60 = np.tile(stats["obsMean"].astype(np.float32), 4)
    std60  = np.tile(np.clip(stats["obsStd"].astype(np.float32), 1e-6, None), 4)
    gLimit = float(stats["gLimit"])
    print(f"[data] {len(demo['obs_raw'])} transitions from {dsPath}")
    return demo, mean60, std60, gLimit


def exportMlp(stateDict, path):
    """Map an FF/Actor state_dict (layer1/2/3) to the l1/l2/l3 npz nnGuide wants."""
    def npy(k):
        return stateDict[k].detach().cpu().numpy()
    np.savez(path,
             l1_weight=npy("layer1.weight"), l1_bias=npy("layer1.bias"),
             l2_weight=npy("layer2.weight"), l2_bias=npy("layer2.bias"),
             l3_weight=npy("layer3.weight"), l3_bias=npy("layer3.bias"))
    print(f"  exported MLP weights -> {path}")


# ----------------------------------------------------------------------
# behavior cloning
# ----------------------------------------------------------------------
def trainBc(demo, mean60, std60, hidden, lr, epochs, batch, bcPath):
    X = ((demo["obs_raw"].astype(np.float32) - mean60) / std60).astype(np.float32)
    Y = demo["action_norm"].astype(np.float32)   # BC targets the normalized action
    n = len(X)
    print(f"[BC] {n} transitions, {epochs} epochs (action-magnitude weighted loss)")
    ff = FfAgent(60, 4, hidden, lr)
    dev = ff._device
    for ep in range(epochs):
        idx = np.random.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            Tx = T.tensor(X[b], dtype=T.float32).to(dev)
            Ty = T.tensor(Y[b], dtype=T.float32).to(dev)
            ff.optimizer.zero_grad()
            pred = ff.net(Tx)
            # weight each sample by mean(|action_norm|) — upweights rare
            # high-accel CLIMB steps that are ~141x underrepresented vs cruise
            w = Ty.abs().mean(dim=1, keepdim=True) + 1e-4  # (B, 1)
            loss = (w * (pred - Ty).pow(2)).mean()
            loss.backward()
            ff.optimizer.step()
            tot += loss.item(); nb += 1
        if ep % max(epochs // 10, 1) == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  wt-mse {tot / max(nb,1):.5f}")
    ff.save(bcPath)
    print(f"[BC] saved -> {bcPath}")
    return ff


# ----------------------------------------------------------------------
# vectorized env helpers (hand-rolled; version-proof)
# ----------------------------------------------------------------------
class VecEnv:
    """Minimal synchronous vector env with manual autoreset + step-limit
    truncation. Returns obs already normalized by MissileEnv's ObsBuilder."""
    def __init__(self, numEnvs, normStatsPath, controlInterval, maxEpSteps, seed):
        self.envs = [MissileEnv(normStatsPath=normStatsPath,
                                controlInterval=controlInterval)
                     for _ in range(numEnvs)]
        self.n = numEnvs
        self.maxEpSteps = maxEpSteps
        self._epStep = np.zeros(numEnvs, dtype=np.int64)
        self._obs = np.stack([e.reset(seed=seed + i)[0]
                              for i, e in enumerate(self.envs)])

    def obs(self):
        return self._obs

    def step(self, actions):
        n = self.n
        nextObs = np.zeros_like(self._obs)
        rewards = np.zeros(n, dtype=np.float32)
        dones   = np.zeros(n, dtype=np.float32)
        finished = []   # episode-end info dicts for logging
        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step(actions[i])
            self._epStep[i] += 1
            done = bool(term or trunc or self._epStep[i] >= self.maxEpSteps)
            rewards[i] = r
            dones[i]   = 1.0 if done else 0.0
            if done:
                finished.append(info)
                o, _ = e.reset()
                self._epStep[i] = 0
            nextObs[i] = o
        self._obs = nextObs
        return rewards, dones, finished


# ----------------------------------------------------------------------
# evaluation (deterministic policy, fresh envs)
# ----------------------------------------------------------------------
def evaluate(agent, normStatsPath, controlInterval, maxEpSteps, nEps, seed):
    misses, speeds, detects, fovLosts = [], [], [], []
    for k in range(nEps):
        env = MissileEnv(normStatsPath=normStatsPath, controlInterval=controlInterval)
        obs, _ = env.reset(seed=seed + 1000 + k)
        for _ in range(maxEpSteps):
            act = agent.evaluate(obs[None, :])[0]
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        misses.append(info["missDist"]); speeds.append(info["impactSpeed"])
        detects.append(1.0 if info["detected"] else 0.0)
        fovLosts.append(1.0 if info["fovLost"] else 0.0)
    print(f"[eval] miss {np.mean(misses):8.1f} m   "
          f"impactSpd {np.mean(speeds):7.1f} m/s   "
          f"detect {np.mean(detects):.2f}   fovLost {np.mean(fovLosts):.2f}")
    return np.mean(misses), np.mean(speeds), np.mean(detects), np.mean(fovLosts)


# ----------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------
def trainPpo(args, bcPath, gLimit):
    statsPath = os.path.join(DATA_DIR, "normStats.npz")
    actLow  = np.array([-gLimit, -gLimit, -gLimit, -np.pi], dtype=np.float32)
    actHigh = np.array([ gLimit,  gLimit,  gLimit,  np.pi], dtype=np.float32)

    vec = VecEnv(args.numEnvs, statsPath, args.controlInterval,
                 args.maxEpSteps, seed=args.seed)

    agent = PpoAgent(60, 4, actLow, actHigh, args.numSteps, args.numEnvs,
                     LR=args.lr, GAMMA=args.gamma, LAM=args.lam, CLIP=args.clip,
                     EPOCHS=args.epochs, MINIBATCH=args.miniBatch,
                     ENT_COEF=args.entCoef, VF_COEF=args.vfCoef,
                     ACTOR_HIDDEN=args.hidden, CRITIC_HIDDEN=2 * args.hidden)
    if bcPath is not None:
        agent.warmStartActor(bcPath)
        agent.setBcReference(bcPath)
        print("[PPO] actor warm-started from BC + BC-aux reference set")

    obs = vec.obs()
    for update in range(args.updates):
        # decay BC-aux coefficient linearly to 0 over the first bcDecay fraction
        frac   = update / max(args.updates * args.bcDecayFrac, 1)
        bcCoef = args.bcCoef0 * max(0.0, 1.0 - frac)
        agent.setBcCoef(bcCoef)

        epInfos = []
        for _ in range(args.numSteps):
            envAct, normAct, logp, val = agent.act(obs)
            rew, done, finished = vec.step(envAct)
            agent.remember(obs, normAct, logp, rew, done, val)
            obs = vec.obs()
            epInfos.extend(finished)

        stats = agent.learn(obs)

        if update % args.logEvery == 0 or update == args.updates - 1:
            if epInfos:
                m = np.mean([i["missDist"] for i in epInfos])
                s = np.mean([i["impactSpeed"] for i in epInfos])
                d = np.mean([1.0 if i["detected"] else 0.0 for i in epInfos])
                f = np.mean([1.0 if i["fovLost"] else 0.0 for i in epInfos])
                epStr = (f"eps {len(epInfos):3d}  miss {m:7.1f}  spd {s:6.1f}  "
                         f"det {d:.2f}  fovLost {f:.2f}")
            else:
                epStr = "no episode completed"
            print(f"[PPO] upd {update:4d}  bcCoef {bcCoef:.3f}  "
                  f"pLoss {stats['policyLoss']:+.4f}  vLoss {stats['valueLoss']:.3f}  "
                  f"ent {stats['entropy']:.3f}  bc {stats['bcLoss']:.4f}  | {epStr}")

        if args.evalEvery > 0 and (update + 1) % args.evalEvery == 0:
            evaluate(agent, statsPath, args.controlInterval,
                     args.maxEpSteps, args.evalEps, args.seed)

    os.makedirs(NET_DIR, exist_ok=True)
    ckpt = os.path.join(NET_DIR, "nnGuidePpo.pth")
    agent.save(ckpt)
    print(f"[PPO] saved checkpoint -> {ckpt}")
    exportMlp(agent.actor.state_dict(),
              os.path.join(NET_DIR, "nnGuideWeights.npz"))
    return agent


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skipBc", action="store_true", help="skip BC, cold PPO")
    p.add_argument("--bcOnly", action="store_true", help="BC then export, no PPO")
    p.add_argument("--dataset", type=str, default=None,
                   help="dataset .npz to train BC on (default demoDataset.npz; "
                        "point at a DAgger-aggregated set to iterate)")
    # BC
    p.add_argument("--bcEpochs", type=int, default=100)
    p.add_argument("--bcBatch",  type=int, default=256)
    p.add_argument("--bcLr",     type=float, default=1e-3)
    # PPO
    p.add_argument("--updates",   type=int, default=1000)
    p.add_argument("--numEnvs",   type=int, default=8)
    p.add_argument("--numSteps",  type=int, default=256)
    p.add_argument("--miniBatch", type=int, default=512)
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--lam",       type=float, default=0.95)
    p.add_argument("--clip",      type=float, default=0.2)
    p.add_argument("--entCoef",   type=float, default=0.01)
    p.add_argument("--vfCoef",    type=float, default=0.5)
    p.add_argument("--hidden",    type=int, default=128)
    p.add_argument("--bcCoef0",     type=float, default=1.0)
    p.add_argument("--bcDecayFrac", type=float, default=0.5)
    # env / loop
    p.add_argument("--controlInterval", type=int, default=5)
    p.add_argument("--maxEpSteps", type=int, default=4000)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--logEvery",   type=int, default=10)
    p.add_argument("--evalEvery",  type=int, default=100)
    p.add_argument("--evalEps",    type=int, default=20)
    args = p.parse_args()

    np.random.seed(args.seed); T.manual_seed(args.seed)
    os.makedirs(NET_DIR, exist_ok=True)

    demo, mean60, std60, gLimit = loadData(args.dataset)
    bcPath = os.path.join(NET_DIR, "nnGuideBc.pth")

    if not args.skipBc:
        trainBc(demo, mean60, std60, args.hidden, args.bcLr,
                args.bcEpochs, args.bcBatch, bcPath)
        exportMlp(T.load(bcPath, map_location="cpu"),
                  os.path.join(NET_DIR, "nnGuideBcWeights.npz"))
    else:
        bcPath = None

    if args.bcOnly:
        print("[done] BC only"); return

    trainPpo(args, bcPath, gLimit)


if __name__ == "__main__":
    main()
