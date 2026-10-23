import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


def simNorm(z: T.Tensor, groupSize: int = 8) -> T.Tensor:
    """SimNorm (TD-MPC2): softmax over fixed-size groups of the latent.

    Projects z onto a product of simplices, which bounds it without a running
    normalizer. This is the structural guard against the all-zeros collapse the
    consistency loss would otherwise admit — a simplex cannot be zero.
    """
    shape = z.shape
    z = z.view(*shape[:-1], shape[-1] // groupSize, groupSize)
    z = F.softmax(z, dim=-1)
    return z.view(*shape)


def symlog(x: T.Tensor) -> T.Tensor:
    """sign(x) * log(1 + |x|). Compresses the terminal spike (~-19500 -> ~-9.9)
    without discarding sign or small-value resolution."""
    return T.sign(x) * T.log1p(T.abs(x))


def symexp(x: T.Tensor) -> T.Tensor:
    """Inverse of symlog."""
    return T.sign(x) * T.expm1(T.abs(x))


class TwoHot(nn.Module):
    """Two-hot encoding over a fixed bin grid in symlog space.

    Classification rather than regression. A regressor torn between 0 and -19500
    outputs -9750, a value that never occurs; a categorical head can hold the
    bimodality instead. Also gives the reward head a genuine distribution to fit,
    which matters here because the environment's reward carries real aleatoric noise
    (unseeded radar measurement noise; reward std ~3.1 on a repeated identical flight).
    """

    def __init__(self, low: float = -12.0, high: float = 12.0, nBins: int = 101):
        super().__init__()
        self.nBins = nBins
        self.register_buffer("bins", T.linspace(low, high, nBins))

    def encode(self, x: T.Tensor) -> T.Tensor:
        """Scalar targets (…,) -> two-hot rows (…, nBins), in symlog space."""
        x = symlog(x).clamp(self.bins[0], self.bins[-1]).unsqueeze(-1)
        # index of the bin immediately below x
        idx = T.clamp(T.searchsorted(self.bins.contiguous(), x.contiguous()) - 1,
                      0, self.nBins - 2)
        lo = self.bins[idx]
        hi = self.bins[idx + 1]
        w = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        out = T.zeros(*x.shape[:-1], self.nBins, device=x.device, dtype=x.dtype)
        out.scatter_(-1, idx, 1.0 - w)
        out.scatter_add_(-1, idx + 1, w)
        return out

    def decode(self, logits: T.Tensor) -> T.Tensor:
        """Logits (…, nBins) -> expected scalar value in real space."""
        p = F.softmax(logits, dim=-1)
        return symexp((p * self.bins).sum(-1))

    @staticmethod
    def loss(logits: T.Tensor, target2h: T.Tensor) -> T.Tensor:
        """Cross-entropy against a soft two-hot target."""
        return -(target2h * F.log_softmax(logits, dim=-1)).sum(-1)


def mlp(inDim: int, hiddenDim: int, outDim: int, nHidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = inDim
    for _ in range(nHidden):
        layers += [nn.Linear(d, hiddenDim), nn.LayerNorm(hiddenDim), nn.Mish()]
        d = hiddenDim
    layers += [nn.Linear(d, outDim)]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """obs -> z. The latent is SimNorm-projected, so it is bounded by construction."""

    def __init__(self, obsDim: int, latentDim: int = 48,
                 hiddenDim: int = 256, groupSize: int = 8):
        super().__init__()
        assert latentDim % groupSize == 0, "latentDim must divide into SimNorm groups"
        self.groupSize = groupSize
        self.net = mlp(obsDim, hiddenDim, latentDim)

    def forward(self, obs: T.Tensor) -> T.Tensor:
        return simNorm(self.net(obs), self.groupSize)


class Dynamics(nn.Module):
    """(z, a) -> z'. The macro-step jump map: one call advances K control steps."""

    def __init__(self, latentDim: int, actDim: int,
                 hiddenDim: int = 256, groupSize: int = 8):
        super().__init__()
        self.groupSize = groupSize
        self.net = mlp(latentDim + actDim, hiddenDim, latentDim)

    def forward(self, z: T.Tensor, a: T.Tensor) -> T.Tensor:
        return simNorm(self.net(T.cat([z, a], dim=-1)), self.groupSize)


class RewardHead(nn.Module):
    """(z, a) -> two-hot logits over block-summed reward."""

    def __init__(self, latentDim: int, actDim: int,
                 hiddenDim: int = 256, nBins: int = 101):
        super().__init__()
        self.net = mlp(latentDim + actDim, hiddenDim, nBins)

    def forward(self, z: T.Tensor, a: T.Tensor) -> T.Tensor:
        return self.net(T.cat([z, a], dim=-1))


class Decoder(nn.Module):
    """z -> physical state, in physical units.

    Replaces the learned reward head (2026-08-13). Reward is then computed by the same
    closed-form function the environment uses, so nothing learned encodes the objective and
    re-tuning it costs no retraining.

    Two jobs beyond reward:
      - it grounds the latent. Consistency loss alone admits a collapse-to-constant solution,
        and SimNorm's simplex structure makes its MSE ~0.0005 from step one whether the model
        is learning or not. A decoder cannot recover range/alt/closingVel from a constant z.
      - it is the only interpretable accuracy metric available -- error in METRES at each
        composition depth, which is what distinguishes "physics degrades under composition"
        from "physics fine, terminal mapping coarse".

    Channels are predicted in a whitened space (targets are standardized before the loss);
    `denorm` maps back to physical units.
    """

    def __init__(self, latentDim: int, nChannels: int, hiddenDim: int = 256):
        super().__init__()
        self.net = mlp(latentDim, hiddenDim, nChannels)

    def forward(self, z: T.Tensor) -> T.Tensor:
        return self.net(z)


class ContinueHead(nn.Module):
    """(z, a) -> logit for P(episode continues past this block).

    RETIRED 2026-08-13 but kept behind a flag. Termination is a known rule on decoded state
    (`alt < 0`, or `range < TERMINAL_RNG and closingVel <= 0`), softened analytically by
    `decodeTargets.pContinue` — no weights needed. This head was the one component measured
    to work (0.98-0.99), so it stays available until the analytic gate is shown to match it
    on the run-3 population.
    """

    def __init__(self, latentDim: int, actDim: int, hiddenDim: int = 256):
        super().__init__()
        self.net = mlp(latentDim + actDim, hiddenDim, 1)

    def forward(self, z: T.Tensor, a: T.Tensor) -> T.Tensor:
        return self.net(T.cat([z, a], dim=-1)).squeeze(-1)
