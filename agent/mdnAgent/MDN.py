import torch as T
import torch.nn as nn

class MDN(nn.Module):
    """Generic Mixture Density Network.

    Outputs K mixture components, each with:
      - mean     (outDim)
      - log_var  (outDim)  — exponentiated at inference for numerical stability
      - logit    (1)       — softmaxed to get mixing weight pi

    Total output size: K * (2 * outDim + 1)
    """
    def __init__(self, inDim:int, outDim:int, K:int,
                 hiddenDim:int=128) -> None:
        super(MDN, self).__init__()
        self.outDim = outDim
        self.K = K

        self.shared = nn.Sequential(
            nn.Linear(inDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        # per-component heads
        self.meanHead   = nn.Linear(hiddenDim, K * outDim)
        self.logVarHead = nn.Linear(hiddenDim, K * outDim)
        self.logitHead  = nn.Linear(hiddenDim, K)

    def forward(self, x:T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """Returns (means, log_vars, logits) each with a K dimension.

        means:    (..., K, outDim)
        log_vars: (..., K, outDim)
        logits:   (..., K)
        """
        h = self.shared(x)
        means   = self.meanHead(h).unflatten(-1, (self.K, self.outDim))
        logVars = self.logVarHead(h).unflatten(-1, (self.K, self.outDim))
        logits  = self.logitHead(h)
        return means, logVars, logits

    @staticmethod
    def nllLoss(target:T.Tensor, means:T.Tensor, logVars:T.Tensor,
                logits:T.Tensor) -> T.Tensor:
        """Negative log-likelihood of target under the mixture.

        target:   (..., outDim)
        means:    (..., K, outDim)
        logVars:  (..., K, outDim)
        logits:   (..., K)
        """
        target = target.unsqueeze(-2)  # (..., 1, outDim)
        vars = T.exp(logVars)          # (..., K, outDim)

        # log N(target | mean, var) per component, summed over outDim
        logProb = -0.5 * (logVars + (target - means)**2 / vars)
        logProb = logProb.sum(dim=-1)  # (..., K)

        # log-sum-exp over mixture components weighted by pi
        logPi = T.log_softmax(logits, dim=-1)
        logMixture = T.logsumexp(logPi + logProb, dim=-1)  # (...)

        return -logMixture.mean()
