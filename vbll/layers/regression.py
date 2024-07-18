import numpy as np
import torch
from dataclasses import dataclass
from ..utils.distributions import Normal, DenseNormal, get_parameterization
from collections.abc import Callable
import torch.nn as nn


def KL(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5*(mse_term + trace_term + logdet_term) # currently exclude constant

@dataclass
class VBLLReturn():
    predictive: Normal | DenseNormal # Could return distribution or mean/cov
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None


class Regression(nn.Module):
    """
    Variational Bayesian Linear Regression

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports 'dense' and 'diagonal'
    prior_scale : float
        Scale of prior covariance matrix
    wishart_scale : float
        Scale of Wishart prior on noise covariance
    dof : float
        Degrees of freedom of Wishart prior on noise covariance
    """
    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=1.,
                 wishart_scale=1e-2,
                 dof=1.):
        super(Regression, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.)/2.
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_scale = prior_scale

        # noise distribution
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad = False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) - 1)

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))
        self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
        if parameterization == 'dense':
          self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features)/in_features)

    def noise_chol(self):
      return torch.exp(self.noise_logdiag)

    def W_chol(self):
      out = torch.exp(self.W_logdiag)
      if self.W_dist == DenseNormal:
        out = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(out)

      return out

    def W(self):
      return self.W_dist(self.W_mean, self.W_chol())

    def noise(self):
      return Normal(self.noise_mean, self.noise_chol())

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out#self.predictive(x), self.predictive(x), self._get_val_loss_fn(x))

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[...,None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5*((W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])) * noise.trace_precision)

            kl_term = KL(W, self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            return -torch.mean(self.predictive(x).log_prob(y))

        return loss_fn