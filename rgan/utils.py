import torch
import math

import torch.nn as nn
from torch.utils.data import Dataset
from torch.distributions import normal

from scipy.stats import beta

import numpy as np


class SamplesDataset(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        res = self.x[idx]
        return res


class MechanisticModel(nn.Module):
    def __init__(self, mm, x_n, y_n, use_norm=True):
        super().__init__()
        self.MM = mm
        self.X_N = x_n
        self.Y_N = y_n
        self.use_norm = use_norm

    def __call__(self, x):
        if self.use_norm:
            return self.forward_n(x)
        else:
            return self.forward(x)

    def forward(self, x):
        return self.MM(x)
    
    def forward_n(self, x_n):
        x = self.X_N.inverse(x_n)
        y = self.MM(x)
        y_n = self.Y_N(y)
        return y_n
    
    def scale_pars(self, x_norm):
        return self.X_N.inverse(x_norm)


class MechanisticModelStochastic(nn.Module):
    def __init__(self, mm, x_n, y_n, noise_N, x_dim, use_norm=True):
        super().__init__()
        self.MM = mm
        self.X_N = x_n
        self.Y_N = y_n
        self.noise_N = noise_N
        self.x_dim = x_dim
        self.use_norm = use_norm

    def __call__(self, x):
        if self.use_norm:
            return self.forward_n(x)
        else:
            return self.forward(x)

    def forward(self, x):
        return self.MM(x)
    
    def forward_n(self, x_n):
        x = self.X_N.inverse(x_n[:, :self.x_dim])
        noise = self.noise_N.inverse(x_n[:, self.x_dim:])
        y = self.MM(torch.concat([x, noise], dim=1))
        y_n = self.Y_N(y)
        return y_n
    
    def scale_pars(self, x_norm):
        return self.X_N.inverse(x_norm)
    

class TorchStandardScaler(nn.Module):
    def __init__(self, mean=0., std=1., dim=1):
        super().__init__()
        self.mean_ = nn.Parameter(torch.ones(dim)*mean, requires_grad=False)
        self.std_ = nn.Parameter(torch.ones(dim)*std, requires_grad=False)

    def __call__(self, x):
        x = self.transform(x)
        return x

    def fit(self, x):
        self.mean_ = nn.Parameter(x.mean(0, keepdim=True), requires_grad=False)
        self.std_ = nn.Parameter(x.std(0, unbiased=False, keepdim=True), requires_grad=False)

    def transform(self, x):
        # print(f"Device mean_: {self.mean_.device}"
        # print(f"Device std_: {self.std_.device}")
        x = x - self.mean_
        x = x / (self.std_)
        return x
    
    def inverse(self, x_n):
        x = x_n * self.std_
        x = x + self.mean_
        return x


class NormToBoundsScaler(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = nn.Parameter(torch.from_numpy(low), requires_grad=False)
        self.high = nn.Parameter(torch.from_numpy(high), requires_grad=False)
        self.delta = nn.Parameter(self.high-self.low, requires_grad=False)
        self.loc = nn.Parameter(torch.zeros(low.shape[0],), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(low.shape[0],), requires_grad=False)
        self.dist = normal.Normal(self.loc, self.scale)
        
    # Takes input as normal with loc 0.0, scale 1.0, converts uniform within par ranges
    def inverse(self, x_norm):
        x_cdf = self.dist.cdf(x_norm)
        return x_cdf * self.delta + self.low

    def __call__(self, x_norm):
        return self.inverse(x_norm=x_norm)


class NormToTransformedBetaScaler(nn.Module):
    def __init__(self, a, b, low, high):
        super().__init__()
        self.a = a
        self.b = b
        self.low = nn.Parameter(torch.from_numpy(low), requires_grad=False)
        self.high = nn.Parameter(torch.from_numpy(high), requires_grad=False)
        self.delta = nn.Parameter(self.high-self.low, requires_grad=False)
        self.loc = nn.Parameter(torch.zeros(low.shape[0],), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(low.shape[0],), requires_grad=False)
        self.dist = normal.Normal(self.loc, self.scale)
        self.beta = beta(a, b)
        self.beta_icdf = IcdfBeta
        
    # Takes input as normal with loc 0.0, scale 1.0, converts to beta within par ranges
    def inverse(self, x_norm):
        x_cdf = self.dist.cdf(x_norm)
        x_beta = self.beta_icdf.apply(x_cdf, 
                                      torch.from_numpy(self.a).type_as(x_cdf), 
                                      torch.from_numpy(self.b).type_as(x_cdf))
        return self.low + (self.delta * x_beta)

    def __call__(self, x_norm):
        return self.inverse(x_norm=x_norm)


# Truncated normal distribution code adapted from:
# https://stackoverflow.com/questions/60233216/how-to-make-a-truncated-normal-distribution-in-pytorch
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> torch.Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# Gradients for inverse cdf of beta distribution using scipy
# Implementation adapted from:
# https://gist.github.com/andiac/dd73b7916f95eba0b20c3a00e405fd55
cpu_det_np = lambda x: x.detach().to('cpu').numpy()

class IcdfBeta(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        return torch.tensor(
                beta.ppf(cpu_det_np(x), cpu_det_np(a), cpu_det_np(b))
            ).type_as(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        Xs, As, Bs = ctx.saved_tensors

        h = 1e-5

        a = cpu_det_np(As)
        b = cpu_det_np(Bs)
        x = cpu_det_np(Xs)
        go = cpu_det_np(grad_outputs)

        small_x = (x < h)
        big_x = (1-x < h)

        small_grad = go * ((beta.ppf(x+h, a, b) - beta.ppf(x, a, b)) / h)
        big_grad = go * ((beta.ppf(x, a, b) - beta.ppf(x-h, a, b)) / h)
        middle_grad = go * ((beta.ppf(x+h, a, b) - beta.ppf(x-h, a, b)) / (2.0 * h))

        grad_xs = np.where(small_x, small_grad, np.where(big_x, big_grad, middle_grad))

        grad_xs = torch.from_numpy(grad_xs).type_as(grad_outputs)

        return grad_xs, None, None
