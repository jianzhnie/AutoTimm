import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import torch
import warnings
from torch import nn
import torch.nn.functional as F

try:
    from pytorch_quantization import nn as quant_nn
except ImportError as e:
    warnings.warn(
        "pytorch_quantization module not found, quantization will not be available"
    )
    quant_nn = None



# SqueezeAndExcitation {{{
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, squeeze, activation):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(in_channels, squeeze)
        self.expand = nn.Linear(squeeze, in_channels)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self._attention(x)

    def _attention(self, x):
        out = torch.mean(x, [2, 3])
        out = self.squeeze(out)
        out = self.activation(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return out


class SqueezeAndExcitationTRT(nn.Module):
    def __init__(self, in_channels, squeeze, activation):
        super(SqueezeAndExcitationTRT, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(in_channels, squeeze, 1)
        self.expand = nn.Conv2d(squeeze, in_channels, 1)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self._attention(x)

    def _attention(self, x):
        out = self.pooling(x)
        out = self.squeeze(out)
        out = self.activation(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        return out


# }}}

# EMA {{{
class EMA:
    def __init__(self, mu, module_ema):
        self.mu = mu
        self.module_ema = module_ema

    def __call__(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            mu = min(self.mu, (1.0 + step) / (10 + step))

        def strip_module(s: str) -> str:
            return s

        mesd = self.module_ema.state_dict()
        with torch.no_grad():
            for name, x in module.state_dict().items():
                if name.endswith("num_batches_tracked"):
                    continue
                n = strip_module(name)
                mesd[n].mul_(mu)
                mesd[n].add_((1.0 - mu) * x)


# }}}

# ONNXSiLU {{{
# Since torch.nn.SiLU is not supported in ONNX,
# it is required to use this implementation in exported model (15-20% more GPU memory is needed)
class ONNXSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ONNXSiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# }}}


class SequentialSqueezeAndExcitation(SqueezeAndExcitation):
    def __init__(self, in_channels, squeeze, activation, quantized=False):
        super().__init__(in_channels, squeeze, activation)
        self.quantized = quantized
        if quantized:
            assert quant_nn is not None, "pytorch_quantization is not available"
            self.mul_a_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
            self.mul_b_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
        else:
            self.mul_a_quantizer = nn.Identity()
            self.mul_b_quantizer = nn.Identity()

    def forward(self, x):
        out = self._attention(x)
        if not self.quantized:
            return out * x
        else:
            x_quant = self.mul_a_quantizer(out)
            return x_quant * self.mul_b_quantizer(x)


class SequentialSqueezeAndExcitationTRT(SqueezeAndExcitationTRT):
    def __init__(self, in_channels, squeeze, activation, quantized=False):
        super().__init__(in_channels, squeeze, activation)
        self.quantized = quantized
        if quantized:
            assert quant_nn is not None, "pytorch_quantization is not available"
            self.mul_a_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
            self.mul_b_quantizer = quant_nn.TensorQuantizer(
                quant_nn.QuantConv2d.default_quant_desc_input
            )
        else:
            self.mul_a_quantizer = nn.Identity()
            self.mul_b_quantizer = nn.Identity()

    def forward(self, x):
        out = self._attention(x)
        if not self.quantized:
            return out * x
        else:
            x_quant = self.mul_a_quantizer(out)
            return x_quant * self.mul_b_quantizer(x)


class StochasticDepthResidual(nn.Module):
    def __init__(self, survival_prob: float):
        super().__init__()
        self.survival_prob = survival_prob
        self.register_buffer("mask", torch.ones(()), persistent=False)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return torch.add(residual, other=x)
        else:
            with torch.no_grad():
                F.dropout(
                    self.mask,
                    p=1 - self.survival_prob,
                    training=self.training,
                    inplace=False,
                )
            return torch.addcmul(residual, self.mask, x)

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1).squeeze(-1)
