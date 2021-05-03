from abc import ABCMeta, abstractmethod
from typing import Tuple
import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from conv_utils import toeplitz_mult_ch_with_stride

class BaseTransform(nn.Module, metaclass=ABCMeta):
    """
    Class for base transformation
    """
    backsub_start_depth = 10
    def __init__(self, input_layer:nn.Module, depth:int, input_shape:Tuple):
        super().__init__()
        self.depth = depth
        self.weight = None
        self.bias = None
        self.ub = None
        self.lb = None
        self.output_shape = None
        self._init(input_layer, input_shape)

    def forward(self, previous_transformer:nn.Module, x_input:torch.Tensor = None, eps:float = None) -> nn.Module:
        if self.depth > 0:
            self._forward(previous_transformer)
        else:
            self._input_forward(x_input, eps)
        return self
    
    @abstractmethod
    def _init(self, input_layer, input_shape):
        """to be implemented in the child class"""
        raise NotImplementedError
    
    @abstractmethod
    def _forward(self, previous_transformer):
        """to be implemented in the child class"""
        raise NotImplementedError

    @abstractmethod
    def _input_forward(self, x_input, eps):
        """to be implemented only in the input forward transformer"""
        raise NotImplementedError

class InputTransform(BaseTransform):

    def _init(self, input_layer, input_shape):
        self.output_shape = input_shape[1:]
    
    def _forward(self, previous_transformer):
        raise NotImplementedError

    def _input_forward(self, x_input, eps):
        # only one sample at a time
        x_input = x_input.squeeze(0)
        self.ub = torch.clamp(x_input + eps, max=1.0)
        self.lb = torch.clamp(x_input - eps, min=0.0)
        self.init_ub = torch.clamp(x_input + eps, max=1.0)
        self.init_lb = torch.clamp(x_input - eps, min=0.0)

class NormalizeTransform(BaseTransform):
    """
    Normalization layer transformation
    """
    def _init(self, input_layer, input_shape):
        self.weight = 1./input_layer.sigma
        self.bias = -1. * input_layer.mean/input_layer.sigma
        self.output_shape = input_shape
    
    def _forward(self, previous_transformer):
        self.previous_transformer = previous_transformer
        shape_matrix = torch.ones(previous_transformer.ub.shape)
        self.upper_weights = self.lower_weights = self.weight*shape_matrix
        self.upper_bias = self.lower_bias = self.bias*shape_matrix

        self.ub = previous_transformer.init_ub * self.upper_weights + self.upper_bias
        self.lb = previous_transformer.init_lb * self.lower_weights + self.lower_bias 

        self.init_ub = previous_transformer.init_ub
        self.init_lb = previous_transformer.init_lb
    
    def _input_forward(self, x_input, eps):
        raise NotImplementedError

class FlattenTransform(BaseTransform):

    def _init(self, input_layer, input_shape):
        self.output_shape = input_shape

    def _forward(self, previous_transformer):
        self.previous_transformer = previous_transformer
        if len(previous_transformer.upper_weights.shape) > 2:
            self.upper_weights = torch.diag(previous_transformer.upper_weights.flatten())
            self.lower_weights = torch.diag(previous_transformer.lower_weights.flatten())
            self.upper_bias = previous_transformer.upper_bias.flatten()
            self.lower_bias = previous_transformer.lower_bias.flatten()

            self.ub = previous_transformer.ub.flatten()
            self.lb = previous_transformer.lb.flatten()
            self.init_ub = previous_transformer.init_ub.flatten()
            self.init_lb = previous_transformer.init_lb.flatten()
        else:
            self.upper_weights = previous_transformer.upper_weights
            self.lower_weights = previous_transformer.lower_weights
            self.upper_bias = previous_transformer.upper_bias
            self.lower_bias = previous_transformer.lower_bias

            self.ub = previous_transformer.ub
            self.lb = previous_transformer.lb
            self.init_ub = previous_transformer.init_ub
            self.init_lb = previous_transformer.init_lb

    def _input_forward(self, x_input, eps):
        raise NotImplementedError

class AffineTransform(BaseTransform):
    """
    Affine Layer transformation
    """

    def _init(self, input_layer, input_shape):
        self.weight = input_layer.weight.data.T
        self.bias = input_layer.bias.data
    
    def _forward(self, previous_transformer):
        pos_w = self.weight >= 0.
        neg_w = self.weight < 0.

        if self.depth <= BaseTransform.backsub_start_depth:
            self.ub = previous_transformer.ub @ (
                        pos_w * self.weight) + previous_transformer.lb @ (neg_w * self.weight)
            self.lb = previous_transformer.lb @ (
                    pos_w * self.weight) + previous_transformer.ub @ (neg_w * self.weight)
            n = self.ub.shape[0]
            self.upper_weights = torch.eye(n, n)
            self.lower_weights = torch.eye(n, n)
            self.upper_bias = torch.zeros(n)
            self.lower_bias = torch.zeros(n)
            self.init_ub = self.ub
            self.init_lb = self.lb

        else:
            self.upper_weights = previous_transformer.upper_weights @ (pos_w * self.weight) + previous_transformer.lower_weights @ (neg_w * self.weight)
            self.upper_bias = previous_transformer.upper_bias @ (pos_w * self.weight) + previous_transformer.lower_bias @ (neg_w * self.weight) + self.bias
            self.lower_weights = previous_transformer.lower_weights @ (pos_w * self.weight) + previous_transformer.upper_weights @ (neg_w * self.weight)
            self.lower_bias = previous_transformer.lower_bias @ (pos_w * self.weight) + previous_transformer.upper_bias @ (neg_w * self.weight) + self.bias

            upos_w = self.upper_weights >= 0.
            uneg_w = self.upper_weights < 0.

            lpos_w = self.lower_weights >= 0.
            lneg_w = self.lower_weights < 0.
            # upper bound
            # print(previous_transformer.init_ub, upos_w, self.upper_weights, self.upper_bias)
            self.ub = previous_transformer.init_ub @ (upos_w * self.upper_weights) +\
                      previous_transformer.init_lb @ (uneg_w * self.upper_weights) + self.upper_bias
            # lower bound
            self.lb = previous_transformer.init_lb @ (lpos_w * self.lower_weights) +\
                      previous_transformer.init_ub @ (lneg_w * self.lower_weights) + self.lower_bias
            # bound for initial input/output remains the same
            self.init_ub = previous_transformer.init_ub
            self.init_lb = previous_transformer.init_lb

    def _input_forward(self, x_input, eps):
        raise NotImplementedError

class ReLUTransform(BaseTransform):
    """
    ReLU layer transformation
    """

    def _init(self, input_layer, input_shape):
        self.output_shape = input_shape
        
    def _forward(self, previous_transformer):

        self.previous_transformer = previous_transformer
        self.upper_lambdas = torch.zeros(previous_transformer.upper_weights.shape[1])
        self.lower_lambdas = torch.zeros(previous_transformer.upper_weights.shape[1])
        self.upper_intercept = torch.zeros(previous_transformer.upper_weights.shape[1])
        self.lower_intercept = torch.zeros(previous_transformer.upper_weights.shape[1])
        # three cases for relu transform
        # if ub < 0
        caseI  = previous_transformer.ub <= 0.
        self.upper_lambdas[caseI] = 0.
        self.lower_lambdas[caseI] = 0.
        
        # if lb > 0
        caseII = previous_transformer.lb >= 0.
        self.upper_lambdas[caseII] = 1.
        self.lower_lambdas[caseII] = 1.

        # if crossing
        caseIII = ~(caseI + caseII)
        self.upper_lambdas[caseIII] = previous_transformer.ub[caseIII]/(previous_transformer.ub - previous_transformer.lb)[caseIII]
        self.upper_intercept[caseIII] = -(previous_transformer.ub * previous_transformer.lb)[caseIII]/(previous_transformer.ub - previous_transformer.lb)[caseIII]
        
        # minimum area for lambda 0 vs 1
        # lambda_1 = previous_transformer.ub[caseIII] > -previous_transformer.lb[caseIII]
        # self.lower_lambdas[caseIII][lambda_1] = 1.0
        if torch.any(caseIII):
            crossing_lambdas = self.get_crossing_lambdas(previous_transformer)
            self.lower_lambdas += crossing_lambdas * caseIII

        if self.depth <= BaseTransform.backsub_start_depth:

            self.ub = torch.zeros_like(previous_transformer.ub)
            self.lb = torch.zeros_like(previous_transformer.lb)
            self.ub[caseI] = 0.
            self.lb[caseI] = 0.
            self.ub[caseII] = previous_transformer.ub[caseII]
            self.lb[caseII] = 0
            self.ub[caseIII] = previous_transformer.ub[caseIII]
            self.lb[caseIII] = -self.lower_lambdas[caseIII] * previous_transformer.lb[caseIII]
            n = self.ub.shape[0]
            self.upper_weights = torch.eye(n, n)
            self.lower_weights = torch.eye(n, n)
            self.upper_bias = torch.zeros(n)
            self.lower_bias = torch.zeros(n)
            self.init_ub = self.ub
            self.init_lb = self.lb

        else: #backsub
            # multiply with previous weights
            self.upper_weights = previous_transformer.upper_weights @ torch.diag(self.upper_lambdas)
            self.lower_weights = previous_transformer.lower_weights @ torch.diag(self.lower_lambdas)
            self.upper_bias = previous_transformer.upper_bias @ torch.diag(self.upper_lambdas) + self.upper_intercept
            self.lower_bias = previous_transformer.lower_bias @ torch.diag(self.lower_lambdas) + self.lower_intercept

            # ub and lb
            upos_w = self.upper_weights >= 0.
            uneg_w = self.upper_weights < 0.

            lpos_w = self.lower_weights >= 0.
            lneg_w = self.lower_weights < 0.

            # upper bound
            self.ub = previous_transformer.init_ub @ (upos_w * self.upper_weights) +\
                      previous_transformer.init_lb @ (uneg_w * self.upper_weights) + self.upper_bias
            # lower bound
            self.lb = previous_transformer.init_lb @ (lpos_w * self.lower_weights) +\
                      previous_transformer.init_ub @ (lneg_w * self.lower_weights) + self.lower_bias

            # bound for initial input/output remains the same
            self.init_ub = previous_transformer.init_ub
            self.init_lb = previous_transformer.init_lb

    def get_crossing_lambdas(self, previous_transformer):
        """
        lower lambdas in case of ub and lb are crossing.
        These will be optimized at each step
        """
        if not hasattr(self, 'opt_params'):
            self.min_area(previous_transformer)
        self.opt_params.data.clamp_(min = 0., max = 1.)
        return self.opt_params

    def min_area(self, previous_transformer):
        """
        Minimum area in case of crossing
        """
        self.opt_params = Variable((previous_transformer.ub > -previous_transformer.lb).float(), requires_grad=True)
        self.opt_params.retain_grad()

    def _input_forward(self, x_input, eps):
        raise NotImplementedError

class ConvTransform(AffineTransform):
    """
    Conv layer transformation
    """

    def _init(self, input_layer, input_shape):

        r = input_layer.in_channels  # channel dim
        m = input_shape[-2]  # first input dim
        n = input_shape[-1]   # second input dim
        t = input_layer.out_channels  # number of filters
        s = input_layer.stride[0]
        self.output_shape = (t, m // s, n // s)
        self.weight = toeplitz_mult_ch_with_stride(input_layer.weight.data, (r, m, n), s).T
        self.bias = torch.repeat_interleave(input_layer.bias.data, self.output_shape[-1] * self.output_shape[-2])
        self.output_shape = (t, m // s, n // s)

def modify_layer(layer: nn.Module, depth: int, input_shape) -> nn.Module:
    # TODO: Implement Conv Transform (maybe convolution as matrix multiplication might help)
    layer_name = layer.__class__.__name__
    modified_layers = {"Linear": AffineTransform, "Normalization": NormalizeTransform,
                       "ReLU": ReLUTransform, "Flatten": FlattenTransform, "Conv2d": ConvTransform}

    if layer_name not in modified_layers:
        print(layer_name)
        return copy.deepcopy(layer)

    return modified_layers[layer_name](layer, depth, input_shape)