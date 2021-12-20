import torch
from torch import nn
from typing import List
from torchinfo import summary
import numpy as np
import os, sys

def prod(tuple):
    prod = 1
    for t in tuple:
        prod *= t
    return prod

def l2_size(n, activation_scale):
    # Gives l2 norm of n-sized tensor with elements of value activation_scale
    return np.sqrt(n*activation_scale**2)

def l2_clip(t, C):
    dims = tuple(range(1, len(t.shape)))
    norm = t.norm(2, dim=dims, keepdim=True).expand(t.shape)
    clipped = torch.where(norm > C, C*(t/norm), t)
    return clipped

def l2_to_l1(l2, n):
    return np.sqrt(n) * l2

def module_requires_grad(module):
    for p in module.parameters():
        if p.requires_grad:
            return True

    return False

def get_layers(module: torch.nn.Module):
    children = list(module.children())
    if len(children) < 1:
        return [module]
    else:
        layers = []
        for c in children:
            layers += get_layers(c)
        return layers


class DummyLayer(nn.Module):
    def forward(self, x):
        return x

class PGCWrapper(nn.Module):
    def __init__(self, pgc, original_module, auto_params=False):
        super().__init__()
        self.module = original_module

        self.pgc = pgc

        self.dummy = DummyLayer()
        self.dummy.register_full_backward_hook(self.backward_hook)

        # Get forward and backward clip params
        p = list(self.module.parameters())
        np = len(p)

        if auto_params:
            # Automatically make up some values based on size of layer weight, input, and output
            self.input_clip_param = l2_size(prod(self.module.in_shape), self.pgc.auto_activation_scale)
            self.pgc.input_clip_params.append(self.input_clip_param)

            if isinstance(self.module, nn.Linear):
                self.pgc.grad_l2_bounds.append(l2_size(p[0].numel(), self.pgc.auto_weight_grad_scale)) # weight
                self.back_clip_param = self.pgc.grad_l2_bounds[self.pgc.parameter_ind] / self.input_clip_param
                self.pgc.back_clip_params.append(self.back_clip_param)
                if np > 1:
                    self.pgc.grad_l2_bounds.append(self.back_clip_param) # bias
            elif isinstance(self.module, nn.Conv2d):
                # for now just do the same for conv weight
                self.pgc.grad_l2_bounds.append(l2_size(p[0].numel(), self.pgc.auto_weight_grad_scale)) # weight
                self.back_clip_param = l2_to_l1(self.pgc.grad_l2_bounds[self.pgc.parameter_ind], prod(self.module.out_shape[1:])) / self.input_clip_param
                self.pgc.back_clip_params.append(self.back_clip_param)
                if np > 1:
                    self.pgc.grad_l2_bounds.append(self.back_clip_param * prod(self.module.out_shape[1:])) # bias (guess)
        else:
            self.input_clip_param = self.pgc.input_clip_params[self.pgc.layer_ind]
            self.back_clip_param = self.pgc.back_clip_params[self.pgc.layer_ind]

            # Calculate max gradient l2 norm for each parameter and save
            if isinstance(self.module, nn.Linear):
                pgc.grad_l2_bounds.append(self.input_clip_param * self.back_clip_param) # weight
                if np > 1:
                    pgc.grad_l2_bounds.append(self.back_clip_param) # bias
            elif isinstance(self.module, nn.Conv2d):
                self.pgc.grad_l2_bounds.append(self.input_clip_param * l2_to_l1(self.back_clip_param, prod(self.module.out_shape[1:]))) # weight
                if np > 1:
                    self.pgc.grad_l2_bounds.append(self.back_clip_param * prod(self.module.out_shape[1:])) # bias

        self.pgc.layer_ind += 1
        self.pgc.parameter_ind += np

    def backward_hook(self, module, grad_input, grad_output):
        if self.pgc.hooks_enabled:
            return [(None if gi is None else l2_clip(gi, self.back_clip_param)) for gi in grad_input]

    def forward(self, x):
        return self.dummy(self.module(l2_clip(x, self.input_clip_param)))

class PropogatingGradClipper:
    def __init__(self, model, back_clip_params=None, input_clip_params=None, auto_activation_scale=0.5, auto_weight_grad_scale=1e-4, device="cpu"):
        # If back_clip_params or input_clip_params is None, will automatically determine them based on layer size and auto params
        self.hooks_enabled = True

        self.parameter_ind = 0
        self.back_clip_params = [] if back_clip_params is None else back_clip_params
        self.layer_ind = 0
        self.input_clip_params = [] if input_clip_params is None else input_clip_params

        self.device = device

        self.auto_activation_scale = auto_activation_scale
        self.auto_weight_grad_scale = auto_weight_grad_scale

        self.grad_l2_bounds = []

        sys.stdout = open(os.devnull, 'w')
        s = summary(model, input_size=(1,1,28,28))
        sys.stdout = sys.__stdout__
        for layer_info in s.summary_list:
            layer_info.module.in_shape = layer_info.input_size[1:]
            layer_info.module.out_shape = layer_info.output_size[1:]

        self.convert(model, auto_params=(back_clip_params is None or input_clip_params is None))

        print("L2 Bounds:",self.grad_l2_bounds)
        print("Backprop Clipping Params:",self.back_clip_params)
        print("Forward Clipping Params:",self.input_clip_params)

    def store_shapes(self, module, inp):
        o = inp
        for layer in get_layers(module):
            layer.in_shape = o.shape[1:]
            o = layer(o)
            layer.out_shape = o.shape[1:]

    def enable_hooks(self):
        self.hooks_enabled = True

    def disable_hooks(self):
        self.hooks_enabled = False

    def convert(self, module, inp_shape=(1,1,28,28), auto_params=False):
        # self.store_shapes(module, torch.zeros(inp_shape, device=self.device))

        for name, m in module.named_modules():
            if module_requires_grad(m) and len(list(m.children())) < 1:
                parent = module
                names = name.split(".")
                for name in names[:-1]:
                    parent = parent._modules[name]

                parent._modules[names[-1]] = PGCWrapper(self, m, auto_params=auto_params).to(self.device)
