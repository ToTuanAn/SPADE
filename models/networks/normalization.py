"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input

def dynamic_attention(q, k, q_prune, k_prune, v, smooth=None, v2=None):
    # q, k, v: b, c, h, w
    b, c_qk, h_q, w_q = q.shape
    h_kv, w_kv = k.shape[2:]
    q = q.view(b, c_qk, h_q * w_q).transpose(-1, -2).contiguous()
    k = k.view(b, c_qk, h_kv * w_kv)
    v = v.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
    q_prune = q_prune.view(b, -1, h_q * w_q).transpose(-1, -2).contiguous()
    k_prune = k_prune.view(b, -1, h_kv * w_kv)
    mask = SignWithSigmoidGrad.apply(torch.matmul(q_prune, k_prune) / k_prune.shape[1])
    # q: b, N_q, c_qk
    # k: b, c_qk, N_kv
    # v: b, N_kv, c_v
    if smooth is None:
        smooth = c_qk ** 0.5
    cor_map = torch.matmul(q, k) / smooth
    attn = torch.softmax(cor_map, dim=-1)
    # attn: b, N_q, N_kv
    masked_attn = attn * mask
    output = torch.matmul(masked_attn, v)
    # output: b, N_q, c_v
    output = output.transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    conf = masked_attn.sum(-1).view(b, 1, h_q, w_q)

    # conf_map = torch.max(cor_map, -1, keepdim=True)[0]
    # conf_map = (conf_map - conf_map.mean(dim=1, keepdim=True)).view(b, 1, h_q, w_q)
    # conf_map = torch.sigmoid(conf_map * 10.0)

    if v2 is not None:
        v2 = v2.view(b, -1, h_kv * w_kv).transpose(-1, -2).contiguous()
        output2 = torch.bmm(torch.softmax(torch.masked_fill(cor_map, mask.bool(), -1e4), dim=-1),
                            v2).transpose(-1, -2).contiguous().view(b, -1, h_q, w_q)
    else:
        output2 = None
    return output, cor_map, conf, output2


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2

        #for real map
        self.real_conv = nn.Conv2d(in_channels=3, out_channels=label_nc, kernel_size=3, padding=1)

        self.f = nn.Conv2d(label_nc, nhidden, kernel_size=ks, stride=1, padding=pw)
        self.g = nn.Conv2d(label_nc, nhidden, kernel_size=ks, stride=1, padding=pw)
        self.h = nn.Conv2d(label_nc, nhidden, kernel_size=ks, stride=1, padding=pw)
        self.f_prune = nn.Conv2d(label_nc, nhidden, kernel_size=ks, stride=1, padding=pw)
        self.g_prune = nn.Conv2d(label_nc, nhidden, kernel_size=ks, stride=1, padding=pw)

        self.alpha = nn.Parameter(torch.Tensor([1]))

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, realmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        realmap = F.interpolate(realmap, size=x.size()[2:], mode='nearest')
        realmap = self.real_conv(realmap)

        # print("Segmap: ", segmap.shape)
        # print("Realmap: ", realmap.shape)

        attn_output, cor_map, conf, output2 = dynamic_attention(
            self.f(segmap), self.g(realmap), self.f_prune(segmap), self.g_prune(realmap), self.h(realmap), None, None)

        actv_beta = self.relu(attn_output)
        
        gamma = self.mlp_gamma(actv_beta)
        beta = self.mlp_beta(actv_beta)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
