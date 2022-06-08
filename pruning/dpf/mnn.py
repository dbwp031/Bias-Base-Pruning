# masked torch.nn layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math


class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)

    def forward(self, input):
        masked_weight = Masker.apply(self.weight, self.mask)
        if self.bias is not None:
            masked_bias = Masker.apply(self.bias, self.mask[:,0,0,0])
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                masked_weight, masked_bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, masked_weight, masked_bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return super(MaskConv2d, self)._conv_forward(input, masked_weight)


class MaskBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                affine=True, track_running_stats=True):
        super(MaskBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.mask = Parameter(torch.ones(self.bias.size()), requires_grad=False)

    def forward(self, input):
        masked_bias = Masker.apply(self.bias, self.mask)
        masked_weight = Masker.apply(self.weight, self.mask)
        
        # masked_bias = self.bias * self.mask
        # masked_weight = self.weight * self.mask

        out =  F.batch_norm(
            input, self.running_mean, self.running_var, masked_weight, masked_bias,
            self.training, self.momentum, self.eps)
        
        return out

class MaskBatchNorm2dv2(nn.BatchNorm2d):
    def __init__(self, max_value, num_features, eps=1e-05, momentum=0.1,
                affine=True, track_running_stats=True):
        super(MaskBatchNorm2dv2, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.mask = Parameter(torch.ones(self.bias.size()), requires_grad=False)
        self.max_value = max_value

    def forward(self, input):
        masked_bias = Masker.apply(self.bias, self.mask * self.max_value)
        masked_weight = Masker.apply(self.weight, self.mask * self.max_value)
        
        out =  F.batch_norm(
            input, self.running_mean, self.running_var, masked_weight, masked_bias,
            self.training, self.momentum, self.eps)
        
        return out


class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias)
        self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)

    def forward(self, input):
        masked_weight = Masker.apply(self.weight, self.mask)
        return F.linear(input, masked_weight, bias)


class MaskConv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw) # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        masked_weight = Masker.apply(self.weight, self.mask)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MaskConv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.mask = Parameter(torch.ones(self.weight.size()), requires_grad=False)

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        masked_weight = Masker.apply(self.weight, self.mask)
        x = F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Module):
    """Identity mapping.
       Send input to output directly.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input