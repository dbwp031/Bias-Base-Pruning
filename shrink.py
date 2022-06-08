import torch
import torch.nn as nn
import models
from copy import deepcopy


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_dict(parent_module, model_dict):
    state_dict = {}
    for key in model_dict:
        state_dict[key] = model_dict[key].size()

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d):
            conv = nn.Conv2d
            
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        # if isinstance(old_module, models.GlobalKernelConv2d):
        #     gkconv = models.GlobalKernelConv2d

        #     s = state_dict[n + '.weight']
        #     in_channels = s[1]
        #     out_channels = s[0]
        #     g = 1
        #     if old_module.kwargs['groups'] > 1:
        #         in_channels = out_channels
        #         g = in_channels
        #     new_gkconv = gkconv(gen=old_module.gen, in_channels=in_channels, out_channels=out_channels, kernel_size = old_module.kernel_size,
        #         stride=old_module.kwargs['stride'], padding=old_module.kwargs['padding'], dilation=old_module.kwargs['dilation'], groups=g)
        #     set_layer(new_module, n, new_gkconv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = nn.Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module

def shrink_state_mv1_cifar(net):
    key_list = list(net.keys())

    pre_filters = torch.tensor(list(range(3)))

    for name in key_list:
        if 'mask' in name:
            conv = net[name.replace('mask', 'weight').replace('bn', 'conv')]
            coef = None
            if conv.size(-1)==3:
                coef = net[name.replace('mask', 'map').replace('bn', 'conv')]
            bn_w = net[name.replace('mask', 'weight')]
            bn_b = net[name.replace('mask', 'bias')]
            mask = net[name]
            bn_rm = net[name.replace('mask', 'running_mean')]
            bn_rv = net[name.replace('mask', 'running_var')]
            filters = mask.sort()[1][-int(mask.sum()):].sort()[0]

            if coef is not None:
                coef_reshape = coef.view(conv.size(0), conv.size(1), -1)
                coef_reshape.data = coef_reshape.data[filters,:,:]
                coef.data = coef_reshape.data.view(-1, coef.size(1))
            bn_w.data = bn_w.data[filters]
            bn_b.data = bn_b.data[filters]
            bn_rm.data = bn_rm.data[filters]
            bn_rv.data = bn_rv.data[filters]

            conv.data = conv.data[filters,:,:,:]
            if 'bn2' in name:
                conv.data = conv.data[:,pre_filters,:,:]

            pre_filters = filters.clone()

        elif 'linear.weight' in name:
            fc = net[name]
            fc.data = fc.data[:, pre_filters]

    new_state_dict = {}

    for key in key_list:
        if 'mask' in key:
            del net[key]
        else:
            new_state_dict[key[7:]] = net[key]

    return new_state_dict


def shrink_state_cifar(net):
    key_list = list(net.keys())

    pre_filters = torch.tensor(list(range(3)))

    for name in key_list:
        if 'mask' in name:
            if 'downsample' in name:
                conv = net[name.replace('1.mask', '0.weight')]
            else:
                conv = net[name.replace('mask', 'weight').replace('bn', 'conv')]
            
            coef = None
            if conv.size(-1)==3 and 'layer' in name and 'map' in name:
                coef = net[name.replace('mask', 'map').replace('bn', 'conv')]
            bn_w = net[name.replace('mask', 'weight')]
            bn_b = net[name.replace('mask', 'bias')]
            mask = net[name]
            bn_rm = net[name.replace('mask', 'running_mean')]
            bn_rv = net[name.replace('mask', 'running_var')]
            filters = mask.sort()[1][-int(mask.sum()):].sort()[0]

            if coef is not None:
                coef_reshape = coef.view(conv.size(0), conv.size(1), -1)
                coef_reshape.data = coef_reshape.data[filters,:,:]
                coef_reshape.data = coef_reshape.data[:,pre_filters,:]
                coef.data = coef_reshape.data.view(-1, coef.size(1))
            conv.data = conv.data[filters,:,:,:]
            if 'downsample' in name:
                conv.data = conv.data[:,temp_pre_filters,:,:]
            else:
                conv.data = conv.data[:,pre_filters,:,:]
            bn_w.data = bn_w.data[filters]
            bn_b.data = bn_b.data[filters]
            bn_rm.data = bn_rm.data[filters]
            bn_rv.data = bn_rv.data[filters]

            if 'bn1' in name:
                temp_pre_filters = pre_filters.clone()
            pre_filters = filters.clone()

        elif 'fc.weight' in name:
            fc = net[name]
            fc.data = fc.data[:, pre_filters]

    new_state_dict = {}

    for key in key_list:
        if 'mask' in key:
            del net[key]
        else:
            new_state_dict[key[7:]] = net[key]

    return new_state_dict

def shrink_state_cifarv2(net):
    key_list = list(net.keys())

    pre_filters = torch.tensor(list(range(3)))

    # for key in net:
    #     print(key, net[key].size())

    fidx_list = []
    for i in range(1, 4):
        mask = torch.ge(torch.sigmoid(net['module.logit'+str(i)].data), 0.5).float()
        num_logit = mask.sum()
        # print(num_logit)
        filteridx = mask.sort()[1][-int(num_logit):].sort()[0]
        fidx_list.append(filteridx)
    chan_dict = {16:0, 32:1, 64:2}

    for name in key_list:
        if 'layer' not in name:
            if 'conv1.weight' in name:
                conv = net[name]
                bn_w = net[name.replace('conv1.weight', 'bn1.weight')]
                bn_b = net[name.replace('conv1.weight', 'bn1.bias')]
                bn_rm = net[name.replace('conv1.weight', 'bn1.running_mean')]
                bn_rv = net[name.replace('conv1.weight', 'bn1.running_var')]
                conv.data = conv.data[fidx_list[0],:,:,:]
                bn_w.data = bn_w.data[fidx_list[0]]
                bn_b.data = bn_b.data[fidx_list[0]]
                bn_rm.data = bn_rm.data[fidx_list[0]]
                bn_rv.data = bn_rv.data[fidx_list[0]]
            elif 'fc.weight' in name:
                fc = net[name]
                fc.data = fc.data[:, fidx_list[2]]
                # print(fc.data.size())
        elif 'layer' in name:
            if 'conv1.weight' in name:
                mask = torch.ge(torch.sigmoid(net[name.replace('conv1.weight', 'blogit')].data), 0.5).float() 
                conv = net[name]
                bn_w = net[name.replace('conv1.weight', 'bn1.weight')]
                bn_b = net[name.replace('conv1.weight', 'bn1.bias')]
                bn_rm = net[name.replace('conv1.weight', 'bn1.running_mean')]
                bn_rv = net[name.replace('conv1.weight', 'bn1.running_var')]
                filters = mask.sort()[1][-int(mask.sum()):].sort()[0]

                conv.data = conv.data[:,fidx_list[chan_dict[conv.size(1)]],:,:]
                conv.data = conv.data[filters,:,:,:]
                bn_w.data = bn_w.data[filters]
                bn_b.data = bn_b.data[filters]
                bn_rm.data = bn_rm.data[filters]
                bn_rv.data = bn_rv.data[filters]
            elif 'conv2.weight' in name:
                conv = net[name]
                bn_w = net[name.replace('conv2.weight', 'bn2.weight')]
                bn_b = net[name.replace('conv2.weight', 'bn2.bias')]
                bn_rm = net[name.replace('conv2.weight', 'bn2.running_mean')]
                bn_rv = net[name.replace('conv2.weight', 'bn2.running_var')]

                conv.data = conv.data[fidx_list[chan_dict[conv.size(0)]],:,:,:]
                conv.data = conv.data[:,filters,:,:]
                bn_w.data = bn_w.data[fidx_list[chan_dict[bn_w.size(0)]]]
                bn_b.data = bn_b.data[fidx_list[chan_dict[bn_b.size(0)]]]
                bn_rm.data = bn_rm.data[fidx_list[chan_dict[bn_rm.size(0)]]]
                bn_rv.data = bn_rv.data[fidx_list[chan_dict[bn_rv.size(0)]]]
            elif 'downsample.0.weight' in name:
                conv = net[name]
                conv = net[name]
                bn_w = net[name.replace('downsample.0.weight', 'downsample.1.weight')]
                bn_b = net[name.replace('downsample.0.weight', 'downsample.1.bias')]
                bn_rm = net[name.replace('downsample.0.weight', 'downsample.1.running_mean')]
                bn_rv = net[name.replace('downsample.0.weight', 'downsample.1.running_var')]

                conv.data = conv.data[fidx_list[chan_dict[conv.size(0)]],:,:,:]
                conv.data = conv.data[:,fidx_list[chan_dict[conv.size(1)]],:,:]
                bn_w.data = bn_w.data[fidx_list[chan_dict[bn_w.size(0)]]]
                bn_b.data = bn_b.data[fidx_list[chan_dict[bn_b.size(0)]]]
                bn_rm.data = bn_rm.data[fidx_list[chan_dict[bn_rm.size(0)]]]
                bn_rv.data = bn_rv.data[fidx_list[chan_dict[bn_rv.size(0)]]]
        
        
    new_state_dict = {}

    for key in key_list:
        if 'logit' in key:
            del net[key]
        else:
            new_state_dict[key[7:]] = net[key]

    return new_state_dict

def shrink_state_imagenet(net, layers):
    key_list = list(net.keys())

    pre_filters = torch.tensor(list(range(3)))

    if layers==50:
        for name in key_list:
            if 'bn2.mask' in name:
                conv = net[name.replace('bn2.mask', 'conv2.weight')]
                bn_w = net[name.replace('mask', 'weight')]
                bn_b = net[name.replace('mask', 'bias')]
                mask = net[name]
                bn_rm = net[name.replace('mask', 'running_mean')]
                bn_rv = net[name.replace('mask', 'running_var')]
                filters = mask.sort()[1][-int(mask.sum()):].sort()[0]

                conv.data = conv.data[filters,:,:,:]
                conv.data = conv.data[:,pre_filters,:,:]
                bn_w.data = bn_w.data[filters]
                bn_b.data = bn_b.data[filters]
                bn_rm.data = bn_rm.data[filters]
                bn_rv.data = bn_rv.data[filters]

                pre_filters = filters.clone()

            elif 'mask' in name:
                if 'downsample' in name:
                    conv = net[name.replace('1.mask', '0.weight')]
                else:
                    conv = net[name.replace('mask', 'weight').replace('bn', 'conv')]
                bn_w = net[name.replace('mask', 'weight')]
                bn_b = net[name.replace('mask', 'bias')]
                mask = net[name]
                bn_rm = net[name.replace('mask', 'running_mean')]
                bn_rv = net[name.replace('mask', 'running_var')]
                filters = mask.sort()[1][-int(mask.sum()):].sort()[0]

                conv.data = conv.data[filters,:,:,:]
                if 'downsample' in name:
                    conv.data = conv.data[:,temp_pre_filters,:,:]
                else:
                    conv.data = conv.data[:,pre_filters,:,:]
                bn_w.data = bn_w.data[filters]
                bn_b.data = bn_b.data[filters]
                bn_rm.data = bn_rm.data[filters]
                bn_rv.data = bn_rv.data[filters]

                if 'bn1' in name:
                    temp_pre_filters = pre_filters.clone()
                pre_filters = filters.clone()
            elif 'fc.weight' in name:
                fc = net[name]
                fc.data = fc.data[:, pre_filters]
    elif layers == 18:
        for name in key_list:
            if 'mask' in name:
                if 'downsample' in name:
                    conv = net[name.replace('1.mask', '0.weight')]
                else:
                    conv = net[name.replace('mask', 'weight').replace('bn', 'conv')]

                bn_w = net[name.replace('mask', 'weight')]
                bn_b = net[name.replace('mask', 'bias')]
                mask = net[name]
                bn_rm = net[name.replace('mask', 'running_mean')]
                bn_rv = net[name.replace('mask', 'running_var')]
                filters = mask.sort()[1][-int(mask.sum()):].sort()[0]

                conv.data = conv.data[filters,:,:,:]
                if 'downsample' in name:
                    conv.data = conv.data[:,temp_pre_filters,:,:]
                else:
                    conv.data = conv.data[:,pre_filters,:,:]
                bn_w.data = bn_w.data[filters]
                bn_b.data = bn_b.data[filters]
                bn_rm.data = bn_rm.data[filters]
                bn_rv.data = bn_rv.data[filters]

                if 'bn1' in name:
                    temp_pre_filters = pre_filters.clone()
                pre_filters = filters.clone()

            elif 'fc.weight' in name:
                fc = net[name]
                fc.data = fc.data[:, pre_filters]

    new_state_dict = {}

    for key in key_list:
        if 'mask' in key:
            del net[key]
        else:
            new_state_dict[key[7:]] = net[key]
    
    return new_state_dict