import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time, os, math


def get_weight_threshold(model, rate, args):
    importance_all = None
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'mask' not in name:
            weights = item.data.view(-1).cpu()
            if args.prune_imptype == 'L1':
                importance = weights.abs().numpy()
            elif args.prune_imptype == 'L2':
                importance = weights.pow(2).numpy()
            elif args.prune_imptype == 'grad':
                grads = item.grad.data.view(-1).cpu()
                importance = grads.abs().numpy()
            elif args.prune_imptype == 'syn':
                grads = item.grad.data.view(-1).cpu()
                importance = (weights * grads).abs().numpy()
            
            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance_all)[int(len(importance_all) * rate)]
    return threshold


def weight_prune(model, threshold, args):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if len(item.size())==4 and 'mask' not in name:
            key = name.replace('weight', 'mask')
            if args.prune_imptype == 'L1':
                mat = item.data.abs()
            elif args.prune_imptype == 'L2':
                mat = item.data.pow(2)
            elif args.prune_imptype == 'grad':
                mat = item.grad.data.abs()
            elif args.prune_imptype == 'syn':
                mat = (item.data * item.grad.data).abs()
            state[key].data.copy_(torch.gt(mat, threshold).float())

def get_BN_mask(model, rate, mask_min, args):
    # fl = open('/root/yujevolume/CoBaL_code/log.txt','a')
    with torch.no_grad():
        idx = 0
        importance_list = None
        inf = 1e+2
        state = model.module.state_dict()

        group_dict = {1:None, 2:None, 3:None, 4:None}
        cnt_dict = {1:0, 2:0, 3:0, 4:0}
        if 'group' not in args.prune_imp:
            if args.arch == 'mobilenet':
                pre_conv_norm = state['conv1.weight'].data.view(32, -1).pow(2).mean(dim=1).cpu()
                pre_bn_norm = state['bn1.weight'].data.pow(2).cpu()

                pre_imp = pre_conv_norm * pre_bn_norm
                for name, item in model.module.named_parameters():
                    if 'layers' in name and 'conv1.weight' in name:
                        conv_norm = item.data.view(item.size(0), -1).pow(2).mean(dim=1).cpu()
                        bn_norm = state[name.replace('conv', 'bn')].data.pow(2).cpu()
                        importance = conv_norm * bn_norm + pre_imp
                    elif 'conv2.weight' in name:
                        if '12' in name:
                            linear_norm = state['linear.weight'].data.pow(2).mean(dim=0).cpu()
                            conv_norm = item.data.view(item.size(0), -1).pow(2).mean(dim=1).cpu()
                            bn_norm = state[name.replace('conv', 'bn')].data.pow(2).cpu()
                            importance = conv_norm * bn_norm + linear_norm
                        else:
                            conv_norm = item.data.view(item.size(0), -1).pow(2).mean(dim=1).cpu()
                            bn_norm = state[name.replace('conv', 'bn')].data.pow(2).cpu()
                            pre_imp = conv_norm * bn_norm
                            continue
                    else:
                        continue
                    importance[importance>=importance.topk(1)[0][0]] = inf
                    if importance_list is None:
                        importance_list = importance.numpy()
                    else:
                        importance_list = np.append(importance_list, importance.numpy())
                threshold = np.sort(importance_list)[int(len(importance_list) * rate)]
                filter_mask = np.greater(importance_list, threshold).astype(int)
                filter_mask[filter_mask==0] = mask_min
                return filter_mask

            for name, item in model.module.named_parameters():
                if len(item.size())==4 and 'weight' in name: # add conv
                    filters = item.data.view(item.size(0), -1)
                    filters_grad = item.grad.data.view(item.size(0), -1)
                    norm = filters.pow(2).mean(dim=1).cpu()
                    grad_norm = filters_grad.pow(2).mean(dim=1).cpu()
                if len(item.size())==1 and 'weight' in name:
                    if 'coba' in args.prune_imp:
                        importance = item.data.pow(2).cpu() * norm
                    elif 'bn' in args.prune_imp:
                        importance = item.data.pow(2).cpu()
                    elif 'conv' in args.prune_imp:
                        importance = norm
                    elif 'grad' in args.prune_imp:
                        importance = item.data.pow(2).cpu() * item.grad.data.pow(2).cpu() * norm * grad_norm

                    importance[importance>=importance.topk(1)[0][0]] = inf
                    if importance_list is None:
                        importance_list = importance.numpy()
                    else:
                        importance_list = np.append(importance_list, importance.numpy())

            threshold = np.sort(importance_list)[int(len(importance_list) * rate)]
            filter_mask = np.greater(importance_list, threshold).astype(int)
            filter_mask[filter_mask==0] = mask_min

            # fl.close()
            return filter_mask
        else:
            ### need to Fix - Yuje
            if args.layers < 50 or 'cifar' in args.dataset:
                for name, item in model.module.named_parameters():
                    # if 'bn' in name:
                    #     print(f"{name}: {item.shape}")
                    if len(item.size())==4 and 'weight' in name: # add conv
                        # print(f"4name:{name}")
                        filters = item.data.view(item.size(0), -1)
                        norm = filters.pow(2).mean(dim=1).cpu()
                        # norm = torch.abs(filters).mean(dim=1).cpu()
                        # norm = filters.cpu()
                    if len(item.size())==1 and 'weight' in name:
                        # print(f"1name:{name}")
                        # print(item)
                        bias_key = name.replace('weight', 'bias')
                        bias = state[bias_key].data.cpu()
                        if 'coba' in args.prune_imp: # 이걸로 하고 있다.
                            # zero = torch.ones_like(bias)*-0.1
                            # importance = item.data.pow(2).cpu() * norm * torch.where(bias>-0.1,bias,zero).pow(2)
                            if args.imp_type == 0:
                                importance = torch.abs(item.data).cpu() * norm# - torch.sign(bias) * bias.pow(2)
                            if args.imp_type == 1:
                                zero = torch.ones_like(bias)*-0.01
                                importance = item.data.pow(2).cpu() * norm * torch.where(bias>-0.01,bias,zero).pow(2)       
                            if args.imp_type == 2:
                                zero = torch.ones_like(bias)*0.01
                                importance = item.data.pow(2).cpu() * norm * torch.where(bias>0.01,bias,zero).pow(2)                         
                            # fl.write(f'{name}\timportance:{importance}\tdata:{item.data.pow(2).cpu()*norm}\t:bias:{bias.pow(2)*args.bias_importance}\n')
                            if args.imp_type == 3:
                                a = item.data.pow(2).cpu()*norm
                                b = torch.sign(bias) * bias.pow(2)
                                asum,bsum = sum(a),sum(b)
                                ratio = asum/bsum
                                importance = a + b*ratio*args.bias_importance
                                
                            if args.imp_type == 4:
                                a = item.data.pow(2).cpu()*norm
                                b = torch.sign(bias)*bias.pow(2)
                                ratio = a/b
                                importance = a + b*ratio*args.bias_importance
                            if args.imp_type == 5:
                                a = item.data.pow(2).cpu()*norm
                                b = bias.pow(2)
                                asum,bsum = sum(a),sum(b)
                                ratio = asum/bsum
                                importance = a + torch.sign(bias)*bias.pow(2)*ratio*args.bias_importance   
                            if args.imp_type == 6:
                                a = item.data.pow(2).cpu()*norm
                                b = bias.pow(2)
                                asum,bsum = sum(a),sum(b)
                                ratio = bsum/asum
                                c = torch.sign(bias)*bias.pow(2)*ratio
                                importance = a + c*args.bias_importance 
                            elif args.imp_type == 7:
                                ratio = bias.pow(2)/(item.data.pow(2).cpu()*norm)
                                importance = item.data.pow(2).cpu()*norm + torch.sign(bias)*bias.pow(2)*ratio*args.bias_importance
    #####################################################################
                            elif args.imp_type == 8:
                                importanance = bias.pow(2)

                            elif args.imp_type == 9:
                                importance = torch.sign(bias)*bias.pow(2)

                            elif args.imp_type == 10:
                                b = bias.pow(2)
                                bsum = sum(b)
                                importance = bias.pow(2)/bsum

                            elif args.imp_type == 11:
                                b = bias.pow(2)
                                bsum = sum(b)
                                importance = (torch.sign(bias)*bias.pow(2))/bsum

                            elif args.imp_type == 12:
                                importance = 1/ (item.data.pow(2).cpu()*norm)
                                
                            elif args.imp_type == 13:
                                importanance = 1/bias.pow(2)

                            elif args.imp_type == 14:
                                importance = item.data.pow(2).cpu()*norm/bias.pow(2)
                            
                            elif args.imp_type == 15:
                                a = item.data.pow(2).cpu()*norm
                                b=bias.pow(2)
                                importance = torch.ones_like(a)*(sum(a)/sum(b))
                            elif args.imp_type == 16:
                                a = item.data.pow(2).cpu()*norm
                                b=bias.pow(2)
                                N = b.shape[0]
                                importance = a/(sum(a)*N) + args.bias_importance*(torch.sign(b)*b)/(N*sum(b))

                                # 수정후
                                # a = item.data.pow(2).cpu()*norm
                                # b=bias.pow(2)
                                # N = b.shape[0]
                                # importance = N*a/(sum(a)) + args.bias_importance*(torch.sign(b)*b*N)/(sum(b))

                            elif args.imp_type == 17:
                                #rw + b 에서 b가 클수록 점수가 안좋다
                                # 값이 이상하게 나오면 스케일링하면서
                                a = item.data.pow(2).cpu()*norm
                                b=bias.pow(2)
                                importance = a + (1/b)*args.bias_importance
                            elif args.imp_type == 18:
                                a = item.data.pow(2).cpu()*norm
                                b=bias.pow(2)
                                N = b.shape[0]
                                importance = a/(N*sum(a)) + args.bias_importance*((N*sum(b)/torch.sign(b)*b))
                            elif args.imp_type == 19:
                                a = item.data.pow(2).cpu()*norm
                                b=bias.pow(2)
                                N = b.shape[0]
                                importance = (N*a/(sum(a)) + (torch.sign(b)*b*N)/(sum(b)))*args.bias_importance

                            elif args.imp_type == 20:
                                importance = item.data.cpu()*norm.mean(dim=1)

                            elif args.imp_type == 21:
                                importance = (item.data.cpu()*norm.mean(dim=1)) + bias
                                
                            elif args.imp_type == 22:
                                a = item.data.pow(2).cpu()*norm
                                b=bias.pow(2)
                                N = b.shape[0]
                                importance = a/(sum(a)) + args.bias_importance*(torch.sign(b)*b)/(sum(b))

                            elif args.imp_type == 23:
                                a = torch.abs(item.data).cpu()*norm
                                b=torch.abs(bias)
                                N = b.shape[0]
                                importance = N*a/(sum(a)) + args.bias_importance*(torch.sign(b)*b*N)/(sum(b))

                            elif args.imp_type == 24:
                                a = torch.abs(item.data).cpu()*norm
                                b=torch.abs(bias)
                                N = b.shape[0]
                                importance = N*a/(sum(a)) + 1/(args.bias_importance*(torch.sign(b)*b*N)/(sum(b)))

                            elif args.imp_type == 25:
                                a = torch.abs(item.data).cpu()*norm
                                b=torch.abs(bias)
                                N = b.shape[0]
                                importance = N*a/(sum(a))* 1/(args.bias_importance*(torch.sign(b)*b*N)/(sum(b)))

                            elif args.imp_type == 26:
                                a = item.data.pow(2).cpu() * norm 
                                b=bias.pow(2)
                                N = b.shape[0]
                                importance = a + args.bias_importance*(torch.sign(b)*b*N)/(sum(b))
    #####################################################################                        
                        elif 'bn' in args.prune_imp:
                            importance = item.data.pow(2).cpu()
                        elif 'conv' in args.prune_imp:
                            importance = norm                    
                            
                        if 'layer' not in name:
                            cnt_dict[1] += 1
                            group_dict[1] = importance.numpy()
                        elif 'layer' in name and ('downsample' in name or 'bn2' in name):
                            layer_num = int(name.split('.')[0][-1])
                            cnt_dict[layer_num] += 1
                            if group_dict[layer_num] is None:
                                group_dict[layer_num] = importance.numpy()
                            else:
                                group_dict[layer_num] += importance.numpy()
                        else:
                            importance[importance>=importance.topk(1)[0][0]] = inf
                            if importance_list is None:
                                importance_list = importance.numpy()
                            else:
                                importance_list = np.append(importance_list, importance.numpy())
            else:
                for name, item in model.module.named_parameters():
                    if len(item.size())==4 and 'weight' in name:
                        filters = item.data.view(item.size(0), -1)
                        norm = filters.pow(2).mean(dim=1).cpu()

                    if len(item.size())==1 and 'weight' in name:
                        if 'coba' in args.prune_imp:
                            importance = (item.data.pow(2).cpu() * norm)
                        elif 'bn' in args.prune_imp:
                            importance = item.data.pow(2).cpu()
                        elif 'conv' in args.prune_imp:
                            importance = norm
                        
                        if 'layer' in name and ('downsample' in name or 'bn3' in name):
                            layer_num = int(name.split('.')[0][-1])
                            cnt_dict[layer_num] += 1
                            if group_dict[layer_num] is None:
                                group_dict[layer_num] = importance.numpy()
                            else:
                                group_dict[layer_num] += importance.numpy()
                        else:
                            if 'layer' not in name:
                                continue
                            importance[importance>=importance.topk(1)[0][0]] = inf
                            if importance_list is None:
                                importance_list = importance.numpy()
                            else:
                                importance_list = np.append(importance_list, importance.numpy())


            shortcut_group1 = (group_dict[1] / cnt_dict[1])
            shortcut_group2 = (group_dict[2] / cnt_dict[2])
            shortcut_group3 = (group_dict[3] / cnt_dict[3])

            if args.dataset == 'imagenet':
                shortcut_group4 = (group_dict[4] / cnt_dict[4])
                importance_all = np.concatenate((importance_list, shortcut_group1, shortcut_group2, shortcut_group3, shortcut_group4))
            else:
                importance_all = np.concatenate((importance_list, shortcut_group1, shortcut_group2, shortcut_group3))
            threshold = np.sort(importance_all)[int(len(importance_all) * rate)]
            mask1 = np.greater(importance_list, threshold).astype(float)
            mask2 = np.greater(shortcut_group1, threshold).astype(float)
            mask3 = np.greater(shortcut_group2, threshold).astype(float)
            mask4 = np.greater(shortcut_group3, threshold).astype(float)

            if args.dataset == 'imagenet':
                mask5 = np.greater(shortcut_group4, threshold).astype(float)
            
            mask1[mask1==0] = mask_min
            mask2[mask2==0] = mask_min
            mask3[mask3==0] = mask_min
            mask4[mask4==0] = mask_min
            if args.dataset == 'imagenet':
                mask5[mask5==0] = mask_min
            
            filter_mask = [mask1, mask2, mask3, mask4]
            if args.dataset == 'imagenet':
                filter_mask.append(mask5)

        return filter_mask

def BN_prune(model, filter_mask, args):
    idx = 0
    state = model.module.state_dict()

    if 'group' not in args.prune_imp:
        if args.arch == 'mobilenet':
            cnt = 0
            for name, item in model.module.named_parameters():
                if 'mask' in name:
                    cnt+=1
                    temp_mask = filter_mask[idx:idx+item.size(0)]
                    item.data = torch.Tensor(temp_mask).cuda()
                    if cnt%2==0:
                        idx+=item.size(0)
        else:
            for name, item in model.module.named_parameters():
                if len(item.size())==1 and 'mask' in name:
                    temp_mask = filter_mask[idx:idx+item.size(0)]
                    item.data = torch.Tensor(temp_mask).cuda()
                    idx+=item.size(0)
    else:
        if args.layers < 50 or 'cifar' in args.dataset:
            for name, item in model.module.named_parameters():
                if len(item.size())==1 and 'mask' in name:
                    if 'layer' not in name:
                        if 'wrn' in args.arch:
                            continue
                            item.data = torch.Tensor(filter_mask[0][idx:idx+item.size(0)]).cuda()
                            idx+=item.size(0)
                        elif 'resnet' in args.arch:
                            item.data = torch.Tensor(filter_mask[1]).cuda()
                    elif 'layer' in name and ('downsample' in name or 'bn2' in name):
                        layer_num = int(name.split('.')[0][-1])
                        item.data = torch.Tensor(filter_mask[layer_num]).cuda()
                    else:
                        item.data = torch.Tensor(filter_mask[0][idx:idx+item.size(0)]).cuda()
                        idx+=item.size(0)
        else:
            for name, item in model.module.named_parameters():
                if len(item.size())==1 and 'mask' in name:
                    if 'layer' in name and ('downsample' in name or 'bn3' in name):
                        layer_num = int(name.split('.')[0][-1])
                        item.data = torch.Tensor(filter_mask[layer_num]).cuda()
                    else:
                        if 'layer' not in name:
                            continue
                        item.data = torch.Tensor(filter_mask[0][idx:idx+item.size(0)]).cuda()
                        idx+=item.size(0)
