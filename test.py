import torch
import numpy as np
import pruning
import models
import argparse

pruner = pruning.dpf
mbn = pruner.mnn.MaskBatchNorm2d
mask = mbn(32)
print(mbn)
print(mask)
exit()
net = pruning.models.rexnet(data='imagenet', width_mult=1.0, depth_mult=1.0, classnum=1000, mnn=pruner.mnn)
dw_cnt = 0
pw_cnt = 0 
for name, item in net.named_parameters():
    print(name, item.size())

exit()


'''

state = torch.load('./pruned_pr36_mg_rexnet1.3_v4.pth', map_location='cpu')
filter_list = state['filter_list']
print(filter_list)
exit()
for key in net:
    print(key, net[key].size())

exit()

model = models.rexnet(data='things', width_mult=1.3, depth_mult=1.0, classnum=41)
state = model.state_dict()

for name, item in model.named_parameters():
    print(name, item.size())

exit()
whole_filters = 0
remain_filters = 0
key_list = list(net.keys())

mask_list_all = []
mask_key = []
for key in net:
    #if 'depthwise_conv.mask' in key:
    if len(net[key].size())==4:
        filters = net[key].view(net[key].size(0),-1)
        weight_len = filters.size(1)
        #whole_filters+=filters.size(0)
        #remain_filters+=(filters.sum(dim=1) / weight_len).sum()
        if 'mask' in key:
            mask_list = []
            for f in range(net[key].size(0)):
                if net[key].data[f,:,:,:].sum() != 0:
                    mask_list.append(f)
            mask_key.append(key)
            mask_list_all.append(mask_list)

idx = 0
mask_idx = 0
for key in net:
    if 'mask' in key:
        for i in range(idx-7,idx+7):
            if len(net[key_list[i]].size())==4:
                if i==idx+6:
                    net[key_list[i]].data = net[key_list[i]].data[:,mask_list_all[mask_idx],:,:]
                else:
                    net[key_list[i]].data = net[key_list[i]].data[mask_list_all[mask_idx],:,:,:]
            elif len(net[key_list[i]].size())==1:
                net[key_list[i]].data = net[key_list[i]].data[mask_list_all[mask_idx]]
          
        net[key_list[idx-1]].data = net[key_list[idx-1]].data[mask_list_all[mask_idx],:,:,:]
        net[key_list[idx-7]].data = net[key_list[idx-7]].data[mask_list_all[mask_idx],:,:,:]
        net[key_list[idx-3]].data = net[key_list[idx-5]].data[mask_list_all[mask_idx]]
        net[key_list[idx-4]].data = net[key_list[idx-5]].data[mask_list_all[mask_idx]]
        net[key_list[idx-5]].data = net[key_list[idx-5]].data[mask_list_all[mask_idx]]
        net[key_list[idx-6]].data = net[key_list[idx-6]].data[mask_list_all[mask_idx]]
        net[key_list[idx+1]].data = net[key_list[idx+1]].data[mask_list_all[mask_idx]]
        net[key_list[idx+2]].data = net[key_list[idx+2]].data[mask_list_all[mask_idx]]
        net[key_list[idx+3]].data = net[key_list[idx+3]].data[mask_list_all[mask_idx]]
        net[key_list[idx+4]].data = net[key_list[idx+4]].data[mask_list_all[mask_idx]]
        net[key_list[idx+6]].data = net[key_list[idx+6]].data[:,mask_list_all[mask_idx],:,:]
        
        mask_idx+=1
    idx+=1

for key in mask_key:
    if 'mask' in key:
        del net[key]
#print(net.keys())
torch.save(net, '/root/volume/AIChallenge_base/pruned_rexnet1.3_v4.pth')
#for name, item in model.named_parameters():
#    print(name)

            #print(key, net[key].size(), (filters.sum(dim=1) / weight_len).sum())
        else:
            #print(key, net[key].size())
            
#print(whole_filters, remain_filters)

pruner = pruning.dpf
net = pruning.models.efficientnet(data='imagenet', width_mult=1, depth_mult=1, efficient_type=0, mnn=pruner.mnn)
for name, item in net.named_parameters():
    print(name, item.size())
#for key in net.state_dict():
#    print(key)


def get_skip_connection(model):
    state = model.state_dict()
    skip_connection = dict()

    for i in range(15):
        key1 = "_blocks."+str(i)+"._project_conv.mask"
        key2 = "_blocks."+str(i+1)+"._project_conv.mask"
        if state[key1].size(0) == state[key2].size(0):
            if state[key1].size(0) not in skip_connection:
                skip_connection[state[key1].size(0)] = [key1, key2]
            else:
                skip_connection[state[key1].size(0)].append(key2)
    print(skip_connection)

#get_skip_connection(net)
'''