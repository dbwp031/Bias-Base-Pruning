import time
import pathlib
from os.path import isfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import models
import config
import pruning
from utils import *
from shrink import *
from data import DataLoader
from get_flops import profile
import copy
import random
import math
from torch.autograd import Variable
from sacred import Experiment
import neptune.new as neptune


# for ignore ImageNet PIL EXIF UserWarning
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

run = neptune.init(
    project="dbwp031/CoBaL",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNzkzYzdkNi00NTMwLTQ2MmUtODdhMC1lNTU3MmI2YzhhNjUifQ==",
)  # your credentials

def hyperparam():
    """
    sacred exmperiment hyperparams
    :return:
    """
    args = config.config()

    return args

def main(args):
    print(args.epochs)
    global arch_name
    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # Dataset loading
    print('==> Load data..')
    start_time = time.time()
    args.image_size=32
    train_loader, val_loader = DataLoader(args.batch_size, args.image_size, args.workers,
                                          args.dataset, args.datapath,
                                          args.cuda)
    target_class_number = len(train_loader.dataset.classes)
    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time//60), elapsed_time%60))
    print('===> Data loaded..')

    # set model name
    arch_name = set_arch_name(args)
    print('\n=> creating model \'{}\''.format(arch_name))
    if not args.prune:  # base
        model = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers, width_mult=args.width_mult, args=args, classnum=target_class_number)
    
    elif args.prune:    # for pruning
        pruner = pruning.__dict__[args.pruner]
        model = pruning.models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                           width_mult=args.width_mult, mnn=pruner.mnn, args=args, classnum=target_class_number)


    if model is None:
        print('==> unavailable model parameters!! exit...\n')
        exit()

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD([param for name, param in model.named_parameters() if 'mask' not in name], lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    scheduler = set_scheduler(optimizer, args)
    
    # set multi-gpu
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()

        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True

    # load a pre-trained model
    if args.load is not None:
        if args.corrp:
            arch_name = 'resnet'+str(args.layers)
            ckpt_file = pathlib.Path('/root/yujevolume/CoBaL_code/checkpoint') / arch_name / args.dataset / args.load
        else:
            ckpt_file = pathlib.Path('/root/yujevolume/CoBaL_code/checkpoint') / arch_name / args.dataset / args.load
        assert isfile(ckpt_file), '==> no checkpoint found \"{}\"'.format(args.load)
        
        print('==> Loading Checkpoint \'{}\''.format(args.load))
        strict = False if args.prune or args.shrink else True  # check pruning
        if args.shrink:
            state = torch.load(ckpt_file)
            if 'cifar' in args.dataset:
                shrinked_state = shrink_state_cifar(state)
            else:
                shrinked_state = shrink_state_imagenet(state, args.layers)
            model = adapt_model_from_dict(model.module, shrinked_state)
            model.load_state_dict(shrinked_state)

            if 'cifar' in args.dataset:
                    input = torch.rand(1, 3, 32, 32).cuda()
                    macs, params = profile(model.cuda(), inputs=(input,))
                    print("MACs : {}, Params : {}".format(macs, params))
            else:
                input = torch.rand(1, 3, 224, 224).cuda()
                macs, params = profile(model.cuda(), inputs=(input,))
                print("MACs : {}, Params : {}".format(macs, params))
            # ex.log_scalar('MACs', macs, 0)
            # ex.log_scalar('params', params, 0)

            if args.cuda:
                torch.cuda.set_device(args.gpuids[0])
                with torch.cuda.device(args.gpuids[0]):
                    model = model.cuda()
                    criterion = criterion.cuda()

                model = nn.DataParallel(model, device_ids=args.gpuids,
                                        output_device=args.gpuids[0])
                cudnn.benchmark = True
        else:
            checkpoint = load_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
        print('==> Loaded Checkpoint \'{}\''.format(args.load))
    

    # for training
    if args.run_type == 'train':
        # init parameters
        start_epoch = 0
        best_acc1 = 0.0
        train_time = 0.0
        validate_time = 0.0

        for epoch in range(start_epoch, args.epochs):

            if args.accel > 0 and epoch < args.accel:
                adjust_learning_rate(optimizer, epoch)
            elif epoch == args.accel:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                
            print('\n==> {}/{} training'.format(
                    arch_name, args.dataset))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))
            
            # train for one epoch
            print('===> [ Training ]')
            start_time = time.time()
            acc1_train, acc5_train = train(args, train_loader,
                epoch=epoch, model=model,
                criterion=criterion, optimizer=optimizer)
            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set
            print('===> [ Validation ]')
            start_time = time.time()
            acc1_valid, acc5_valid = validate(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)
            run['log/acc1_valid'].log(acc1_valid)
            run['log/acc5_valid'].log(acc5_valid)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate this epoch'.format(
                elapsed_time))

            # learning rate schduling
            scheduler.step()

            acc1_train = round(acc1_train.item(), 4)
            acc5_train = round(acc5_train.item(), 4)
            acc1_valid = round(acc1_valid.item(), 4)
            acc5_valid = round(acc5_valid.item(), 4)

            # remember best Acc@1 and save checkpoint and summary csv file
            state = model.state_dict()

            is_best = acc1_valid >= best_acc1
            best_acc1 = max(acc1_valid, best_acc1)
            if is_best:
                best_epoch = epoch
            if is_best or epoch>=args.milestones[1]:
                # is_best = acc1_valid >= best_acc1
                # best_acc1 = max(acc1_valid, best_acc1)

                if args.pruner == 'dpf':
                    savename = args.save+str(args.prune_type)+'fr'+str(args.prune_freq)+'pr'+str(args.prune_rate)+args.prune_imp+args.prune_imptype+'.pth'
                elif args.pruner == 'static' or args.pruner=='staticv2':
                    savename = args.save+args.pruner+'pr'+str(args.prune_rate)+'time_'+str(args.date)+'type_'+args.prune_imptype+'.pth'
                save_model(arch_name, args.dataset, state, savename)
                # check MACs & Params
            tmp_model = copy.deepcopy(model.module)
            if args.prune and args.prune_type=='structured':
                tmp_model = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers, 
                                        width_mult=args.width_mult, 
                                       args=args, classnum=target_class_number)
                if 'cifar' in args.dataset:
                    if 'mobilenet' in args.arch:
                        shrinked_state = shrink_state_mv1_cifar(model.state_dict())
                    elif 'resnetv2' == args.arch:
                        shrinked_state = shrink_state_cifarv2(model.state_dict())
                    else:
                        shrinked_state = shrink_state_cifar(model.state_dict())
                else:
                    shrinked_state = shrink_state_imagenet(model.state_dict(), args.layers)
                tmp_model = adapt_model_from_dict(tmp_model, shrinked_state)
            tmp_model = tmp_model.cuda()
            
            if 'cifar' in args.dataset:
                input = torch.rand(1, 3, 32, 32).cuda()
                macs, params = profile(tmp_model, inputs=(input,))
                print("MACs : {}, Params : {}".format(macs, params))
                run['log/MACS'].log(macs)
                run['log/params'].log(params)
            else:
                input = torch.rand(1, 3, 224, 224).cuda()
                macs, params = profile(tmp_model, inputs=(input,))
                print("MACs : {}, Params : {}".format(macs, params))
            # ex.log_scalar('MACs', macs, epoch)
            # ex.log_scalar('params', params, epoch)

            # end of one epoch
            print()



        args.checking_epoch = 0
        # calculate the total training time 
        avg_train_time = train_time / (args.epochs - start_epoch)
        avg_valid_time = validate_time / (args.epochs - start_epoch)
        total_train_time = train_time + validate_time
        print('====> average training time each epoch: {:,}m {:.2f}s'.format(
            int(avg_train_time//60), avg_train_time%60))
        print('====> a`verage validation time each epoch: {:,}m {:.2f}s'.format(
            int(avg_valid_time//60), avg_valid_time%60))
        print('====> training time: {}h {}m {:.2f}s'.format(
            int(train_time//3600), int((train_time%3600)//60), train_time%60))
        print('====> validation time: {}h {}m {:.2f}s'.format(
            int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
        print('====> total training time: {}h {}m {:.2f}s'.format(
            int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))
        f = open('/root/yujevolume/CoBaL_code/best_accuracy.txt','a')
        f.write(f'imptype: {args.imp_type} prune-rate:{args.prune_rate}, best epoch: {best_epoch} best_acc1:{best_acc1},  Param:{params},  Macs:{macs}, test-name:{args.test_name}, date:{args.date} \n')
        f.close()
        return best_acc1
    
    elif args.run_type == 'evaluate':   # for evaluation
        # for evaluation on validation set
        print('\n===> [ Evaluation ]')
        start_time = time.time()

        if 'cifar' in args.dataset:
            input = torch.rand(1, 3, 32, 32).cuda()
            macs, params = profile(model.module, inputs=(input,))
            print("MACs : {}, Params : {}".format(macs, params))
        else:
            input = torch.rand(1, 3, 224, 224).cuda()
            macs, params = profile(model.module, inputs=(input,))
            print("MACs : {}, Params : {}".format(macs, params))

        model = model.cuda()
        acc1, acc5 = validate(args, val_loader, None, model, criterion)
        elapsed_time = time.time() - start_time
        print('====> {:.2f} seconds to evaluate this model\n'.format(
            elapsed_time))
        
        acc1 = round(acc1.item(), 4)
        acc5 = round(acc5.item(), 4)

        # save the result
        ckpt_name = '{}-{}-{}'.format(arch_name, args.dataset, args.load[:-4])
        save_eval([ckpt_name, acc1, acc5])
        
        return acc1
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'


def train(args, train_loader, epoch, model, criterion, optimizer, **kwargs):
    # fl = open('/root/yujevolume/CoBaL_code/log.txt','a')

    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))
    
    args.checking_epoch=epoch

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        # # for pruning
        if args.prune and args.pruner == 'dpf' and (i+1)%args.prune_freq==0 and (epoch+1)<=args.milestones[1]:
            target_sparsity = args.prune_rate - args.prune_rate * (1 - (epoch+1)/args.milestones[1])**3

            if args.prune_type == 'structured':
                mask_min = math.exp(-(epoch+1)*9/args.milestones[1])
                if (epoch+1) >= args.milestones[1]:
                    mask_min = 0
                # fl.write(f"Epoch: {args.checking_epoch}\n")
                filter_mask = pruning.get_BN_mask(model, target_sparsity, mask_min, args)
                pruning.BN_prune(model, filter_mask, args)

            elif args.prune_type == 'unstructured':
                threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                pruning.weight_prune(model, threshold, args)
        
        output = model(input)
        
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

        end = time.time()
    run['log/loss'].log(loss)
    run['log/acc1'].log(top1.avg)
    run['log/acc5'].log(top5.avg)
    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    # ex.log_scalar('train.loss', losses.avg, epoch)
    # ex.log_scalar('train.top1', top1.avg.item(), epoch)
    # ex.log_scalar('train.top5', top5.avg.item(), epoch)

    return top1.avg, top5.avg


def validate(args, val_loader, epoch, model, criterion):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        output_label = []
        target_label = []
        end = time.time()
        if args.pruner=='static':
            pruning.mones(model, args)
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # f1score
            pred = torch.argmax(output, dim=-1)
            output_label += pred.detach().cpu().tolist()
            target_label += target.detach().cpu().tolist()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()
        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))


    # logging at sacred
    # ex.log_scalar('test.loss', losses.avg, epoch)
    # ex.log_scalar('test.top1', top1.avg.item(), epoch)
    # ex.log_scalar('test.top5', top5.avg.item(), epoch)

    return top1.avg, top5.avg


if __name__ == '__main__':
    start_time = time.time()
    args = hyperparam()
    for k, v in args.__dict__.items():
        run[f'config/{k}']=v
    main(args)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))


def extract_data(model):
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