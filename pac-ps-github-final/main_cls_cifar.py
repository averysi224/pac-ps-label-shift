import os, sys
import argparse
from turtle import color
import numpy as np
import pdb

import torch as tc
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, TensorDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import util
import data
import model
import uncertainty
import matplotlib.pyplot as plt

def run(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    batch_size = 200
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    trainset = tc.utils.data.Subset(trainset, range(30001, 50000))
    testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)
    fullset = ConcatDataset([trainset, testset])
    full_set_len = len(fullset)
    src_base_set, tar_base_set = random_split(fullset, [int(full_set_len*0.5), full_set_len - int(full_set_len*0.5)])
    src_loader_unshifted = tc.utils.data.DataLoader(src_base_set, batch_size=batch_size, shuffle=True, num_workers=2)
    tar_loader_unshifted = tc.utils.data.DataLoader(tar_base_set, batch_size=32, shuffle=True, num_workers=2)

    src_x, src_y = [], []
    for x, y in src_loader_unshifted:
        src_x.append(x)
        src_y.append(y)
    src_x = np.concatenate(src_x)
    src_y = np.concatenate(src_y)

    # sample source cal
    imp = [0.1]*10
    imp = imp / np.array(imp).max()
    n = 2700 
    target_numbers = [int(n*r) for r in imp]
    accept = [] 
    for i_label in range(10):
        class_data_idx = np.where(src_y == i_label)[0]
        class_label = np.random.choice(class_data_idx, target_numbers[i_label])
        accept.append(class_label)
    
    accept = np.concatenate(accept)
    src_x = tc.Tensor(src_x[accept])
    src_y = tc.Tensor(src_y[accept]).long()
    src_cal = TensorDataset(src_x, src_y)
    print("Source calibration size: ", len(src_cal))
    # debug with same val and test
    src_calloader = tc.utils.data.DataLoader(src_cal, batch_size=batch_size, shuffle=False, num_workers=2)

    tar_x, tar_y = [], []
    for x, y in tar_loader_unshifted:
        tar_x.append(x)
        tar_y.append(y)
    tar_x = np.concatenate(tar_x)
    tar_y = np.concatenate(tar_y)
    # sample tar cal, large shifts
    imp = [0.1]*3 + [0.6]+[0.1]*6
    imp = imp / np.array(imp).max()
    n = 8000   
    target_numbers = [int(n*r) for r in imp]
    accept = [] 
    for i_label in range(10):
        class_data_idx = np.where(tar_y == i_label)[0]
        class_label = np.random.choice(class_data_idx, target_numbers[i_label])
        accept.append(class_label)
    accept = np.concatenate(accept)
    tar_x1 = tc.Tensor(tar_x[accept])
    tar_y1 = tc.Tensor(tar_y[accept]).long()
    tar_cal = TensorDataset(tar_x1, tar_y1)
    print("Target calibration size: ", len(tar_cal))
    tar_calloader = tc.utils.data.DataLoader(tar_cal, batch_size=batch_size, shuffle=False, num_workers=2)

    # sample tar test
    accept = [] 
    for i_label in range(10):
        class_data_idx = np.where(tar_y == i_label)[0]
        class_label = np.random.choice(class_data_idx, target_numbers[i_label])
        accept.append(class_label)
    
    accept = np.concatenate(accept)
    tar_x = tc.Tensor(tar_x[accept])
    tar_y = tc.Tensor(tar_y[accept]).long()
    tar_test = TensorDataset(tar_x, tar_y)
    print("Target testset size: ", len(tar_test))
    # debug with same val and test
    tar_testloader = tc.utils.data.DataLoader(tar_test, batch_size=batch_size, shuffle=False, num_workers=2)

    mdl = models.resnet50(pretrained=False)
    mdl.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify the final fully connected layer according to the number of classes
    num_features = mdl.fc.in_features
    mdl.fc = nn.Linear(num_features, 10)
    mdl = mdl.cuda()

    mdl.load_state_dict(tc.load('snapshots_models/resnet50_cifar10_56.pth'))
    mdl.eval()
    wt = None
    if args.train_predset.method == 'pac_predset_CP':
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_CP(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, dataset_name=args.data.tar)
        # evaluate
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'pac_predset_rejection':
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=None)
        l.train(tar_testloader, tar_calloader, dataset_name=args.data.tar, wt=None)   # src_train, src_val, tar
        # evaluate
        l.test(tar_calloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'pac_ps_maxiw':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_maxiw(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, tar_calloader, dataset_name=args.data.tar)   # src_train, src_val, tar
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'pac_predset_worst_rejection':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_worst_rejection(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, tar_calloader, dataset_name=args.data.tar)
        # evaluate
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'cp_ls':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m, T_size=10)
        l = uncertainty.CP_Constructor(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, tar_calloader, dataset_name=args.data.tar) 
        # evaluate
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)

def parse_args():
    ## init a parser
    parser = argparse.ArgumentParser(description='learning')

    ## meta args
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--train_iw', action='store_true')
    #parser.add_argument('--estimate', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=100)
    parser.add_argument('--data.n_workers', type=int, default=0)
    parser.add_argument('--data.src', type=str, required=True)
    parser.add_argument('--data.tar', type=str, required=True)
    parser.add_argument('--data.n_labels', type=int)
    parser.add_argument('--data.img_size', type=int, nargs=3)
    parser.add_argument('--data.dim', type=int, nargs='*')
    parser.add_argument('--data.aug_src', type=str, nargs='*')
    parser.add_argument('--data.aug_tar', type=str, nargs='*')
    parser.add_argument('--data.n_train_src', type=int)
    parser.add_argument('--data.n_train_tar', type=int)
    parser.add_argument('--data.n_val_src', type=int)
    parser.add_argument('--data.n_val_tar', type=int)
    parser.add_argument('--data.n_test_src', type=int)
    parser.add_argument('--data.n_test_tar', type=int)
    parser.add_argument('--data.seed', type=lambda v: None if v=='None' else int(v), default=0)
    parser.add_argument('--data.load_feat', type=str)

    ## model args
    parser.add_argument('--model.base', type=str)
    parser.add_argument('--model.base_feat', type=str)
    parser.add_argument('--model.path_pretrained', type=str)
    parser.add_argument('--model.feat_dim', type=int)
    parser.add_argument('--model.cal', type=str, default='Temp')
    parser.add_argument('--model.sd', type=str, default='MidFNN')
    parser.add_argument('--model.sd_cal', type=str, default='HistBin')
    parser.add_argument('--model.normalize', action='store_true')
    parser.add_argument('--model.iw_true', action='store_true')

    parser.add_argument('--model_sd.path_pretrained', type=str)
    parser.add_argument('--model_iwcal.n_bins', type=int, default=10) ## can be changed depending on binning scheme
    parser.add_argument('--model_iwcal.delta', type=float)

    ## predset model args
    parser.add_argument('--model_predset.eps', type=float, default=0.1)
    parser.add_argument('--model_predset.alpha', type=float, default=0.1)
    parser.add_argument('--model_predset.delta', type=float, default=1e-5)
    parser.add_argument('--model_predset.m', type=int)

    ## train args
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')

    ## calibration args
    parser.add_argument('--cal.method', type=str, default='HistBin')
    parser.add_argument('--cal.rerun', action='store_true')
    parser.add_argument('--cal.load_final', action='store_true')
    parser.add_argument('--cal.optimizer', type=str, default='SGD')
    parser.add_argument('--cal.n_epochs', type=int, default=100)
    parser.add_argument('--cal.lr', type=float, default=0.01)
    parser.add_argument('--cal.momentum', type=float, default=0.9)
    parser.add_argument('--cal.weight_decay', type=float, default=0.0)
    parser.add_argument('--cal.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--cal.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--cal.val_period', type=int, default=1)    

    ## train args for a source discriminator
    parser.add_argument('--train_sd.rerun', action='store_true')
    parser.add_argument('--train_sd.load_final', action='store_true')
    parser.add_argument('--train_sd.optimizer', type=str, default='SGD')
    parser.add_argument('--train_sd.n_epochs', type=int, default=100)
    parser.add_argument('--train_sd.lr', type=float, default=0.01)
    parser.add_argument('--train_sd.momentum', type=float, default=0.9)
    parser.add_argument('--train_sd.weight_decay', type=float, default=0.0)
    parser.add_argument('--train_sd.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train_sd.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--train_sd.val_period', type=int, default=1)

    ## calibration args for a source discriminator
    parser.add_argument('--cal_sd.method', type=str, default='HistBin')
    parser.add_argument('--cal_sd.rerun', action='store_true')
    parser.add_argument('--cal_sd.resume', action='store_true')
    parser.add_argument('--cal_sd.load_final', action='store_true')
    ## histbin parameters
    parser.add_argument('--cal_sd.delta', type=float, default=1e-5)
    parser.add_argument('--cal_sd.estimate_rate', action='store_true')
    parser.add_argument('--cal_sd.cal_target', type=int, default=1)
    ## temp parameters
    parser.add_argument('--cal_sd.optimizer', type=str, default='SGD')
    parser.add_argument('--cal_sd.n_epochs', type=int, default=100) 
    parser.add_argument('--cal_sd.lr', type=float, default=0.01)
    parser.add_argument('--cal_sd.momentum', type=float, default=0.9)
    parser.add_argument('--cal_sd.weight_decay', type=float, default=0.0)
    parser.add_argument('--cal_sd.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--cal_sd.lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--cal_sd.val_period', type=int, default=1)    

    ## iw calibration args
    parser.add_argument('--cal_iw.method', type=str, default='HistBin')
    parser.add_argument('--cal_iw.rerun', action='store_true')
    parser.add_argument('--cal_iw.load_final', action='store_true')
    parser.add_argument('--cal_iw.smoothness_bound', type=float, default=0.001)

    ## uncertainty estimation args
    parser.add_argument('--train_predset.method', type=str, default='pac_predset')
    parser.add_argument('--train_predset.rerun', action='store_true')
    parser.add_argument('--train_predset.load_final', action='store_true')
    parser.add_argument('--train_predset.binary_search', action='store_true')
    parser.add_argument('--train_predset.bnd_type', type=str, default='direct')

    parser.add_argument('--train_predset.T_step', type=float, default=1e-7) 
    parser.add_argument('--train_predset.T_end', type=float, default=np.inf)
    parser.add_argument('--train_predset.eps_tol', type=float, default=1.5)
        
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')
    
    ## set loggers
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))
    
    ## print args
    util.print_args(args)
    
    return args    
    

if __name__ == '__main__':
    args = parse_args()
    run(args)


