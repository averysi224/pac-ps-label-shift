import os, sys
import argparse
import numpy as np

import torch as tc
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import util
import data
import model
import uncertainty
import pdb

import pandas as pd

def get_loader(image_names_all, labels_all, target_numbers, tfm):
    accept = [] 
    labels_all_num = np.argmax(labels_all, 1)
    for i_label in range(6):
        class_data_idx = np.where(labels_all_num == i_label)[0]
        class_label = np.random.choice(class_data_idx, target_numbers[i_label])
        accept.append(class_label)
    accept = np.concatenate(accept, 0)
    train_image_names, train_labels = image_names_all[accept], labels_all[accept]
    tar_set = data.ChestXray14Dataset(train_image_names, train_labels, tfm, '/data1/wenwens/CXR8/images/images/', 224, percentage=1)
    loader = DataLoader(tar_set, batch_size=64, num_workers=6)
    return loader


def run(args):
    test_df = pd.read_csv('data/chestxray_single/test_list_single.csv', header=None, sep=' ')
    test_image_names = test_df.iloc[:, 0].values
    test_labels = test_df.iloc[:, 1:].values

    val_df = pd.read_csv('data/chestxray_single/val_list_single.csv', header=None, sep=' ')
    val_image_names = val_df.iloc[:, 0].values
    val_labels = val_df.iloc[:, 1:].values

    image_names_all = np.concatenate([test_image_names, val_image_names], 0)
    labels_all = np.concatenate((test_labels, val_labels),0)

    labels_all = labels_all[:,[0,2,3,4,5,7]]
    valid = labels_all.sum(1) > 0
    image_names_all = image_names_all[valid]
    labels_all = labels_all[valid]
    
    src_split_indices = np.random.choice(range(len(labels_all)), len(labels_all)//2, replace=False)
    tar_split_indices = list(set(range(len(labels_all))) - set(src_split_indices))

    src_image_names_all = image_names_all[src_split_indices]
    src_labels_all = labels_all[src_split_indices]
    tar_image_names_all = image_names_all[tar_split_indices]
    tar_labels_all = labels_all[tar_split_indices]

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: tc.stack([toTensor(crop) for crop in crops])),
        transforms.Lambda(lambda crops: tc.stack([normalize(crop) for crop in crops]))
    ])
    
    src_calloader = get_loader(src_image_names_all, src_labels_all, [12800]*4+[3200,12800], tfm)
    tar_calloader = get_loader(tar_image_names_all, tar_labels_all, [3200]*4+[19200,3200], tfm)
    tar_testloader = get_loader(tar_image_names_all, tar_labels_all, [320]*4+[1920,320], tfm)

    mdl = model.ChexNet(trained=True).cuda()
    mdl = tc.nn.DataParallel(mdl).cuda()
    mdl.load_state_dict(tc.load('snapshots_models/chestxray_new251.pth'))
    mdl.eval()
    
    if args.train_predset.method == 'pac_predset_CP':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_CP(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, dataset_name=args.data.tar)
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)

    elif args.train_predset.method == 'pac_predset_rejection':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, tar_calloader, dataset_name=args.data.tar)   # src_train, src_val, tar
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)
    
    elif args.train_predset.method == 'pac_predset_worst_rejection':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_worst_rejection(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, tar_calloader, dataset_name=args.data.tar)
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)
    # else:
    #     raise NotImplementedError
    elif args.train_predset.method == 'pac_ps_maxiw':
        # iw_max baseline
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_maxiw(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_calloader, tar_calloader, dataset_name=args.data.tar)   # src_train, src_val, tar
        # evaluate
        l.test(tar_testloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'cp_ls':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m, T_size=6)
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
