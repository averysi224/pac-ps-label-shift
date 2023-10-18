import os, sys
import argparse
from turtle import color
import numpy as np
import torch as tc

import util
import data
import model
import uncertainty
import pdb

device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')
    
def run(args):
    # 0.1 0.9 is the correct one
    ds_src = data.Heart(root='/data5/wenwens/heart', imp=[0.5]*2, n=150, m=150, part=0)
    ds_tar = data.Heart(root='/data5/wenwens/heart', imp=[0.1, 0.9], n=150, m=150, part=1)
    ds_test = data.Heart(root='/data5/wenwens/heart', imp=[0.1, 0.9], n=75, m=75, part=1)

    src_valloader = ds_src.train
    tar_testloader = ds_tar.train
    test_testloader = ds_test.train
    
    r1, r2 = ds_src.ratio, ds_tar.ratio
    wt = np.array([r2[0]/r1[0], r2[1]/r1[1]])

    mdl = model.CDCNet()
    mdl = tc.nn.DataParallel(mdl).cuda()
    mdl.load_state_dict(tc.load("snapshots_models/cdc_cls_model.pth"))
    mdl.eval()

    if args.train_predset.method == 'pac_predset_CP':
        # construct a prediction set
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_CP(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_valloader, dataset_name=args.data.tar)
        # evaluate
        l.test(test_testloader, ld_name=args.data.tar, verbose=True)

    elif args.train_predset.method == 'pac_predset_rejection':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_rejection(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_valloader, tar_testloader, dataset_name=args.data.tar)   # src_train, src_val, tar
        # evaluate
        l.test(test_testloader, ld_name=args.data.tar, verbose=True)
    
    elif args.train_predset.method == 'pac_predset_worst_rejection':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_worst_rejection(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_valloader, tar_testloader, dataset_name=args.data.tar)
        # l.train(ds_src.val)  
        # evaluate
        l.test(test_testloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'pac_ps_maxiw':
        # iw_max baseline
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m)
        l = uncertainty.PredSetConstructor_maxiw(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_valloader, tar_testloader, dataset_name=args.data.tar)   # src_train, src_val, tar
        # evaluate
        l.test(test_testloader, ld_name=args.data.tar, verbose=True)
    elif args.train_predset.method == 'cp_ls':
        mdl_predset = model.PredSetCls(mdl, args.model_predset.eps, args.model_predset.delta, args.model_predset.m, T_size=2)
        l = uncertainty.CP_Constructor(mdl_predset, args.train_predset, model_iw=None)
        l.train(src_valloader, tar_testloader, dataset_name=args.data.tar) 
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


