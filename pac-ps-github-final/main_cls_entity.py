import os, sys
import argparse
import numpy as np

import torchvision.transforms as transforms
from robustness.tools.breeds_helpers import make_entity13, print_dataset_info, ClassHierarchy
from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from tqdm import tqdm
import torch as tc
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

import util
import data
import model
import uncertainty
import pdb

import pandas as pd

class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        # logger.debug(f"IDx recieved {idx}")
        # logger.debug(f"Indices type {type(self.indices[idx])} value {self.indices[idx]}")
        x = self.dataset[self.indices[idx]]

        if self.transform is not None:
            transformed_img = self.transform(x[0])

            return transformed_img, x[1], x[2:]

        else:
            return x

    @property
    def y_array(self):
        return self.dataset.y_array[self.indices]

    def __len__(self):
        return len(self.indices)

def dataset_with_targets(cls):
    """
    Modifies the dataset class to return target
    """

    def y_array(self):
        return np.array(self.targets).astype(int)

    return type(cls.__name__, (cls,), {"y_array": property(y_array)})

def get_entity13(
    root_dir=None,
    transforms=None,
    split_indices=None,
):
    return get_breeds(
        root_dir,
        transforms,
        split_indices,
    )

def get_breeds(
    root_dir=None,
    transforms=None,
    split_indices=None,
):
    root_dir = f"{root_dir}/imagenet/"
    ret = make_entity13(f"{root_dir}/imagenet_hierarchy/", split=None)
    ImageFolder = dataset_with_targets(folder.ImageFolder)

    source_label_mapping = get_label_mapping("custom_imagenet", ret[1][0])
    sourceset = ImageFolder(
        f"{root_dir}/imagenetv1/train/", label_mapping=source_label_mapping
    )
    source_trainset = Subset(
        sourceset, split_indices, transform=transforms,
    )

    return source_trainset

def get_shifted_loader(base_set_y_array, imp, n, base_split_indices, data_transforms):
    imp = imp / np.array(imp).max()
    target_numbers = [int(n*r) for r in imp]
    
    src_val_list = [] 
    for i_label in range(len(imp)):
        class_data_idx = np.where(base_set_y_array == i_label)[0]
        class_label = np.random.choice(class_data_idx, target_numbers[i_label])
        src_val_list.append(base_split_indices[class_label])
    
    src_dataset = get_entity13(
            root_dir='/data1/wenwens/',
            transforms=data_transforms,
            split_indices = np.concatenate(src_val_list),
        )
    print("Calibration size: ", len(src_dataset))
    for i in range(13):
        print((src_dataset.y_array == i).sum(),(src_dataset.y_array == i).sum())
    loader = DataLoader(src_dataset, batch_size=256, shuffle=True, num_workers=2)
    return loader

def run(args):
    # data_dir = '/data1/wenwens/imagenet/imagenetv1/'
    num_classes = 13
    data_transforms = transforms.Compose([
        transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias='warn'),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])]
    )
    
    # Import ResNet50 model pretrained on ImageNet
    mdl = models.resnet50(pretrained=True)
    mdl.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    num_features = mdl.fc.in_features
    mdl.fc = nn.Linear(num_features, num_classes)

    # Move the model to GPU if available
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    mdl = nn.DataParallel(mdl).to(device)
    mdl.load_state_dict(tc.load('snapshots_models/resnet50_entity13.pth')) 

    src_split = np.array(range(1, 334700, 4))
    tar_split = np.array(range(3, 334700, 4))

    src_dataset_unshift = get_entity13(
            root_dir='/data1/wenwens/',
            transforms=data_transforms,
            split_indices = src_split,
        )
    
    tar_dataset_unshift = get_entity13(
            root_dir='/data1/wenwens/',
            transforms=data_transforms,
            split_indices = tar_split,
        )

    src_calloader = get_shifted_loader(src_dataset_unshift.y_array, [1.]*num_classes, 4000, src_split, data_transforms)
    imp_tar = [0.2]+[0.1]*4+[0.3]+[0.1]+[0.2]+[1.2]+[0.1]*4
    tar_calloader = get_shifted_loader(tar_dataset_unshift.y_array, imp_tar, 9000, tar_split, data_transforms)
    tar_testloader = get_shifted_loader(tar_dataset_unshift.y_array, imp_tar, 2000, tar_split, data_transforms)
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
