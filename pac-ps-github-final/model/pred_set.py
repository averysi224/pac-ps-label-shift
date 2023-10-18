import torch as tc
import torch.nn as nn

from .util import *
from tqdm import tqdm
##
## predictive confidence set
##
class PredSet(nn.Module):
    
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0, T_size=None):
        super().__init__()
        self.mdl = mdl
        if T_size == None:
            self.T = nn.Parameter(tc.tensor(0.0))
        else:
            self.T = nn.Parameter(tc.zeros([T_size, 1]))
        self.eps = nn.Parameter(tc.tensor(eps), requires_grad=False)
        self.delta = nn.Parameter(tc.tensor(delta), requires_grad=False)
        self.n = nn.Parameter(tc.tensor(n), requires_grad=False)

    
class PredSetCls(PredSet):
    """
    T \in [0, \infty]
    """
    def __init__(self, mdl, eps=0.0, delta=0.0, n=0, T_size=None):
        super().__init__(mdl, eps, delta, n, T_size)
        self.sft = nn.Softmax(dim=1)

    ## normal forward
    def forward_all(self, x, y=None, e=0.0):
        with tc.no_grad():
            pred_y_soft = self.sft(self.mdl(x))
            pred_y = tc.argmax(pred_y_soft, axis=1)
            logp = pred_y_soft.log() 
            logp = logp + tc.rand_like(logp)*e # break the tie
            if y is not None:
                logp = logp.gather(1, y.view(-1, 1)).squeeze(1)
                        
        return -logp, pred_y_soft, pred_y
    
    ## agnews forward
    def forward_all_ag(self, x, mask, y=None, e=0.0):
        with tc.no_grad():
            pred_y_soft = self.sft(self.mdl(x, mask))
            pred_y = tc.argmax(pred_y_soft, axis=1)
            logp = pred_y_soft.log() 
            logp = logp + tc.rand_like(logp)*e # break the tie
            if y is not None:
                logp = logp.gather(1, y.view(-1, 1)).squeeze(1)
                        
        return -logp, pred_y_soft, pred_y

    ## normal forward
    def forward(self, x, y=None, e=0.0):
        with tc.no_grad():
            logp = self.sft(self.mdl(x)).log()
            logp = logp + tc.rand_like(logp)*e # break the tie
            if y is not None:
                logp = logp.gather(1, y.view(-1, 1)).squeeze(1)
                        
        return -logp

    def forward_ag(self, x, mask, y=None, e=0.0):
        with tc.no_grad():
            logp = self.sft(self.mdl(x, mask)).log()
            logp = logp + tc.rand_like(logp)*e # break the tie
            if y is not None:
                logp = logp.gather(1, y.view(-1, 1)).squeeze(1)
        return -logp

    def set(self, x, mask=None, e=0.0):
        with tc.no_grad():
            if mask is not None:
                nlogp=self.forward_ag(x, mask)
            else:
                nlogp = self.forward(x)
            s = nlogp <= self.T
        return s

    
    def membership(self, x, y, mask=None):
        with tc.no_grad():
            s = self.set(x, mask)
            membership = s.gather(1, y.view(-1, 1)).squeeze(1)
        return membership

    
    def size(self, x, y=None, mask=None):
        with tc.no_grad():
            sz = self.set(x, mask).sum(1).float()
        return sz

    def prepare_iw(self, src_val, tar):
        ypred_s, ypred_s_soft = [], []
        ypred_t, ypred_t_soft = [], []
        yval = []
        f_nll_list = []

        for data in tqdm(src_val):
            x, y = data[0].cuda(), data[1].cuda()
            f_nll_i, f_i, f_hard = self.forward_all(x, y)
            yval.append(y)
            ypred_s.append(f_hard)
            ypred_s_soft.append(f_i)
            f_nll_list.append(f_nll_i)
        for data in tqdm(tar):
            x, y = data[0].cuda(), data[1].cuda()
            _, f_i, f_hard = self.forward_all(x)
            ypred_t.append(f_hard)
            ypred_t_soft.append(f_i)

        yval = tc.cat(yval).cpu().numpy()
        f_nll_list = tc.cat(f_nll_list)
        # Converting to numpy array for later convenience
        ypred_s = tc.cat(ypred_s).cpu().numpy()  #.asnumpy() # hard val 
        ypred_s_soft = tc.cat(ypred_s_soft).cpu().numpy()   #.asnumpy()
        ypred_t = tc.cat(ypred_t).cpu().numpy()   #.asnumpy() # hard test
        ypred_t_soft = tc.cat(ypred_t_soft).cpu().numpy()   #.asnumpy()

        return ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list

    def prepare_iw_chx(self, src_val, tar):
        ypred_s, ypred_s_soft = [], []
        ypred_t, ypred_t_soft = [], []
        yval = []
        f_nll_list = []

        for x, y in tqdm(src_val):
            x, y = x.cuda(), y.cuda().argmax(1)
            bs, cs, c, h, w = x.shape
            x = x.view(-1, c, h, w)
            f_nll_i, f_i, f_hard = self.forward_all(x, y)
            yval.append(y)
            ypred_s.append(f_hard)
            ypred_s_soft.append(f_i)
            f_nll_list.append(f_nll_i)

        for x, y in tqdm(tar): 
            x, y = x.cuda(), y.cuda().argmax(1)
            bs, cs, c, h, w = x.shape
            x = x.view(-1, c, h, w)
            _, f_i, f_hard = self.forward_all(x)
            ypred_t.append(f_hard)
            ypred_t_soft.append(f_i)

        yval = tc.cat(yval).cpu().numpy()
        f_nll_list = tc.cat(f_nll_list)
        # Converting to numpy array for later convenience
        ypred_s = tc.cat(ypred_s).cpu().numpy()  #.asnumpy() # hard val 
        ypred_s_soft = tc.cat(ypred_s_soft).cpu().numpy()   #.asnumpy()
        ypred_t = tc.cat(ypred_t).cpu().numpy()   #.asnumpy() # hard test
        ypred_t_soft = tc.cat(ypred_t_soft).cpu().numpy()   #.asnumpy()

        return ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list
        
    def prepare_iw_ag(self, src_val, tar):
        ypred_s, ypred_s_soft = [], []
        ypred_t, ypred_t_soft = [], []
        yval = []
        f_nll_list = []

        for x, mask, y in tqdm(src_val):
            x, y = x.cuda(), y.cuda()
            mask = mask.cuda()
            f_nll_i, f_i, f_hard = self.forward_all_ag(x, mask, y)
            yval.append(y)
            ypred_s.append(f_hard)
            ypred_s_soft.append(f_i)
            f_nll_list.append(f_nll_i)

        for x, mask, y in tqdm(tar): # TODO: check td validation set
            x, y = x.cuda(), y.cuda()
            mask = mask.cuda()
            _, f_i, f_hard = self.forward_all_ag(x, mask)
            ypred_t.append(f_hard)
            ypred_t_soft.append(f_i)

        yval = tc.cat(yval).cpu().numpy()
        f_nll_list = tc.cat(f_nll_list)
        # Converting to numpy array for later convenience
        ypred_s = tc.cat(ypred_s).cpu().numpy()  #.asnumpy() # hard val 
        ypred_s_soft = tc.cat(ypred_s_soft).cpu().numpy()   #.asnumpy()
        ypred_t = tc.cat(ypred_t).cpu().numpy()   #.asnumpy() # hard test
        ypred_t_soft = tc.cat(ypred_t_soft).cpu().numpy()   #.asnumpy()

        return ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list

    def get_nlog(self, ld, dataset_name, e=0.0):
        f_nll_list = []
        if dataset_name == "Cifar10" or dataset_name == "Heart" or dataset_name == "Entity":
            with tc.no_grad():
                for data in ld:
                    x, y = data[0].cuda(), data[1].cuda()
                    f_nll_i = self.forward(x, y)
                    f_nll_list.append(f_nll_i)
            
        elif dataset_name == "ChestXray":
            for x, y in tqdm(ld):
                x, y = x.cuda(), y.cuda().argmax(1)
                bs, cs, c, h, w = x.shape
                x = x.view(-1, c, h, w)
                f_nll_i = self.forward(x, y)
                f_nll_list.append(f_nll_i)

        elif dataset_name == "AGNews":
            for x, mask, y in tqdm(ld):
                x, y =  x.cuda(), y.cuda()
                mask = mask.cuda()
                f_nll_i = self.forward_ag(x, mask, y, e=e)
                f_nll_list.append(f_nll_i)
        else:
            print("Unknown Dataset!")

        f_nll_list = tc.cat(f_nll_list)
        return f_nll_list