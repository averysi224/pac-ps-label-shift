import numpy as np
import math

import torch as tc

from learning import *
from uncertainty import *
from .util import *
import pdb
from .labelshift import *
        
class PredSetConstructor_rejection(PredSetConstructor):
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)

        
    def train(self, src_val, tar, dataset_name, wt=None): # td should be source val for computing lambda
        m, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: m = {m}, eps = {eps:.2e}, delta = {delta:.2e}")
        # load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
        if dataset_name == "Cifar10" or dataset_name == "Heart" or dataset_name == "Entity":
            ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list = self.mdl.prepare_iw(src_val, tar)
        elif dataset_name == "ChestXray":
            ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list = self.mdl.prepare_iw_chx(src_val, tar)
        elif dataset_name == "AGNews":
            ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list = self.mdl.prepare_iw_ag(src_val, tar)
        else:
            print("Unknown Dataset!")

        num_labels = ypred_t_soft.shape[1]
        wt = estimate_labelshift_ratio(yval, ypred_s_soft, ypred_t_soft, num_labels)
        print(wt)

        w_list = tc.zeros_like(f_nll_list)
        for i in range(len(w_list)):
            w_list[i] = wt[yval[i]]
        U = tc.rand(len(w_list), device=self.params.device) # sample only once
        ## find the smallest prediction set by line-searching over T
        T, T_step, T_end, T_opt_nll = 0., self.params.T_step*50, self.params.T_end, np.inf
        accept =  U < (w_list / wt.max())
        f_nll_list_tar = f_nll_list[accept]
        
        while T <= T_end:
            T_nll = -math.log(T) if T>0 else np.inf
            ## CP bound
            error_U = (f_nll_list_tar > T_nll).sum().float()
            k_U, n_U, delta_U = error_U.item(), len(f_nll_list_tar), delta
            U_bnd = bci_clopper_pearson_worst(k_U, n_U, delta_U)
            
            ## check condition
            if U_bnd <= eps:
                T_opt_nll = T_nll

            elif k_U >50 and U_bnd >= 1.5*eps: ## no more search if the upper bound is too large
                break

            print(f'[m = {m}, n = {n_U}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                  f'T_opt = {math.exp(-T_opt_nll):.4f}, k = {k_U}, error_UCB = {U_bnd:.4f}')

            T += T_step

        ## save
        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.to(self.params.device)

        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    
