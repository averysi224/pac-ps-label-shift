import numpy as np
import torch as tc

from learning import *
from uncertainty import *
from .util import *
from .labelshift import *
import pdb

class PredSetConstructor_worst_rejection(PredSetConstructor):
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)
        
    def train(self, src_val, tar, dataset_name): # td should be source val for computing lambda
        m, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: m = {m}, eps = {eps:.2e}, delta = {delta:.2e}")
        # load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True

        ## precompute -log f(y|x) and w(x)
        
        if dataset_name == "Cifar10" or dataset_name == "Heart" or dataset_name == "Entity":
            ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list = self.mdl.prepare_iw(src_val, tar)
        elif dataset_name == "ChestXray":
            ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list = self.mdl.prepare_iw_chx(src_val, tar)
        elif dataset_name == "AGNews":
            ypred_s, ypred_s_soft, ypred_t, ypred_t_soft, yval, f_nll_list = self.mdl.prepare_iw_ag(src_val, tar)
        else:
            print("Unknown Dataset!")
        num_labels = ypred_t_soft.shape[1]
        
        n_list = confusion_matrix(yval, ypred_s, num_labels).flatten() * ypred_s.shape[0] 
        n_list = np.array([k if k > 0 else 1 for k in n_list])
        itv_rate = [bci_clopper_pearson(k, int(n_list.sum()), delta /(num_labels*(num_labels+1)+1)) for k in n_list]
        lm = [ii[0] for ii in itv_rate]
        um = [ii[1] for ii in itv_rate]
        lm, um = np.reshape(np.array(lm), [num_labels, num_labels]), np.reshape(np.array(um), [num_labels, num_labels])
        mu_t = calculate_marginal(ypred_t, num_labels) * len(ypred_t)
        mu_t_bound = [bci_clopper_pearson(k, len(ypred_t), delta /(num_labels*(num_labels+1)+1)) for k in mu_t]
        print(mu_t_bound)
        lmq = np.array([mu_t_bound[i][0][0] for i in range(num_labels)])[...,np.newaxis]
        umq = np.array([mu_t_bound[i][1][0] for i in range(num_labels)])[...,np.newaxis]
        lm = np.concatenate([lm, lmq],1)
        um = np.concatenate([um, umq],1)
        g1, g2 = myGauss(lm, um)
        g1, g2 = tc.Tensor(np.array(g1)[..., np.newaxis]).cuda(), tc.Tensor(np.array(g2)[..., np.newaxis]).cuda()
        ## save
        print(g1,g2)
        yval_cuda = tc.Tensor(yval).cuda()    
        w_list = tc.zeros_like(f_nll_list)

        U = tc.rand(yval.shape, device=self.params.device) # sample only once
        T, T_step, T_end, T_opt_nll = 0., self.params.T_step*5, self.params.T_end, np.inf
        while T <= T_end:
            T_nll = -math.log(T) if T>0 else np.inf
            ## find the worst IW
            i_err = f_nll_list > T_nll

            for i_label in range(num_labels):
                w_list[i_err * (yval_cuda == i_label)] = g2[i_label]
                w_list[~i_err * (yval_cuda == i_label)] = g1[i_label]

            iw_max = w_list.max()
            ## run rejection sampling for target labeled examples
            i_accept = U <= (w_list/iw_max)

            f_nll_list_tar = f_nll_list[i_accept]
            error_U = (f_nll_list_tar > T_nll).sum().float()
            k_U, n_U, delta_U = error_U.item(), len(f_nll_list_tar), delta / (num_labels*(num_labels+1)+1)
            U_bnd = bci_clopper_pearson_worst(k_U, n_U, delta_U)        
            print(f'[m = {m}, n = {n_U}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                f'T_opt = {math.exp(-T_opt_nll):.4f}, k = {k_U}, error_UCB = {U_bnd:.4f}')        
            ## check condition
            if U_bnd <= eps:
                T_opt_nll = T_nll

            elif k_U >50 and U_bnd >= 1.5*eps: ## no more search if the upper bound is too large
                break

            T += T_step

        print(f'[m = {m}, n = {n_U}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                    f'T_opt = {math.exp(-T_opt_nll):.4f}, k = {k_U}, error_UCB = {U_bnd:.4f}')

        ## save
        self.mdl.T.data = tc.Tensor([T_opt_nll])
        self.mdl.to(self.params.device)

        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    