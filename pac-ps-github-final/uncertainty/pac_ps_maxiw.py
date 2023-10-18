import numpy as np

from learning import *
from uncertainty import *
from .util import *
from .labelshift import *


class PredSetConstructor_maxiw(PredSetConstructor_CP):
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)

        
    def train(self, src_val, tar, dataset_name):
        # this is only for 
        m, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: m = {m}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
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

        n_list = confusion_matrix(yval, ypred_s, num_labels).flatten() * ypred_s.shape[0] 
        n_list = np.array([k if k > 0 else 1 for k in n_list])
        itv_rate = [bci_clopper_pearson(k, int(n_list.sum()), delta / (num_labels*(num_labels+1)+1)) for k in n_list]
        lm = [ii[0] for ii in itv_rate]
        um = [ii[1] for ii in itv_rate]
        lm, um = np.reshape(np.array(lm), [num_labels, num_labels]), np.reshape(np.array(um), [num_labels, num_labels])
        mu_t = calculate_marginal(ypred_t, num_labels) * len(ypred_t)
        mu_t_bound = [bci_clopper_pearson(k, len(ypred_t), delta / (num_labels*(num_labels+1)+1)) for k in mu_t]
        lmq = np.array([mu_t_bound[i][0][0] for i in range(num_labels)])[...,np.newaxis]
        umq = np.array([mu_t_bound[i][1][0] for i in range(num_labels)])[...,np.newaxis]
        lm = np.concatenate([lm, lmq],1)
        um = np.concatenate([um, umq],1)
        g1, g2 = myGauss(lm, um)
        g1, g2 = np.array(g1), np.array(g2)

        self.mdl.eps.data = self.mdl.eps.data / max(g2)
        self.mdl.delta.data = self.mdl.delta.data / (num_labels*(num_labels+1)+1)
        
        super().train(src_val, dataset_name, f_nll_list)

        return True
        
        
    