import numpy as np
import math
import torch as tc

from learning import *
from uncertainty import *
from .util import *
    

class PredSetConstructor_CP(PredSetConstructor):
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)

        
    def train(self, ld, dataset_name, f_nll_list=None):
        m, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: m = {m}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
        if f_nll_list is None:
            f_nll_list = self.mdl.get_nlog(ld, dataset_name)

        ## line search over T
        T, T_step, T_end, T_opt_nll = 0., self.params.T_step*50, self.params.T_end, np.inf
        while T <= T_end:
            T_nll = -np.log(T).astype(np.float32) if T>0 else np.inf
            ## CP bound
            error_U = (f_nll_list > T_nll).sum().float()
            k_U, n_U, delta_U = error_U.int().item(), len(f_nll_list), delta 
            # print(f'[Clopper-Pearson parametes] k={k_U}, n={n_U}, delta={delta_U}')
            U = bci_clopper_pearson_worst(k_U, n_U, delta_U)

            if U <= eps:
                T_opt_nll = T_nll

            elif k_U > 50 and U >= 1.5*eps: ## no more search if the upper bound is too large
                break

            print(f'[m = {m}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                  f'T_opt = {math.exp(-T_opt_nll):.4f}, #error = {k_U}, error_emp = {k_U/n_U:.4f}, U = {U:.6f}')
            T += T_step        
        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.to(self.params.device)

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
