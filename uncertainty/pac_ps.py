import os
from learning import *
import numpy as np
import pickle
import types

import torch as tc
from .util import *
from tqdm import tqdm

def geb_VC(delta, n, d=1.0):
    n = float(n)
    g = np.sqrt(((np.log((2*n)/d) + 1.0) * d + np.log(4/delta))/n)
    return g

def geb_iw_finite(delta, m, n_C, M, d2_max):
    m, n_C, M, d2_max = float(m), float(n_C), float(M), float(d2_max)
    g = 2.0*M*(np.log(n_C) + np.log(1.0/delta)) / 3.0 / m + np.sqrt( 2.0*d2_max*(np.log(n_C) + np.log(1.0/delta))/m )
    return g

def log_factorial(n):
    #log_f = tc.arange(n, 0, -1).float().log().sum()
    log_f = np.sum(np.log(np.arange(n, 0, -1.0)))
    return log_f

def log_n_choose_k(n, k):
    if k == 0:
        #return tc.tensor(1)
        return 1
    else:
        res = np.sum(np.log(np.arange(n, n-k, -1.0))) - log_factorial(k)
        return res

    
def half_line_bound_upto_k(n, k, eps):
    assert(eps > 0.0)
    ubs = []
    #eps = tc.tensor(eps)
    for i in np.arange(0, k+1):
        bc_log = log_n_choose_k(n, i)
        #log_ub = bc_log + eps.log()*i + (1.0-eps).log()*(n-i)
        #ubs.append(log_ub.exp().unsqueeze(0))
        log_ub = bc_log + np.log(eps)*i + np.log(1.0-eps)*(n-i)
        ubs.append([np.exp(log_ub)])
    ubs = np.concatenate(ubs)
    ub = np.sum(ubs)
    return ub


def binedges_equalmass(x, n_bins):
    n = len(x)
    return np.interp(np.linspace(0, n, n_bins + 1),
                     np.arange(n),
                     np.sort(x))



class PredSetConstructor(BaseLearner):
    def __init__(self, model, params=None, model_iw=None, iw_max=None, name_postfix=None):
        super().__init__(model, params, name_postfix)
        self.mdl_iw = model_iw
        self.iw_max = iw_max
        
        if params:
            base = os.path.join(
                params.snapshot_root,
                params.exp_name,
                f"model_params{'_'+name_postfix if name_postfix else ''}_n_{self.mdl.n.item()}_eps_{self.mdl.eps.item()}_delta_{self.mdl.delta.item()}")
            self.mdl_fn_best = base + '_best'
            self.mdl_fn_final = base + '_final'
            self.mdl.to(self.params.device)
        else:
            self.params = types.SimpleNamespace(device=tc.device('cpu'))

        
    def _compute_error_permissive_VC(self, eps, delta, n):
        g = geb_VC(delta, n)    
        error_per = eps - g
        return round(error_per*n) if error_per >= 0.0 else None
    
    def _compute_error_permissive_direct(self, eps, delta, n):
        k_min = 0
        k_max = n
        bnd_min = half_line_bound_upto_k(n, k_min, eps)
        if bnd_min > delta:
            return None
        assert(bnd_min <= delta)
        k = n
        while True:
            # choose new k
            k_prev = k
            #k = (T(k_min + k_max).float()/2.0).round().long().item()
            k = round(float(k_min + k_max)/2.0)
        
            # terinate condition
            if k == k_prev:
                break
        
            # check whether the current k satisfies the condition
            bnd = half_line_bound_upto_k(n, k, eps)
            if bnd <= delta:
                k_min = k
            else:
                k_max = k

        # confirm that the solution satisfies the condition
        k_best = k_min
        assert(half_line_bound_upto_k(n, k_best, eps) <= delta)
        return k_best

    
    def _find_opt_T(self, ld, n, error_perm):
        nlogp = []
        for x, y in ld:
            x = to_device(x, self.params.device)
            y = to_device(y, self.params.device)
            #x, y = x.to(self.params.device), y.to(self.params.device)
            nlogp_i = self.mdl(x, y)
            nlogp.append(nlogp_i)
        nlogp = tc.cat(nlogp)
        assert(n == len(nlogp))
        nlogp_sorted = nlogp.sort(descending=True)[0]
        T_opt = nlogp_sorted[error_perm]

        return T_opt
              
    def train(self, ld):
        n, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
        
        ## compute permissive error
        if self.params.bnd_type == 'VC':
            error_permissive = self._compute_error_permissive_VC(eps, delta, n)
        elif self.params.bnd_type == 'direct':
            error_permissive = self._compute_error_permissive_direct(eps, delta, n)
        else:
            raise NotImplementedError
        
        if error_permissive is None:
            print("## construction failed: too strict parameters")
            return False
        
        T_opt = self._find_opt_T(ld, n, error_permissive)
        self.mdl.T.data = T_opt
        self.mdl.to(self.params.device)
        print(f"error_permissive = {error_permissive}, T_opt = {T_opt}")

        ## save
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True
        
        
    def test(self, ld, ld_name, verbose=False):

        ## compute set size and error
        fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'stats_pred_set.pk')
        if os.path.exists(fn) and not self.params.rerun:
            res = pickle.load(open(fn, 'rb'))
            error = res['error_test']
            size = res['size_test']
        else:
            size, error = [], []
            with tc.no_grad():
                if ld_name == "AGNews":
                    for x, mask, y in tqdm(ld):
                        x, y =  x.cuda(), y.cuda()
                        mask = mask.cuda()
                        size_i = loss_set_size(x, y, self.mdl, reduction='none', device=self.params.device, mask=mask)['loss']
                        error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device, mask=mask)['loss']
                        size.append(size_i)
                        error.append(error_i)
                elif ld_name == "ChestXray":
                    for x, y in tqdm(ld):
                        x, y = to_device(x, self.params.device), to_device(y, self.params.device).argmax(1)
                        bs, cs, c, h, w = x.shape
                        x = x.view(-1, c, h, w)
                        size_i = loss_set_size(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                        error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                        size.append(size_i)
                        error.append(error_i)
                else:
                    for data in ld:
                        x, y = to_device(data[0], self.params.device), to_device(data[1], self.params.device)
                        size_i = loss_set_size(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                        error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                        size.append(size_i)
                        error.append(error_i)

            size, error = tc.cat(size), tc.cat(error)
            
            pickle.dump({'error_test': error, 'size_test': size, 'n': self.mdl.n, 'eps': self.mdl.eps, 'delta': self.mdl.delta}, open(fn, 'wb'))

        if verbose:
            mn = size.min()
            Q1 = size.kthvalue(int(round(size.size(0)*0.25)))[0]
            Q2 = size.median()
            Q3 = size.kthvalue(int(round(size.size(0)*0.75)))[0]
            mx = size.max()
            av = size.mean()

            print(f'[test: {ld_name}, n = {self.mdl.n.item()}, eps = {self.mdl.eps.item():.2e}, delta = {self.mdl.delta.item():.2e}',  #, T = {(-self.mdl.T.data).exp():.5f}] '
                f'error = {error.mean():.4f}, min = {mn}, 1st-Q = {Q1}, median = {Q2}, 3rd-Q = {Q3}, max = {mx}, mean = {av:.2f}'
            )
        return size.mean(), error.mean()

