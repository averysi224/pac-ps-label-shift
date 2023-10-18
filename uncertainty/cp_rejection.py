from bdb import set_trace
import os, sys
import numpy as np
import pickle
import torch as tc

from learning import *
from uncertainty import *
from .util import *
from .labelshift import *
from tqdm import tqdm

def compute_quanile_weighted_dist(raw_residuals, weights, alpha):
    if raw_residuals.ndim > 1:
        residuals = raw_residuals.ravel()
    else:
        residuals = np.copy(raw_residuals)
    if 1-alpha > sum(weights):
        return 1
    else:
        # sort residuals in non-decreasing order
        res_order = np.argsort(residuals)
        res_sorted = residuals[res_order]
        weights_ordered = weights[res_order]

        # compute cumulative sums
        cum_weights = np.cumsum(weights_ordered)

        # find the index of the first exceeding value
        index = np.argmax(cum_weights >= 1-alpha)

        return res_sorted[index]

class CP_Constructor(PredSetConstructor):
    # code adapted from WCP author implementation
    def __init__(self, model, params=None, model_iw=None, name_postfix=None):
        super().__init__(model=model, params=params, model_iw=model_iw, name_postfix=name_postfix)

    def get_scores(self, y_true, num_of_preds, prob_sort, cumulative_probs, ranks, U_rand=None):
        # identufy how the true classes were ranked
        true_ranks = np.array([ranks[i, y_true[i]]
                               for i in range(num_of_preds)])

        # get cumulative prob that includes term corresponding to
        # the correct class
        prob_cum = np.array([cumulative_probs[i, true_ranks[i]]
                             for i in range(num_of_preds)]).reshape(-1, 1)

        # get probabilities assigned to the true classes
        prob_for_pred_class = np.array(
            [prob_sort[i, true_ranks[i]] for i in range(num_of_preds)]).reshape(-1, 1)

        more_likely_probs = prob_cum - prob_for_pred_class
        U = np.copy(U_rand)
        scores = more_likely_probs + U * prob_for_pred_class

        return scores

    
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
        ypred_s_soft = ypred_s_soft + 1e-4 * (np.random.uniform(size=ypred_s_soft.shape) - 0.5)
        ypred_t_soft = ypred_t_soft + 1e-4 * (np.random.uniform(size=ypred_t_soft.shape) - 0.5)

        # yval is just the ground truth label of the validation set
        # soft and hard both work well
        wt = estimate_labelshift_ratio(yval, ypred_s, ypred_t, num_labels)
        self.imp_weights = wt
        self.alpha = eps

        U_rand = np.random.uniform(size=[len(ypred_s), 1])
        # sort predicted probabilities for each point in decreasing order
        prob_sort = -np.sort(-ypred_s_soft, axis=1)
        # sort predicted classes for each point from most likely to least likely
        classes_sort = np.argsort(-ypred_s_soft, axis=1)
        # rank classes for each point from most likely to least likely
        # i-th entry in a row corresponds to how likely it was ranked to be the true one
        ranks = np.empty_like(classes_sort)
        for i in range(ypred_s_soft.shape[0]):
            ranks[i, classes_sort[i]] = np.arange(num_labels)
        cumulative_probs = prob_sort.cumsum(axis=1)
        scores = self.get_scores(yval, yval.shape[0], prob_sort, cumulative_probs, ranks, U_rand)
        weights = self.imp_weights[yval]
        self.alpha_corrected = np.zeros_like(self.imp_weights)
        # in this case we have different thresholds for different labels
        for cur_label in range(num_labels):
            cur_weights = weights / (sum(weights) + self.imp_weights[cur_label])

            self.alpha_corrected[cur_label] = compute_quanile_weighted_dist(
                scores, cur_weights, self.alpha)

        self.mdl.T.data = tc.tensor(self.alpha_corrected)
        self.mdl.to(self.params.device)

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
            ypred_t_soft, y_true_t = [], []

            if ld_name == "ChestXray":
                for x, y in tqdm(ld): # td validation set
                    x, y = to_device(x, self.params.device), to_device(y, self.params.device).argmax(1)
                    bs, cs, c, h, w = x.shape
                    x = x.view(-1, c, h, w)
                    _, f_i, f_hard = self.mdl.forward_all(x)
                    ypred_t_soft.append(f_i)
                    y_true_t.append(y)
            if ld_name == "AGNews":
                for x, mask, y in ld: 
                    x, y = to_device(x, self.params.device), to_device(y, self.params.device)
                    mask = to_device(mask, self.params.device)
                    _, f_i, _ = self.mdl.forward_all_ag(x, mask)
                    y_true_t.append(y)
                    ypred_t_soft.append(f_i)
            else:
                for data in ld: # td validation set
                    x, y = to_device(data[0], self.params.device), to_device(data[1], self.params.device)
                    _, f_i, f_hard = self.mdl.forward_all(x)
                    ypred_t_soft.append(f_i)
                    y_true_t.append(y)

            num_labels = ypred_t_soft[0].shape[1]
            y_true_t = tc.cat(y_true_t)
            ypred_t_soft = tc.cat(ypred_t_soft).cpu().numpy()  
            ypred_t_soft = ypred_t_soft + 1e-4 * (np.random.uniform(size=ypred_t_soft.shape) - 0.5)

            U_rand_test = np.random.uniform(size=[len(y_true_t), 1])
            # here I saved numpy cpu array
            error, size = self.evaluate_ps(y_true_t.cpu().numpy(), num_labels, ypred_t_soft, U_rand=U_rand_test)
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

            ## plot results

            
        return size.mean(), error.mean()

    def predict_sets(self, probs, U_rand=None):
        """
        Predict sets on unseen data
        """

        # predict probabilities
        # probs = self.mdl(X_test)

        num_of_preds, num_of_classes = probs.shape

        # sort predicted probabilities for each point in decreasing order
        prob_sort = -np.sort(-probs, axis=1)

        # sort predicted classes for each point in decreasing order from most likely
        classes_sort = np.argsort(-probs, axis=1)

        # get cumulative probs of most likely classes
        cumulative_probs = prob_sort.cumsum(axis=1)

        # get cumulative probs of exceeding (more likely) classes
        more_likely_probs = cumulative_probs - prob_sort

        # ranked more_likely_probs
        # i-th entry in a row corresponds to the the total mass of labels that are more likely than class i
        more_likely_probs_ranked = np.empty_like(more_likely_probs)
        for i in range(num_of_preds):
            more_likely_probs_ranked[i, classes_sort[i]] = more_likely_probs[i]
        U = np.random.uniform(size=[num_of_preds, 1])
        test_points_scores = more_likely_probs_ranked + U * probs
        sets = [np.sort(np.arange(num_of_classes)[cur_prob <= self.alpha_corrected[:,0]]) for cur_prob in test_points_scores]

        return sets

    def evaluate_ps(self, y_test, num_of_classes, probs, U_rand=None):
        """
        Evaluate quality of the PS wrapper
        """
        # predict sets
        pred_sets = self.predict_sets(probs, U_rand)

        num_of_preds = len(pred_sets)

        # evaluate coverage
        error = 1-tc.Tensor([y_test[i] in pred_sets[i]
                            for i in range(num_of_preds)])
        # evaluate size
        length = tc.Tensor([len(cur_set) for cur_set in pred_sets])
        return error, length
        