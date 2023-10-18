import numpy as np

#---------------------- utility functions used ----------------------------
def idx2onehot(a,k):
    a=a.astype(int)
    b = np.zeros((a.size, k))
    b[np.arange(a.size), a] = 1
    return b


def confusion_matrix(ytrue, ypred,k):
    # C[i,j] denotes the frequency of ypred = i, ytrue = j.
    n = ytrue.size
    C = np.dot(idx2onehot(ypred,k).T,idx2onehot(ytrue,k))
    return C/n

def confusion_matrix_exclude(ytrue, ypred, target_label):
    # C[i,j] denotes the frequency of ypred = i, ytrue = j.
    n = ytrue.size
    # n_index = (ypred>1)
    # ypred[n_index] = abs(1-ytrue[n_index])
    ypred = (ypred != target_label).astype(int)
    ytrue = (ytrue != target_label).astype(int)
    
    C = np.dot(idx2onehot(ypred,2).T,idx2onehot(ytrue,2))
    return C/n

def confusion_matrix_probabilistic(ytrue, ypred,k):
    # Input is probabilistic classifiers in forms of n by k matrices
    n,d = np.shape(ypred)
    C = np.dot(ypred.T, idx2onehot(ytrue,k))
    return C/n


def calculate_marginal(y,k):
    mu = np.zeros(shape=(k,1))
    for i in range(k):
        mu[i] = np.count_nonzero(y == i)
    return mu/np.size(y)

def calculate_marginal_exclude(y, target_label):
    # target label will be 0, other 1
    y = (y != target_label).astype(int)
    mu = np.zeros(shape=(2,1))
    for i in range(2):
        mu[i] = np.count_nonzero(y == i)
    return mu/np.size(y)

def calculate_marginal_probabilistic(y,k):
    return np.mean(y,axis=0)

def estimate_labelshift_ratio(ytrue_s, ypred_s, ypred_t,k):
    if ypred_s.ndim == 2: # this indicates that it is probabilistic, for soft 
        C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
        mu_t = calculate_marginal_probabilistic(ypred_t, k)
    else:
        C = confusion_matrix(ytrue_s, ypred_s,k)
        mu_t = calculate_marginal(ypred_t, k)
    lamb = (1/min(len(ypred_s),len(ypred_t)))
    wt = np.linalg.solve(np.dot(C.T, C)+lamb*np.eye(k), np.dot(C.T, mu_t))
    return wt

def estimate_target_dist(wt, ytrue_s,k):
    ''' Input:
    - wt:    This is the output of estimate_labelshift_ratio)
    - ytrue_s:      This is the list of true labels from validation set

    Output:
    - An estimation of the true marginal distribution of the target set.
    '''
    mu_t = calculate_marginal(ytrue_s,k)
    return wt*mu_t

# functions that convert beta to w and converge w to a corresponding weight function.
def beta_to_w(beta, y, k):
    w = []
    for i in range(k):
        w.append(np.mean(beta[y.astype(int) == i]))
    w = np.array(w)
    return w

# a function that converts w to beta.
def w_to_beta(w,y):
    return w[y.astype(int)]

def w_to_weightfunc(w):
    return lambda x, y: w[y.astype(int)]

def myGauss(lm, um):
    # eliminate columns
    for col in range(len(lm[0])):
        for row in range(col+1, len(lm)):
            factor_l, factor_u = lm[row][col] / um[col][col], um[row][col] / lm[col][col]
            # assume minus lower
            r_l = [(rowValue * -factor_l) for rowValue in lm[col]]
            # assume minus upper
            r_u = [(rowValue * -factor_u) for rowValue in um[col]]
            # TODO: check eliminated 0
            # modify row m
            um[row] = [sum(pair) for pair in zip(um[row], r_l)]
            um[row][col] = 0
            lm[row] = [sum(pair) for pair in zip(lm[row], r_u)]
            lm[row][col] = 0
    # now backsolve by substitution
    ans_l, ans_u = [], []
    um = um[::-1]
    lm = lm[::-1]
    ###########################
    for sol in range(len(lm)):
        if sol == 0:
            ans_l.append(lm[sol][-1] / um[sol][-2])
            ans_u.append(um[sol][-1] / lm[sol][-2])
        else:
            inner_l, inner_u = 0, 0
            # substitute in all known coefficients
            for x in range(sol):
                inner_l += (ans_l[x]*lm[sol][-2-x])
                inner_u += (ans_u[x]*um[sol][-2-x])
            # the equation is now reduced to ax + b = c form
            # solve with (c - b) / a
            ans_u.append((um[sol][-1]-inner_l)/lm[sol][-sol-2])
            ans_l.append((lm[sol][-1]-inner_u)/um[sol][-sol-2])
    ans_l.reverse()
    ans_u.reverse()
    return ans_l, ans_u
    

#----------------------------------------------------------------------------