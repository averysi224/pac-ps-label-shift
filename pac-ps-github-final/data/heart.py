import os
import pickle
import numpy as np
import pandas

import torch as tc
from torch.utils.data import DataLoader

DATAFILES = [
    'LLCP2011.XPT',
    'LLCP2012.XPT',
    'LLCP2013.XPT',
    'LLCP2014.XPT',
    'LLCP2015.XPT',
    'LLCP2016.XPT',
    'LLCP2017.XPT',
    'LLCP2018.XPT',
    'LLCP2019.XPT',
    'LLCP2020.XPT',
]

STATEKEY = '_STATE'
LABELKEY = 'CVDINFR4' # Ever Diagnosed with Heart Attack

def load_data(root, fn='data.pk'):
    if os.path.exists(os.path.join(root, fn)):
        return pickle.load(open(os.path.join(root, fn), 'rb'))
    
    years = [fn[4:8] for fn in DATAFILES]

    data = [pandas.read_sas(os.path.join(root, fn)) for fn in DATAFILES]
    keys = [set(d.keys()) for d in data]

    # find common keys
    common_keys = set.intersection(*keys)

    # remove unnecessary keys: IDATE, IYEAR, IMONTH, IDAY, SEQNO
    common_keys = common_keys.difference({'IDATE', 'IYEAR', 'IMONTH', 'IDAY', 'SEQNO'})
    
    common_keys = list(common_keys)
    # print(f'n_common_keys = {len(common_keys)}')
    # print('common_keys =', common_keys)
    
    # remove keys with many nan
    common_val_keys = []
    for k in common_keys:
        rate_inval = np.mean([np.isnan(d[k].to_numpy()).mean() for d in data])
        if rate_inval <= 0.05:
            common_val_keys.append(k)
    print(f'n_common_val_keys = {len(common_val_keys)}')
    print('common_val_keys =', common_val_keys)
    assert(STATEKEY in common_val_keys)
    assert(LABELKEY in common_val_keys)

    # remove data with nan
    data_common_val = [d[common_val_keys].to_numpy() for d in data]
    data_common_val = [d[~np.isnan(d.sum(1))] for d in data_common_val]
    
    # remove idk labels
    ind_label = np.argmax(np.array(common_val_keys) == LABELKEY)
    data_common_val = [d[(d[:, ind_label] == 1) | (d[:, ind_label] == 2)] for d in data_common_val]
    for i, d in enumerate(data_common_val):
        d[d[:, ind_label]==2, ind_label] = 0 # 1: yes, 0: no
        data_common_val[i] = d
    
    # return
    ind_state = np.argmax(np.array(common_val_keys) == STATEKEY)
    assert(ind_state != ind_label)
    states_per_year = [d[:, ind_state].astype(int) for d in data_common_val]
    labels_per_year = [d[:, ind_label].astype(int) for d in data_common_val]
    examples_per_year = [np.delete(d, [ind_state, ind_label], axis=1) for d in data_common_val]

    def get_task_ids(states_per_year):
        strid2id = {}
        cnt = 0
        task_ids = []
        for i, states in enumerate(states_per_year):
            task_ids_i = []
            for state in states:
                strid = f'{i}_{state}'
                if strid in strid2id:
                    pass
                else:
                    strid2id[strid] = cnt
                    cnt = cnt + 1
                task_ids_i.append(strid2id[strid])
            task_ids.append(task_ids_i)
        return task_ids
    
    #task_ids_per_year = get_task_ids(states_per_year)
    
    assert(all([x.shape[0] ==  y.shape[0] for x, y in zip(examples_per_year, labels_per_year)]))
    assert(all([~np.isnan(e.sum()) for e in examples_per_year]))

    for yr, y, s in zip(years, labels_per_year, states_per_year):
        print(f'[year = {yr}] n_states = {len(set(s))}, n_examples = {len(y)}, '
              f'n_pos = {(y==1).sum()} ({float((y==1).sum())/float(len(y))*100.0:.2f}%), n_neg = {(y==0).sum()} ({float((y==0).sum())/float(len(y))*100.0:.2f}%)')

    pickle.dump((years, states_per_year, examples_per_year, labels_per_year), open(os.path.join(root, fn), 'wb'))

    return years, states_per_year, examples_per_year, labels_per_year

    

class HeartDataset:

    def __init__(self, x, y, mean, std):
        self.x = np.concatenate(x, 0)
        self.y = np.concatenate(y, 0)
        self.mean = mean
        self.std = std

        
    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        x = (x - self.mean) / self.std
        x = tc.tensor(x).float()
        y = tc.tensor(y).long()
        return x, y


class YearStateSampler:

    def __init__(self, state, n_ways, n_datasets, n_shots):
        self.n_datasets = n_datasets
        self.n_ways = n_ways
        self.batch_size = n_shots # could be n_shots_adapt + n_shots_{train,val,test}
        
        self.ind2yearstate = []
        for ind_year, s_year in enumerate(state):
            for s in s_year:
                self.ind2yearstate.append(f'{ind_year}_{s}')
        self.ind2yearstate = np.array(self.ind2yearstate)
                
        self.yearstate2ind = []
        for yearstate in set(self.ind2yearstate):
            ind = np.argwhere(self.ind2yearstate == yearstate).reshape(-1)
            ind = tc.from_numpy(ind)
            if len(ind) >= self.batch_size*self.n_ways:
                self.yearstate2ind.append(ind)

        assert len(self.yearstate2ind) >= self.n_datasets, f'should hold {len(self.yearstate2ind)} >= {self.n_datasets}'

        
    def __len__(self):
        return self.n_datasets

    
    def __iter__(self):
        # tc.manual_seed(0)
        # np.random.seed(0)
        for _ in range(self.n_datasets):
            ind_yearstate = tc.randint(len(self.yearstate2ind), (1,)).item()
            # print(ind_yearstate)
            ind = self.yearstate2ind[ind_yearstate]            
            ind_batch = ind[tc.randperm(len(ind))[:self.batch_size*self.n_ways]]
            
            yield ind_batch
            
            
    
class Heart:
    def __init__(self, root, split_ratio={'train': 0.7, 'val': 0.3, 'test': 0.1}, part=0, imp=[0.5, 0.5], n=500, m=200):
        n_ways = 2
        num_workers = 4
        seed = 1

        # load and split data
        years, states_per_year, examples_per_year, labels_per_year = load_data(root)
        n_total = len(years)
        n_train = int(n_total * split_ratio['train'])
        n_val = int(n_total * split_ratio['val'])
        
        states_per_year = np.split(states_per_year, [n_train, n_train+n_val])
        examples_per_year = np.split(examples_per_year, [n_train, n_train+n_val])
        labels_per_year = np.split(labels_per_year, [n_train, n_train+n_val])

        states_per_year = states_per_year[part]
        examples_per_year = examples_per_year[part]
        labels_per_year = labels_per_year[part]

        def split(batch, n_samples_adapt):
            x = tc.stack([x for x, y in batch], 0)
            y = tc.stack([y for x, y in batch], 0)

            x = x[n_samples_adapt:]
            y = y[n_samples_adapt:]
            return x, y

        def get_mean_std(examples):
            d = np.concatenate(examples_per_year, 0)
            m = np.mean(d, axis=0).astype(float)
            s = np.std(d, axis=0).astype(float)
            return m, s
        
        states_train, examples_train, labels_train = [], [], []
        self.imp = imp / np.array(imp).max()

        tar_num = [[], []]
        sum0, sum1 = 0, 0
        for yy in range(len(labels_per_year)):
            tar_num[0].append(int(np.sum(labels_per_year[yy] == 0)*self.imp[0]))
            tar_num[1].append(int(np.sum(labels_per_year[yy] == 1)*self.imp[1]))
            sum0 += int(np.sum(labels_per_year[yy] == 0)*self.imp[0])
            sum1 += int(np.sum(labels_per_year[yy] == 1)*self.imp[1])

        self.ratio = (sum0/(sum0+sum1), sum1/(sum0+sum1))

        accepts = []
        nn_train = 7 if part == 0 else 3
        for yy in range(nn_train):
            y_label = []
            for i_label in range(2):
                class_data_idx = np.where(labels_per_year[yy] == i_label)[0]
                class_label = np.random.choice(class_data_idx, tar_num[i_label][yy])
                y_label.append(class_label)
            accepts.append(np.concatenate(y_label, 0))

        # remove half negative
        for ii in range(nn_train):
            states_train.append(states_per_year[ii][accepts[ii]])
            examples_train.append(examples_per_year[ii][accepts[ii]])
            labels_train.append(labels_per_year[ii][accepts[ii]])

        mean_train, std_train = get_mean_std(examples_per_year)
        ds = HeartDataset(examples_train, labels_train, mean_train, std_train)
        sampler = YearStateSampler(states_train, n_ways, n, m)
        self.train = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, 10*n_ways), num_workers=num_workers, pin_memory=True)


    


