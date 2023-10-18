import numpy as np

import torch as tc
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from pathlib import Path

import sys
sys.path.append('..')

import data
import model
import pandas as pd
# device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')

    
test_df = pd.read_csv('../data/chestxray_single/test_list_single.csv', header=None, sep=' ')
test_image_names = test_df.iloc[:, 0].values
test_labels = test_df.iloc[:, 1:].values

val_df = pd.read_csv('../data/chestxray_single/val_list_single.csv', header=None, sep=' ')
val_image_names = val_df.iloc[:, 0].values
val_labels = val_df.iloc[:, 1:].values

train_df = pd.read_csv('../data/chestxray_single/train_list_single.csv', header=None, sep=' ')
train_image_names = train_df.iloc[:, 0].values
train_labels = train_df.iloc[:, 1:].values

labels_all = train_labels[:,[0,2,3,4,5,7]]
valid = labels_all.sum(1) > 0
image_names_all = train_image_names[valid]
labels_all = labels_all[valid]
labels_all_num = np.argmax(labels_all, 1)

imp_src = tc.Tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 
imp_src = imp_src / imp_src.max()
target_numbers = [int(1600*r) for r in imp_src] # 25200
accept = [] 
for i_label in range(6):
    class_data_idx = np.where(labels_all_num == i_label)[0]
    class_label = np.random.choice(class_data_idx, target_numbers[i_label])
    accept.append(class_label)
accept = np.concatenate(accept, 0)
train_image_names, train_labels = image_names_all[accept], labels_all[accept]

image_names_all = np.concatenate([test_image_names, val_image_names], 0)
labels_all = np.concatenate((test_labels, val_labels),0)
labels_all_num = np.argmax(labels_all, 1)
target_numbers = [int(320*r) for r in imp_src] # 25200
accept = [] 
for i_label in range(6):
    class_data_idx = np.where(labels_all_num == i_label)[0]
    class_label = np.random.choice(class_data_idx, target_numbers[i_label])
    accept.append(class_label)
accept = np.concatenate(accept, 0)
test_image_names, test_labels = image_names_all[accept], labels_all[accept]

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
toTensor = transforms.ToTensor()

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: tc.stack([toTensor(crop) for crop in crops])),
    transforms.Lambda(lambda crops: tc.stack([normalize(crop) for crop in crops]))
])

mdl = model.ChexNet(trained=True, path=Path('../snapshots_models'))
mdl = tc.nn.DataParallel(mdl).cuda()

tar_set = data.ChestXray14Dataset(test_image_names, test_labels, tfm, '/data1/wenwens/CXR8/images/images/', 224, percentage=1)
src_set = data.ChestXray14Dataset(train_image_names, train_labels, tfm, '/data1/wenwens/CXR8/images/images/', 224, percentage=1)

trainloader = DataLoader(src_set, batch_size=16, num_workers=6)
valloader = DataLoader(tar_set, batch_size=16, num_workers=6)

optimizer = tc.optim.SGD(mdl.parameters(), lr=0.0003, momentum=0.9, weight_decay=0.0005)
cnt, cnt_all = tc.zeros([6]).cuda(), tc.zeros([6]).cuda()
print(len(train_labels))

for epoch in range(4):
    loss_cul = 0
    cnt = 0
    for i, (x, y) in enumerate(trainloader):
        x, y = x.cuda(), y.cuda().argmax(1)
        bs, cs, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        optimizer.zero_grad()
        output = mdl(x)
        cnt += (output.argmax(1) ==y).sum()
        loss = tc.nn.functional.cross_entropy(output, y)
        loss_cul += loss.detach().cpu()
        loss.backward()
        optimizer.step()

    print("Epoch ", epoch, loss_cul/ len(trainloader))
    if epoch % 1 == 0:
        tc.save(mdl.state_dict(), "chestxray_new{}.pth".format(24+epoch))
    print(cnt)
mdl.eval()

cnt, cnt_all = tc.zeros([6]).cuda(), tc.zeros([6]).cuda()
with tc.no_grad():
    for i, (x, y) in enumerate(trainloader):
        x, y = x.cuda(), y.cuda().argmax(1)
        bs, cs, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        output = mdl(x)
        for i_label in range(6):
            cnt[i_label] += tc.logical_and(output.argmax(1) ==y, y == i_label ).sum()
            cnt_all[i_label] += ( y == i_label ).sum()
# label-wise accuracy rate 0.8940 90.64
print("Training Acc: ", [cnt[i].item()/cnt_all[i].item() for i in range(6)])

cnt, cnt_all = tc.zeros([6]).cuda(), tc.zeros([6]).cuda()
with tc.no_grad():
    for i, (x, y) in enumerate(valloader):
        x, y = x.cuda(), y.cuda().argmax(1)
        bs, cs, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        output = mdl(x)
        for i_label in range(6):
            cnt[i_label] += tc.logical_and(output.argmax(1) ==y, y == i_label ).sum()
            cnt_all[i_label] += ( y == i_label ).sum()
print("Validation Acc: ", [cnt[i].item()/cnt_all[i].item() for i in range(6)])