
from turtle import color
import numpy as np
import torch as tc
import sys
sys.path.append('..')

import data
import model

device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')
# balance data
ds_train = data.Heart(root='/data5/wenwens/heart', imp=[0.1, 0.9], n=150, m=150, part=0)
ds_val = data.Heart(root='/data5/wenwens/heart', imp=[0.1, 0.9], n=75, m=75, part=1)

train_loader = ds_train.train
val_loader = ds_val.train

mdl = model.CDCNet()
mdl = tc.nn.DataParallel(mdl).cuda()
optimizer = tc.optim.SGD(mdl.parameters(), lr=0.003, momentum=0.9)
criterion = tc.nn.CrossEntropyLoss()

def train_loop(dataloader, mdl, loss_fn, optimizer):
    mdl.train()
    size = len(dataloader) * 64
    for batch, (X, y) in enumerate(dataloader):
        # pdb.set_trace()
        X = X.to(device)
        # y = y.type(torch.LongTensor)
        y = y.to(device)

        # Compute prediction and loss
        pred = mdl(X)
        # if pred 0, add loss
        loss = loss_fn(pred, y) 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            batch_size = len(X)
            size = len(dataloader)*batch_size
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        

def test_loop(dataloader, mdl, loss_fn):
    mdl.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    gt, predy = [], []
    with tc.no_grad():
        for X, y in dataloader:
            # torch.Size([64, 17])
            X = X.to(device)
            y = y.to(device)
            pred = mdl(X)
            predy.append(pred.argmax(1))
            gt.append(y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(tc.float).sum().item()
        batch_size = len(X)
    size = len(dataloader)*batch_size
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return tc.cat(gt).cpu().numpy(), tc.cat(predy).cpu().numpy()

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, mdl, criterion, optimizer)
    test_loop(val_loader, mdl, criterion)

tc.save(mdl.state_dict(), "cdc_cls_large.pth")
