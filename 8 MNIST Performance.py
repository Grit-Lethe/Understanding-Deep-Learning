import torch, torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

train_data=torchvision.datasets.MNIST(
    root="MNIST",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
test_data=torchvision.datasets.MNIST(
    root="MNIST",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

Di=784
Dk=100
Do=10
model=torch.nn.Sequential(torch.nn.Linear(Di,Dk),
                          torch.nn.ReLU(),
                          torch.nn.Linear(Dk,Do),
                          torch.nn.ReLU())

def WeightsInit(layer_in):
    if isinstance(layer_in,nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
model.apply(WeightsInit)

loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.05,momentum=0.9)
scheduler=StepLR(optimizer,step_size=10,gamma=0.5)

n_epoch=5
loss_train=np.zeros((n_epoch))
error_train=np.zeros((n_epoch))
loss_test=np.zeros((n_epoch))
error_test=np.zeros((n_epoch))

for epoch in range(n_epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        data=data.reshape(-1,Di)
        pred=model(data)
        loss=loss_function(pred,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 ==0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch,batch_idx*len(data),len(train_dataloader.dataset),loss.item()))
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_dataloader:
            data=data.reshape(-1,Di)
            output=model(data)
            loss=loss_function(output,target)
            pred=output.data.max(1,keepdim=True)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
    test_loss/=len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_dataloader.dataset),100. * correct / len(test_dataloader.dataset)))
