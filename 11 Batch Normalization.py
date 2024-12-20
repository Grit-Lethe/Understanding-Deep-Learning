import numpy as np
import torchvision
import torch,torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

batch_size_train=64
batch_size_test=1000
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
train_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST',
                                                                    train=True,
                                                                    download=True,
                                                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                              torchvision.transforms.Normalize((0.1307,),(0.3081,))])),
                                                                    batch_size=batch_size_train,
                                                                    shuffle=True)
test_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST',
                                                                    train=False,
                                                                    download=True,
                                                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                              torchvision.transforms.Normalize((0.1307,),(0.3081,))])),
                                                                    batch_size=batch_size_test,
                                                                    shuffle=True)

def WeightsInit(layer_in):
    if isinstance(layer_in,nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

# def RunOneStep(epoch,model,train_loader):
#     loss_function=nn.CrossEntropyLoss()
#     optimizer=torch.optim.SGD(model.parameters(),lr=0.05,momentum=0.9)
#     model.apply(WeightsInit)
#     model=model.to(device)
#     for batch_idx,(data,target) in enumerate(train_loader):
#         data=data.reshape(-1,784)
#         data=data.to(device)
#         target=target.to(device)
#         optimizer.zero_grad()
#         output=model(data)
#         loss=loss_function(output,target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx%100==0:
#             print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset),loss.item()))
        

class ResidualNetwork(nn.Module):
    def __init__(self,input_size,output_size,hidden_size=1000):
        super(ResidualNetwork,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,hidden_size)
        self.linear3=nn.Linear(hidden_size,hidden_size)
        self.linear4=nn.Linear(hidden_size,hidden_size)
        self.linear5=nn.Linear(hidden_size,hidden_size)
        self.linear6=nn.Linear(hidden_size,hidden_size)
        self.linear7=nn.Linear(hidden_size,output_size)
        self.batchnorm1=nn.BatchNorm1d(hidden_size)
        self.batchnorm2=nn.BatchNorm1d(hidden_size)
        self.batchnorm3=nn.BatchNorm1d(hidden_size)
        self.batchnorm4=nn.BatchNorm1d(hidden_size)
        self.batchnorm5=nn.BatchNorm1d(hidden_size)
        self.batchnorm6=nn.BatchNorm1d(hidden_size)
    
    def forward(self,x):
        f=self.batchnorm1(self.linear1(x))
        res1=f+self.batchnorm2(self.linear2(f.relu()))
        res2=res1+self.batchnorm3(self.linear3(res1.relu()))
        res3=res2+self.batchnorm4(self.linear4(res2.relu()))
        res4=res3+self.batchnorm5(self.linear5(res3.relu()))
        res5=res4+self.batchnorm6(self.linear6(res4.relu()))
        res6=self.linear7(res5)
        return res6

model=ResidualNetwork(784,10)
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.05,momentum=0.9)
model.apply(WeightsInit)
model=model.to(device)

def Train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.reshape(-1,784)
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=loss_function(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx%100==0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset),loss.item()))

def Test():
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data=data.reshape(-1,784)
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            test_loss+=loss_function(output,target).item()
            pred=output.data.max(1,keepdim=True)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
        test_loss/=len(test_loader.dataset)
        print('\nTest set: Avg.loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))

Test()
n_epochs=10
for epoch in range(1,n_epochs+1):
    Train(epoch)
    Test()
