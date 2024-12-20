import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

Di=784
Do=10
model1layer=nn.Sequential(nn.Linear(Di, 300),
                          nn.ReLU(),
                          nn.Linear(300, Do))
model2layer=nn.Sequential(nn.Linear(Di, Do))
model3layer=nn.Sequential(nn.Linear(Di, Do))
model4layer=nn.Sequential(nn.Linear(Di, Do))


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

# model.apply(weights_init)
# optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
# model=model.to(device)

def Train(model, train_data, epochs, device):
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.0)
    model.apply(weights_init)
    errors_train=np.zeros((epochs))
    model.train()
    for epoch in range(epochs):
        for batch_idx,(data,target) in enumerate(train_data):
            data=data.reshape(-1,784)
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=model(data)
            loss=loss_function(output, target)
            _,predicted=torch.max(output.data,1)
            errors_train[epoch]=100-100*(predicted==target).float().sum()/len(target)
            loss.backward()
            optimizer.step()
            if epoch%1==0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset),loss.item()))
                print('Epoch %d, Errors %3.3f'%(epoch, errors_train[epoch]))
    return errors_train

model4layer=model4layer.to(device)
errors4=Train(model4layer, train_loader, epochs=20, device=device)
model3layer=model3layer.to(device)
errors3=Train(model3layer, train_loader, epochs=20, device=device)
model2layer=model2layer.to(device)
errors2=Train(model2layer, train_loader, epochs=20, device=device)
model1layer=model1layer.to(device)
errors1=Train(model1layer, train_loader, epochs=20, device=device)

fig, ax=plt.subplots()
ax.plot(errors1, 'r-', label='one layer')
ax.plot(errors2, 'r-', label='two layer')
ax.plot(errors3, 'r-', label='three layer')
ax.plot(errors4, 'r-', label='four layer')
ax.set_ylim(0,100)
ax.set_xlabel('Epochs')
ax.set_ylabel('Percent Error')
plt.show()
