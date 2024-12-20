import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

examples=enumerate(test_loader)
batch_idx,(example_data,example_targets)=next(examples)
fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.pool=nn.MaxPool2d(2)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.drop=nn.Dropout2d(p=0.1)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(320,50)
        self.linear2=nn.Linear(50,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.pool(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.drop(x)
        x=self.pool(x)
        x=self.relu(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=F.log_softmax(x)
        return x

def WeightsInit(layer_in):
    if isinstance(layer_in,nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

model=Net()
model.apply(WeightsInit)
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
model=model.to(device)

def Train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
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
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,size_average=False).item()
            pred=output.data.max(1,keepdim=True)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
        test_loss/=len(test_loader.dataset)
        print('\nTest set: Avg.loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))

Test()
n_epochs=10
for epoch in range(1,n_epochs+1):
    Train(epoch)
    Test()
