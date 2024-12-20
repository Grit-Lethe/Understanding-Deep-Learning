import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size_train=64
batch_size_test=1000
train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST',
                               train=True,
                               download=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307),(0.3081))])),
    batch_size=batch_size_train,
    shuffle=True)
test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST',
                               train=False,
                               download=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307),(0.3081))])),
    batch_size=batch_size_test,
    shuffle=True)

examples=enumerate(test_loader)
batch_idx, (example_data, example_targets)=next(examples)
fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title('Ground Truth: {}'.format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.drop=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)
    
    def forward(self, x):
        x=self.conv1(x)
        x=F.max_pool2d(x,2)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.drop(x)
        x=F.max_pool2d(x,2)
        x=F.relu(x)
        x=x.flatten(1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x=F.log_softmax(x)
        return x

def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

model=Net()
model.apply(weights_init)
optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
model=model.to(device)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%100==0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), loss.item()))

def test():
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            test_loss=F.nll_loss(output, target, size_average=False).item()
            pred=output.data.max(1,keepdim=True)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
        test_loss/=len(test_loader.dataset)
        print('\nTest Set: Avg.loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100.*correct/len(test_loader.dataset)))

test()
n_epochs=3
for epoch in range(1,n_epochs+1):
    train(epoch)
    test()

output=model(example_data.to(device))
fig=plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title('Prediction: {}'.format(output.data.max(1,keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()

def fgsm_attack(x, epsilon, dldx):
    sign_dldx=torch.sign(dldx)
    x_modified=x+epsilon*sign_dldx
    return x_modified

no_examples=3
epsilon=0.5
for i in range(no_examples):
    optimizer.zero_grad()
    x=example_data[i,:,:,:]
    x=x[None,:,:,:]
    x=x.to(device)
    x.requires_grad=True
    y=torch.ones(1,dtype=torch.long)*example_targets[i]
    y=y.to(device)
    output=model(x)
    loss=F.nll_loss(output,y)
    loss.backward()
    dldx=x.grad.data
    x_prime=fgsm_attack(x,epsilon,dldx)
    output_prime=model(x_prime)
    x_prime=x_prime.cpu()
    x=x.cpu()
    x=x.detach().numpy()
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.tight_layout()
    plt.imshow(x[0][0], cmap='gray', interpolation='none')
    plt.title("Original Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][0].item()))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.tight_layout()
    plt.imshow(x_prime[0][0].detach().numpy(), cmap='gray', interpolation='none')
    plt.title("Perturbed Prediction: {}".format(
        output_prime.data.max(1, keepdim=True)[1][0].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
