import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df=pd.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        label=self.data_df.iloc[index,0]
        label=torch.from_numpy(np.array([[label]]))
        target=torch.zeros((10))
        target[label]=1.0
        image_values=torch.FloatTensor(self.data_df.iloc[index,1:].values)/255.0
        return label, image_values, target
    pass

    def PlotImage(self, index):
        arr=self.data_df.iloc[index, 1:].values.reshape(28,28)
        plt.title("label="+str(self.data_df.iloc[index, 0]))
        plt.imshow(arr, interpolation='none', cmap='Blues')
        plt.show()
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(784+10,200),
                                 nn.LeakyReLU(0.02),
                                 nn.LayerNorm(200),
                                 nn.Linear(200,1),
                                 nn.Sigmoid())
        self.loss_function=nn.BCELoss()
        self.optimizer=torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter=0
        self.progress=[]
        pass

    def forward(self, image_tensor, label_tensor):
        inputs=torch.cat((image_tensor, label_tensor))
        output=self.model(inputs)
        return output
    
    def train(self, inputs, label_tensor, targets):
        outputs=self.forward(inputs, label_tensor)
        loss=self.loss_function(outputs, targets)
        self.counter+=1
        if (self.counter%10==0):
            self.progress.append(loss.item())
            pass
        if (self.counter%5000==0):
            print("Counter = ",self.counter)
            pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def PlotProgress(self):
        df=pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.25,0.5,1.0,5.0))
        plt.show()
        pass

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(100+10,200),
                                 nn.LeakyReLU(0.02),
                                 nn.LayerNorm(200),
                                 nn.Linear(200,784),
                                 nn.Sigmoid())
        self.optimizer=torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter=0
        self.progress=[]
        pass

    def forward(self, seed_tensor, label_tensor):
        inputs=torch.cat((seed_tensor, label_tensor))
        output=self.model(inputs)
        return output

    def train(self, DNET, inputs, label_tensor, targets):
        g_output=self.forward(inputs, label_tensor)
        d_output=DNET.forward(g_output, label_tensor)
        loss=DNET.loss_function(d_output, targets)
        self.counter+=1
        if (self.counter%10==0):
            self.progress.append(loss.item())
            pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def PlotProgress(self):
        df=pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.25,0.5,1.0,5.0))
        plt.show()
        pass

    def PlotImage(self, label):
        label_tensor=torch.zeros((10))
        label_tensor[label]=1.0
        label_tensor=label_tensor.to(device)
        f, axarr=plt.subplots(2,3, figsize=(16,8))
        for i in range(2):
            for j in range(3):
                axarr[i,j].imshow(GNET.forward(GenerateRandomSeed(100).to(device), label_tensor).detach().cpu().numpy().reshape(28,28), interpolation='none', cmap='Blues')
                pass
            pass
        plt.show()
        pass

def GenerateRandomSeed(size):
    random_data=torch.randn(size)
    return random_data

def GenerateRandomOneHot(size):
    label_tensor=torch.zeros((size))
    random_idx=random.randint(0, size-1)
    label_tensor[random_idx]=1.0
    return label_tensor

mnist_dataset=MnistDataset('./MNIST/mnist_train.csv')
DNET=Discriminator()
DNET=DNET.to(device)
GNET=Generator()
GNET=GNET.to(device)
epochs=10
for epoch in range(epochs):
    print("Epoch = ", epoch+1)
    for label, image_data_tensor, label_tensor in mnist_dataset:
        label=label.to(device)
        image_data_tensor=image_data_tensor.to(device)
        label_tensor=label_tensor.to(device)
        DNET.train(image_data_tensor, label_tensor, torch.cuda.FloatTensor([1.0]))
        random_label=GenerateRandomOneHot(10).to(device)
        DNET.train(GNET.forward(GenerateRandomSeed(100).to(device), random_label).detach(), random_label, torch.cuda.FloatTensor([0.0]))
        random_label=GenerateRandomOneHot(10).to(device)
        GNET.train(DNET, GenerateRandomSeed(100).to(device), random_label, torch.cuda.FloatTensor([1.0]))
        pass
    pass

DNET.PlotProgress()
GNET.PlotProgress()
GNET.PlotImage(0)
GNET.PlotImage(1)
GNET.PlotImage(2)
GNET.PlotImage(3)
GNET.PlotImage(4)
GNET.PlotImage(5)
GNET.PlotImage(6)
GNET.PlotImage(7)
GNET.PlotImage(8)
GNET.PlotImage(9)

# f, axarr=plt.subplots(2,3, figsize=(16,8))
# for i in range(2):
#     for j in range(3):
#         output=GNET.forward(GenerateRandomImage(100).to(device))
#         img=output.detach().cpu().numpy().reshape(28,28)
#         axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
#         pass
#     pass
# plt.show()

# count=0.0
# seed1=GenerateRandomSeed(100)
# seed2=GenerateRandomSeed(100)
# f, axarr=plt.subplots(3,4, figsize=(16,8))
# for i in range(3):
#     for j in range(4):
#         seed=seed1+(seed2-seed1)/11*count
#         seed=seed.to(device)
#         output=GNET.forward(seed)
#         img=output.detach().cpu().numpy().reshape(28,28)
#         axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
#         count=count+1
#         pass
#     pass
# plt.show()
