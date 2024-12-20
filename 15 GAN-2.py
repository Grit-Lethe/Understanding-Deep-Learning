import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import numpy as np

def GenerateReal():
    real_data=torch.FloatTensor([rd.uniform(0.8,1.0),
                                 rd.uniform(0.0,0.2),
                                 rd.uniform(0.8,1.0),
                                 rd.uniform(0.0,0.2),
                                 rd.uniform(0.8,1.0),
                                 rd.uniform(0.0,0.2),
                                 rd.uniform(0.8,1.0),
                                 rd.uniform(0.0,0.2),
                                 rd.uniform(0.8,1.0),
                                 rd.uniform(0.0,0.2)])
    return real_data
# A=GenerateReal()
# print(A)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(10,5),
                                 nn.Sigmoid(),
                                 nn.Linear(5,1),
                                 nn.Sigmoid())
        self.loss_function=nn.MSELoss()
        self.optimizer=torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter=0
        self.progress=[]
        pass

    def forward(self, inputs):
        output=self.model(inputs)
        return output
    
    def train(self, inputs, targets):
        outputs=self.forward(inputs)
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
        df.plot(ylim=(0,1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.25,0.5))
        plt.show()
        pass

def GenerateRandom(size):
    random_data=torch.rand(size)
    return random_data

DNET=Discriminator()
# for i in range(10000):
#     DNET.train(GenerateReal(), torch.FloatTensor([1.0]))
#     DNET.train(GenerateRandom(10), torch.FloatTensor([0.0]))
#     pass
# DNET.PlotProgress()
# print(DNET.forward(GenerateReal()).item())
# print(DNET.forward(GenerateRandom(10)).item())

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(1,5),
                                 nn.Sigmoid(),
                                 nn.Linear(5,10),
                                 nn.Sigmoid())
        self.optimizer=torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter=0
        self.progress=[]
        pass

    def forward(self, inputs):
        output=self.model(inputs)
        return output

    def train(self, DNET, inputs, targets):
        g_output=self.forward(inputs)
        d_output=DNET.forward(g_output)
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
        df.plot(ylim=(0,1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.25,0.5))
        plt.show()
        pass

GNET=Generator()
# A=GNET.forward(torch.FloatTensor([0.5]))
# print(A)

image_list=[]
for i in range(10000):
    DNET.train(GenerateReal(), torch.FloatTensor([1.0]))
    DNET.train(GNET.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
    GNET.train(DNET, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
    if (i%1000==0):
        image_list.append(GNET.forward(torch.FloatTensor([0.5])).detach().numpy())
    pass
DNET.PlotProgress()
GNET.PlotProgress()
plt.figure(figsize=(8,8))
plt.imshow(np.array(image_list).T, interpolation='none', cmap='Blues')
plt.show()
