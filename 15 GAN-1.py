import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df=pd.read_csv('./MNIST/mnist_train.csv', header=None)
# row=0
# data=df.iloc[row]
# label=data[0]
# img=data[1:].values.reshape(28,28)
# plt.title("label="+str(label))
# plt.imshow(img, interpolation='none', cmap='Blues')
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(784,200),
                                 nn.LeakyReLU(0.02),
                                 nn.LayerNorm(200),
                                 nn.Linear(200,10),
                                 nn.Sigmoid())
        self.loss_function=nn.BCELoss()
        self.optimizer=torch.optim.Adam(self.parameters())
        self.counter=0
        self.progress=[]

    def forward(self, inputs):
        output=self.model(inputs)
        return output
    
    def train(self, inputs, targets):
        outputs=self.forward(inputs)
        loss=self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.counter+=1
        if (self.counter%10==0):
            self.progress.append(loss.item())
            pass
        if (self.counter%10000==0):
            print("Counter=",self.counter)
            pass

    def PlotProgress(self):
        df=pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0,1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.25,0.5))
        plt.show()
        pass

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df=pd.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        label=self.data_df.iloc[index,0]
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

mnist_dataset=MnistDataset('./MNIST/mnist_train.csv')
# mnist_dataset.PlotImage(100)
C=Net()
epochs=3
for i in range(epochs):
    print('Training Epoch ',i+1, 'of ',epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor, target_tensor)
        pass
    pass
C.PlotProgress()

mnist_test_dataset=MnistDataset('./MNIST/mnist_test.csv')
# record=19
# image_data=mnist_test_dataset[record][1]
# output=C.forward(image_data)
# pd.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1))
# plt.show()

score=0
items=0
for label, image_data_tensor, target_tensor in mnist_test_dataset:
    answer=C.forward(image_data_tensor).detach().numpy()
    # answer=np.argmax(np.squeeze(answer))
    if (answer.argmax()==label):
        score+=1
        pass
    items+=1
    pass
print(score, items, score/items)
