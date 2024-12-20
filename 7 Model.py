import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

Di,Dk,Do=10,40,5
model=nn.Sequential(nn.Linear(Di,Dk),
                    nn.ReLU(),
                    nn.Linear(Dk,Dk),
                    nn.ReLU(),
                    nn.Linear(Dk,Do))

def WeightsInit(layer_in):
    if isinstance(layer_in,nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

model.apply(WeightsInit)
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
scheduler=StepLR(optimizer,step_size=10,gamma=0.5)

x=torch.randn(100,Di)
y=torch.randn(100,Do)
data_loader=DataLoader(TensorDataset(x,y),batch_size=10,shuffle=True)

for epoch in range(1000):
    epoch_loss=0.0
    for i, data in enumerate(data_loader):
        x_batch,y_batch=data
        optimizer.zero_grad()
        pred=model(x_batch)
        loss=criterion(pred,y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    print(f'Epoch {epoch:5d}, Loss {epoch_loss:.3f}')
    scheduler.step()
