import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def GetRealDataBatch(n_sample):
    np.random.seed(0)
    x_true=np.random.normal(size=(1, n_sample))+7.5
    return x_true

def Generator(z, theta):
    x_gen=z+theta
    return x_gen

def Sig(data_in):
    out=1.0/(1.0+np.exp(-data_in))
    return out

def Discriminator(x, phi0, phi1):
    y=Sig(phi0+phi1*x)
    return y

def DrawDataModel(x_real, x_syn, phi0=None, phi1=None):
    fix,ax=plt.subplots()
    for x in x_syn:
        ax.plot([x,x], [0,0.25], color='#f47a60')
    for x in x_real:
        ax.plot([x,x], [0,0.25], color='#7fe7dc')
    if phi0 is not None:
        x_model=np.arange(0, 10, 0.01)
        y_model=Discriminator(x_model, phi0, phi1)
        ax.plot(x_model, y_model, color='#dddddd')
    ax.set_xlim([0,10])
    ax.set_ylim([0,1])
    plt.show()

x_real=GetRealDataBatch(10)
# theta=3.0
# np.random.seed(1)
z=np.random.normal(size=(1,10))
# x_syn=Generator(z, theta)
# phi0=-2
# phi1=1
# DrawDataModel(x_real, x_syn, phi0, phi1)

def ComputeDiscriminatorLoss(x_real, x_syn, phi0, phi1):
    y_real=Discriminator(x_real, phi0, phi1)
    y_real=torch.from_numpy(y_real)
    y_syn=Discriminator(x_syn, phi0, phi1)
    y_syn=torch.from_numpy(y_syn)
    loss_real=nn.BCELoss()(y_real, torch.ones_like(y_real))
    loss_syn=nn.BCELoss()(y_syn, torch.zeros_like(y_syn))
    loss=(loss_real+loss_syn)
    loss=loss.numpy()
    return loss

# loss=ComputeDiscriminatorLoss(x_real, x_syn, phi0, phi1)
# print("True Loss=13.81475710851447, Your Loss=",loss)

def ComputeDiscriminatorGradient(x_real, x_syn, phi0, phi1):
    delta=0.00001
    loss1=ComputeDiscriminatorLoss(x_real, x_syn, phi0, phi1)
    loss2=ComputeDiscriminatorLoss(x_real, x_syn, phi0+delta, phi1)
    loss3=ComputeDiscriminatorLoss(x_real, x_syn, phi0, phi1+delta)
    dl_dphi0=(loss2-loss1)/delta
    dl_dphi1=(loss3-loss1)/delta
    return dl_dphi0, dl_dphi1

def UpdateDiscriminator(x_real, x_syn, n_iter, phi0, phi1):
    alpha=0.1
    print("Initial Discriminator Loss = ",ComputeDiscriminatorLoss(x_real, x_syn, phi0, phi1))
    for iter in range(n_iter):
        dl_dphi0, dl_dphi1=ComputeDiscriminatorGradient(x_real, x_syn, phi0, phi1)
        phi0=phi0-alpha*dl_dphi0
        phi1=phi1-alpha*dl_dphi1
    print("Final Discriminator Loss = ",ComputeDiscriminatorLoss(x_real, x_syn, phi0, phi1))
    return phi0, phi1

# n_iter=100
# print("Initial Parameters (phi0, phi1) ",phi0, phi1)
# phi0, phi1=UpdateDiscriminator(x_real, x_syn, n_iter, phi0, phi1)
# print("Final Parameters (phi0, phi1) ",phi0, phi1)
# DrawDataModel(x_real, x_syn, phi0, phi1)

def ComputeGeneratorLoss(z, theta, phi0, phi1):
    x_syn=Generator(z, theta)
    p=Discriminator(x_syn, phi0, phi1)
    loss=np.sum(-np.log(p))
    # p=torch.from_numpy(p)
    # loss=-torch.mean(torch.log(p))
    # loss=loss.numpy()
    return loss

# loss=ComputeGeneratorLoss(z, theta, -2, 1)
# print("True Loss = 13.78437035945412, Your loss=", loss)

def ComputeGeneratorGradient(z, theta, phi0, phi1):
    delta=0.0001
    loss1=ComputeGeneratorLoss(z, theta, phi0, phi1)
    loss2=ComputeGeneratorLoss(z, theta+delta, phi0, phi1)
    dl_dtheta=(loss2-loss1)/delta
    return dl_dtheta

def UpdateGenerator(z, theta, n_iter, phi0, phi1):
    alpha=0.02
    print("Initial Generator Loss = ",ComputeGeneratorLoss(z, theta, phi0, phi1))
    for iter in range(n_iter):
        dl_dtheta=ComputeGeneratorGradient(z, theta, phi0, phi1)
        theta=theta-alpha*dl_dtheta
    print("Final Generator Loss = ",ComputeGeneratorLoss(z, theta, phi0, phi1))
    return theta

# n_iter=100
# theta=3.0
# print("Theta Before ",theta)
# theta=UpdateGenerator(z, theta, n_iter, phi0, phi1)
# print("Theta After ",theta)
# x_syn=Generator(z, theta)
# DrawDataModel(x_real, x_syn, phi0, phi1)

theta=3.0
phi0=-2.0
phi1=1.0
n_iter_discrim=300
n_iter_gen=30
print("Final Parameters (phi0, phi1) ",phi0, phi1)
for c_gan_iter in range(5):
    x_syn=Generator(z, theta)
    DrawDataModel(x_real, x_syn, phi0, phi1)
    print("Updating Discriminator")
    phi0, phi1=UpdateDiscriminator(x_real, x_syn, n_iter_discrim, phi0, phi1)
    DrawDataModel(x_real, x_syn, phi0, phi1)
    print("Updating Generator")
    theta=UpdateGenerator(z, theta, n_iter_gen, phi0,phi1)
