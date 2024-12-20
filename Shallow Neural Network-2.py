import numpy as np
import matplotlib.pyplot as plt

def draw2Dfunction(ax,x1_mesh,x2_mesh,y):
    pos=ax.contourf(x1_mesh,x2_mesh,y,levels=256,cmap='hot',vmin=-10.0,vmax=10.0)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    levels=np.arange(-10,10,1.0)
    ax.contour(x1_mesh,x2_mesh,y,levels,cmap='winter')

def plotneural2inputs(x1,x2,y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3):
    fig,ax=plt.subplots(3,3)
    fig.set_size_inches(8.5,8.5)
    fig.tight_layout(pad=3.0)
    draw2Dfunction(ax[0,0],x1,x2,pre1)
    ax[0,0].set_title('Preactivation')
    draw2Dfunction(ax[0,1],x1,x2,pre2)
    ax[0,1].set_title('Preactivation')
    draw2Dfunction(ax[0,2],x1,x2,pre3)
    ax[0,2].set_title('Preactivation')
    draw2Dfunction(ax[1,0],x1,x2,act1)
    ax[1,0].set_title('Activation')
    draw2Dfunction(ax[1,1],x1,x2,act2)
    ax[1,1].set_title('Activation')
    draw2Dfunction(ax[1,2],x1,x2,act3)
    ax[1,2].set_title('Activation')
    draw2Dfunction(ax[2,0],x1,x2,wact1)
    ax[2,0].set_title('Weighted Act')
    draw2Dfunction(ax[2,1],x1,x2,wact2)
    ax[2,1].set_title('Weighted Act')
    draw2Dfunction(ax[2,2],x1,x2,wact3)
    ax[2,2].set_title('Weighted Act')
    plt.show()
    fig,ax=plt.subplots()
    draw2Dfunction(ax,x1,x2,y)
    ax.set_title('Network Output,$y$')
    ax.set_aspect(1.0)
    plt.show()

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def shallow213(x1,x2,activation_fn,phi0,phi1,phi2,phi3,theta10,theta11,theta12,theta20,theta21,theta22,theta30,theta31,theta32):
    pre1=theta10+theta11*x1+theta12*x2
    pre2=theta20+theta21*x1+theta22*x2
    pre3=theta30+theta31*x1+theta32*x2
    act1=activation_fn(pre1)
    act2=activation_fn(pre2)
    act3=activation_fn(pre3)
    wact1=phi1*act1
    wact2=phi2*act2
    wact3=phi3*act3
    y=phi0+wact1+wact2+wact3
    return y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3

theta10=-4.0
theta11=0.9
theta12=0.0
theta20=5.0
theta21=-0.9
theta22=-0.5
theta30=-7.0
theta31=0.5
theta32=0.9
phi0=0.0
phi1=-2.0
phi2=2.0
phi3=1.5
x1=np.arange(0.0,10.0,0.1)
x2=np.arange(0.0,10.0,0.1)
x1,x2=np.meshgrid(x1,x2)
y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3=shallow213(x1,x2,ReLU,phi0,phi1,phi2,phi3,theta10,theta11,theta12,theta20,theta21,theta22,theta30,theta31,theta32)
plotneural2inputs(x1,x2,y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3)

def plotneural2inputs2outputs(x1,x2,y1,y2,pre1,pre2,pre3,act1,act2,act3,wact11,wact12,wact13,wact21,wact22,wact23):
    fig,ax=plt.subplots(4,3)
    fig.set_size_inches(8.5,8.5)
    fig.tight_layout(pad=3.0)
    draw2Dfunction(ax[0,0],x1,x2,pre1)
    ax[0,0].set_title('Preactivation')
    draw2Dfunction(ax[0,1],x1,x2,pre2)
    ax[0,1].set_title('Preactivation')
    draw2Dfunction(ax[0,2],x1,x2,pre3)
    ax[0,2].set_title('Preactivation')
    draw2Dfunction(ax[1,0],x1,x2,act1)
    ax[1,0].set_title('Activation')
    draw2Dfunction(ax[1,1],x1,x2,act2)
    ax[1,1].set_title('Activation')
    draw2Dfunction(ax[1,2],x1,x2,act3)
    ax[1,2].set_title('Activation')
    draw2Dfunction(ax[2,0],x1,x2,wact11)
    ax[2,0].set_title('Weighted Act 1')
    draw2Dfunction(ax[2,1],x1,x2,wact12)
    ax[2,1].set_title('Weighted Act 1')
    draw2Dfunction(ax[2,2],x1,x2,wact13)
    ax[2,2].set_title('Weighted Act 1')
    draw2Dfunction(ax[3,0],x1,x2,wact21)
    ax[2,0].set_title('Weighted Act 2')
    draw2Dfunction(ax[3,1],x1,x2,wact22)
    ax[2,1].set_title('Weighted Act 2')
    draw2Dfunction(ax[3,2],x1,x2,wact23)
    ax[2,2].set_title('Weighted Act 2')
    plt.show()
    fig,ax=plt.subplots()
    draw2Dfunction(ax,x1,x2,y1)
    ax.set_title('Network Output,$y1$')
    ax.set_aspect(1.0)
    plt.show()
    fig,ax=plt.subplots()
    draw2Dfunction(ax,x1,x2,y2)
    ax.set_title('Network Output,$y2$')
    ax.set_aspect(1.0)
    plt.show()

def shallow223(x1,x2,activation_fn,phi10,phi11,phi12,phi13,phi20,phi21,phi22,phi23,theta10,theta11,theta12,theta20,theta21,theta22,theta30,theta31,theta32):
    pre1=theta10+theta11*x1+theta12*x2
    pre2=theta20+theta21*x1+theta22*x2
    pre3=theta30+theta31*x1+theta32*x2
    act1=activation_fn(pre1)
    act2=activation_fn(pre2)
    act3=activation_fn(pre3)
    wact11=phi11*act1
    wact12=phi12*act2
    wact13=phi13*act3
    wact21=phi21*act1
    wact22=phi22*act2
    wact23=phi23*act3
    y1=phi10+wact11+wact12+wact13
    y2=phi20+wact21+wact22+wact23
    return y1,y2,pre1,pre2,pre3,act1,act2,act3,wact11,wact12,wact13,wact21,wact22,wact23

theta10=-4.0
theta11=0.9
theta12=0.0
theta20=5.0
theta21=-0.9
theta22=-0.5
theta30=-7
theta31=0.5
theta32=0.9
phi10=0.0
phi11=-2.0
phi12=2.0
phi13=1.5
phi20=-2.0
phi21=-1.0
phi22=-2.0
phi23=0.8
x1=np.arange(0.0,10.0,0.1)
x2=np.arange(0.0,10.0,0.1)
x1,x2=np.meshgrid(x1,x2)
y1,y2,pre1,pre2,pre3,act1,act2,act3,wact11,wact12,wact13,wact21,wact22,wact23=shallow223(x1,x2,ReLU,phi10,phi11,phi12,phi13,phi20,phi21,phi22,phi23,theta10,theta11,theta12,theta20,theta21,theta22,theta30,theta31,theta32)
plotneural2inputs2outputs(x1,x2,y1,y2,pre1,pre2,pre3,act1,act2,act3,wact11,wact12,wact13,wact21,wact22,wact23)
