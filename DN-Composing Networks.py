import numpy as np
import matplotlib.pyplot as plt

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def shallow113(x,activation_fn,phi0,phi1,phi2,phi3,theta10,theta11,theta20,theta21,theta30,theta31):
    pre1=theta10+theta11*x
    pre2=theta20+theta21*x
    pre3=theta30+theta31*x
    act1=activation_fn(pre1)
    act2=activation_fn(pre2)
    act3=activation_fn(pre3)
    wact1=phi1*act1
    wact2=phi2*act2
    wact3=phi3*act3
    y=phi0+wact1+wact2+wact3
    return y

def PlotNeuralTwoComponents(xin,net1out,net2out,net12out=None):
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(8.5,8.5)
    fig.tight_layout(pad=3.0)
    ax[0].plot(xin,net1out,'r')
    ax[0].set_xlabel('Net 1 Input')
    ax[0].set_ylabel('Net 1 Output')
    ax[0].set_xlim([-1,1])
    ax[0].set_ylim([-1,1])
    ax[0].set_aspect(1.0)
    ax[1].plot(xin,net2out,'b')
    ax[1].set_xlabel('Net 2 Input')
    ax[1].set_ylabel('Net 2 Output')
    ax[1].set_xlim([-1,1])
    ax[1].set_ylim([-1,1])
    ax[1].set_aspect(1.0)
    plt.show()
    if net12out is not None:
        fig,ax=plt.subplots()
        ax.plot(xin,net12out,'g')
        ax.set_xlabel('Net 1 Input')
        ax.set_ylabel('Net 2 Output')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect(1.0)
        plt.show()

n1theta10=0.0
n1theta11=-1.0
n1theta20=0.0
n1theta21=1.0
n1theta30=-0.67
n1theta31=1.0
n1phi0=1.0
n1phi1=-2.0
n1phi2=-3.0
n1phi3=9.3
n2theta10=-0.6
n2theta11=-1.0
n2theta20=0.2
n2theta21=1.0
n2theta30=-0.5
n2theta31=1.0
n2phi0=0.5
n2phi1=-1.0
n2phi2=-1.50
n2phi3=2.0
x=np.arange(-1,1,0.001)
net1out=shallow113(x,ReLU,n1theta10,n1theta11,n1theta20,n1theta21,n1theta30,n1theta31,n1phi0,n1phi1,n1phi2,n1phi3)
net2out=shallow113(x,ReLU,n2theta10,n2theta11,n2theta20,n2theta21,n2theta30,n2theta31,n2phi0,n2phi1,n2phi2,n2phi3)
PlotNeuralTwoComponents(x,net1out,net2out)

net12out=shallow113(net1out,ReLU,n2theta10,n2theta11,n2theta20,n2theta21,n2theta30,n2theta31,n2phi0,n2phi1,n2phi2,n2phi3)
PlotNeuralTwoComponents(x,net1out,net2out,net12out)
