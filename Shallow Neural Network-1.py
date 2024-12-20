import numpy as np
import matplotlib.pyplot as plt

def ReLU(preactivation):
    activation=np.clip(preactivation,0,None)
    return activation

z=np.arange(-5,5,0.1)
ReLU_z=ReLU(z)

fig,ax=plt.subplots()
ax.plot(z,ReLU_z,'r-')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_xlabel('z')
ax.set_ylabel('ReLU[z]')
ax.set_aspect('equal')
plt.show()

def shallow_1_1_3(x,activation_fn,phi0,phi1,phi2,phi3,theta10,theta11,theta20,theta21,theta30,theta31):
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
    return y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3

def plot_neural(x,y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3,plot_all=False,x_data=None,y_data=None):
    if plot_all:
        fig,ax=plt.subplots(3,3)
        fig.set_size_inches(8.5,8.5)
        fig.tight_layout(pad=3.0)
        ax[0,0].plot(x,pre1,'r-')
        ax[0,0].set_ylabel('Preactivation')
        ax[0,1].plot(x,pre2,'b-')
        ax[0,1].set_ylabel('Preactivation')
        ax[0,2].plot(x,pre3,'g-')
        ax[0,2].set_ylabel('Preactivation')
        ax[1,0].plot(x,act1,'r-')
        ax[1,0].set_ylabel('Activation')
        ax[1,1].plot(x,act2,'b-')
        ax[1,1].set_ylabel('Activation')
        ax[1,2].plot(x,act3,'g-')
        ax[1,2].set_ylabel('Activation')
        ax[2,0].plot(x,wact1,'r-')
        ax[2,0].set_ylabel('Weighted Act')
        ax[2,1].plot(x,wact2,'b-')
        ax[2,1].set_ylabel('Weighted Act')
        ax[2,2].plot(x,wact3,'g-')
        ax[2,2].set_ylabel('Weighted Act')
        for plot_y in range(3):
            for plot_x in range(3):
                ax[plot_y,plot_x].set_xlim([0,1])
                ax[plot_x,plot_y].set_ylim([-1,1])
                ax[plot_y,plot_x].set_aspect(0.5)
            ax[2,plot_y].set_xlabel('Input, $x$')
        plt.show()
    fig,ax=plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $x$')
    ax.set_xlim([0,1])
    ax.set_ylim([-1,1])
    ax.set_aspect(0.5)
    if x_data is not None:
        ax.plot(x_data,y_data,'mo')
        for i in range(len(x_data)):
            ax.plot(x_data[i],y_data[i])
    plt.show()

theta10=0.3
theta11=-1.0
theta20=-1.0
theta21=2.0
theta30=-0.5
theta31=0.65
phi0=-0.3
phi1=2.0
phi2=-1.0
phi3=7.0
x=np.arange(0,1,0.01)
y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3=shallow_1_1_3(x,ReLU,phi0,phi1,phi2,phi3,theta10,theta11,theta20,theta21,theta30,theta31)
plot_neural(x,y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3,plot_all=True)

def least_squares_loss(y_train,y_predict):
    loss=(y_predict-y_train)**2
    loss=np.sum(loss)
    return loss

x_train=np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train=np.array([-0.15934537,0.18195445,0.451270150,0.13921448,0.09366691,0.30567674,\
                    0.372291170,0.40716968,-0.08131792,0.41187806,0.36943738,0.3994327,\
                    0.019062570,0.35820410,0.452564960,-0.0183121,0.02957665,-0.24354444, \
                    0.148038840,0.26824970])

y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3=shallow_1_1_3(x,ReLU,phi0,phi1,phi2,phi3,theta10,theta11,theta20,theta21,theta30,theta31)
plot_neural(x,y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3,plot_all=True)

y_predict,*_=shallow_1_1_3(x_train,ReLU,phi0,phi1,phi2,phi3,theta10,theta11,theta20,theta21,theta30,theta31)

loss=least_squares_loss(y_train,y_predict)
print('Your Loss=%3.3f, True value=9.385'%(loss))
