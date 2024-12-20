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
    return y,pre1,pre2,pre3,act1,act2,act3,wact1,wact2,wact3

def PlotNeural(x,y):
    fig,ax=plt.subplots()
    ax.plot(x.T,y.T)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_aspect(1.0)
    plt.show()

n1_theta_10 = 0.0
n1_theta_11 = -1.0
n1_theta_20 = 0
n1_theta_21 = 1.0
n1_theta_30 = -0.67
n1_theta_31 =  1.0
n1_phi_0 = 1.0
n1_phi_1 = -2.0
n1_phi_2 = -3.0
n1_phi_3 = 9.3
n1_in=np.arange(-1,1,0.01).reshape([1,-1])
n1_out, *_ = shallow113(n1_in, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11, n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31)
PlotNeural(n1_in,n1_out)

beta_0 = np.zeros((3,1))
omega_0 = np.zeros((3,1))
beta_1 = np.zeros((1,1))
omega_1 = np.zeros((1,3))

beta_0[0,0]=n1_theta_10
beta_0[1,0]=n1_theta_20
beta_0[2,0]=n1_theta_30
omega_0[0,0]=n1_theta_11
omega_0[1,0]=n1_theta_21
omega_0[2,0]=n1_theta_31
beta_1[0,0]=n1_phi_0
omega_1[0,0]=n1_phi_1
omega_1[0,1]=n1_phi_2
omega_1[0,2]=n1_phi_3

n_data=n1_in.size
n_dim_in=1
n1_in_mat=np.reshape(n1_in,(n_dim_in,n_data))
h1=ReLU(np.matmul(beta_0,np.ones((1,n_data)))+np.matmul(omega_0,n1_in_mat))
n1_out=np.matmul(beta_1,np.ones((1,n_data)))+np.matmul(omega_1,h1)
PlotNeural(n1_in,n1_out)
