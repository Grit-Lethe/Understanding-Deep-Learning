import numpy as np
import matplotlib.pyplot as plt

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def shallow1133(x,activation_fn,phi,psi,theta):
    layer1pre1=theta[1,0]+theta[1,1]*x
    layer1pre2=theta[2,0]+theta[2,1]*x
    layer1pre3=theta[3,0]+theta[3,1]*x
    h1=activation_fn(layer1pre1)
    h2=activation_fn(layer1pre2)
    h3=activation_fn(layer1pre3)
    layer2pre1=psi[1,0]+psi[1,1]*h1+psi[1,2]*h2+psi[1,3]*h3
    layer2pre2=psi[2,0]+psi[2,1]*h1+psi[2,2]*h2+psi[2,3]*h3
    layer2pre3=psi[3,0]+psi[3,1]*h1+psi[3,2]*h2+psi[3,3]*h3
    h1prime=activation_fn(layer2pre1)
    h2prime=activation_fn(layer2pre2)
    h3prime=activation_fn(layer2pre3)
    phi1h1prime=phi[1]*h1prime
    phi2h2prime=phi[2]*h2prime
    phi3h3prime=phi[3]*h3prime
    y=phi[0]+phi1h1prime+phi2h2prime+phi3h3prime
    return y,layer2pre1,layer2pre2,layer2pre3,h1prime,h2prime,h3prime,phi1h1prime,phi2h2prime,phi3h3prime

def PlotNeuralTwoLayers(x,y,layer2pre1,layer2pre2,layer2pre3,h1prime,h2prime,h3prime,phi1h1prime,phi2h2prime,phi3h3prime):
    fig, ax = plt.subplots(3,3)
    fig.set_size_inches(8.5, 8.5)
    fig.tight_layout(pad=3.0)
    ax[0,0].plot(x,layer2pre1,'r-')
    ax[0,0].set_ylabel(r'$\psi_{10}+\psi_{11}h_{1}+\psi_{12}h_{2}+\psi_{13}h_3$')
    ax[0,1].plot(x,layer2pre2,'b-')
    ax[0,1].set_ylabel(r'$\psi_{20}+\psi_{21}h_{1}+\psi_{22}h_{2}+\psi_{23}h_3$')
    ax[0,2].plot(x,layer2pre3,'g-')
    ax[0,2].set_ylabel(r'$\psi_{30}+\psi_{31}h_{1}+\psi_{32}h_{2}+\psi_{33}h_3$')
    ax[1,0].plot(x,h1prime,'r-')
    ax[1,0].set_ylabel(r"$h_{1}^{'}$")
    ax[1,1].plot(x,h2prime,'b-')
    ax[1,1].set_ylabel(r"$h_{2}^{'}$")
    ax[1,2].plot(x,h3prime,'g-')
    ax[1,2].set_ylabel(r"$h_{3}^{'}$")
    ax[2,0].plot(x,phi1h1prime,'r-')
    ax[2,0].set_ylabel(r"$\phi_1 h_{1}^{'}$")
    ax[2,1].plot(x,phi2h2prime,'b-')
    ax[2,1].set_ylabel(r"$\phi_2 h_{2}^{'}$")
    ax[2,2].plot(x,phi3h3prime,'g-')
    ax[2,2].set_ylabel(r"$\phi_3 h_{3}^{'}$")

    for plot_y in range(3):
        for plot_x in range(3):
            ax[plot_y,plot_x].set_xlim([0,1]);ax[plot_x,plot_y].set_ylim([-1,1])
            ax[plot_y,plot_x].set_aspect(0.5)
        ax[2,plot_y].set_xlabel(r'Input, $x$');
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel(r'Input, $x$'); ax.set_ylabel(r'Output, $y$')
    ax.set_xlim([0,1]);ax.set_ylim([-1,1])
    ax.set_aspect(0.5)
    plt.show()

theta = np.zeros([4,2])
psi = np.zeros([4,4])
phi = np.zeros([4,1])

theta[1,0] =  0.3
theta[1,1] = -1.0
theta[2,0]= -1.0
theta[2,1] = 2.0
theta[3,0] = -0.5
theta[3,1] = 0.65
psi[1,0] = 0.3
psi[1,1] = 2.0
psi[1,2] = -1.0
psi[1,3]=7.0
psi[2,0] = -0.2
psi[2,1] = 2.0
psi[2,2] = 1.2
psi[2,3]=-8.0
psi[3,0] = 0.3
psi[3,1] = -2.3
psi[3,2] = -0.8
psi[3,3]=2.0
phi[0] = 0.0
phi[1] = 0.5
phi[2] = -1.5
phi[3] = 2.2
x=np.arange(0,1,0.001)
y,layer2pre1,layer2pre2,layer2pre3,h1prime,h2prime,h3prime,phi1h1prime,phi2h2prime,phi3h3prime=shallow1133(x,ReLU,phi,psi,theta)
PlotNeuralTwoLayers(x,y,layer2pre1,layer2pre2,layer2pre3,h1prime,h2prime,h3prime,phi1h1prime,phi2h2prime,phi3h3prime)
