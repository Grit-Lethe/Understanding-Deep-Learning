# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:20:11 2024

@author: 32116
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([0.03,0.19,0.34,0.46,0.78,0.81,1.08,1.18,1.39,1.60,1.65,1.90])
y=np.array([0.67,0.85,1.05,1.00,1.40,1.50,1.30,1.54,1.55,1.68,1.73,1.60])
#print(x)
#print(y)

def f(x,phi0,phi1):
    y=phi1*x+phi0
    return y

def plot(x,y,phi0,phi1):
    fig,ax=plt.subplots()
    ax.scatter(x,y)
    plt.xlim([0,2.0])
    plt.ylim([0,2.0])
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $y%')
    x_line=np.arange(0,2,0.01)
    y_line=f(x_line,phi0,phi1)
    plt.plot(x_line,y_line,'b-',lw=2)
    plt.show()

phi0=0.4
phi1=0.2
plot(x,y,phi0,phi1)

def compute_loss(x,y,phi0,phi1):
    loss=0
    for i in range(len(x)):
        y_hat=f(x[i],phi0,phi1)
        loss_val=(y[i]-y_hat)**2
        loss=loss+loss_val
    return loss

loss=compute_loss(x,y,phi0,phi1)
print(f'Your Loss={loss:3.2f}, ground truth=7.07')

phi0=1.60
phi1=-0.8
plot(x,y,phi0,phi1)
loss=compute_loss(x,y,phi0,phi1)
print(f'Your Loss={loss:3.2f}, ground truth=10.28')

phi0_mesh,phi1_mesh=np.meshgrid(np.arange(0.0,2.0,0.02),np.arange(-1.0,1.0,0.02))
all_losses=np.zeros_like(phi1_mesh)
for indices,temp in np.ndenumerate(phi1_mesh):
    all_losses[indices]=compute_loss(x,y,phi0_mesh[indices],phi1_mesh[indices])

fig=plt.figure()
ax=plt.axes()
fig.set_size_inches(7,7)
levels=40
ax.contourf(phi0_mesh,phi1_mesh,all_losses,levels)
levels=40
ax.contour(phi0_mesh,phi1_mesh,all_losses,levels,colors=['#80808080'])
ax.set_ylim([1,-1])
ax.set_xlabel(r'Intercept, $\phi_0$')
ax.set_ylabel(r'Slope,$\phi_1$')
ax.plot(phi0,phi1,'ro')
plt.show()
