import numpy as np
import matplotlib.pyplot as plt

def GaussPDF(z, mu, sigma):
    pr_z=np.exp(-0.5*(z-mu)*(z-mu)/(sigma*sigma))/(np.sqrt(2*3.1415926)*sigma)
    return pr_z

z=np.arange(-3,3,0.01)
pr_z=GaussPDF(z,0,1)
fig, ax=plt.subplots()
ax.plot(z, pr_z)
ax.set_xlim([-3,3])
ax.set_xlabel('$z$')
ax.set_ylabel('$Pr(z)$')
plt.show()

def f(z):
    x1=6/(1+np.exp(-(z-0.25)*0.15))-3
    x2=z
    p=z*z/9
    x=(1-p)*x1+p*x2
    return x

def DFDZ(z):
    return (f(z+0.0001)+f(z-0.0001))/0.0002

x=f(z)
fig, ax=plt.subplots()
ax.plot(z, x)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_xlabel('Latent Variable, $z$')
ax.set_ylabel('Observed Variable, $x$')
plt.show()

x=f(z)
pr_x=GaussPDF(x,0,1)
fig, ax=plt.subplots()
ax.plot(x, pr_x)
ax.set_xlim([-3,3])
ax.set_ylim([0,0.5])
ax.set_xlabel('$x$')
ax.set_ylabel('$Pr(x)$')
plt.show()

np.random.seed(1)
n_sample=20
