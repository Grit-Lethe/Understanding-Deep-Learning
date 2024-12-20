import numpy as np
import matplotlib.pyplot as plt

x=[5.2,5.3,5.4,5.1,10.1,10.3,9.9,10.3,3.2,3.4,3.3,3.1]
x=np.array(x)

fig,ax=plt.subplots()
ax.plot(x,'k')
ax.set_xlim(0,11)
ax.set_ylim(0,12)
plt.show()

def Conv311(x_in,omega):
    n=len(x_in)
    m=len(omega)
    x_out=np.zeros(n-m+1)
    for i in range(n-m+1):
        x_out[i]=np.sum(x_in[i:i+m]*omega)
    return x_out

def Conv321(x_in,omega):
    length_out=int(np.ceil(len(x_in)-len(omega)+1)/2.0)+1
    x_out=np.zeros(length_out)
    for i in range(length_out-1):
        start=i*2
        end=start+len(omega)
        x_out[i]=np.sum(x_in[start:end]*omega)
    return x_out

omega=[-0.5,0.0,0.5,0.2,0.4]
omega=np.array(omega)
h=Conv321(x,omega)

fig,ax=plt.subplots()
ax.plot(x,'k',label='before')
ax.plot(h,'r',label='after')
ax.set_xlim(0,11)
ax.set_ylim(0,12)
ax.legend()
plt.show()
