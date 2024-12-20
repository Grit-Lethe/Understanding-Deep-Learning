import numpy as np
import matplotlib.pyplot as plt

def LinearFunction1D(x,beta,omega):
    y=omega*x+beta
    return y

x=np.arange(0.0,10.0,0.01)
beta=0.0
omega=1.0
y=LinearFunction1D(x,beta,omega)
fig,ax=plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([0,10])
ax.set_xlim([0,10])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

def Draw2DFunction(x1_mesh,x2_mesh,y):
    fig,ax=plt.subplots()
    fig.set_size_inches(7,7)
    pos=ax.contourf(x1_mesh,x2_mesh,y,levels=256,cmap='hot',vmin=-10.0,vmax=10.0)
    fig.colorbar(pos,ax=ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    levels=np.arange(-10,10,1.0)
    ax.contour(x1_mesh,x2_mesh,y,levels,cmap='winter')
    plt.show()

def LinearFunction2D(x1,x2,beta,omega1,omega2):
    x1,x2=np.meshgrid(x1,x2)
    y=omega1*x1+omega2*x2+beta
    return y

x1=np.arange(0.0,10.0,0.1)
x2=np.arange(0.0,10.0,0.1)
beta=0.0
omega1=1.0
omega2=-0.5
y=LinearFunction2D(x1,x2,beta,omega1,omega2)
Draw2DFunction(x1,x2,y)

def LinearFunction3D(x1,x2,x3,beta,omega1,omega2,omega3):
    x1,x2,x3=np.meshgrid(x1,x2,x3)
    y=omega1*x1+omega2*x2+omega3*x3+beta
    return y

beta1=0.5
beta2=0.2
omega11=-1.0
omega12=0.4
omega13=-0.3
omega21=0.1
omega22=0.1
omega23=1.2
x1=4
x2=-1
x3=2
y1=LinearFunction3D(x1,x2,x3,beta1,omega11,omega12,omega13)
y2=LinearFunction3D(x1,x2,x3,beta2,omega21,omega22,omega23)
print("Individual equations")
print('y1=%3.3f\ny2=%3.3f'%((y1,y2)))
beta_vec=np.array([[beta1],[beta2]])
omega_mat=np.array([[omega11,omega12,omega13],[omega21,omega22,omega23]])
x_vec=np.array([[x1],[x2],[x3]])
y_vec=beta_vec+np.matmul(omega_mat,x_vec)
print("Matrix/vector form")
print('y1=%3.3f\ny2=%3.3f'%((y_vec[0][0],y_vec[1][0])))

x=np.arange(-5.0,5.0,0.01)
y=np.exp(x)
fig,ax=plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([0,100])
ax.set_xlim([-5,5])
ax.set_xlabel('x')
ax.set_ylabel('exp[x]')
plt.show()

x=np.arange(0.01,5.0,0.01)
y=np.log(x)
fig,ax=plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([-5,5])
ax.set_xlim([0,5])
ax.set_xlabel('x')
ax.set_ylabel('log[x]')
plt.show()
