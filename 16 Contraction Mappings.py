import numpy as np
import matplotlib.pyplot as plt

def f(z):
    y=0.3+0.5*z+0.02*np.sin(z*15)
    return y

def DrawFunction(f, fixed_point=None):
    z=np.arange(0,1,0.01)
    z_prime=f(z)
    fig, ax=plt.subplots()
    ax.plot(z, z_prime, 'c-')
    if fixed_point!=None:
        ax.plot(fixed_point, fixed_point, 'ro')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Input, $z$')
    ax.set_ylabel('Output, f$[z]$')
    plt.show()

DrawFunction(f)

def FixedPointIteration(f, z0):
    max_iterations=20
    iteration=0
    z_out=z0
    while iteration<max_iterations:
        z_out=f(z_out)
        print(f"Iteration {iteration}: z={z_out}")
        iteration+=1
    return z_out

z=FixedPointIteration(f, 0.2)
DrawFunction(f, z)

def f2(z):
    y=0.7-0.6*z+0.03*np.sin(z*15)
    return y

DrawFunction(f2)
z=FixedPointIteration(f2, 0.9)
DrawFunction(f2, z)

def f3(z):
    y=-0.2+1.5*z+0.1*np.sin(z*15)
    return y

DrawFunction(f3)
z=FixedPointIteration(f3, 0.7)
DrawFunction(f3, z)

def f4(z):
    y=-0.3+0.5*z+0.02*np.sin(z*15)
    return y

def FixedPointIteration1(f, y, z0):
    z_out=z0+f(y)
    return z_out

def DrawFunction2(f, y, fixed_point=None):
    z=np.arange(0,1,0.01)
    z_prime=z+f(z)
    fig, ax=plt.subplots()
    ax.plot(z, z_prime, 'c-')
    ax.plot(z, y-f(z), 'r-')
    ax.plot([0,1], [0,1], 'k--')
    if fixed_point!=None:
        ax.plot(fixed_point, y, 'ro')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Input, $z$')
    ax.set_ylabel('Output, z+f$[z]$')
    plt.show()

y=0.8
z=FixedPointIteration1(f4, y, 0.2)
DrawFunction2(f4, y, z)
