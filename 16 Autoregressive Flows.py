import numpy as np
import matplotlib.pyplot as plt

def g(h, phi):
    K=len(phi)
    i=int(h*K)
    if i==0:
        h_prime=(h*K)*phi[i]
    else:
        h_prime=np.sum(phi[:i])+(h*K-i)*phi[i]
    return h_prime

phi=np.array([0.2,0.1,0.4,0.05,0.25])
h=np.arange(0,1,0.01)
h_prime=np.zeros_like(h)
for i in range(len(h)):
    h_prime[i]=g(h[i], phi)
fig, ax=plt.subplots()
ax.plot(h, h_prime, 'b-')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('Input, $h$')
ax.set_ylabel('Output, $hprime$')
plt.show()

def GInverse(h_prime, phi):
    h_low=0
    h_mid=0.5
    h_high=0.999
    thresh=0.0001
    c_iter=0
    while (c_iter<20 and h_high-h_low>thresh):
        h_prime_low=g(h_low, phi)
        h_prime_mid=g(h_mid, phi)
        h_prime_high=g(h_high, phi)
        if h_prime_mid<h_prime:
            h_low=h_mid
        else:
            h_high=h_mid
        h_mid=h_low+(h_high-h_low)/2
        c_iter+=1
    return h_mid

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def Softmax(x):
    x=np.exp(x)
    x=x/np.sum(x)
    return x

def Getphi():
    return np.array([0.2,0.1,0.4,0.05,0.25])

def ShallowNetwork1(h1, n_hidden=10):
    n_input=1
    np.random.seed(n_input)
    beta0=np.random.normal(size=(n_hidden,1))
    omega0=np.random.normal(size=(n_hidden, n_input))
    beta1=np.random.normal(size=(5,1))
    omega1=np.random.normal(size=(5,n_hidden))
    y=Softmax(beta1+omega1@ReLU(beta0+omega0@np.array([[h1]])))
    return y

def ShallowNetwork2(h1, h2, n_hidden=10):
    n_input=2
    np.random.seed(n_input)
    beta0=np.random.normal(size=(n_hidden,1))
    omega0=np.random.normal(size=(n_hidden, n_input))
    beta1=np.random.normal(size=(5,1))
    omega1=np.random.normal(size=(5,n_hidden))
    y=Softmax(beta1+omega1@ReLU(beta0+omega0@np.array([[h1],[h2]])))
    return y

def ShallowNetwork3(h1, h2, h3, n_hidden=10):
    n_input=3
    np.random.seed(n_input)
    beta0=np.random.normal(size=(n_hidden,1))
    omega0=np.random.normal(size=(n_hidden, n_input))
    beta1=np.random.normal(size=(5,1))
    omega1=np.random.normal(size=(5,n_hidden))
    y=Softmax(beta1+omega1@ReLU(beta0+omega0@np.array([[h1],[h2],[h3]])))
    return y

def Forward(h1, h2, h3, h4):
    h_prime1=g(h1, Getphi())
    h_prime2=g(h2, ShallowNetwork1(h1))
    h_prime3=g(h3, ShallowNetwork2(h1, h2))
    h_prime4=g(h4, ShallowNetwork3(h1, h2, h3))
    return h_prime1, h_prime2, h_prime3, h_prime4

def Backward(h1_prime, h2_prime, h3_prime, h4_prime):
    h1=GInverse(h1_prime, Getphi())
    h2=GInverse(h2_prime, ShallowNetwork1(h1))
    h3=GInverse(h3_prime, ShallowNetwork2(h1, h2))
    h4=GInverse(h4_prime, ShallowNetwork3(h1, h2, h3))
    return h1, h2, h3, h4

h1=0.22
h2=0.41
h3=0.83
h4=0.53
print("Original h values %3.3f,%3.3f,%3.3f,%3.3f"%(h1,h2,h3,h4))
h1_prime, h2_prime, h3_prime, h4_prime = Forward(h1,h2,h3,h4)
print("h_prime values %3.3f,%3.3f,%3.3f,%3.3f"%(h1_prime,h2_prime,h3_prime,h4_prime))
h1,h2,h3,h4 =  Backward(h1_prime,h2_prime,h3_prime,h4_prime)
print("Reconstructed h values %3.3f,%3.3f,%3.3f,%3.3f"%(h1,h2,h3,h4))
