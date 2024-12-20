import numpy as np
import matplotlib.pyplot as plt

def InitParams(K,D):
    np.random.seed(1)
    D_i=1
    D_o=1
    sigma_sq_omega=1.0/D
    all_weights=[None]*(K+1)
    all_bias=[None]*(K+1)
    all_weights[0]=np.random.normal(size=(D,D_i))*np.sqrt(sigma_sq_omega)
    all_weights[-1]=np.random.normal(size=(D_o,D))*np.sqrt(sigma_sq_omega)
    all_bias[0]=np.random.normal(size=(D,1))*np.sqrt(sigma_sq_omega)
    all_bias[-1]=np.random.normal(size=(D_o,1))*np.sqrt(sigma_sq_omega)
    for layer in range(1,K):
        all_weights[layer]=np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
        all_bias[layer]=np.random.normal(size=(D,1))*np.sqrt(sigma_sq_omega)
    return all_weights,all_bias

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def ForwardPass(net_input,all_weights,all_bias):
    K=len(all_weights)-1
    all_f=[None]*(K+1)
    all_h=[None]*(K+1)
    all_h[0]=net_input
    for layer in range(K):
        all_f[layer]=all_bias[layer]+np.matmul(all_weights[layer],all_h[layer])
        all_h[layer+1]=ReLU(all_f[layer])
    all_f[K]=all_bias[K]+np.matmul(all_weights[K],all_h[K])
    net_output=all_f[K]
    return net_output,all_f,all_h

def IndicatorFunction(x):
    x_in=np.array(x)
    x_in[x_in>=0]=1
    x_in[x_in<0]=0
    return x_in

def CalcInputOutputGradient(x_in,all_weights,all_bias):
    y,all_f,all_h=ForwardPass(x_in,all_weights,all_bias)
    K=len(all_weights)-1
    all_dl_dweights=[None]*(K+1)
    all_dl_dbias=[None]*(K+1)
    all_dl_df=[None]*(K+1)
    all_dl_dh=[None]*(K+1)
    all_dl_df[K]=np.ones_like(all_f[K])
    for layer in range(K,-1,-1):
        all_dl_dbias[layer]=np.array(all_dl_df[layer])
        all_dl_dweights[layer]=np.matmul(all_dl_df[layer],all_h[layer].transpose())
        all_dl_dh[layer]=np.matmul(all_weights[layer].transpose(),all_dl_df[layer])
        if layer>0:
            all_dl_df[layer-1]=IndicatorFunction(all_f[layer-1])*all_dl_dh[layer]
    return all_dl_dh[0],y

D=200
K=3
all_weights,all_bias=InitParams(K,D)
x=np.ones((1,1))
dydx,y=CalcInputOutputGradient(x,all_weights,all_bias)
delta=0.0000001
x1=x
y1,*_=ForwardPass(x1,all_weights,all_bias)
x2=x+delta
y2,*_=ForwardPass(x2,all_weights,all_bias)
dydx_fd=(y2-y1)/delta
print("Gradient calculation=%f, Finite difference gradient=%f"%(dydx.squeeze(),dydx_fd.squeeze()))

def plot_derivatives(K, D):
    all_weights, all_biases = InitParams(K,D)
    x_in = np.arange(-2,2, 4.0/256.0)
    x_in = np.resize(x_in, (1,len(x_in)))
    dydx,y = CalcInputOutputGradient(x_in, all_weights, all_biases)
    fig,ax = plt.subplots()
    ax.plot(np.squeeze(x_in), np.squeeze(dydx), 'b-')
    ax.set_xlim(-2,2)
    ax.set_xlabel(r'Input, $x$')
    ax.set_ylabel(r'Gradient, $dy/dx$')
    ax.set_title('No layers = %d'%(K))
    plt.show()

D=200
K=1
plot_derivatives(K,D)

def Autocorr(dydx):
    ac=np.correlate(dydx,dydx,mode='same')
    return ac

def plot_autocorr(K,D):
    all_weights,all_bias=InitParams(K,D)
    x_in=np.arange(-2.0,2.0,4.0/256)
    x_in=np.resize(x_in,(1,len(x_in)))
    dydx,y=CalcInputOutputGradient(x_in,all_weights,all_bias)
    ac=Autocorr(np.squeeze(dydx))
    ac=ac/ac[128]
    y=ac[128:]
    x=np.squeeze(x_in)[128:]
    fig,ax=plt.subplots()
    ax.plot(x,y,'b-')
    ax.set_xlim([0,2])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('No layers=%d'%(K))
    plt.show()

D=200
K=1
plot_autocorr(K,D)
D=200
K=50
plot_autocorr(K,D)
