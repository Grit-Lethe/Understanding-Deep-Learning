import numpy as np
import matplotlib.pyplot as plt

def InitParams(K,D,sigma_sq_omega):
    np.random.seed(0)
    Di=1
    Do=1
    all_weights=[None]*(K+1)
    all_bias=[None]*(K+1)
    all_weights[0]=np.random.normal(size=(D,Di))*np.sqrt(sigma_sq_omega)
    all_weights[-1]=np.random.normal(size=(Do,D))*np.sqrt(sigma_sq_omega)
    all_bias[0]=np.zeros((D,1))
    all_bias[-1]=np.zeros((Do,1))
    for layer in range(1,K):
        all_weights[layer]=np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
        all_bias[layer]=np.zeros((D,1))
    return all_weights,all_bias

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def ComputeNetworkOutput(net_input,all_weights,all_bias):
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

K=5
D=8
Di=1
Do=1
sigma_sq_omega=1.0
all_weights,all_bias=InitParams(K,D,sigma_sq_omega)
n_data=1000
data_in=np.random.normal(size=(1,n_data))
net_output,all_f,all_h=ComputeNetworkOutput(data_in,all_weights,all_bias)
for layer in range(1,K+1):
    print("Layer %d, std of hidden units = %3.3f"%(layer, np.std(all_h[layer])))

def LeastSquaresLoss(net_output,y):
    a=np.sum((net_output-y)*(net_output-y))
    return a

def DLossDOutput(net_output,y):
    b=2*(net_output-y)
    return b

def IndicatorFunction(x):
    x_in=np.array(x)
    x_in[x_in>=0]=1
    x_in[x_in<0]=0
    return x_in

def BackwardPass(all_weights,all_bias,all_f,all_h,y):
    K=len(all_weights)-1
    all_dl_dweights=[None]*(K+1)
    all_dl_dbias=[None]*(K+1)
    all_dl_df=[None]*(K+1)
    all_dl_dh=[None]*(K+1)
    all_dl_df[K]=np.array(DLossDOutput(all_f[K],y))
    for layer in range(K,-1,-1):
        all_dl_dbias[layer]=np.array(all_dl_df[layer])
        all_dl_dweights[layer]=np.matmul(all_dl_df[layer],all_h[layer].transpose())
        all_dl_dh[layer]=np.matmul(all_weights[layer].transpose(),all_dl_df[layer])
        if layer > 0:
            all_dl_df[layer-1]=IndicatorFunction(all_f[layer-1])*all_dl_dh[layer]
    return all_dl_dweights,all_dl_dbias,all_dl_dh,all_dl_df

K=5
D=8
Di=1
Do=1
sigma_sq_omega=1.0
all_weights,all_bias=InitParams(K,D,sigma_sq_omega)
n_data=100
aggregate_dl_df=[None]*(K+1)
for layer in range(1,K):
    aggregate_dl_df[layer]=np.zeros((D,n_data))
for c_data in range(n_data):
    data_in=np.random.normal(size=(1,1))
    y=np.zeros((1,1))
    net_output,all_f,all_h=ComputeNetworkOutput(data_in,all_weights,all_bias)
    all_dl_dweights,all_dl_dbias,all_dl_dh,all_dl_df=BackwardPass(all_weights,all_bias,all_f,all_h,y)
    for layer in range(1,K):
        aggregate_dl_df[layer][:,c_data]=np.squeeze(all_dl_df[layer])
for layer in range(1,K):
    print("Layer %d, std of dl_dh = %3.3f"%(layer, np.std(aggregate_dl_df[layer].ravel())))
