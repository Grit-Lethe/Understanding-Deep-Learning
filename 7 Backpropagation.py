import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
K=5
D=6
Di=1
Do=1
all_weights=[None]*(K+1)
all_bias=[None]*(K+1)
all_weights[0]=np.random.normal(size=(D,Di))
all_weights[-1]=np.random.normal(size=(Do,D))
all_bias[0]=np.random.normal(size=(D,1))
all_bias[-1]=np.random.normal(size=(Do,1))
for layer in range(1,K):
    all_weights[layer]=np.random.normal(size=(D,D))
    all_bias[layer]=np.random.normal(size=(D,1))

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
    return net_output, all_f, all_h

net_input=np.ones((Di,1))*1.2
net_output,all_f,all_h=ComputeNetworkOutput(net_input,all_weights,all_bias)
print("True output = %3.3f, Your answer = %3.3f"%(1.907, net_output[0,0]))

def LeastSquaresLoss(net_output,y):
    lsl=np.sum((net_output-y)*(net_output-y))
    return lsl

def DLossDOutput(net_output,y):
    dldo=2*(net_output-y)
    return dldo

y=np.ones((Do,1))*20.0
loss=LeastSquaresLoss(net_output,y)
print("y = %3.3f Loss = %3.3f"%(y, loss))

def IndicatorFunction(x):
    x_in=np.array(x)
    x_in[x_in>=0]=1
    x_in[x_in<0]=0
    return x_in

def BackwardPass(all_weights,all_bias,all_f,all_h,y):
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
    return all_dl_dweights,all_dl_dbias

all_dl_dweights, all_dl_dbias = BackwardPass(all_weights, all_bias, all_f, all_h, y)
np.set_printoptions(precision=3)
all_dl_dweights_fd=[None]*(K+1)
all_dl_dbias_fd=[None]*(K+1)
delta_fd=0.000001
for layer in range(K):
    dl_dbias=np.zeros_like(all_dl_dbias[layer])
    for row in range(all_bias[layer].shape[0]):
        all_bias_copy=[np.array(x) for x in all_bias]
        all_bias_copy[layer][row]+=delta_fd
        network_output1,*_=ComputeNetworkOutput(net_input,all_weights,all_bias_copy)
        network_output2,*_=ComputeNetworkOutput(net_input,all_weights,all_bias)
        dl_dbias[row]=(LeastSquaresLoss(network_output1,y)-LeastSquaresLoss(network_output2,y))/delta_fd
    all_dl_dbias_fd[layer]=np.array(dl_dbias)
    print("------------------")
    print("Bias %d, derivatives from backprop:"%(layer))
    print(all_dl_dbias[layer])
    print("Bias %d, derivatives from finite differences"%(layer))
    print(all_dl_dbias_fd[layer])
    if np.allclose(all_dl_dbias_fd[layer],all_dl_dbias[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
        print("Success!  Derivatives match.")
    else:
        print("Failure!  Derivatives different.")

for layer in range(K):
    dl_dweight  = np.zeros_like(all_dl_dweights[layer])
    for row in range(all_weights[layer].shape[0]):
        for col in range(all_weights[layer].shape[1]):
            all_weights_copy = [np.array(x) for x in all_weights]
            all_weights_copy[layer][row][col] += delta_fd
            network_output_1, *_ = ComputeNetworkOutput(net_input, all_weights_copy, all_bias)
            network_output_2, *_ = ComputeNetworkOutput(net_input, all_weights, all_bias)
            dl_dweight[row][col] = (LeastSquaresLoss(network_output_1, y) - LeastSquaresLoss(network_output_2,y))/delta_fd
    all_dl_dweights_fd[layer] = np.array(dl_dweight)
    print("-----------------------------------------------")
    print("Weight %d, derivatives from backprop:"%(layer))
    print(all_dl_dweights[layer])
    print("Weight %d, derivatives from finite differences"%(layer))
    print(all_dl_dweights_fd[layer])
    if np.allclose(all_dl_dweights_fd[layer],all_dl_dweights[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
        print("Success!  Derivatives match.")
    else:
        print("Failure!  Derivatives different.")
