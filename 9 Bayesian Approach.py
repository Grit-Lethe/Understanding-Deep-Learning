import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

def TrueFunction(x):
    y=np.exp(np.sin(x*(2*3.1415926)))
    return y

def GenerateData(n_data,sigma_y=0.3):
    x=np.ones(n_data)
    for i in range(n_data):
        x[i]=np.random.uniform(i/n_data,(i+1)/n_data,1)
    y=np.ones(n_data)
    for i in range(n_data):
        y[i]=TrueFunction(x[i])
        y[i]+=np.random.normal(0,sigma_y,1)
    return x,y

def PlotFunction(x_func,y_func,x_data=None,y_data=None,x_model=None,y_model=None,sigma_func=None,sigma_model=None):
    fig,ax=plt.subplots()
    ax.plot(x_func,y_func,'k')
    if sigma_func is not None:
        ax.fill_between(x_func,y_func-2*sigma_func,y_func+2*sigma_func,color='lightgray')
    if x_data is not None:
        ax.plot(x_data,y_data,'o',color='#d18362')
    if x_model is not None:
        ax.plot(x_model,y_model,'-',color='#7fe7de')
    if sigma_model is not None:
        ax.fill_between(x_model,y_model-2*sigma_model,y_model+2*sigma_model,color='lightgray')
    ax.set_xlim(0,1)
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $y$')
    plt.show()

x_func=np.linspace(0,1.0,100)
y_func=TrueFunction(x_func)
sigma_func=0.3
n_data=15
x_data,y_data=GenerateData(n_data,sigma_func)
PlotFunction(x_func,y_func,x_data,y_data,sigma_func=sigma_func)

def Network(x,beta,omega):
    n_hidden=omega.shape[0]
    y=np.zeros_like(x)
    for c_hidden in range(n_hidden):
        line_vals=x-c_hidden/n_hidden
        h=line_vals*(line_vals>0)
        y=y+omega[c_hidden]*h
    y=y+beta
    return y

def ComputeH(x_data,n_hidden):
    psi1=np.ones((n_hidden+1,1))
    psi0=np.linspace(0.0,1.0,num=n_hidden,endpoint=False)*-1
    n_data=x_data.size
    H=np.ones((n_hidden+1,n_data))
    for i in range(n_hidden):
        for j in range(n_data):
            H[i,j]=psi1[i]*x_data[j]+psi0[i]
            if H[i,j]<0:
                H[i,j]=0
    return H

def ComputeParamMeanCovar(x_data,y_data,n_hidden,sigma_sq,sigma_p_sq):
    H=ComputeH(x_data,n_hidden)
    H_mean=np.mean(H,axis=1,keepdims=True)
    H_centered=H-H_mean
    phi_covar=np.matmul(H_centered.T,H_centered)/(H_centered.shape[1]-1)
    phi_mean=np.mean(H,axis=1,keepdims=True)
    return phi_mean,phi_covar

n_hidden=5
sigma_sq=sigma_func*sigma_func
sigma_p_sq=1000
phi_mean,phi_covar=ComputeParamMeanCovar(x_data,y_data,n_hidden,sigma_sq,sigma_p_sq)
x_model=x_func
y_model_mean=Network(x_model,phi_mean[-1],phi_mean[0:n_hidden])
PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model_mean)

# phi_sample1,phi_sample2=ComputeParamMeanCovar(x_data,y_data,n_hidden,sigma_sq,sigma_p_sq)
# y_model_sample1=Network(x_model,phi_sample1[-1],phi_sample1[0:n_hidden])
# y_model_sample2=Network(x_model,phi_sample2[-1],phi_sample2[0:n_hidden])
# PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model_sample1)
# PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model_sample2)

def Inference(x_star,x_data,y_data,sigma_sq,sigma_p_sq,n_hidden):
    h_star=ComputeH(x_star,n_hidden)
    H=ComputeH(x_data,n_hidden)
    H_transpose=H.T
    H_H=np.dot(H_transpose,H)
    H_H+=sigma_p_sq*np.eye(n_hidden)
    H_H_inv=np.linalg.inv(H_H)
    beta=np.dot(np.dot(H_H_inv,H_transpose),y_data)
    y_star_mean=np.dot(h_star,beta)
    y_star_var=sigma_sq+np.dot(h_star,np.dot(H_H_inv,h_star.T))
    return y_star_mean,y_star_var

x_model=x_func
y_model=np.zeros_like(x_model)
y_model_std=np.zeros_like(x_model)
for c_model in range(len(x_model)):
    y_star_mean,y_star_var=Inference(x_model[c_model]*np.ones((1,1)),x_data,y_data,sigma_sq,sigma_p_sq,n_hidden)
    y_model[c_model]=y_star_mean
    y_model_std[c_model]=np.sqrt(y_star_var)
PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model,sigma_model=y_model_std)
