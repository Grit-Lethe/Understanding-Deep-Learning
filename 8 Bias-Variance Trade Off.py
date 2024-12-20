import numpy as np
import matplotlib.pyplot as plt

def TrueFunction(x):
    y=np.exp(np.sin(x*(2*3.1413)))
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
    ax.plot(x_func,y_func,'k-')
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
np.random.seed(1)
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

def FitModelClosedForm(x,y,n_hidden):
    n_data=len(x)
    A=np.ones((n_data,n_hidden+1))
    for i in range(n_data):
        for j in range(1,n_hidden+1):
            A[i,j]=x[i]-(j-1)/n_hidden
            if A[i,j]<0:
                A[i,j]=0
    beta_omega=np.linalg.lstsq(A,y,rcond=None)[0]
    beta=beta_omega[0]
    omega=beta_omega[1:]
    return beta,omega

beta,omega=FitModelClosedForm(x_data,y_data,n_hidden=3)
x_model=np.linspace(0,1,100)
y_model=Network(x_model,beta,omega)
PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model)

def GetModelMeanVariance(n_data,n_datasets,n_hidden,sigma_func):
    y_model_all=np.zeros((n_datasets,n_data))
    for c_dataset in range(n_datasets):
        x_data=np.random.normal(0,1,n_data)
        y_data=np.random.normal(0,sigma_func,n_data)
        beta,omega=FitModelClosedForm(x_data,y_data,n_hidden)
        y_model=Network(x_data,beta,omega)
        y_model_all[c_dataset,:]=y_model
    mean_model=np.mean(y_model_all,axis=0)
    std_model=np.std(y_model_all,axis=0)
    return mean_model,std_model

n_datasets=100
n_data=100
sigma_func=0.3
n_hidden=5
mean_model,std_model=GetModelMeanVariance(n_data,n_datasets,n_hidden,sigma_func)
PlotFunction(x_func,y_func,x_model=x_model,y_model=mean_model,sigma_model=std_model)
