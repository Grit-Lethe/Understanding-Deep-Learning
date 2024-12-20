import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

def TrueFunction(x):
    y=np.exp(np.sin(x*(2*3.1416)))
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
    reg_value=0.00001
    regMat=reg_value*np.identity(n_hidden+1)
    regMat[0,0]=0
    ATA=np.matmul(np.transpose(A),A)+regMat
    ATAInv=np.linalg.inv(ATA)
    ATAInvAT=np.matmul(ATAInv,np.transpose(A))
    beta_omega=np.matmul(ATAInvAT,y)
    beta=beta_omega[0]
    omega=beta_omega[1:]
    return beta,omega

beta,omega=FitModelClosedForm(x_data,y_data,n_hidden=14)
x_model=np.linspace(0,1,100)
y_model=Network(x_model,beta,omega)
PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model)
mean_sq_error=np.mean((y_model-y_func)*(y_model-y_func))
print(f"Mean Square Error={mean_sq_error:3.3f}")

n_model=4
all_y_model=np.zeros((n_model,len(y_model)))
for c_model in range(n_model):
    resampled_indices=np.random.choice(n_data,size=n_data,replace=True)
    x_data_resampled=x_data[resampled_indices]
    y_data_resampled=y_data[resampled_indices]
    beta,omega=FitModelClosedForm(x_data_resampled,y_data_resampled,n_hidden=14)
    y_model_resampled=Network(x_model,beta,omega)
    all_y_model[c_model,:]=y_model_resampled
    PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model_resampled)
    mean_sq_error=np.mean((y_model_resampled-y_func)*(y_model_resampled-y_func))
    print("Mean Square Error={mean_sq_error:3.3f}")

y_model_media=np.median(all_y_model,axis=0)
PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model_media)
mean_sq_error = np.mean((y_model_media-y_func) * (y_model_media-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")

y_model_mean=np.mean(all_y_model,axis=0)
PlotFunction(x_func,y_func,x_data,y_data,x_model,y_model_mean)
mean_sq_error = np.mean((y_model_mean-y_func) * (y_model_mean-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")
