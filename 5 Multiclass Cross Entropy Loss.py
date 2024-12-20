import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def ShallowNN(x,beta0,omega0,beta1,omega1):
    n_data=x.size
    x=np.reshape(x,(1,n_data))
    h1=ReLU(np.matmul(beta0,np.ones((1,n_data)))+np.matmul(omega0,x))
    model_out=np.matmul(beta1,np.ones((1,n_data)))+np.matmul(omega1,h1)
    return model_out

def GetParameters():
    beta0=np.array([[0.3],[-1.0],[0.5]])
    omega0=np.array([[-1.0],[1.8],[0.65]])
    beta1=np.array([[2.0],[-2.0],[0.0]])
    omega1=np.array([[-24.0,-8.0,50.0],
                     [-2.0,8.0,-30.0],
                     [16.0,-8.0,-8.0]])
    return beta0,omega0,beta1,omega1

def PlotMulticlassClassification(x_model,out_model,lambda_model,x_data=None,y_data=None,title=None):
    n_data=len(x_model)
    n_class=3
    x_model=np.squeeze(x_model)
    out_model=np.reshape(out_model,(n_class,n_data))
    lambda_model=np.reshape(lambda_model,(n_class,n_data))
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(7.0,3.5)
    fig.tight_layout(pad=3.0)
    ax[0].plot(x_model,out_model[0,:],'r')
    ax[0].plot(x_model,out_model[1,:],'g')
    ax[0].plot(x_model,out_model[2,:],'b')
    ax[0].set_xlabel('Input, $x$')
    ax[0].set_ylabel('Model Output')
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([-4,4])
    if title is not None:
        ax[0].set_title(title)
    ax[1].plot(x_model,lambda_model[0,:],'r')
    ax[1].plot(x_model,lambda_model[1,:],'g')
    ax[1].plot(x_model,lambda_model[2,:],'b')
    ax[1].set_xlabel('Input, $x$')
    ax[1].set_ylabel('$\lambda$ or Pr(y=k|x)')
    ax[1].set_xlim([0,1])
    ax[1].set_ylim([-4,4])
    if title is not None:
        ax[1].set_title(title)
    if x_data is not None:
        for i in range(len(x_data)):
            if y_data[i]==0:
                ax[1].plot(x_data[i],-0.05,'r.')
            if y_data[i]==1:
                ax[1].plot(x_data[i],-0.05,'g.')
            if y_data[i]==2:
                ax[1].plot(x_data[i],-0.05,'b.')
    plt.show()

def Softmax(model_out):
    exp_model_out=np.exp(model_out)
    sum_exp_model_out=np.sum(exp_model_out,axis=0,keepdims=True)
    softmax_model_out=exp_model_out/sum_exp_model_out
    return softmax_model_out

x_train = np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train = np.array([2,0,1,2,1,0,\
                    0,2,2,0,2,0,\
                    2,0,1,2,1,2, \
                    1,0])
beta0,omega0,beta1,omega1=GetParameters()
x_model=np.arange(0,1,0.01)
model_out=ShallowNN(x_model,beta0,omega0,beta1,omega1)
lambda_model=Softmax(model_out)
PlotMulticlassClassification(x_model,model_out,lambda_model,x_train,y_train)

def CategoricalDistribution(y,lambda_param):
    return np.array([lambda_param[row,i] for i,row in enumerate(y)])
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.2,CategoricalDistribution(np.array([[0]]),np.array([[0.2],[0.5],[0.3]]))))
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.5,CategoricalDistribution(np.array([[1]]),np.array([[0.2],[0.5],[0.3]]))))
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.3,CategoricalDistribution(np.array([[2]]),np.array([[0.2],[0.5],[0.3]]))))

