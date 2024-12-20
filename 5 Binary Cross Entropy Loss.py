import numpy as np
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
    beta0=np.zeros((3,1))
    omega0=np.zeros((3,1))
    beta1=np.zeros((1,1))
    omega1=np.zeros((1,3))
    beta0[0,0]=0.3
    beta0[1,0]=-1.0
    beta0[2,0]=-0.5
    omega0[0,0]=-1.0
    omega0[1,0]=1.8
    omega0[2,0]=0.65
    beta1[0,0]=2.6
    omega1[0,0]=-24.0
    omega1[0,1]=-8.0
    omega1[0,2]=50.0
    return beta0,omega0,beta1,omega1

def PlotBinaryClassification(x_model,out_model,lambda_model,x_data=None,y_data=None,title=None):
    x_model=np.squeeze(x_model)
    out_model=np.squeeze(out_model)
    lambda_model=np.squeeze(lambda_model)
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(7.0,3.5)
    fig.tight_layout(pad=3.0)
    ax[0].plot(x_model,out_model)
    ax[0].set_xlabel(r'Input, $x$')
    ax[0].set_ylabel(r'Model Output')
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([-4,4])
    if title is not None:
        ax[0].set_title(title)
    ax[1].plot(x_model,lambda_model)
    ax[1].set_xlabel(r'Input, $x$')
    ax[1].set_ylabel(r'$\lambda$ or Pr(y=1|x)')
    ax[1].set_xlim([0,1])
    ax[1].set_ylim([-0.05,1.05])
    if title is not None:
        ax[1].set_title(title)
    if x_data is not None:
        ax[1].plot(x_data,y_data,'ko')
    plt.show()

def Sigmoid(model_out):
    sig_model_out=1/(1+np.exp(-model_out))
    return sig_model_out

x_train=np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                  0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                  0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                  0.87168699,0.58858043])
y_train=np.array([0,1,1,0,0,1,\
                  1,0,0,1,0,1,\
                  0,1,1,0,1,0, \
                  1,1])
beta0,omega0,beta1,omega1=GetParameters()
x_model=np.arange(0,1,0.01)
model_out=ShallowNN(x_model,beta0,omega0,beta1,omega1)
lambda_model=Sigmoid(model_out)
PlotBinaryClassification(x_model,model_out,lambda_model,x_train,y_train)

def BernoulliDistribution(y,lambda_param):
    # a=[1-lambda_param,lambda_param]
    # b=[1-y,y]
    # c=np.power(a,b)
    # prob=np.prod(c)
    prob=np.where(y==1,lambda_param,1-lambda_param)
    return prob

print("Correct answer = %3.3f, Your answer = %3.3f"%(0.8,BernoulliDistribution(0,0.2)))
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.2,BernoulliDistribution(1,0.2)))

def ComputeLikehood(y_train,lambda_param):
    probabilities=[BernoulliDistribution(y,lambda_param) for y in y_train]
    likehood=np.prod(probabilities)
    return likehood

model_out=ShallowNN(x_train,beta0,omega0,beta1,omega1)
lambda_train=Sigmoid(model_out)
likehood=ComputeLikehood(y_train,lambda_train)
print("Correct answer = %9.9f, Your answer = %9.9f"%(0.000070237,likehood))

def ComputeNegativeLogLikehood(y_train,lambda_param):
    probabilities=[BernoulliDistribution(y,lambda_param) for y in y_train]
    probabilities=np.log(probabilities)
    nll=-np.sum(probabilities)
    return nll

model_out=ShallowNN(x_train,beta0,omega0,beta1,omega1)
lambda_train=Sigmoid(model_out)
nll=ComputeNegativeLogLikehood(y_train,lambda_train)
print("Correct answer = %9.9f, Your answer = %9.9f"%(9.563639387,nll))

beta1vals=np.arange(-2.0,6.0,0.01)
likehoods=np.zeros_like(beta1vals)
nlls=np.zeros_like(beta1vals)
for count in range(len(beta1vals)):
    beta1[0,0]=beta1vals[count]
    model_out=ShallowNN(x_train,beta0,omega0,beta1,omega1)
    lambda_train=Sigmoid(model_out)
    likehoods[count]=ComputeLikehood(y_train,lambda_train)
    nlls[count]=ComputeNegativeLogLikehood(y_train,lambda_train)
    if count%200==0:
        model_out=ShallowNN(x_model,beta0,omega0,beta1,omega1)
        lambda_model=Sigmoid(model_out)
        PlotBinaryClassification(x_model,model_out,lambda_model,x_train,y_train,title="beta_1[0]=%3.3f"%(beta1[0,0]))

fig, ax = plt.subplots()
fig.tight_layout(pad=5.0)
likelihood_color = 'tab:red'
nll_color = 'tab:blue'
ax.set_xlabel('beta_1[0]')
ax.set_ylabel('likelihood', color = likelihood_color)
ax.plot(beta1vals, likehoods, color = likelihood_color)
ax.tick_params(axis='y', labelcolor=likelihood_color)
ax1 = ax.twinx()
ax1.plot(beta1vals, nlls, color = nll_color)
ax1.set_ylabel('negative log likelihood', color = nll_color)
ax1.tick_params(axis='y', labelcolor = nll_color)
plt.axvline(x = beta1vals[np.argmax(likehoods)], linestyle='dotted')
plt.show()

print("Maximum likelihood = %f, at beta_1=%3.3f"%( (likehoods[np.argmax(likehoods)],beta1vals[np.argmax(likehoods)])))
print("Minimum negative log likelihood = %f, at beta_1=%3.3f"%( (nlls[np.argmin(nlls)],beta1vals[np.argmin(nlls)])))
beta1[0,0] = beta1vals[np.argmin(nlls)]
model_out = ShallowNN(x_model, beta0, omega0, beta1, omega1)
lambda_model = Sigmoid(model_out)
PlotBinaryClassification(x_model, model_out, lambda_model, x_train, y_train, title="beta_1[0]=%3.3f"%(beta1[0,0]))
