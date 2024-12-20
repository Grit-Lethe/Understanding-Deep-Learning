import numpy as np
import matplotlib.pyplot as plt
import math

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def ShallowNN(x,beta0,omega0,beta1,omega1):
    n_data=x.size
    x=np.reshape(x,(1,n_data))
    h1=ReLU(np.matmul(beta0,np.ones((1,n_data))))+np.matmul(omega0,x)
    y=np.matmul(beta1,np.ones((1,n_data)))+np.matmul(omega1,h1)
    return y

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
    beta1[0,0]=0.1
    omega1[0,0]=-2.0
    omega1[0,1]=-1.0
    omega1[0,2]=7.0
    return beta0,omega0,beta1,omega1

def PlotUnivariateRegression(x_model,y_model,x_data=None,y_data=None,sigma_model=None,title=None):
    x_model=np.squeeze(x_model)
    y_model=np.squeeze(y_model)
    fig,ax=plt.subplots()
    ax.plot(x_model,y_model)
    if sigma_model is not None:
        ax.fill_between(x_model,y_model-2*sigma_model,y_model+2*sigma_model,color='lightgray')
    ax.set_xlabel(r'Input, $x$')
    ax.set_ylabel(r'Output, $y$')
    ax.set_xlim([0,1])
    ax.set_ylim([-1,1])
    ax.set_aspect(0.5)
    if title is not None:
        ax.set_title(title)
    if x_data is not None:
        ax.plot(x_data,y_data,'ko')
    plt.show()

x_train=np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train=np.array([-0.25934537,0.18195445,0.651270150,0.13921448,0.09366691,0.30567674,\
                    0.372291170,0.20716968,-0.08131792,0.51187806,0.16943738,0.3994327,\
                    0.019062570,0.55820410,0.452564960,-0.1183121,0.02957665,-1.24354444, \
                    0.248038840,0.26824970])
beta0,omega0,beta1,omega1=GetParameters()
sigma=0.2
x_model=np.arange(0,1,0.01)
y_model=ShallowNN(x_model,beta0,omega0,beta1,omega1)
PlotUnivariateRegression(x_model,y_model,x_train,y_train,sigma_model=sigma)

def NormalDistribution(y,mu,sigma):
    C1=1/(sigma*np.sqrt(2*math.pi))
    C2=-((y-mu)**2)/(2*sigma*sigma)
    E=C1*np.exp(C2)
    prob=E
    return prob

print("Correct Answer=%3.3f, Your Answer=%3.3f"%(0.119,NormalDistribution(1,-1,2.3)))

y_gauss=np.arange(-5,5,0.1)
mu0=0
sigma0=1.0
gauss_prob=NormalDistribution(y_gauss,mu0,sigma0)
fig,ax=plt.subplots()
ax.plot(y_gauss,gauss_prob)
ax.set_xlabel(r'Input, $y$')
ax.set_ylabel(r'Probability $Pr(y)$')
ax.set_xlim([-5,5])
ax.set_ylim([0,1.0])
plt.show()

def ComputeLikehood(y_train,mu,sigma):
    Prob=NormalDistribution(y_train,mu,sigma)
    likehood=np.prod(Prob)
    return likehood

mu_pred1=ShallowNN(x_train,beta0,omega0,beta1,omega1)
sigma1=0.2
likehood=ComputeLikehood(y_train,mu_pred1,sigma1)
print("Correct Answer=%9.9f, Your Answer=%9.9f"%(0.000010624,likehood))

def CNLLH(y_train,mu,sigma):
    Prob=NormalDistribution(y_train,mu,sigma)
    L=np.log(Prob)
    nll=-np.sum(L)
    return nll

mu_pred2=ShallowNN(x_train,beta0,omega0,beta1,omega1)
sigma2=0.2
nll=CNLLH(y_train,mu_pred2,sigma2)
print("Correct Answer=%9.9f, Your Answer=%9.9f"%(11.452419564,nll))

def CSOS(y_train,y_pred):
    SOS=np.sum((y_train-y_pred)**2)
    return SOS

y_pred=mu_pred=ShallowNN(x_train,beta0,omega0,beta1,omega1)
SumS=CSOS(y_train,y_pred)
print("Correct answer = %9.9f, Your answer = %9.9f"%(2.020992572,SumS))

beta1vals=np.arange(0.0,1.0,0.01)
likehoods=np.zeros_like(beta1vals)
nlls=np.zeros_like(beta1vals)
sum_squares=np.zeros_like(beta1vals)
sigma=0.2
for count in range(len(beta1vals)):
    beta1[0,0]=beta1vals[count]
    mu_pred=y_pred=ShallowNN(x_train,beta0,omega0,beta1,omega1)
    likehoods[count]=ComputeLikehood(y_train,mu_pred,sigma)
    nlls[count]=CNLLH(y_train,mu_pred,sigma)
    sum_squares[count]=CSOS(y_train,y_pred)
    if count%20==0:
        y_model=ShallowNN(x_model,beta0,omega0,beta1,omega1)
        PlotUnivariateRegression(x_model,y_model,x_train,y_train,sigma_model=sigma,title='beta1=%3.3f'%(beta1[0,0]))

fig, ax = plt.subplots(1,2)
fig.set_size_inches(10.5, 5.5)
fig.tight_layout(pad=10.0)
likelihood_color = 'tab:red'
nll_color = 'tab:blue'
ax[0].set_xlabel('beta_1[0]')
ax[0].set_ylabel('likelihood', color = likelihood_color)
ax[0].plot(beta1vals, likehoods, color = likelihood_color)
ax[0].tick_params(axis='y', labelcolor=likelihood_color)
ax00 = ax[0].twinx()
ax00.plot(beta1vals, nlls, color = nll_color)
ax00.set_ylabel('negative log likelihood', color = nll_color)
ax00.tick_params(axis='y', labelcolor = nll_color)
plt.axvline(x = beta1vals[np.argmax(likehoods)], linestyle='dotted')
ax[1].plot(beta1vals, sum_squares); ax[1].set_xlabel('beta_1[0]'); ax[1].set_ylabel('sum of squares')
plt.show()
