import numpy as np

np.random.seed(1)
N=8
D=4
A = np.array([[0,1,0,1,0,0,0,0],
              [1,0,1,1,1,0,0,0],
              [0,1,0,0,1,0,0,0],
              [1,1,0,0,1,0,0,0],
              [0,1,1,1,0,1,0,1],
              [0,0,0,0,1,0,1,1],
              [0,0,0,0,0,1,0,0],
              [0,0,0,0,1,1,0,0]])
X=np.random.normal(size=(D,N))

omega=np.random.normal(size=(D,D))
beta=np.random.normal(size=(D,1))
phi=np.random.normal(size=(1,2*D))

def Softmax(data_in):
    exp_values=np.exp(data_in)
    denom=np.sum(exp_values,axis=0)
    denom=np.matmul(np.ones((data_in.shape[0],1)),denom[np.newaxis,:])
    softmax=exp_values/denom
    return softmax

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def GraphAttention(X,omega,beta,phi,A):
    i=np.ones((1,8))
    X_prime=np.dot(beta,i)+np.dot(omega,X)
    S=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            # xm=np.zeros((4,1))
            # xn=np.zeros((4,1))
            # xm=X_prime[:,i]
            # xm=xm.reshape(-1,1)
            # xn=X_prime[:,j]
            # xn=xn.reshape(-1,1)
            # x=np.vstack((xm,xn))
            x=np.vstack((X_prime[:,i].reshape(-1,1),X_prime[:,j].reshape(-1,1)))
            S[i,j]=np.squeeze(ReLU(phi@x))
    I=np.eye(A.shape[0])
    AI=A+I
    zero_positions=AI==0
    S[zero_positions]=-1e20
    attention=Softmax(S)
    attentions=X_prime@attention
    output=ReLU(attentions)
    return output

np.set_printoptions(precision=3)
output=GraphAttention(X,omega,beta,phi,A)
print("Correct answer is:")
print("[[0.    0.028 0.37  0.    0.97  0.    0.    0.698]")
print(" [0.    0.    0.    0.    1.184 0.    2.654 0.  ]")
print(" [1.13  0.564 0.    1.298 0.268 0.    0.    0.779]")
print(" [0.825 0.    0.    1.175 0.    0.    0.    0.  ]]]")
print("Your answer is:")
print(output)
