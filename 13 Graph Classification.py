import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

G=nx.Graph()
G.add_edge('0:H','2:C')
G.add_edge('1:H','2:C')
G.add_edge('3:H','2:C')
G.add_edge('2:C','5:C')
G.add_edge('4:H','5:C')
G.add_edge('6:C','5:C')
G.add_edge('7:O','5:C')
G.add_edge('8:H','7:O')
nx.draw(G,nx.spring_layout(G,seed=0),with_labels=True,node_size=600)
plt.show()

A=np.array([[0,0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [1,1,0,1,0,1,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0],
            [0,0,1,0,1,0,1,1,0],
            [0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,1,0]])

X=np.zeros((118,9))
X[0,0]=1
X[5,1]=1
X[7,2]=1
# print(X[0:15,:])

def ReLU(preactivation):
    activation=preactivation.clip(0.0)
    return activation

def Sigmoid(x):
    y=1.0/(1.0+np.exp(-x))
    return y

K=3
D=200
np.random.seed(1)
omega0=np.random.normal(size=(D,118))*2.0/D
beta0=np.random.normal(size=(D,1))*2.0/D
omega1=np.random.normal(size=(D,D))*2.0/D
beta1=np.random.normal(size=(D,1))*2.0/D
omega2=np.random.normal(size=(D,D))*2.0/D
beta2=np.random.normal(size=(D,1))*2.0/D
omega3=np.random.normal(size=(1,D))
beta3=np.random.normal(size=(1,1))

def GraphNeuralNetwork(A,X,omega0,beta0,omega1,beta1,omega2,beta2,omega3,beta3):
    l=np.ones((9,1))
    h1=ReLU(beta0@(l.T)+omega0@X@(A+np.eye(9)))
    h2=ReLU(beta1@(l.T)+omega1@h1@(A+np.eye(9)))
    h3=ReLU(beta2@(l.T)+omega2@h2@(A+np.eye(9)))
    # h4=ReLU(beta3.T+omega3@h3@(A+np.eye(9)))
    f=Sigmoid(beta3+omega3@h3@(l/9))
    return f

f=GraphNeuralNetwork(A,X,omega0,beta0,omega1,beta1,omega2,beta2,omega3,beta3)
print("Your Value is %3f: "%(f[0,0]),"True Value of f: 0.310843")

P = np.array([[0,1,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,1],
              [1,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1,0,0]])
A_permuted=(P.T)@A@P
X_permuted=X@P
f=GraphNeuralNetwork(A_permuted,X_permuted,omega0,beta0,omega1,beta1,omega2,beta2,omega3,beta3)
print("Your Value is %3f: "%(f[0,0]),"True Value of f: 0.310843")
