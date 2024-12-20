import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
N=6
D=8
X=np.random.normal(size=(D,N))
# print(X)

H=2
H_D=int(D/H)
np.random.seed(0)
omega_q1=np.random.normal(size=(H_D,D))
omega_k1=np.random.normal(size=(H_D,D))
omega_v1=np.random.normal(size=(H_D,D))
beta_q1=np.random.normal(size=(H_D,1))
beta_k1=np.random.normal(size=(H_D,1))
beta_v1=np.random.normal(size=(H_D,1))
omega_q2=np.random.normal(size=(H_D,D))
omega_k2=np.random.normal(size=(H_D,D))
omega_v2=np.random.normal(size=(H_D,D))
beta_q2=np.random.normal(size=(H_D,1))
beta_k2=np.random.normal(size=(H_D,1))
beta_v2=np.random.normal(size=(H_D,1))
omega_c=np.random.normal(size=(D,D))

def Softmax_cols(data_in):
    exp_values=np.exp(data_in)
    denom=np.sum(exp_values,axis=0)
    softmax=exp_values/denom
    return softmax

def MSSA(X,omega_v1,omega_q1,omega_k1,beta_v1,beta_q1,beta_k1,omega_v2,omega_q2,omega_k2,beta_v2,beta_q2,beta_k2,omega_c):
    queries1=beta_q1+np.dot(omega_q1,X)
    queries2=beta_q2+np.dot(omega_q2,X)
    keys1=beta_k1+np.dot(omega_k1,X)
    keys2=beta_k2+np.dot(omega_k2,X)
    values1=beta_v1+np.dot(omega_v1,X)
    values2=beta_v2+np.dot(omega_v2,X)
    attention1=Softmax_cols(np.dot(keys1.T,queries1))
    attention2=Softmax_cols(np.dot(keys2.T,queries2))
    head1=np.dot(values1,attention1)
    head2=np.dot(values2,attention2)
    head=np.hstack((head1.T,head2.T))
    X_prime=np.dot(omega_c,head.T)
    return X_prime

X_prime=MSSA(X,omega_v1,omega_q1,omega_k1,beta_v1,beta_q1,beta_k1,omega_v2,omega_q2,omega_k2,beta_v2,beta_q2,beta_k2,omega_c)
np.set_printoptions(precision=3)
print("Your Answer: ")
print(X_prime)

print("True values:")
print("[[-21.207  -5.373 -20.933  -9.179 -11.319 -17.812]")
print(" [ -1.995   7.906 -10.516   3.452   9.863  -7.24 ]")
print(" [  5.479   1.115   9.244   0.453   5.656   7.089]")
print(" [ -7.413  -7.416   0.363  -5.573  -6.736  -0.848]")
print(" [-11.261  -9.937  -4.848  -8.915 -13.378  -5.761]")
print(" [  3.548  10.036  -2.244   1.604  12.113  -2.557]")
print(" [  4.888  -5.814   2.407   3.228  -4.232   3.71 ]")
print(" [  1.248  18.894  -6.409   3.224  19.717  -5.629]]")
