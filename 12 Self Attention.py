import numpy as np

np.random.seed(3)
N=3
D=4
all_x=[]
for n in range(N):
    all_x.append(np.random.normal(size=(D,1)))
print(all_x)

np.random.seed(0)
omega_q=np.random.normal(size=(D,D))
omega_k=np.random.normal(size=(D,D))
omega_v=np.random.normal(size=(D,D))
beta_q=np.random.normal(size=(D,1))
beta_k=np.random.normal(size=(D,1))
beta_v=np.random.normal(size=(D,1))

all_queries=[]
all_keys=[]
all_values=[]
for x in all_x:
    query=beta_q+np.dot(omega_q,x)
    key=beta_k+np.dot(omega_k,x)
    values=beta_v+np.dot(omega_v,x)
    all_queries.append(query)
    all_keys.append(key)
    all_values.append(values)

def Softmax(items_in):
    exps=np.exp(items_in)
    items_out=exps/(np.sum(exps,axis=0))
    return items_out

all_x_prime=[]
for n in range(N):
    all_km_qn=[]
    for key in all_keys:
        dot_product=np.dot(key.T,all_queries[n])
        all_km_qn.append(dot_product)
    attention=Softmax(np.squeeze(all_km_qn))
    print("Attentions for output ",n)
    print(attention)
    x_prime=np.dot(attention,np.squeeze(all_values))
    all_x_prime.append(x_prime)

print("x_prime_0_calculated:", all_x_prime[0].transpose())
print("x_prime_0_true: [[ 0.94744244 -0.24348429 -0.91310441 -0.44522983]]")
print("x_prime_1_calculated:", all_x_prime[1].transpose())
print("x_prime_1_true: [[ 1.64201168 -0.08470004  4.02764044  2.18690791]]")
print("x_prime_2_calculated:", all_x_prime[2].transpose())
print("x_prime_2_true: [[ 1.61949281 -0.06641533  3.96863308  2.15858316]]")

def SoftmaxCols(data_in):
    exp_values=np.exp(data_in)
    denom=np.sum(exp_values,axis=0)
    denom=np.matmul(np.ones((data_in.shape[0],1)),denom[np.newaxis,:])
    softmax=exp_values/denom
    return softmax

def SelfAttention(X,omega_v,omega_q,omega_k,beta_v,beta_q,beta_k):
    queries=beta_q+np.dot(omega_q,X)
    keys=beta_k+np.dot(omega_k,X)
    values=beta_v+np.dot(omega_v,X)
    attention=SoftmaxCols(np.dot(keys.T,queries))
    X_prime=np.dot(values,attention)
    return X_prime

X=np.zeros((D,N))
X[:,0]=np.squeeze(all_x[0])
X[:,1]=np.squeeze(all_x[1])
X[:,2]=np.squeeze(all_x[2])
X_prime=SelfAttention(X,omega_v,omega_q,omega_k,beta_v,beta_q,beta_k)
print(X_prime)
