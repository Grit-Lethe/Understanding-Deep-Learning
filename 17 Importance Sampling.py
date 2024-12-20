import numpy as np
import matplotlib.pyplot as plt

def f(y):
    Y=np.exp(-(y-1)**4)
    return Y

def Pry(y):
    pry=(1/np.sqrt(2*np.pi))*np.exp(-0.5*y*y)
    return pry

fig, ax=plt.subplots()
y=np.arange(-10,10,0.01)
ax.plot(y, f(y), 'r-', label='f$[y]$')
ax.plot(y, Pry(y), 'b-', label='$Pr(y)$')
ax.set_xlabel('$y$')
ax.legend()
plt.show()

def ComputeExpectation(n_samples):
    y_i=np.random.normal(size=(n_samples,1))
    expectation=f(y_i)
    expectation=np.mean(expectation)
    return expectation

np.random.seed(0)
n_samples=100000000
expected_f=ComputeExpectation(n_samples)
print("Your value: ", expected_f, ", True value:  0.43160702267383166")

def ComputeMeanVariance(n_sample):
    n_estimate=10000
    estimates=np.zeros((n_estimate,1))
    for i in range(n_estimate):
        estimates[i]=ComputeExpectation(n_sample.astype(int))
    return np.mean(estimates), np.var(estimates)

n_sample_all = np.array([1.,2,3,4,5,6,7,8,9,10,15,20,25,30,45,50,60,70,80,90,100,150,200,250,300,350,400,450,500])
mean_all=np.zeros_like(n_sample_all)
variance_all=np.zeros_like(n_sample_all)
for i in range(len(n_sample_all)):
    print("Computing mean and variance for expectation with %d samples"%(n_sample_all[i]))
    mean_all[i],variance_all[i]=ComputeMeanVariance(n_sample_all[i])

fig, ax=plt.subplots()
ax.semilogx(n_sample_all, mean_all, 'r-', label='Mean Estimate')
ax.fill_between(n_sample_all, mean_all-2*np.sqrt(variance_all), mean_all+2*np.sqrt(variance_all))
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Mean of Estimate')
ax.plot([0,500], [0.43160702267383166,0.43160702267383166], 'k--', label='True Value')
ax.legend()
plt.show()

def f2(y):
    Y=20.466*np.exp(-(y-3)**4)
    return Y

fig, ax=plt.subplots()
y=np.arange(-10,10,0.01)
ax.plot(y, f2(y), 'r-', label='f$[y]$')
ax.plot(y, Pry(y), 'b-', label='$Pr(y)$')
ax.set_xlabel('$y$')
ax.legend()
plt.show()

def ComputeExpectation2(n_samples):
    y=np.random.normal(size=(n_samples,1))
    expectation=np.mean(f2(y))
    return expectation

n_samples=100000000
expected_f2=ComputeExpectation2(n_samples)
print("Expected value: ", expected_f2)

def ComputeMeanVariance2(n_sample):
    n_estimate=10000
    estimates=np.zeros((n_estimate,1))
    for i in range(n_estimate):
        estimates[i]=ComputeExpectation2(n_sample.astype(int))
    return np.mean(estimates), np.var(estimates)

mean_all2=np.zeros_like(n_sample_all)
variance_all2=np.zeros_like(n_sample_all)
for i in range(len(n_sample_all)):
    print("Computing variance for expectation with %d samples"%(n_sample_all[i]))
    mean_all2[i], variance_all2[i]=ComputeMeanVariance2(n_sample_all[i])

fig,ax1 = plt.subplots()
ax1.semilogx(n_sample_all, mean_all,'r-',label='mean estimate')
ax1.fill_between(n_sample_all, mean_all-2*np.sqrt(variance_all), mean_all+2*np.sqrt(variance_all))
ax1.set_xlabel("Number of samples")
ax1.set_ylabel("Mean of estimate")
ax1.plot([1,500],[0.43160702267383166,0.43160702267383166],'k--',label='true value')
ax1.set_ylim(-5,6)
ax1.set_title("First function")
ax1.legend()

fig2,ax2 = plt.subplots()
ax2.semilogx(n_sample_all, mean_all2,'r-',label='mean estimate')
ax2.fill_between(n_sample_all, mean_all2-2*np.sqrt(variance_all2), mean_all2+2*np.sqrt(variance_all2))
ax2.set_xlabel("Number of samples")
ax2.set_ylabel("Mean of estimate")
ax2.plot([0,500],[0.43160428638892556,0.43160428638892556],'k--',label='true value')
ax2.set_ylim(-5,6)
ax2.set_title("Second function")
ax2.legend()
plt.show()

