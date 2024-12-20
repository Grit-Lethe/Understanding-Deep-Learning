import numpy as np
import matplotlib.pyplot as plt
from math import comb

def NumberRegions(Di,D):
    N=0
    for j in range(Di+1):
        Ni=comb(D,j)
        N=N+Ni
    return N

N=NumberRegions(10,50)
print(f"Di=10, D=50, Number of Regions={int(N)}, True Value=13432735556")

try:
    N=NumberRegions(10,8)
    print(f"Di=10, D=8, Number of Regions={int(N)}, True Value=256")
except Exception as error:
    print("An Exception Occurred: ",error)

D=8
Di=10
N=np.power(2,D)
N2=NumberRegions(D,D)
print(f"Di=10, D=8, Number of Regions={int(N)}, Number of Regions={int(N2)}, True value=256")

dims=np.array([1,5,10,50,100])
regions=np.zeros((dims.shape[0],1000))
for c_dim in range(dims.shape[0]):
    Di=dims[c_dim]
    print(f"Counting Regions for {Di} Input Dimensions")
    for D in range(1000):
        regions[c_dim,D]=NumberRegions(np.min([Di,D]),D)
fig,ax=plt.subplots()
ax.semilogy(regions[0,:],'k')
ax.semilogy(regions[1,:],'b')
ax.semilogy(regions[2,:],'m')
ax.semilogy(regions[3,:],'c')
ax.semilogy(regions[4,:],'y')
ax.legend(['$Di$=1','$Di$=5','$Di$=10','$Di$=50','$Di$=100'])
ax.set_xlabel("Number of Hidden Units, D")
ax.set_ylabel("Number of Regions, N")
plt.xlim([0,1000])
plt.ylim([1e1,1e150])
plt.show()

def NumberParameters(Di,D):
    N=(Di+2)*D+1
    return N

N=NumberParameters(10,8)
print(f"Di=10, D=8, Number of Parameters={int(N)}, True Value=97")

dims=np.array([1,5,10,50,100])
regions=np.zeros((dims.shape[0],200))
params=np.zeros((dims.shape[0],200))
for c_dim in range(dims.shape[0]):
    Di=dims[c_dim]
    print(f"Counting Regions for {Di} Input Dimensions")
    for c_hidden in range(1,200):
        D=int(c_hidden*500/Di)
        params[c_dim,c_hidden]=Di*D+D+D+1
        regions[c_dim,c_hidden]=NumberRegions(np.min([Di,D]),D)
fig,ax=plt.subplots()
ax.semilogy(params[0,:],regions[0,:],'k')
ax.semilogy(params[1,:],regions[1,:],'b')
ax.semilogy(params[2,:],regions[2,:],'m')
ax.semilogy(params[3,:],regions[3,:],'c')
ax.semilogy(params[4,:],regions[4,:],'y')
ax.legend(['$Di$=1','$Di$=5','$Di$=10','$Di$=50','$Di$=100'])
ax.set_xlabel("Number of Parameters, D")
ax.set_ylabel("Number of Regions, N")
plt.xlim([0,100000])
plt.ylim([1e1,1e150])
plt.show()
