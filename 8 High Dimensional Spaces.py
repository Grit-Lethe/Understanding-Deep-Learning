import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci

np.random.seed(0)
n_data=1000
n_dim=2
x_2d=np.random.normal(size=(n_dim,n_data))
n_dim=100
x_100d=np.random.normal(size=(n_dim,n_data))
n_dim=1000
x_1000d=np.random.normal(size=(n_dim,n_data))

def DistanceRatio(x):
    x=np.array(x)
    dist_matrix=np.sqrt(np.sum((x[:,np.newaxis,:]-x[:,:,np.newaxis])**2,axis=2))
    np.fill_diagonal(dist_matrix,np.inf)
    smallest_dist=np.inf
    largest_dist=0
    for i in range(dist_matrix.shape[0]):
        for j in range(i+1,dist_matrix.shape[1]):
            if dist_matrix[i,j]<smallest_dist:
                smallest_dist=dist_matrix[i,j]
            if dist_matrix[i,j]>largest_dist:
                largest_dist=dist_matrix[i,j]
    dist_ratio=largest_dist/smallest_dist
    return dist_ratio

print('Ratio of largest to smallest distance 2D: %3.3f'%(DistanceRatio(x_2d)))
print('Ratio of largest to smallest distance 100D: %3.3f'%(DistanceRatio(x_100d)))
print('Ratio of largest to smallest distance 1000D: %3.3f'%(DistanceRatio(x_1000d)))

def VolumeOfHypersphere(diameter,dimensions):
    pi=np.pi
    volume=(pi**(dimensions/2))/sci.gamma(dimensions/2+1)*(diameter/2)**dimensions
    return volume

diameter=1.0
for c_dim in range(1,11):
    print("Volume of unit diameter hypersphere in %d dimensions is %3.3f"%(c_dim, VolumeOfHypersphere(diameter, c_dim)))

def GPVIOP(dimensions):
    pi=np.pi
    diameter1=1.0
    diameter2=0.99
    volume1=(pi**(dimensions/2))/sci.gamma(dimensions/2+1)*(diameter1/2)**dimensions
    volume2=(pi**(dimensions/2))/sci.gamma(dimensions/2+1)*(diameter2/2)**dimensions
    ratio=(volume1-volume2)/volume1
    return ratio

for c_dim in [1,2,10,20,50,100,150,200,250,300]:
  print('Proportion of volume in outer 1 percent of radius in %d dimensions =%3.3f'%(c_dim, GPVIOP(c_dim)))
