import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

orig_4_4=np.array([[1,3,5,3],[6,2,0,8],[4,6,1,4],[2,8,0,3]])

def Subsample(x_in):
    x_out=np.zeros((int(np.ceil(x_in.shape[0]/2)),int(np.ceil(x_in.shape[1]/2))))
    for i in range(x_out.shape[0]):
        for j in range(x_out.shape[1]):
            start_i=i*2
            start_j=j*2
            patch=x_in[start_i:start_i+2,start_j:start_j+2]
            x_out[i,j]=np.mean(patch)
    return x_out

print("Original:")
print(orig_4_4)
print("Subsampled:")
print(Subsample(orig_4_4))

image=Image.open('test_image.png')
data=asarray(image)
data_subsample=Subsample(data)
plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(data_subsample, cmap='gray')
plt.show()
data_subsample2 = Subsample(data_subsample)
plt.figure(figsize=(5,5))
plt.imshow(data_subsample2, cmap='gray')
plt.show()
data_subsample3 = Subsample(data_subsample2)
plt.figure(figsize=(5,5))
plt.imshow(data_subsample3, cmap='gray')
plt.show()

def Maxpool(x_in):
    x_out=np.zeros((int(np.ceil(x_in.shape[0]/2)),int(np.ceil(x_in.shape[1]/2))))
    for i in range(0,x_in.shape[0],2):
        for j in range(0,x_in.shape[1],2):
            window=x_in[i:i+2,j:j+2]
            x_out[int(i/2),int(j/2)]=np.max(window)
    return x_out

print("Original:")
print(orig_4_4)
print("Maxpooled:")
print(Maxpool(orig_4_4))

data_maxpool=Maxpool(data)
plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool, cmap='gray')
plt.show()
data_maxpool2 = Maxpool(data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool2, cmap='gray')
plt.show()
data_maxpool3 = Maxpool(data_maxpool2)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool3, cmap='gray')
plt.show()

def Meanpool(x_in):
    x_out=np.zeros((int(np.ceil(x_in.shape[0]/2)),int(np.ceil(x_in.shape[1]/2))))
    for i in range(x_out.shape[0]):
        for j in range(x_out.shape[1]):
            start_i=i*2
            start_j=j*2
            window=x_in[start_i:start_i+2,start_j:start_j+2]
            x_out[i,j]=np.mean(window)
    return x_out

def Bilinear(x_in):
    h=x_in.shape[0]
    w=x_in.shape[1]
    x_out=np.zeros((h*2,w*2))
    for i in range(h*2):
        for j in range(w*2):
            x=i/2
            y=j/2
            x0=int(np.floor(x))
            x1=min(x0+1,h-1)
            y0=int(np.floor(y))
            y1=min(y0+1,w-1)
            dx=x-x0
            dy=y-y0
            w00=(1-dx)*(1-dy)
            w10=dx*(1-dy)
            w01=(1-dy)*dy
            w11=dx*dy
            x_out[i,j]=w00*x_in[x0,y0]+w10*x_in[x1,y0]+w01*x_in[x0,y1]+w11*x_in[x1,y1]
    return x_out

print("Original:")
print(orig_4_4)
print("Bilinear:")
print(Bilinear(orig_4_4))

data_bilinear = Bilinear(data);

plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear, cmap='gray')
plt.show()
data_bilinear2 = Bilinear(data_bilinear)
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear2, cmap='gray')
plt.show()
