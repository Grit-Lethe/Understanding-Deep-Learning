import numpy as np
import torch
np.set_printoptions(precision=3,floatmode="fixed")
torch.set_printoptions(precision=3)

def ConvPytorch(image,conv_weights,stride=1,pad=1):
    image_tensor=torch.from_numpy(image)
    conv_weights_tensor=torch.from_numpy(conv_weights)
    output_tensor=torch.nn.functional.conv2d(image_tensor,conv_weights_tensor,stride=stride,padding=pad)
    return (output_tensor.numpy())

def ConvNumpy1(image,weights,pad=1):
    if pad !=0:
        image=np.pad(image,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    batchSize,channelsIn,imageHeightIn,imageWidthIn=image.shape
    channelsOut,channelsIn,kernelHeight,kernelWidth=weights.shape
    imageHeightOut=np.floor(1+imageHeightIn-kernelHeight).astype(int)
    imageWidthOut=np.floor(1+imageWidthIn-kernelWidth).astype(int)
    out=np.zeros((batchSize,channelsOut,imageHeightOut,imageWidthOut),dtype=np.float32)
    for c_y in range(imageHeightOut):
        for c_x in range(imageHeightOut):
            for c_kernel_y in range(kernelHeight):
                for c_kernel_x in range(kernelWidth):
                    orig_y=c_y+c_kernel_y
                    orig_x=c_x+c_kernel_x
                    this_pxiel_value=image[0,0,orig_y,orig_x]
                    this_weight=weights[0,0,c_kernel_y,c_kernel_x]
                    out[0,0,c_y,c_x]+=np.sum(this_pxiel_value*this_weight) 
    return out

np.random.seed(1)
n_batch=1
image_height=4
image_width=6
channels_in=1
kernel_size=3
channels_out=1
input_image=np.random.normal(size=(n_batch,channels_in,image_height,image_width))
conv_weights=np.random.normal(size=(channels_out,channels_in,kernel_size,kernel_size))
conv_results_pytorch=ConvPytorch(input_image,conv_weights,stride=1,pad=1)
print("PyTorch Results")
print(conv_results_pytorch)
print("Your results")
conv_results_numpy=ConvNumpy1(input_image,conv_weights)
print(conv_results_numpy)
