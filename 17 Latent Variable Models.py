import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.colors import ListedColormap
from matplotlib import cm

def f(z):
    x1=np.exp(np.sin(2+z*3.675))*0.5
    x2=np.cos(2+z*2.85)
    return x1, x2

def Draw3DProjection(z, pr_z, x1, x2):
    alpha=pr_z/np.max(pr_z)
    ax=plt.axes(projection='3d')
    fig=plt.gcf()
    fig.set_size_inches(18.5,10.5)
    for i in range(len(z)-1):
        ax.plot([z[i], z[i+1]], [x1[i], x1[i+1]], [x2[i], x2[i+1]], 'r-', alpha=pr_z[i])
    ax.set_xlabel('$z$')
    ax.set_ylabel('$x1$')
    ax.set_zlabel('$x2$')
    ax.set_xlim(-3,3)
    ax.set_ylim(0,2)
    ax.set_zlim(-1,1)
    ax.set_box_aspect((3,1,1))
    plt.show()

def GetPrior(z):
    y=scipy.stats.multivariate_normal.pdf(z)
    return y

z=np.arange(-3.0,3.0,0.01)
pr_z=GetPrior(z)
x1, x2=f(z)
Draw3DProjection(z, pr_z, x1, x2)

def PlotHeatmap(x1_mesh, x2_mesh, y_mesh, x1_samples=None, x2_samples=None, title=None):
    my_colormap_vals_hex =('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', '320f0a', '330f0b', '34100b', '35110c', '36110d', '37120e', '38120f', '39130f', '3a1410', '3b1411', '3c1511', '3d1612', '3e1613', '3f1713', '401714', '411814', '421915', '431915', '451a16', '461b16', '471b17', '481c17', '491d18', '4a1d18', '4b1e19', '4c1f19', '4d1f1a', '4e201b', '50211b', '51211c', '52221c', '53231d', '54231d', '55241e', '56251e', '57261f', '58261f', '592720', '5b2821', '5c2821', '5d2922', '5e2a22', '5f2b23', '602b23', '612c24', '622d25', '632e25', '652e26', '662f26', '673027', '683027', '693128', '6a3229', '6b3329', '6c342a', '6d342a', '6f352b', '70362c', '71372c', '72372d', '73382e', '74392e', '753a2f', '763a2f', '773b30', '783c31', '7a3d31', '7b3e32', '7c3e33', '7d3f33', '7e4034', '7f4134', '804235', '814236', '824336', '834437', '854538', '864638', '874739', '88473a', '89483a', '8a493b', '8b4a3c', '8c4b3c', '8d4c3d', '8e4c3e', '8f4d3f', '904e3f', '924f40', '935041', '945141', '955242', '965343', '975343', '985444', '995545', '9a5646', '9b5746', '9c5847', '9d5948', '9e5a49', '9f5a49', 'a05b4a', 'a15c4b', 'a35d4b', 'a45e4c', 'a55f4d', 'a6604e', 'a7614e', 'a8624f', 'a96350', 'aa6451', 'ab6552', 'ac6552', 'ad6653', 'ae6754', 'af6855', 'b06955', 'b16a56', 'b26b57', 'b36c58', 'b46d59', 'b56e59', 'b66f5a', 'b7705b', 'b8715c', 'b9725d', 'ba735d', 'bb745e', 'bc755f', 'bd7660', 'be7761', 'bf7862', 'c07962', 'c17a63', 'c27b64', 'c27c65', 'c37d66', 'c47e67', 'c57f68', 'c68068', 'c78169', 'c8826a', 'c9836b', 'ca846c', 'cb856d', 'cc866e', 'cd876f', 'ce886f', 'ce8970', 'cf8a71', 'd08b72', 'd18c73', 'd28d74', 'd38e75', 'd48f76', 'd59077', 'd59178', 'd69279', 'd7937a', 'd8957b', 'd9967b', 'da977c', 'da987d', 'db997e', 'dc9a7f', 'dd9b80', 'de9c81', 'de9d82', 'df9e83', 'e09f84', 'e1a185', 'e2a286', 'e2a387', 'e3a488', 'e4a589', 'e5a68a', 'e5a78b', 'e6a88c', 'e7aa8d', 'e7ab8e', 'e8ac8f', 'e9ad90', 'eaae91', 'eaaf92', 'ebb093', 'ecb295', 'ecb396', 'edb497', 'eeb598', 'eeb699', 'efb79a', 'efb99b', 'f0ba9c', 'f1bb9d', 'f1bc9e', 'f2bd9f', 'f2bfa1', 'f3c0a2', 'f3c1a3', 'f4c2a4', 'f5c3a5', 'f5c5a6', 'f6c6a7', 'f6c7a8', 'f7c8aa', 'f7c9ab', 'f8cbac', 'f8ccad', 'f8cdae', 'f9ceb0', 'f9d0b1', 'fad1b2', 'fad2b3', 'fbd3b4', 'fbd5b6', 'fbd6b7', 'fcd7b8', 'fcd8b9', 'fcdaba', 'fddbbc', 'fddcbd', 'fddebe', 'fddfbf', 'fee0c1', 'fee1c2', 'fee3c3', 'fee4c5', 'ffe5c6', 'ffe7c7', 'ffe8c9', 'ffe9ca', 'ffebcb', 'ffeccd', 'ffedce', 'ffefcf', 'fff0d1', 'fff2d2', 'fff3d3', 'fff4d5', 'fff6d6', 'fff7d8', 'fff8d9', 'fffada', 'fffbdc', 'fffcdd', 'fffedf', 'ffffe0')
    my_colormap_vals_dec = np.array([int(element,base=16) for element in my_colormap_vals_hex])
    r = np.floor(my_colormap_vals_dec/(256*256))
    g = np.floor((my_colormap_vals_dec - r *256 *256)/256)
    b = np.floor(my_colormap_vals_dec - r * 256 *256 - g * 256)
    my_colormap = ListedColormap(np.vstack((r,g,b)).transpose()/255.0)
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    ax.contourf(x1_mesh,x2_mesh,y_mesh,256,cmap=my_colormap)
    ax.contour(x1_mesh,x2_mesh,y_mesh,8,colors=['#80808080'])
    if title is not None:
        ax.set_title(title);
    if x1_samples is not None:
        ax.plot(x1_samples, x2_samples, 'c.')
    ax.set_xlim([-0.5,2.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    plt.show()

def GetLikehood(x1_mesh, x2_mesh, z_val):
    x1, x2=f(z_val)
    mn=scipy.stats.multivariate_normal([x1, x2], [[sigma_sq, 0], [0, sigma_sq]])
    pr_x1_x2_given_z_val=mn.pdf(np.dstack((x1_mesh, x2_mesh)))
    return pr_x1_x2_given_z_val

sigma_sq=0.04
z_val=1.8
x1_mesh, x2_mesh=np.meshgrid(np.arange(-0.5, 2.5, 0.01), np.arange(-1.5, 1.5, 0.01))
pr_x1_x2_given_z_val=GetLikehood(x1_mesh, x2_mesh, z_val)
PlotHeatmap(x1_mesh, x2_mesh, pr_x1_x2_given_z_val, title="Conditional distribution $Pr(x_1,x_2|z)$")

pr_x1_x2=np.zeros((len(x1_mesh), len(x2_mesh)))
for z in np.arange(-3.0,3.0,0.01):
    pr_x1_x2+=GetLikehood(x1_mesh, x2_mesh, z)*GetPrior(z)*0.01
PlotHeatmap(x1_mesh, x2_mesh, pr_x1_x2, title="Data density $Pr(x_1,x_2)$")

def DrawSamples(n_sample):
    z_samples=np.random.normal(loc=0, scale=1, size=n_sample)
    x1_val, x2_val=f(z_samples)
    x1_samples=np.random.normal(loc=x1_val, scale=1, size=n_sample)
    x2_samples=np.random.normal(loc=x2_val, scale=1, size=n_sample)
    return x1_samples, x2_samples

x1_samples, x2_samples=DrawSamples(5000)
PlotHeatmap(x1_mesh, x2_mesh, pr_x1_x2, x1_samples, x2_samples, title="Data density $Pr(x_1,x_2)$")

def GetPosterior(x1, x2):
    z=np.arange(-3,3,0.01)
    pr_z_given_x1_x2 = np.zeros_like(z)
    
    # 假设x1和x2是独立同分布的，且与z有关
    # 这里我们使用正态分布作为似然函数
    # 先验分布是均匀的，因此我们可以直接计算似然
    for i, z_val in enumerate(z):
        # 计算给定z值时，x1和x2的似然
        x1_val, x2_val=f(z_val)
        likelihood_x1 = scipy.stats.norm.pdf(x1, loc=x1_val, scale=1)
        likelihood_x2 = scipy.stats.norm.pdf(x2, loc=x2_val, scale=1)
        
        # 将似然相乘（在这个简单例子中，我们假设x1和x2的似然是独立的）
        joint_likelihood = likelihood_x1 * likelihood_x2
        
        # 归一化后验分布
        pr_z_given_x1_x2[i] = joint_likelihood
    
    # 归一化后验分布
    pr_z_given_x1_x2 /= np.sum(pr_z_given_x1_x2)
    
    return z, pr_z_given_x1_x2

x1=0.9
x2=-0.9
z, pr_z_given_x1_x2=GetPosterior(x1, x2)
fig, ax = plt.subplots()
ax.plot(z, pr_z_given_x1_x2, 'r-')
ax.set_xlabel("Latent variable $z$")
ax.set_ylabel("Posterior probability $Pr(z|x_{1},x_{2})$")
ax.set_xlim([-3,3])
ax.set_ylim([0,1.5 * np.max(pr_z_given_x1_x2)])
plt.show()
