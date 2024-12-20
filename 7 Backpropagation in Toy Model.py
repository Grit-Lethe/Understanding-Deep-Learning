import numpy as np

def Function(x,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3):
    y=beta3+omega3*np.cos(beta2+omega2*np.exp(beta1+omega1*np.sin(beta0+omega0*x)))
    return y

def Loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3):
    diff=Function(x,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3)-y
    return diff*diff

beta0=1.0
beta1=2.0
beta2=-3.0
beta3=0.4
omega0=0.1
omega1=-0.4
omega2=2.0
omega3=3.0
x=2.3
y=2.0
lifunc=Loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3)
print('li=%3.3f'%lifunc)

dldbeta3_func = 2 * (beta3 +omega3 * np.cos(beta2 + omega2 * np.exp(beta1+omega1 * np.sin(beta0+omega0 * x)))-y)
dldomega0_func = -2 *(beta3 +omega3 * np.cos(beta2 + omega2 * np.exp(beta1+omega1 * np.sin(beta0+omega0 * x)))-y) * \
              omega1 * omega2 * omega3 * x * np.cos(beta0 + omega0 * x) * np.exp(beta1 +omega1 * np.sin(beta0 + omega0 * x)) *\
              np.sin(beta2 + omega2 * np.exp(beta1+ omega1* np.sin(beta0+omega0 * x)))

dldomega0_fd = (Loss(x,y,beta0,beta1,beta2,beta3,omega0+0.00001,omega1,omega2,omega3)-Loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3))/0.00001
print('dydomega0: Function value = %3.3f, Finite difference value = %3.3f'%(dldomega0_func,dldomega0_fd))

f0=beta0+omega0*x
h1=np.sin(f0)
f1=beta1+omega1*h1
h2=np.exp(f1)
f2=beta2+omega2*h2
h3=np.cos(f2)
f3=beta3+omega3*h3
li=(f3-y)**2
print("f0: true value = %3.3f, your value = %3.3f"%(1.230, f0))
print("h1: true value = %3.3f, your value = %3.3f"%(0.942, h1))
print("f1: true value = %3.3f, your value = %3.3f"%(1.623, f1))
print("h2: true value = %3.3f, your value = %3.3f"%(5.068, h2))
print("f2: true value = %3.3f, your value = %3.3f"%(7.137, f2))
print("h3: true value = %3.3f, your value = %3.3f"%(0.657, h3))
print("f3: true value = %3.3f, your value = %3.3f"%(2.372, f3))
print("li original = %3.3f, l_i from forward pass = %3.3f"%(lifunc, li))

dldf3=2*(f3-y)
dldh3=omega3*dldf3
dldf2=-np.sin(f2)*dldh3
dldh2=omega2*dldf2
dldf1=np.exp(f1)*dldh2
dldh1=omega1*dldf1
dldf0=np.cos(f0)*dldh1
print("dldf3: true value = %3.3f, your value = %3.3f"%(0.745, dldf3))
print("dldh3: true value = %3.3f, your value = %3.3f"%(2.234, dldh3))
print("dldf2: true value = %3.3f, your value = %3.3f"%(-1.683, dldf2))
print("dldh2: true value = %3.3f, your value = %3.3f"%(-3.366, dldh2))
print("dldf1: true value = %3.3f, your value = %3.3f"%(-17.060, dldf1))
print("dldh1: true value = %3.3f, your value = %3.3f"%(6.824, dldh1))
print("dldf0: true value = %3.3f, your value = %3.3f"%(2.281, dldf0))

dldbeta3=dldf3
dldomega3=h3*dldf3
dldbeta2=dldf2
dldomega2=h2*dldf2
dldbeta1=dldf1
dldomega1=h1*dldf1
dldbeta0=dldf0
dldomega0=x*dldf0
print('dldbeta3: Your value = %3.3f, True value = %3.3f'%(dldbeta3, 0.745))
print('dldomega3: Your value = %3.3f, True value = %3.3f'%(dldomega3, 0.489))
print('dldbeta2: Your value = %3.3f, True value = %3.3f'%(dldbeta2, -1.683))
print('dldomega2: Your value = %3.3f, True value = %3.3f'%(dldomega2, -8.530))
print('dldbeta1: Your value = %3.3f, True value = %3.3f'%(dldbeta1, -17.060))
print('dldomega1: Your value = %3.3f, True value = %3.3f'%(dldomega1, -16.079))
print('dldbeta0: Your value = %3.3f, True value = %3.3f'%(dldbeta0, 2.281))
print('dldomega0: Your value = %3.3f, Function value = %3.3f, Finite difference value = %3.3f'%(dldomega0, dldomega0_func, dldomega0_fd))
