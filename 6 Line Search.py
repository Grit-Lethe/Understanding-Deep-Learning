import numpy as np
import matplotlib.pyplot as plt

def LossFunction(phi):
    return 1-0.5*np.exp(-(phi-0.65)*(phi-0.65)/0.1)-0.45*np.exp(-(phi-0.35)*(phi-0.35)/0.02)

def DrawFunction(LossFunction,a=None,b=None,c=None,d=None):
    phi_plot=np.arange(0,1,0.01)
    fig,ax=plt.subplots()
    ax.plot(phi_plot,LossFunction(phi_plot),'r-')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$L[\phi]$')
    if a is not None and b is not None and c is not None and d is not None:
        plt.axvspan(a,d,facecolor='k',alpha=0.2)
        ax.plot([a,a],[0,1],'b-')
        ax.plot([b,b],[0,1],'b-')
        ax.plot([c,c],[0,1],'b-')
        ax.plot([d,d],[0,1],'b-')
    plt.show()

DrawFunction(LossFunction)

def LineSearch(LossFunction,thresh=0.0001,max_iter=10,draw_flag=False):
    a=0
    b=0.33
    c=0.66
    d=1.0
    n_iter=0
    while np.abs(b-c)>thresh and n_iter<max_iter:
        n_iter=n_iter+1
        lossa=LossFunction(a)
        lossb=LossFunction(b)
        lossc=LossFunction(c)
        lossd=LossFunction(d)
        if draw_flag:
            DrawFunction(LossFunction,a,b,c,d)
        print('Iter %d, a=%3.3f, b=%3.3f, c=%3.3f, d=%3.3f'%(n_iter, a,b,c,d))
        if lossa<lossb and lossc and lossd:
            b=(a+b)/2
            c=(a+c)/2
            d=(a+d)/2
        if lossb<lossc:
            d=c
            b=(2*a+d)/3
            c=(a+2*d)/3
        if lossc<lossb:
            a=b
            b=(2*a+d)/3
            c=(a+2*d)/3
    soln=(b+c)/2
    return soln
soln = LineSearch(LossFunction, draw_flag=True)
print('Soln = %3.3f, loss = %3.3f'%(soln,LossFunction(soln)))
