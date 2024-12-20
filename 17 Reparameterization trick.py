import numpy as np
import matplotlib.pyplot as plt

def ComputeExpectation(phi, n_samples):
    mu=phi*phi*phi
    sigma=(np.exp(phi))
    samples=np.random.normal(mu, sigma, size=(n_samples,1))
    expected_f=samples**2+np.sin(samples)
    expected_f_given_phi=np.mean(expected_f)
    return expected_f_given_phi

np.random.seed(0)
phi1=0.5
n_samples=100000000
expected_f_given_phi1=ComputeExpectation(phi1, n_samples)
print("Your value: ", expected_f_given_phi1, ", True value:  2.7650801613563116")

phi_vals=np.arange(-1.5,1.5,0.05)
expected_vals=np.zeros_like(phi_vals)
n_samples=1000000
for i in range(len(phi_vals)):
    expected_vals[i]=ComputeExpectation(phi_vals[i], n_samples)
fig, ax=plt.subplots()
ax.plot(phi_vals, expected_vals, 'r-')
ax.set_xlabel(r'Parameter $\phi$')
ax.set_ylabel(r'$\mathbb{E}_{Pr(x|\phi)}[f[x]]$')
plt.show()

def ComputeDfDxStar(x_star):
    deriv=2*x_star+np.cos(x_star)
    return deriv

def ComputeDxstarDphi(epsilon_star, phi):
    deriv=epsilon_star*np.exp(phi)+3*phi*phi
    return deriv

def ComputeDerivativeofExpectation(phi, n_samples):
    epsilon_star=np.random.normal(size=(n_samples,1))
    x_star=epsilon_star*np.exp(phi)+phi**3
    dxstardphi=ComputeDxstarDphi(epsilon_star, phi)
    dfdxstar=ComputeDfDxStar(x_star)
    dfdphi=dfdxstar*dxstardphi
    df_dphi=np.mean(dfdphi)
    return df_dphi

np.random.seed(0)
phi1=0.5
n_samples=100000000
deriv=ComputeDerivativeofExpectation(phi1, n_samples)
print("Your value: ", deriv, ", True value:  5.726338035051403")

phi_vals=np.arange(-1.5,1.5,0.05)
deriv_vals=np.zeros_like(phi_vals)
n_samples=1000000
for i in range(len(phi_vals)):
    deriv_vals[i]=ComputeDerivativeofExpectation(phi_vals[i], n_samples)
fig, ax=plt.subplots()
ax.plot(phi_vals, deriv_vals, 'r-')
ax.set_xlabel(r'Parameter $\phi$')
ax.set_ylabel(r'$\mathbb{E}_{Pr(x|\phi)}[f[x]]$')
plt.show()

def DlogprxGivenPhi(x, phi):
    y=np.log((1/(np.exp(phi)*np.sqrt(2*3.1415926)))*np.exp(-((x-phi**3)**2)/(2*(np.exp(phi))**2)))
    deriv=(np.log((1/(np.exp(phi+0.000001)*np.sqrt(2*3.1415926)))*np.exp(-((x-(phi+0.000001)**3)**2)/(2*(np.exp(phi+0.000001))**2)))-y)/0.000001
    return deriv

def ComputeDerivativeofExpectationScoreFunction(phi, n_samples):
    mu=phi**3
    sigma=np.exp(phi)
    samples=np.random.normal(mu, sigma, size=(n_samples,1))
    expected_f=samples**2+np.sin(samples)
    ef=expected_f*DlogprxGivenPhi(samples, phi)
    deriv=np.mean(ef)
    return deriv

np.random.seed(0)
phi1=0.5
n_samples=100000000
deriv=ComputeDerivativeofExpectationScoreFunction(phi1, n_samples)
print("Your value: ", deriv, ", True value:  5.724609927313369")

phi_vals=np.arange(-1.5,1.5,0.05)
deriv_vals=np.zeros_like(phi_vals)
n_samples=1000000
for i in range(len(phi_vals)):
    deriv_vals[i]=ComputeDerivativeofExpectationScoreFunction(phi_vals[i], n_samples)
fig, ax=plt.subplots()
ax.plot(phi_vals, deriv_vals, 'r-')
ax.set_xlabel(r'Parameter $\phi$')
ax.set_ylabel(r'$\mathbb{E}_{Pr(x|\phi)}[f[x]]$')
plt.show()

n_estimate=100
n_sample=1000
phi=0.3
reparam_estimates=np.zeros((n_estimate,1))
score_function_estimates=np.zeros((n_estimate,1))
for i in range(n_estimate):
    reparam_estimates[i]=ComputeDerivativeofExpectation(phi, n_samples)
    score_function_estimates[i]=ComputeDerivativeofExpectationScoreFunction(phi, n_samples)
print("Variance of reparameterization estimator", np.var(reparam_estimates))
print("Variance of score function estimator", np.var(score_function_estimates))
