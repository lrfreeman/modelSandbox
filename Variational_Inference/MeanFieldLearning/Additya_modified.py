# Modified code from Author: Additya Singh's code

# Os Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from tqdm import tqdm
import argparse

# Custom
from genimages import gen_images

def safe_log(a, tol=1e-100):
    '''Useful helper function to prevent nans'''
    return np.where(a > tol, np.log(a), -1e20)

def calc_free_energy(X, mu, sigma, pie, lam, xmu_corr=None, mu_corr=None):
    '''
    Calculate the free energy given the current parameters and (approx) posterior

    args:
        Parameters of model:
        X: N x D, dataset
        mu: D x K, factor means
        pie: K, factor prior probabilities of being present

        Paramteres of posterior:
        lam: N x K, bernoulli parameters of fully factorized posterior

        Optional precomputed values:
        xmu_corr: N x K, X@mu
        mu_corr: K x K, mu.T@mu

    returns: int, free energy
    '''
    N,_ = lam.shape
    D = X.shape[1]

    # Avoid performing recomputation
    if xmu_corr is None:
        xmu_corr = X @ mu # N x K
    if mu_corr is None:
        mu_corr = mu.T @ mu # K x K

    constantEnergy = -N*D*safe_log(2*np.pi)/2 - N*D*safe_log(sigma) - np.sum(X**2)/(2*(sigma**2))
    
    # Selects the diagonal of mu_corr, and converts it to a row vector using [None,:]
    muCorMinusxmuCor = np.diag(mu_corr)[None,:] - 2 * xmu_corr
    
    # Lam @ mu_corr includes when i=j, so we subtract the diagonal where i=j, and use np,diag twice to ensure the diagonal is of correct shape
    lamjmuimuj = lam @ mu_corr - lam @ np.diag(np.diag(mu_corr))
    
    FreeEnergy = constantEnergy - np.sum(lam*(muCorMinusxmuCor+lamjmuimuj)/(2*sigma**2) 
                 - lam*safe_log(lam/pie[None,:]) 
                 - (1-lam)*safe_log((1-lam)/(1-pie[None,:])) )
                 
    return FreeEnergy

def e_step(method, X, mu, sigma, pie, **kwargs):
    '''
    Performs the e-step by using a variational posterior inference method

    args:
        X: N x D, dataset
        mu: D x K, factor means
        pie: K, factor prior probabilities of being present
        kwargs: Other arguments that are passed directly to the inference methods

    returns: dict, aux
        dict['ES']: N x K matrix, expectations of each factor
        dict['ESS']: K x K matrix, expectations of each pair of factors, 
                    summed over data points
        aux: Dictionary of auxiliary info. This could for example hold information
                like free energy traces that we might want to plot
    '''

    if method == 'vi':
        out = mean_field(X, mu, sigma, pie, **kwargs)
        lam = out.pop('posterior')

        # Compute ESS, vectorising lam @ lam
        lamlam = np.einsum('nK,nk->Kk', lam, lam)
        
        # Set the diagonal to not be squares
        lamlam[np.arange(K),np.arange(K)] = np.sum(lam,axis=0)
        return {'ES': lam, 'ESS': lamlam}, out
    
    raise NotImplementedError

def mean_field(X,mu,sigma,pie,**kwargs):
    '''
    Computes mean-field variational approximation, parametrized by lambda.

    Involves fixed point iteration where all lambda_in are updated
    simulatenously based on the previous values (simultaneous updates
    allow for efficient vectorization)

    Args:
        X: NxD data matrix
        mu: DxK matrix of means
        sigma: float, variance param
        pie: K vector of priors on s
        kwargs: dict of additional params, required to contain:
            lambda0: NxK initial values for lambda, each between 0 and 1
            tol: float, threshold
            maxsteps: int, maximum number of steps of the fixed point equations

    returns: a dict containing
        posterior is NxK, converged params
        free_energy_trace is a trajectory of lower bounds on the likelihood per 
            step of fixed point iteration. the length of F is thus at most maxsteps 
    '''
    lambda0 = kwargs['lambda0']
    maxsteps = kwargs['maxsteps']
    tol = 1e-5 if 'tol' not in kwargs else kwargs['tol']

    N,_ = X.shape
    K = mu.shape[1]
    xmu_corr = X @ mu # N x K
    mu_corr = mu.T @ mu # K x K

    lam = lambda0
    F_vals = [-1e20]
        
    for it in range(maxsteps):
        F = calc_free_energy(X, mu, sigma, pie, lam, xmu_corr=xmu_corr, mu_corr=mu_corr)

        # IF free energy changes by less than tol, we have converged
        if np.abs((F-F_vals[-1])/F_vals[-1]) < tol:
            F_vals.append(F)
            break

        F_vals.append(F)
        
        # For every data point, update the lambda values
        for n in range(N):
            for i in range(K):
                lam[n,i] = pie[i]/(pie[i]+(1-pie[i])*np.exp((mu_corr[i,i] - 2*xmu_corr[n,i] + 2*np.sum(lam[n,:]*mu_corr[i,:]) - 2*lam[n,i]*mu_corr[i,i])/(2*sigma**2)))
    return {'posterior': lam, 'free_energy_trace': F_vals[1:]}

def m_step(X, ES, ESS):
    '''
    Computes M-step updates for parameters.

    Args:
    X: NxD data matrix
    ES: NxK matrix of E_q[s]
    ESS: KxK matrix of E_q[s@s.T], summed over N
    pie: K vector of priors on s

    Returns:
    mu: DxK matrix of new mu_i values
    sigma: float, new std dev
    pie: K vector of new prior probs
    '''
    mu = np.linalg.solve(ESS, ES.T@X).T
    sigma = np.sqrt((np.sum(X**2) + np.sum((mu.T@mu) * ESS) - 2*np.sum((X@mu) * ES)) / np.prod(X.shape))
    pie = np.mean(ES,axis=0)
        
    return mu, sigma, pie

def learn_bin_factors(e_method, 
                      X, K, 
                      iterations,
                      pie_init=None,
                      mu_init=None, mu_scale=None,
                      use_last=False,
                      max_e_steps=50,
                      tol=1e-10,rs=0):
    
    N, _ = X.shape
    np.random.seed(rs)

    if pie_init is not None:
        pie = np.ones(K)*pie_init
        
    else:
        pie = np.random.random(K)
        
    if mu_init == 'equal':
        mu = np.tile(np.mean(X, axis=0)[:,None], (1,K))
    else:
        # Every X should be made of roughly K/2 components at initialization
        # Thus, we sample the mu's to be normally distributed according to X/(K/2)
        # So that at leat the magnitudes line up initially
        mu = np.mean(X, axis=0)[:,None] + np.std(X, axis=0)[:,None]*np.random.randn(K)[None,:]
    if mu_scale:
        mu /= mu_scale
    else:
        mu /= np.sum(pie)
    # A very generous estimate of sigma
    sigma = (np.max(X)-np.min(X))/4

    lambda_init = np.random.random((N,K))
    lambda_init /= np.sum(lambda_init,axis=1)[:,None]
    lam = lambda_init

    estep_Fs = []
    mstep_Fs = []
    E_curves = []

    for i in tqdm(range(iterations)):
        mstep_Fs.append(calc_free_energy(X, mu, sigma, pie, lam))

        lambda0 = lam if use_last else lambda_init
        expectations, aux = e_step(e_method, X, mu, sigma, pie, maxsteps=max_e_steps, tol=tol, lambda0=lambda0)
        lam = expectations['ES']
        E_curves.append(aux['free_energy_trace'])
        estep_Fs.append(E_curves[-1][-1])

        mu, sigma, pie = m_step(X, expectations['ES'], expectations['ESS'])

    return mu, sigma, pie, lam, mstep_Fs, estep_Fs, E_curves

def create_parser():
    """A set of command line arguments that are used to run the model

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='fit_em', choices=['vis_gt', 'analyze_gt', 'fit_em', 'search', 'test_sigma'], help='Mode to run script with')
    parser.add_argument('-s', '--sigma', type=float, default=0.1, help='Sigma of data to use')
    parser.add_argument('-p', '--posterior', type=str, default='vi', choices=['vi', 'ep', 'exact'], help='Fitter to use for posterior')
    
    # params of fitter
    parser.add_argument('-k', '--num_factors', type=int, default=8, help='Number of factors to fit')
    parser.add_argument('-n', '--niters', type=int, default=50, help='Number of iterations of em to run')
    parser.add_argument('-ne', '--neiters', type=int, default=10, help='Number of iterations to run in e-step')

    # other params (mode specific typically)
    parser.add_argument('--num_x_to_try', type=int, default=5, help='How many x to fit posteriors to in vis_gt mode')
    parser.add_argument('--start_ind', type=int, default=0, help='Which x to start at')

    return parser

if __name__ == "__main__":
    parser = create_parser()
    opts = parser.parse_args()

    np.random.seed(1)
    X = gen_images()
    K = opts.num_factors
    niters = opts.niters
    neiters = opts.neiters

    mode = opts.mode

    if mode == 'fit_em':

        mu, sigma, pie, lam, mstep_Fs, estep_Fs, E_curves = learn_bin_factors(opts.posterior, X, K, niters, mu_init='equal', pie_init=0.5, max_e_steps=neiters, rs=0)
        fig = plt.figure(constrained_layout=True)

        gs = GridSpec(5, K, figure=fig)
        ax1 = fig.add_subplot(gs[:2, :])
        np.vstack((mstep_Fs,estep_Fs)).T.reshape(-1)
        ax1.plot(np.arange(len(mstep_Fs)+len(estep_Fs))/2, np.vstack((mstep_Fs,estep_Fs)).T.reshape(-1), 'g-*', label='M-step updates')
        ax1.legend()
        ax2 = fig.add_subplot(gs[2:4, :])
        cmap = cm.get_cmap('hsv')
        denom = 1.25*len(E_curves)
        for i, curve in enumerate(E_curves):
            ax1.plot(i+0.5, estep_Fs[i], c=cmap(i/denom), marker='*')
            ax2.plot(curve, c=cmap(i/denom))
            ax2.plot(len(curve)-1,curve[-1],c=cmap(i/denom), marker='*')
        axs = [fig.add_subplot(gs[4, i]) for i in range(K)]
        for i in range(8):
            axs[i].imshow(mu[:,i].reshape(4,4), cmap='gray', vmin=np.min(mu), vmax=np.max(mu),interpolation='none')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(r'$\pi_{} = {:.3f}$'.format(i, pie[i]))
        plt.show()


