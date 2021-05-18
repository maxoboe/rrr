"""
Max Vilgalys, 4/15/21
This script estimates a minimal linear approximator to the Riesz Representer
Follows Chernozhukov, Newey, Singh 2018, 
De-Biased Machine Learning of Global and Local
Parameters Using Regularized Riesz Representers

The test focuses on two cases for m, but could handle more general cases 
"""
import numpy as np 
from scipy.stats import norm
import copy


def get_MG(Y, X, m):
    """
    Builds matrices M, N, G for the general functional m, 
    with gamma = b(x) = x
    """
    n = X.shape[0]
    p = X.shape[1]

    M=np.zeros((p, n))

    for i in range(n):
        M[:, i]=m(Y[i],X[i,:])
    G_hat = X.T @ X / n
    M_hat = np.mean(M, 1) # Row means 
    return M_hat, G_hat, X # uncomment to return B as well 

def RMD_lasso(M, G, D, _lambda=0, l=0.1, control = {"max_iter":1000, "optTol":1e-5, "zeroThreshold":1e-6},
     beta_start = None):
    p = G.shape[1] # num columns 
    Gt = G
    Mt = M
    L = np.concatenate([np.array([l]), np.ones(p-1)])
    lambda_vec = _lambda*L*D
    if beta_start is None: # Warm start; allows passing in previous beta
        beta = np.zeros(p)
    else:
        beta = beta_start
    wp = beta
    mm = 1
    while mm < control['max_iter']:
        beta_old = copy.deepcopy(beta)
        for j in range(p):
            rho = Mt[j] - Gt[j, :] @ beta + Gt[j,j]*beta[j]
            z = Gt[j,j]
            if np.isnan(rho):
                beta[j] = 0
                continue
            if rho < -1 * lambda_vec[j]:
                beta[j] = (rho+lambda_vec[j])/z
            if (np.abs(rho) <= lambda_vec[j]):
                beta[j] = 0
            if (rho > lambda_vec[j]):
                beta[j] = (rho-lambda_vec[j])/z
        wp = np.c_[wp, beta]
        if (np.nansum(np.abs(beta - beta_old)) < control['optTol']):
            break
        mm = mm + 1
    w = beta
    w[abs(w) < control['zeroThreshold']] = 0
    return w, wp, mm # returns coefficients, list of past coefficients, and number of steps 

def get_D(Y, X, rho_hat,m):
    n = X.shape[0] # num rows
    p = X.shape[1] # num columns
    df = np.zeros((p, n))
    for i in range(n):
        df[:, i] = X[i, :] * np.array(rho_hat @ X[i, :]) - m(Y[i],X[i,:])
    df = df**2
    D2 = np.mean(df, 1) # Takes row means of df
    D = np.sqrt(D2)
    return D


def RMD_stable(Y, X, p0,D_LB,D_add,max_iter,config,m):
    n = X.shape[0] # num rows
    p = X.shape[1] # num columns

    # First, find low-dimensional moments
    X0 = X[:, 0:p0]
    M_hat0, G_hat0, B0 = get_MG(Y, X0, m)

    # initial estimate
    rho_hat0 = np.linalg.solve(G_hat0, M_hat0)
    rho_hat = np.concatenate([rho_hat0, np.zeros(p - G_hat0.shape[1])])

    # Full moments
    M_hat, G_hat, B = get_MG(Y, X)
    # penalty term 
    _lambda = config['c'] * norm.ppf(1 - config['alpha'] / (2 * p)) / np.sqrt(n)
    ###########
    # alpha_hat
    ###########
    diff_rho=1
    k=1
    while (diff_rho>config['tol']) & (k<=max_iter):
        # previous values
        rho_hat_old=copy.deepcopy(rho_hat)
        
        # normalization
        D_hat_rho=get_D(Y, X,m)
        D_hat_rho=np.maximum(D_LB,D_hat_rho) 
        D_hat_rho=D_hat_rho+D_add
        rho_hat = RMD_lasso(M_hat, G_hat, D_hat_rho, _lambda, config['l'])[0]
        # difference
        diff_rho=np.linalg.norm(rho_hat-rho_hat_old, 2)
        k=k+1
    # print('k: ' + str(k))
    return rho_hat

def alpha_hat(Y, X, m_method=None, m=None, direction=None):
    assert((m is None) or (m_method is None))
    # m should be either a name to one of the labeled methods, or a function of {Y,X,gamma}
    if m_method=='diff':
        # In CNS' notation, this is the case where t(x) = l(x) = 1 and b(x) = x
        # if b includes nonlinear elements, a transformation is required such that 
        # m(Y,X) gives d(gamma(X))/d(X)
        assert(direction is not None)
        assert(len(direction) == X.shape[1])
        def m(y, x):
            return direction
    elif m_method=='regression':
        def m(y, x):
            return y * x
    # Check that m method is specified
    assert(m is not None)

    p = X.shape[1]
    n = X.shape[0]

    #p0=dim(X0) used in low-dim dictionary in the stage 1 tuning procedure
    p0=np.ceil(p/4) #p/2 works for low p
    if (p>60):
      p0=np.ceil(p/40)
    p0 = int(p0)
    D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
    D_add=.2 #each diagonal entry of \hat{D} increased by D_add. 0.1 for 0, 0,.2 otw
    max_iter=100 #max number iterations in Dantzig selector iteration over estimation and weights


    ###########
    # alpha hat
    ###########
    config = {'l':0.1,  'c':0.5, 'alpha':0.1, 'tol':1e-6}

    rho_hat = RMD_stable(Y, X, p0, D_LB, D_add, max_iter, config, m)
    def alpha_hat(x):
        return x @ rho_hat

    return alpha_hat