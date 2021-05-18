import numpy as np
from numpy import linalg as LA

def two_norm(x): 
    return LA.norm(x, 2)

def one_norm(x):
    return LA.norm(x, 1)

def one_norm_grad(x):
    return np.sign(x)

def b(x):
    return x

def m_diff(y, x, x_up, x_down, delta, gamma):
    return (gamma(x_up) - gamma(x_down)) / delta 

def m2(y, x, x_up, x_down, delta, gamma):
    return y * gamma(x)

# m2<-function(y,x,x.up,x.down,delta,gamma){
#     return(y*gamma(x))
# }
 
def psi_tilde(y, x, x_up, x_down, delta, m, alpha, gamma):
    return m(y, x, x_up, x_down, delta, gamma) + alpha(x) * (y - gamma(x))

def psi_tilde_bias(y, x, x_up, x_down, delta, m, alpha, gamma):
    return m(y, x, x_up, x_down, delta, gamma)

def get_MNG(Y, X, X_up, X_down, delta):
    """
    Finds partial difference, using m_diff
    """
    n = X.shape[0]
    p = X.shape[1]

    M=np.zeros((p, n))
    N=np.zeros((p, n))

    for i in range(n):
        N[:,i]=Y[i]*X[i,:] #since m2(w,b)=y*x
        M[:, i] = (X_up[i,:]-X_down[i,:])/delta
    G_hat = X.T @ X / n
    M_hat = np.mean(M, 1) # Row means 
    N_hat = np.mean(N, 1)
    return M_hat, N_hat, G_hat, X # uncomment to return B as well 

# get_MNG<-function(Y,X,X.up,X.down,delta){
    
#     p=ncol(X)
#     n=nrow(X)
    
#     M=matrix(0,p,n)
#     N=matrix(0,p,n)
    
#     for (i in 1:n){ #simplifications since dictionary b is the identity
#         M[,i]=(X.up[i,]-X.down[i,])/delta #since m(w,b)=(x.up-x.down)/delta
#         N[,i]=Y[i]*X[i,] #since m2(w,b)=y*x
#     }
    
#     M_hat=rowMeans(M) #since m(w,b)=dx
#     N_hat=rowMeans(N)
#     G_hat=t(X)%*%X/n
    
#     return(list(M_hat,N_hat,G_hat,X))
# }