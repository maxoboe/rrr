#stage 1 function
import numpy as np
import pandas as pd
base = "/Users/max/Documents/ag_production_functions/code/riesz_representer/rr_python/" 
import os
import sys
import scipy.sparse as sc_sparse
from scipy.stats import norm
import copy

sys.path.append(os.path.abspath(base))
import config
import primitives

def RMD_dantzig(M, G, D, _lambda=0, sparse=True):
    return "not yet implemented; use RMD_lasso instead "

def RMD_lasso(M, G, D, _lambda=0, control = {"max_iter":1000, "optTol":1e-5, "zeroThreshold":1e-6},
     beta_start = None):
    p = G.shape[1] # num columns 
    Gt = G
    Mt = M
    L = np.concatenate([np.array([config.l]), np.ones(p-1)])
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

def get_D(Y,X,X_up,X_down,delta,m,rho_hat):
    n = X.shape[0] # num rows
    p = X.shape[1] # num columns
    df = np.zeros((p, n))
    for i in range(n):
        df[:, i] = X[i, :] * np.array(rho_hat @ X[i, :]) - m(Y[i], X[i, :], X_up[i, :], X_down[i, :], delta, primitives.b)
    df = df**2
    D2 = np.mean(df, 1) # Takes row means of df
    D = np.sqrt(D2)
    return D

# get_D <- function(Y,X,X.up,X.down,delta,m,rho_hat){
#     n=dim(X)[1]
#     p=dim(X)[2]
    
#     df=matrix(0,p,n)
#     for (i in 1:n){
#         df[,i]=X[i,]*as.vector(rho_hat %*% X[i,])-m(Y[i],X[i,],X.up[i,],X.down[i,],delta,b)
#     }
#     df=df^2
#     D2=rowMeans(df)
    
#     D=sqrt(D2)
#     return(D) #pass around D as vector
# }



def RMD_stable(Y,X,X_up,X_down,delta,p0,D_LB,D_add,max_iter,is_alpha,is_lasso):
    n = X.shape[0] # num rows
    p = X.shape[1] # num columns

    # First, find low-dimensional moments
    X0 = X[:, 0:p0]
    X0_up = X_up[:, 0:p0]
    X0_down = X_down[:, 0:p0]
    M_hat0, N_hat0, G_hat0, B0 = primitives.get_MNG(Y, X0, X0_up, X0_down, delta)

    # initial estimate
    rho_hat0 = np.linalg.solve(G_hat0, M_hat0)
    rho_hat = np.concatenate([rho_hat0, np.zeros(p - G_hat0.shape[1])]) 
    beta_hat0 = np.linalg.solve(G_hat0, N_hat0)
    beta_hat = np.concatenate([beta_hat0, np.zeros(p - G_hat0.shape[1])]) 

    # Full moments
    M_hat, N_hat, G_hat, B = primitives.get_MNG(Y, X, X_up, X_down, delta)
    # penalty term 
    _lambda = config.c * norm.ppf(1 - config.alpha / (2 * p)) / np.sqrt(n)
    if is_alpha:
        ###########
        # alpha_hat
        ###########
        diff_rho=1
        k=1
        while (diff_rho>config.tol) & (k<=max_iter):
            # previous values
            rho_hat_old=copy.deepcopy(rho_hat)
            
            # normalization
            D_hat_rho=get_D(Y,X,X_up,X_down,delta,primitives.m_diff,rho_hat_old)
            D_hat_rho=np.maximum(D_LB,D_hat_rho) 
            D_hat_rho=D_hat_rho+D_add
            if is_lasso:
                rho_hat = RMD_lasso(M_hat, G_hat, D_hat_rho, _lambda)[0]
            else:
                return "dantzig selector not implemented"
                beta_hat = RMD_dantzig(M_hat, G_hat, D_hat_rho, _lambda)[0]
            # difference
            diff_rho=primitives.two_norm(rho_hat-rho_hat_old)
            k=k+1
        print('k: ' + str(k))
        return rho_hat
    else:
        ###########
        # gamma_hat
        ###########
        diff_beta=1
        k=0
        while (diff_beta>config.tol) & (k<=max_iter):
            # previous values
            beta_hat_old=copy.deepcopy(beta_hat)
            
            # normalization
            D_hat_beta=get_D(Y,X,X_up,X_down,delta,primitives.m2,beta_hat_old)
            D_hat_beta=np.maximum(D_LB,D_hat_beta) # What is this? 
            D_hat_beta=D_hat_beta+D_add
            if is_lasso:
                beta_hat = RMD_lasso(N_hat, G_hat, D_hat_beta, _lambda)[0]
            else:
                return "dantzig selector not implemented"
                beta_hat = RMD_dantzig(N_hat, G_hat, D_hat_beta, _lambda)[0]
            # difference
            diff_beta=primitives.two_norm(beta_hat-beta_hat_old)
            k=k+1
        print('k: ' + str(k))
        return beta_hat

arg_Forest = {"clas_nodesize":1, "reg_nodesize":5, "ntree":1000, "na_action":"na_omit", "replace":True}
arg_Nnet = {'size':8, 'maxit':1000, 'decay':0.01, 'MaxNWts':10000, 'trace':False}

def get_stage_1(Y,X,X_up,X_down,delta,p0,D_LB,D_add,max_iter,alpha_estimator,gamma_estimator):
    n = X.shape[0] # num rows
    p = X.shape[1] # num columns
    MNG = primitives.get_MNG(Y, X, X_up, X_down, delta)
    B = MNG[3]

    ###########
    # alpha hat
    ###########
    rho_hat = RMD_stable(Y, X, X_up, X_down, delta, p0, D_LB, D_add, max_iter,
             is_alpha=1, is_lasso=(alpha_estimator == 'lasso'))
    def alpha_hat(x):
        return primitives.b(x) @ rho_hat

    ###########
    # gamma hat
    ###########
    if gamma_estimator == "dantzig":
        beta_hat=RMD_stable(Y, X, X_up, X_down, delta, p0, D_LB, D_add, max_iter,
             is_alpha=0, is_lasso=0)
        def gamma_hat(x):
            return primitives.b(x) @ beta_hat
    elif gamma_estimator == "lasso":
        beta_hat=RMD_stable(Y, X, X_up, X_down, delta, p0, D_LB, D_add, max_iter,
             is_alpha=0, is_lasso=1)
        def gamma_hat(x):
            return primitives.b(x) @ beta_hat
    elif gamma_estimator == "frst":
        return "random forest not implemented"
    elif gamma_estimator == "nnet":
        return "neural net not implemented "
    return alpha_hat, gamma_hat 

# get_stage1<-function(Y,X,X.up,X.down,delta,p0,D_LB,D_add,max_iter,alpha_estimator,gamma_estimator){
    
#     n=dim(X)[1]
#     p=dim(X)[2]
#     MNG<-get_MNG(Y,X,X.up,X.down,delta)
#     B=MNG[[4]]
    
#     ###########
#     # alpha hat
#     ###########
#     if(alpha_estimator==0){ # dantzig
        
#         rho_hat=RMD_stable(Y,X,X.up,X.down,delta,p0,D_LB,D_add,max_iter,1,0)
#         alpha_hat<-function(x){
#             return(b(x)%*%rho_hat)
#         }
        
#     } else if(alpha_estimator==1){ # lasso
        
#         rho_hat=RMD_stable(Y,X,X.up,X.down,delta,p0,D_LB,D_add,max_iter,1,1)
#         alpha_hat<-function(x){
#             return(b(x)%*%rho_hat)
#         }
        
#     }
    
#     ###########
#     # gamma hat
#     ###########
#     if(gamma_estimator==0){ # dantzig
        
#         beta_hat=RMD_stable(Y,X,X.up,X.down,delta,p0,D_LB,D_add,max_iter,0,0)
#         gamma_hat<-function(x){
#             return(b(x)%*%beta_hat)
#         }
        
#     } else if(gamma_estimator==1){ # lasso
        
#         beta_hat=RMD_stable(Y,X,X.up,X.down,delta,p0,D_LB,D_add,max_iter,0,1)
#         gamma_hat<-function(x){ 
#             return(b(x)%*%beta_hat)
#         }
        
#     } else if(gamma_estimator==2){ # random forest
        
#         forest<- do.call(randomForest, append(list(x=B,y=Y), arg_Forest))
#         gamma_hat<-function(x){
#             return(predict(forest,newdata=b(x), type="response"))
#         }
        
#     } else if(gamma_estimator==3){ # neural net
        
#         # scale down, de-mean, run NN, scale up, remean so that NN works well
#         maxs_B <- apply(B, 2, max)
#         mins_B <- apply(B, 2, min)
        
#         maxs_Y<-max(Y)
#         mins_Y<-min(Y)
        
#         # hack to ensure that constant covariates do not become NA in the scaling
#         const=maxs_B==mins_B
#         keep=(1-const)*1:length(const)
        
#         NN_B<-B
#         NN_B[,keep]<-scale(NN_B[,keep], center = mins_B[keep], scale = maxs_B[keep] - mins_B[keep])
        
#         NN_Y<-scale(Y, center = mins_Y, scale = maxs_Y - mins_Y)
        
#         nn<- do.call(nnet, append(list(x=NN_B,y=NN_Y), arg_Nnet)) #why is it degenerate with fitted.values=1?
#         gamma_hat<-function(x){
            
#             test<-t(as.vector(x))
#             NN_b<-test
#             NN_b[,keep]<-scale(t(NN_b[,keep]), 
#                                                  center = mins_B[keep], 
#                                                  scale = maxs_B[keep] - mins_B[keep])
            
#             NN_Y_hat<-predict(nn,newdata=NN_b)
#             Y_hat=NN_Y_hat*(maxs_Y-mins_Y)+mins_Y
            
#             return(Y_hat)
#         }
        
#     }
    
#     return(list(alpha_hat,gamma_hat))
    
# }
