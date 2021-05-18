"""
Implements the second stage of the average derivative finding
Main change to Rahul's implementation: 
For the neural net case, gamma is now an object with 
predict and derivative functions. This allows me to use 
the gradients learned during training of the neural network
in estimating the average derivatives. 
"""

import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import KFold

base = "/Users/max/Documents/ag_production_functions/code/riesz_representer/rr_python/" 

import os
import sys
sys.path.append(os.path.abspath(base))
import config
import stage_1
import primitives

def rrr(Y, X, X_up, X_down, delta, p0, D_LB, D_add, max_iter, alpha_estimator, gamma_estimator, bias):
    n=X.shape[0]
    kf = KFold(n_splits=config.L, shuffle=True, random_state=config.random)
    Psi_tilde = None
    for l_index, nl_index in kf.split(X):
        Y_l = Y[l_index]
        Y_nl = Y[nl_index]

        X_l = X[l_index, :]
        X_nl = X[nl_index, :]

        X_up_l = X_up[l_index, :]
        X_up_nl = X_up[nl_index, :]
        
        X_down_l = X_down[l_index, :]
        X_down_nl = X_down[nl_index, :]

        n_l = X_l.shape[0]
        n_nl = X_nl.shape[0]

        alpha_hat, gamma_hat = stage_1.get_stage_1(Y_nl,X_nl,X_up_nl,X_down_nl,delta,
            p0,D_LB,D_add,max_iter,alpha_estimator,gamma_estimator)
        if gamma_estimator == "nnet": #here, the derivative is learned during training 
            def m(y, x, x_up, x_down, delta, gamma):
                return gamma_hat.get_derivative(x)
        else: # Otherwise, we use the partial difference to approximate m()
            def m(y, x, x_up, x_down, delta, gamma):
                return primitives.m_diff(y, x, x_up, x_down, delta, gamma)
        gamma_predictor = gamma_hat.predict if gamma_estimator == 3 else gamma_hat
        if bias:
            Psi_tilde_l = [primitives.psi_tilde_bias(Y_l[i],X_l[i, :],X_up_l[i, :],X_down_l[i, :],
                    delta,m,alpha_hat,gamma_predictor) for i in range(n_l)]# naive plug-in estimator
        else:
            Psi_tilde_l = [primitives.psi_tilde(Y_l[i],X_l[i, :],X_up_l[i, :],X_down_l[i, :],
                    delta,m,alpha_hat,gamma_predictor) for i in range(n_l)]# DML approach
        if Psi_tilde is None:
            Psi_tilde = Psi_tilde_l
        else:
            Psi_tilde = np.concatenate([Psi_tilde, Psi_tilde_l])
    print(Psi_tilde)
    avg_derivative=np.mean(Psi_tilde)
  
    #influences
    Psi=Psi_tilde-avg_derivative

    var=np.mean(Psi**2)
    se=np.sqrt(var/n)
    return (n, avg_derivative, se)
 