"""
This document is adapted from the implementation of https://arxiv.org/pdf/1802.08667.pdf
by Rahul Singh, available here: https://github.com/r4hu1-5in9h/rrr/

This implementation is coded in Python, and calculates the debiased average derivative of the machine learner.
It extends the original implementation to find the debiased average derivative for a neural network 
directly from the automatic derivatives of the network, instead of the partial differences approach. 
"""

base = "/Users/max/Dropbox (MIT)/GitHub/rrr/rr_python/" 

import os
import sys
import numpy as np
import pandas as pd 
sys.path.append(os.path.abspath(base))


# rm(list=ls())

# library("foreign")
# library("dplyr")
# library("ggplot2")
# library("quantreg")

# library("MASS")
# library("glmnet") #lasso, group lasso, and ridge, for outcome models, LPM, mlogit. Also, unpenalized mlogit via glmnet is far faster than the mlogit package.
# library("grplasso")   #glmnet group lasso requires the same n per group (ie multi-task learning), which is perfect for mlogit but wrong for the outcome model.
# # library("mlogit")   #slower than unpenalized estimation in glmnet, but glmnet won't fit only an intercept
# library("nnet")   #quicker multinomial logit
# library("randomForest")
# library("gglasso")
# library("plotrix")
# library("gridExtra")

# setwd("~/Documents/research/rrr_gas_blackbox")

#######################
# clean and format data
#######################

# Implements get_data
import gen_dataset

gas_df = pd.read_stata(base + 'gasoline_final_tf1.dta')
take_logs = ['gas', 'price', 'income', 'age', 'distance']
# take logs of continuous variables; part of pre-processing
for label in take_logs:
	gas_df[label] = np.log(gas_df[label])

spec=1
# specification
if spec == 1:
    factor_list = 'C(driver) + C(hhsize) + C(month) + C(prov) + C(year)'
    rhs_termlist = 'price + I(price**2) + (price : (' + factor_list + ')) + (I(price**2) : (' \
        + factor_list + ')) + ' + factor_list + ' + urban + youngsingle + distance + age + I(age**2) + income + I(income**2)'
elif spec == 0:
    factor_list = 'C(driver) + C(hhsize) + C(month) + C(prov) + C(year) + age + I(age**2) + income + I(income**2)'
    rhs_termlist = 'price + I(price**2) + (price : (' + factor_list + ')) + (I(price**2) : (' \
                + factor_list + ')) + ' + factor_list + ' + urban + youngsingle + distance + I(distance**2)'

quintile=0
output_var = 'gas'
x_vars = ['price']

Y, X, X_up, X_down, delta = gen_dataset.gen_dataset(rhs_termlist, gas_df, output_var, x_vars, quintile )
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

# ###########
# # algorithm
# ###########

alpha_estimator="lasso"
gamma_estimator="lasso"
bias=0
#alpha_estimator: 0 dantzig, 1 lasso
#gamma_estimator: 0 dantzig, 1 lasso, 2 rf, 3 nn

# set.seed(1) # for sample splitting

import estimate_alpha_up_down
alpha = estimate_alpha_up_down.alpha_hat(Y, X, X_up, X_down, delta, p0, D_LB, D_add, max_iter, m_method='diff', direction=np.ones(p))

print(np.mean([alpha(x) for x in X]))

