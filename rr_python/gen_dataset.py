import pandas as pd
import numpy as np 
from patsy import (ModelDesc, EvalEnvironment, Term, EvalFactor, LookupFactor, dmatrices, dmatrix)


def gen_dataset(rhs_termlist, df, output_var, x_vars, quintile=0, quintile_var=''):
    N = len(df)

    # output var
    Y = df[output_var].values
    
    # ensure diff memory location
    df_up = df.copy(deep=True)
    df_down = df.copy(deep=True)
    
    # construct df.up and df.down
    prices = df['price']
    delta = np.std(prices)/4

    df_up['price'] = prices + delta/2
    df_down['price'] = prices - delta/2

    df['price2']=df['price']**2
    df_up['price2']=df_up['price']**2
    df_down['price2']=df_down['price']**2

    # Generate DMatrices using input rhs_termlist 
    regressors = dmatrix(rhs_termlist, data = df, return_type='matrix')
    regressors_up = dmatrix(rhs_termlist, data = df_up, return_type='matrix')
    regressors_down = dmatrix(rhs_termlist, data = df_down, return_type='matrix')

    # check
    assert(regressors.shape == regressors_up.shape )
    assert(regressors.shape == regressors_down.shape )
    print(regressors.shape)

    # Divide into quintiles by assigned variable 
    if (quintile>0):
        assert(quintile_var != '')
        q = pd.qcut(df[quintile_var], 5, labels=False)
        Y_q = Y[q == quintile - 1]
        regressors_q = regressors[q == quintile - 1]
        regressors_up_q = regressors_up[q == quintile - 1]
        regressors_down_q = regressors_down[q == quintile - 1]
        return (Y_q, regressors_q, regressors_up_q, regressors_down_q, delta)
    else:
        return (Y, regressors, regressors_up, regressors_down, delta)   
