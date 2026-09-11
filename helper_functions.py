# Example to collect all helper functions into a separate file
# then import them to use in main file



import numpy as np, numpy.linalg as la, time


# 1 ######################################################################
# Bayesain OLS regression (Gaussian prior for beta, Inverse gamma prior 
# for residual variance)
def Bayesian_OLS(y,x,beta_var,pa,pb,burnin,ndraws,rseed):   
# Inputs:
# y: a n-by-1 vector of target
# x: a n-by-m matrix of regressors
# beta_var: a scalar of the Gaussian prior variance of linear coef
# pa: a scalar of the Inverse Gamma prior shape of residual variance
# pb: a scalar of the Inverse Gamma prior scale of residual variance
# burnin: a scalar of the number of burn-ins
# ndraws: a scalar of the number of effective draws
# rseed: a scalar of the random number generator seed
#
# Outputs:
# draws_coef: a ndraws-by-m matrix of linear coef draws
# draws_s2: a ndraws-by-1 vector of residual variance draws

    (n,m) = x.shape
    if len(y.shape) == 1: #if y is 1D, reshape to 2D
        yy = y.reshape(n,1)
    else:
        yy = y
        
    
    rng = np.random.default_rng(rseed)
    beta = np.sqrt(beta_var) * rng.normal(0,1,(m,1))
    s2 = 1/rng.gamma(pa,1/pb)
    
    
    draws_coef = np.zeros((ndraws,m))
    draws_s2 = np.zeros((ndraws,1))
    ntotal = burnin + ndraws
    time_start = time.time()
    for drawi in range(ntotal):
        # Linear Coef
        Binv = (1/beta_var)*np.eye(m) + x.T@x
        Binvb = x.T@yy
        tmp = rng.multivariate_normal(Binvb.reshape(-1),Binv)
        beta = la.solve(Binv,tmp).reshape(m,1)
        
        # Residual Variance
        resid = yy - x@beta
        pa_new = pa + 0.5*n
        pb_new = pb + 0.5*(resid.T@resid)
        s2 = 1/rng.gamma(pa_new,1/pb_new)
        
        if drawi >= burnin:
            idx = drawi-burnin
            draws_coef[idx,:] = beta.T
            draws_s2[idx] = s2
            
        if round(drawi/2000) == (drawi/2000):
            time_count = time.time()
            print(drawi,' draws are completed after ',time_count-time_start,' seconds')
    
    return draws_coef, draws_s2
        
    
        
# 2 ######################################################################
# PCA transformatio
def PCA_trans(x, mu, sigma):   
# Inputs:
# x: a n-by-m matrix of raw data
# mu: a m-by-1 vector to demean x
# sigma: a m-by-1 vector to scale x
#
# Outputs:
# f: a n-by-m matrix of transformed data
# coef: a m-by-m matrix of coefficients f*coef = (x-mu)/sigma
# var_explained: a m-by-1 vector of percent variance explained by f

    (n,m) = x.shape
    if m == 1:
        raise ValueError("Need more than 1 column in x") 
    
    a = np.ones((n,1))
    mat_mean = np.kron(a,mu.T)
    mat_std = np.kron(a,sigma.T)
    xx = (x - mat_mean)/mat_std
    
    (u,d,vv) = la.svd(xx)
    coef = vv
    if m < n:
        dmat = np.vstack((np.diag(d),np.zeros((n-m,m))))
    else:
        if m == n:
            dmat = np.diag(d)
        else:
            dmat = np.hstack((np.diag(d),np.zeros((n,m-n))))
                
    f = u@dmat
    var_explained = d**2/sum(d**2)
    return f, coef, var_explained
        
    

