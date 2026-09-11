## Import data from Excel sheet, do some analysis

# %pwd # current folder
# %cd C:\Users\hezho\Documents\Learning Programming\Python\2026Feb
# %clear # clear console (the variables are not cleared, just the console)
# %reset # clear all variables

import numpy as np, pandas as pd, matplotlib.pyplot as plt 
from numpy import linalg as la
from scipy.optimize import minimize, Bounds, LinearConstraint
import time


# Import data
rawdata = pd.read_excel('Equity_Qtrly_Github.xlsx', sheet_name='Data',
                        usecols='A,B,C:N',nrows=296) #nrows does not count header
print()
print('data type = ', type(rawdata))
print('data shape = ', rawdata.shape)
print('first few obs: \n', rawdata.iloc[:3,:6])
print()



# Process data
(ntotal,mtotal) = rawdata.shape
ntrain = round(0.8*ntotal)
ntest = ntotal - ntrain
print('(ntrain,ntest) = (',ntrain,',',ntest,')')
print()

datecol = rawdata.iloc[:,0]
y = rawdata.iloc[:,1]
x = rawdata.iloc[:,2:mtotal]
xfull = x.copy(); # be carefull here to avoid view
xfull.insert(loc=0,column='Const',value=np.ones(ntotal))
m = x.shape[1]
mfull = m + 1

ytrain = y.iloc[:ntrain]
ytest = y.iloc[ntrain:ntotal]
xtrain = x.iloc[:ntrain]
xtest = x.iloc[ntrain:ntotal]
xfulltrain = xfull.iloc[:ntrain]
xfulltest = xfull.iloc[ntrain:ntotal]

# del xfull['Const'] # delete the column 'Const' of x
# del rawdata #clear a variable


# Check data
print('mean,std,skew,kurt of y:\n ',y.mean(),y.std(),y.skew(),y.kurt())
print()
for j in range(m+1):
    if j==0:
        plt.subplot(4,4,1)
        plt.plot(y)
        plt.title(y.name)
    else:
        plt.subplot(4,4,j+1)
        plt.scatter(x.iloc[:,j-1],y,color='r',s=5)
        plt.title(x.columns[j-1])



# Simple OLS regression
xx = xfull.iloc[:,0:3].to_numpy() #better to use numpy instead of pandas for matrix operations
yy = y.to_numpy().reshape(len(y),1)
coef_ols = la.solve(xx.T@xx,xx.T@yy)



# Bayesian regression
burnin = 1000
ndraws = 10000
ntotal = burnin + ndraws
KK = xx.shape[1]
coefmat = np.zeros((ndraws,KK))
rng = np.random.default_rng(100) #set seed
eps = rng.normal(0,1,(ntotal,KK))

resid = yy - xx@coef_ols
s = resid.T@resid/ntotal
Binv = 0.01*np.eye(KK) + xx.T@xx/s
Binvhalf = la.cholesky(Binv)
Binvb = xx.T@yy/s

time_start = time.time()
for drawi in range(ntotal):
    epsi = eps[drawi,:].reshape(KK,1) #covert 1D to 2D
    tmp = Binvb + Binvhalf@epsi
    coefi = la.solve(Binv,tmp)
    if drawi >= burnin:
        coefmat[drawi-burnin,:] = coefi.T 
    if round(drawi/2000) == (drawi/2000):
        time_count = time.time()
        print(drawi,' draws are completed after ',time_count-time_start,' seconds')
coef_bayes = coefmat.mean(0).reshape(KK,1)
print()


# Try least squares minimization
def ols_objective(beta,x,y): 
    resid = y - x@beta.reshape(len(beta),1)
    return resid.T@resid/len(y)
beta0 = np.zeros((KK,1)).reshape(-1) #beta must be 1D to be used in minimize
opt_ret = minimize(ols_objective, beta0, args=(xx, yy),options={'disp': True})
coef_optimize = opt_ret.x.reshape(KK,1)



# Compare the results
print()
print('Compare results: OLS(direct formula), Bayesian, OLS(minimization)')
print(np.hstack((coef_ols,coef_bayes,coef_optimize)))  



# Write results
coef_mat = pd.DataFrame(np.hstack((coef_ols,coef_bayes,coef_optimize)))  
coef_mat.to_excel('coef_est.xlsx',sheet_name='coef',startrow=1,startcol=1,\
                  header=False, index=False)
  
    
    


