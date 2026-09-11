# Useful Basics



import pandas as pd, matplotlib.pyplot as pl, numpy as np, numpy.linalg as la
from scipy.optimize import minimize, Bounds, LinearConstraint
import helper_functions as hp

##### Part 1: Basic Operations

# %pwd #current folder
# %cd C:\Users\hezho\Documents\Learning Programming\Python\2026Feb
# %clear #clear console (the variables are not cleared, just the console)
# %reset #clear all variables

#1.1 Matrix Operations
tmp1 = np.array([[1,2],[2,5]]) #create a 2-by-2 matrix [1 2;2 5]
tmp2 = np.zeros((1,2)) + np.ones((1,2))
tmp3 = np.eye(2) 
tmp = np.vstack((tmp1,tmp2,tmp3)) #stack to form a 3-by-2 matrix
tmp_add = tmp + np.tile(np.ones((5,1)),(1,2)) #note the tile function
tmp_mul = tmp*2 #element-wise multiplication
tmp_div = tmp/2 #element-wise division
tmp_prod = tmp.T @ tmp #matrix product

tmp = np.array([[1,2],[2,5]])
tmpvec = np.array([[0.1],[0.2]])
la.det(tmp) #matrix determinant
la.cholesky(tmp) #Chol decomp (lower half matrix)
la.solve(tmp,tmpvec) #inv(tmp)*tmpvec
(tmp_eigval,tmp_eigvec) = la.eig(tmp) #eigvec is column-wise
(u,d,vtrans) = la.svd(tmp) #note "d" is a 1-dimension array



#1.2 Random Numbers
rng = np.random.default_rng(123456)
nr = 1000
tmp = rng.standard_normal((nr,1))
tmp = rng.normal(0,1,(nr,1))
tmp = rng.uniform(0,1,(nr,1))
tmp = rng.gamma(2,3,(nr,1)) #mean is 2*3=6
tmp = rng.beta(1,3,(nr,1))
mu = np.array([1,2])
sigma = np.array([[1,2],[2,5]])
tmp = rng.multivariate_normal(mu,sigma,nr)



#1.3. String Vector
ns = 3 #string vector length
tmp_list = [" "]*ns #create an empty string vector as a list
for i in range(ns):
    tmp_list[i] = 'yes' + str(i) #each entry combines a number index
tmp_list.append('yes1') #add an entry to the list

tmp_np = np.array(tmp_list) #use np to search over string vector 
tmp_idxvec = np.where(tmp_np == 'yes1') #note "tmp_idx" is a vector
tmp_idx = np.min(tmp_idxvec)



##### Part 2: A Simple Regression Example

#2.1 Read Data
read_file = 'Equity_Qtrly_Github.xlsx'
read_sheet = 'Data' 
rawdata_pd = pd.read_excel(read_file, sheet_name=read_sheet,
                        usecols='A,B,C:N',nrows=296) #nrows does not count headers

(n,ncol) = rawdata_pd.shape
datevec = np.array(rawdata_pd.iloc[:,0]) #snap dates
y = np.array(rawdata_pd.iloc[:,1]).reshape(n,1) #force a column vector
xnc = np.array(rawdata_pd.iloc[:,2:ncol]) 
name_y = rawdata_pd.columns[1] #process column names
name_xnc = np.array(rawdata_pd.columns[2:ncol])
mx = xnc.shape[1]

x = np.hstack((np.ones((n,1)), xnc)) #add the intercept
name_x = np.insert(name_xnc,0,'Const')
m = x.shape[1]



#2.2 Plot Data
pl.subplot(2,2,1)
pl.plot(y)
pl.title(name_y)

for j in range(2):
    pl.subplot(2,2,j+2)
    pl.plot(xnc[:,j])
    pl.title(name_xnc[j])

pl.subplot(2,2,4)
pl.hist(xnc[:,3],20)
pl.title(name_xnc[3])



#2.3 Simple OLS Regression
coef_ols = la.solve(x.T@x,x.T@y)



# Bayesian regression
rseed = 123456
burnin = 1000
ndraws = 10000
beta_var = 100
pa = 0.01
pb = 0.01
(draws_coef, draws_s2) = hp.Bayesian_OLS(y,x,beta_var,pa,pb,burnin,ndraws,rseed)
coef_bayes = np.mean(draws_coef,axis=0).reshape(m,1)



# Try least squares minimization
def ols_objective(beta,xx,yy): 
    resid = yy - xx@beta.reshape(len(beta),1)
    return resid.T@resid/len(y)
beta0 = np.zeros((m,1)).reshape(-1) #beta must be 1D to be used in minimize

opt = minimize(ols_objective, beta0, args=(x, y),options={'disp': True}) #unconstrained
coef_optimize = opt.x.reshape(m,1)

lb = -10*np.array(m)
ub = 10*np.array(m)
bounds = Bounds(lb,ub)
A = np.vstack((np.ones((1,m)), -1*np.ones((1,m))))
bl = np.array([-100,-100])
bu = -1*bl
linear_constraint = LinearConstraint(A, bl, bu)
opt2 = minimize(ols_objective, beta0, args=(x, y), 
                bounds=bounds, constraints=linear_constraint,
                method='trust-constr',options={'disp': True}) #constrained
coef_optimize2 = opt2.x.reshape(m,1)


#2.4 Write data
write_file = pd.ExcelWriter(read_file,mode='a',engine='openpyxl',
                            if_sheet_exists = 'replace') 
# default mode is "w" that will clear existing contents in the target file
write_sheet = 'Coef'
coef = np.hstack((coef_ols,coef_bayes,coef_optimize,coef_optimize2))
name_coef = np.array(['OLS','Bayes','Optimize(Unconstrained)','Optimize(Constrained)'])
coef_pd = pd.DataFrame(coef)
coef_pd.columns = name_coef
coef_pd.index = name_x
coef_pd.to_excel(write_file, sheet_name = write_sheet, startrow=0, startcol=0,
                 header=True, index=True)
write_file.close()



