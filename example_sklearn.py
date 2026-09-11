'''
Compare conventional ML algorithms for regression in sklearn.
'''

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pandas import read_excel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline


#1. Read data
#Note: no need to add ones to X; individual algorithms will handle it.
file_dir = 'C:\\Users\\hezho\\Documents\\Learning Programming\\Python\\2026Feb\\'
file_name = 'Equity_Qtrly_Github.xlsx'
rawdata = read_excel(file_dir+file_name, sheet_name='Data',
                        usecols='A,B,C:N',nrows=296)
print('data shape = ', rawdata.shape)
print('first few obs: \n', rawdata.iloc[:3,:6])
y = rawdata.iloc[:,1].to_numpy()
X = rawdata.iloc[:,2:rawdata.shape[1]].to_numpy()
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2,
#                                                    shuffle=True,random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2,
                                                    shuffle=False)

#2. Preparation                                                    
compare_score = {} #to store test scores
compare_ypred = pd.DataFrame(y_test,columns=['Actual']) #to store predicted y 
def MAE(xtrue,xpred):
    n = len(xtrue)
    if n == len(xpred):
        eps = np.abs(xtrue - xpred)
        return(eps/n)
    else:
        raise ValueError("len(xtrue) ~= len(xpred)")       
def MSE(xtrue,xpred):
    n = len(xtrue)
    if n == len(xpred):
        eps = xtrue - xpred
        return(eps@eps/n)
    else:
        raise ValueError("len(xtrue) ~= len(xpred)") 
rseed = np.random.RandomState(100) #global random seed for reproducibility 
    

#3.1 Ridge regression
mlname = 'Ridge'
from sklearn.linear_model import Ridge
pipe_ml = make_pipeline(StandardScaler(), Ridge())
#para_grid = {'ridge__alpha': np.arange(0.01, 1.01, 0.01)} #alpha is like the inverse of prior variance
#para_grid = {'ridge__alpha':[0.001, 0.01, 0.1, 1]}
para_grid = {'ridge__alpha':np.logspace(-3,0,4)}
#gs = GridSearchCV(pipe_ml, para_grid, cv=5, scoring='neg_mean_squared_error') #k fold
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
'''train1 = list(range(200))
train2 = list(range(200,len(y_train)))
holdout = [(train1,train2)]
gs = GridSearchCV(pipe_ml, para_grid, cv=holdout, scoring='neg_mean_squared_error')''' #hold out
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred 


#3.2 Lasso regression
mlname = 'Lasso'
from sklearn.linear_model import Lasso
pipe_ml = make_pipeline(StandardScaler(), Lasso())
para_grid = {'lasso__alpha':np.logspace(-3,0,4)}
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred


#3.3 Gaussian process (special in that it uses ML to optimize hyper-params)
mlname = 'GP'
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK, DotProduct as DP
kn = DP(0,'fixed')+RBF(0.1,(1e-5,1e2)) + WK(0.01,(1e-8,1e2))
pipe_ml = make_pipeline(StandardScaler(), GPR(kn,alpha=0, n_restarts_optimizer=10, random_state=rseed))
#para_grid = {'GaussianProcessRegressor__kernel':[RBF()]}
#gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs = pipe_ml
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred
print('initial kernel = ', gs.named_steps['gaussianprocessregressor'].kernel) #initial kernel
print('optimized kernel = ', gs.named_steps['gaussianprocessregressor'].kernel_) #optimized kernel


#3.4 Kernel ridge regression
mlname = 'KernelRidge'
from sklearn.kernel_ridge import KernelRidge
pipe_ml = make_pipeline(StandardScaler(), KernelRidge(kernel='rbf'))
para_grid = {'kernelridge__gamma':np.logspace(-3,3,5),
             'kernelridge__alpha':np.logspace(-3,3,5)} #exp(-gamma*||x-y||^2), alpha is ridge penalty
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred


#3.5 Support vector regression
mlname = 'SupportVector'
from sklearn.svm import SVR
pipe_ml = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
para_grid = {'svr__gamma':np.logspace(-3,3,5),
             'svr__C':np.logspace(-3,3,5)} #exp(-gamma*||x-y||^2)
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred


#3.6 Random forest
mlname = 'RandomForest'
from sklearn.ensemble import RandomForestRegressor as RF
pipe_ml = make_pipeline(StandardScaler(), RF(random_state = rseed))
para_grid = {'randomforestregressor__n_estimators':[50, 80, 100, 120, 150],
             'randomforestregressor__max_features':[0.3, 0.5, 0.8, 1.0]} 
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred


#3.7 Gradient boosting
mlname = 'GradientBoosting'
from sklearn.ensemble import GradientBoostingRegressor as GB
pipe_ml = make_pipeline(StandardScaler(), GB(subsample=0.8, max_features=0.8, random_state = rseed))
para_grid = {'gradientboostingregressor__n_estimators':[50, 80, 100, 120, 150],
             'gradientboostingregressor__learning_rate':[0.01, 0.05, 0.1]} 
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred


#3.8 Neural network
mlname = 'NeuralNetwork'
from sklearn.neural_network import MLPRegressor as NN
pipe_ml = make_pipeline(StandardScaler(), NN(max_iter=500, early_stopping=True, random_state = rseed))
para_grid = {'mlpregressor__hidden_layer_sizes':[(3000,), (3000,10)]} 
gs = GridSearchCV(pipe_ml, para_grid, cv=TimeSeriesSplit(n_splits=5,test_size=1), scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
y_test_pred = gs.predict(X_test)
test_score = MSE(y_test, y_test_pred)
print(f'{mlname} test score = {test_score:.6f}')
compare_score[mlname] = test_score
compare_ypred[mlname] = y_test_pred