"""
Thesis Chapter 2 - Housing Price Prediction
Name : Shukhrat Khuseynov
ID   : 0070495
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def report (ytest, ypredict):
    """ reporting scores """
    
    corr = np.corrcoef(ytest, ypredict)
    r2 = r2_score(ytest, ypredict)
    rmse = mean_squared_error(ytest, ypredict, squared=False)
    print("\nCorrelation:\n", corr)
    #print("\nR^2:\n", r2)
    print("\nRMSE:\n", rmse)
    return (corr, r2, rmse)

def gridsearch (model, param, Xtrain, ytrain):
    """ implementing the process of GridSearchCV """
    
    grid = GridSearchCV(model, param, cv=5, scoring = 'neg_mean_squared_error', refit=True, verbose=1)
    grid.fit(Xtrain, ytrain)

    print("\n", grid.best_score_)
    print("\n", grid.best_params_)
    print("\n", grid.best_estimator_)


# reading the data
df = pd.read_csv('flats_moscow.csv')

# checking variable types
print(df.info())

# checking whether there is any null element
print(df.isnull().values.any())

# detecting columns with diferent values for each row
print(df.loc[:, (df.nunique()==df.shape[0])].columns)

# dropping the id variable
df.drop(['Unnamed: 0'], axis=1, inplace=True)


# initiating variables for the models
X = df.loc[:, df.columns != 'price']
y = df.price
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=0)

# scaling the data
scale = MinMaxScaler(feature_range=(0,1))
scale.fit(Xtrain)
Xtrain = scale.transform(Xtrain)
Xtest = scale.transform(Xtest)


# Models:

print("\n\nLinear regression")
from sklearn.linear_model import LinearRegression

reg = LinearRegression(fit_intercept=True)
reg.fit(Xtrain, ytrain)
ypredict = reg.predict(Xtest)

reg_corr, reg_r2, reg_rmse = report (ytest, ypredict)


print("\n\nGaussian Process regression")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RationalQuadratic, RBF
# tuned manually

kernel = RationalQuadratic()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1.7, random_state=0)
gpr.fit(Xtrain, ytrain)
ypredict = gpr.predict(Xtest)

gpr_corr, gpr_r2, gpr_rmse = report (ytest, ypredict)


print("\n\nK Nearest Neighbors regression")
from sklearn.neighbors import KNeighborsRegressor

# param = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# gridsearch(KNeighborsRegressor(weights='distance'), param, Xtrain, ytrain)
# choosing 8, local extremum

knn = KNeighborsRegressor(n_neighbors=8, weights='distance')
knn.fit(Xtrain, ytrain)
ypredict = knn.predict(Xtest)

knn_corr, knn_r2, knn_rmse = report (ytest, ypredict)


print("\n\nRandom Forest regression")
from sklearn.ensemble import RandomForestRegressor

# param = {'n_estimators': [114, 115, 116]}
# gridsearch(RandomForestRegressor(random_state=0), param, Xtrain, ytrain)
# choosing 115, local extremum

rf = RandomForestRegressor(n_estimators=115, random_state=0)
rf.fit(Xtrain, ytrain) 
ypredict = rf.predict(Xtest)

rf_corr, rf_r2, rf_rmse = report (ytest, ypredict)


print("\n\nSupport Vector Machines regression")
from sklearn.svm import SVR

# param = {'C': [10000], 'gamma': ['auto'], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
# gridsearch(SVR(), param, Xtrain, ytrain)
# choosing 10000, greater value does not improve the model significantly

svm = SVR(kernel='rbf', C=10000, gamma='auto')
svm.fit(Xtrain, ytrain)
ypredict = svm.predict(Xtest)

svm_corr, svm_r2, svm_rmse = report (ytest, ypredict)


print("\n\nNeural Networks regression")
from sklearn.neural_network import MLPRegressor

# tuned manually

nn = MLPRegressor(hidden_layer_sizes=(150, 150, 150, 150, 150), max_iter=500, random_state=0)  
nn.fit(Xtrain, ytrain)
ypredict = nn.predict(Xtest)

nn_corr, nn_r2, nn_rmse = report (ytest, ypredict)


print("\n\nExtreme Gradient Booster regression")
from xgboost import XGBRegressor

# param = {'n_estimators': [20, 21, 22, 25]}
# gridsearch(XGBRegressor(), param, Xtrain, ytrain)
# choosing 21, local extremum

xgb = XGBRegressor(n_estimators=21, random_state=0)
xgb.fit(Xtrain, ytrain)
ypredict = xgb.predict(Xtest)

xgb_corr, xgb_r2, xgb_rmse = report (ytest, ypredict)


# plotting RMSE and correlation

import matplotlib.pyplot as plt

models = ['Reg', 'GPR', 'KNN', 'RF', 'SVM', 'NN', 'XGB']
rmse = [reg_rmse, gpr_rmse, knn_rmse, rf_rmse, svm_rmse, nn_rmse, xgb_rmse]
corr = [reg_corr[0,1], gpr_corr[0,1], knn_corr[0,1], rf_corr[0,1], svm_corr[0,1], nn_corr[0,1], xgb_corr[0,1]]

# RMSE
fig1 = plt.figure()
plt.bar(models, rmse)
plt.ylim(0, 40)
plt.text(3.7, min(rmse)+2, '25.29', fontdict={'weight':'semibold'})

plt.title("Model comparison")
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.savefig('RMSE.png')
plt.show()

# Correlation
fig2 = plt.figure()
plt.bar(models, corr, color='maroon')
plt.ylim(0, 1.1)
plt.text(3.7, max(corr)+0.05, '0.872', fontdict={'weight':'semibold'})

plt.title("Model comparison")
plt.xlabel("Models")
plt.ylabel("Correlation coefficient")
plt.savefig('Corr.png')
plt.show()


# SVM seems to be slightly better than other models both in terms of RMSE and correlation.

svm_corr, svm_r2, svm_rmse = report (ytest, ypredict)


# The end.