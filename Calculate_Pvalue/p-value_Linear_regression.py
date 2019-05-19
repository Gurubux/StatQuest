# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:32:25 2019

@author: Guru
"""
"""
Source : 
    https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
"""
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt



diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print("summary()\n",est2.summary())
print("pvalues\n",est2.pvalues)
print("tvalues\n",est2.tvalues)
print("rsquared\n",est2.rsquared)
print("rsquared_adj\n",est2.rsquared_adj)
for attr in dir(est2):
    if not attr.startswith('_'):
        print(attr)
predictions = est2.predict(X2)

print(est2.predict(X2[:3,:]))
from sklearn.metrics import r2_score
print("r2_score",r2_score(y,predictions))
#slope, intercept, r_value, p_value, std_err = stats.linregress(X2[:,1],y)
#print(slope, intercept, r_value, p_value, std_err)
"""
# Plot all independent variables on plot to see check the linearity
for i in range(len(diabetes.feature_names)):
    plt.scatter(X[:,i],y)
    plt.xlabel(diabetes.feature_names[i])
    plt.show()

"""

"""
plt.scatter(X[:,1],y)
plt.show()
plt.scatter(X[:,2],y)
plt.show()
plt.scatter(X[:,3],y)
plt.show()
plt.scatter(X[:,4],y)
plt.show()
plt.scatter(X[:,5],y)
plt.show()
plt.scatter(X[:,5],y)
plt.show()
plt.scatter(X[:,5],y)
plt.show()
plt.scatter(X[:,5],y)
plt.show()
plt.scatter(X[:,5],y)
plt.show()
plt.scatter(X[:,5],y)
plt.show()"""



lm = LinearRegression()
lm.fit(X,y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

#newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
newX = pd.DataFrame(X2)


MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
# Standar Error, t-values, p-values, stats.t.cdf
var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)

#Large values of t indicate that the null hypothesis can be rejected and that the corresponding coefficient is not zero. The second column, p-value, expresses the results of the hypothesis test as a significance level. Conventionally, p-values smaller than 0.05 are taken as evidence that the population coefficient is nonzero.