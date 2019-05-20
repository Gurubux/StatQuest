# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:05:05 2019

@author: Guru
"""
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
from sklearn.linear_model  import LinearRegression
import statsmodels.api as sm
## Here's the data from the example:
mouse = pd.DataFrame({"weight":[0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3],
  "sizes":[1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3]})
print(mouse)


#init_notebook_mode()

## plot a x/y scatter plot with the data
trace0 = Scatter(
    x=mouse.weight,
    y=mouse.sizes,
    mode='markers')

# create a "linear model" - that is, do the regression
X2 = sm.add_constant(mouse.iloc[:,0:1].values)
est = sm.OLS(mouse.iloc[:,1].values, X2)
est2  = est.fit()

## generate a summary of the regression
print("summary()\n",est2.summary())


"""
# create a "linear model" - that is, do the regression
lm = LinearRegression()
lm.fit(mouse.iloc[:,0:1].values,mouse.iloc[:,1].values)

# add the regression line to our x/y scatter plot
trace1 = Scatter(
    x = mouse.weight,
    y = lm.predict(mouse.iloc[:,0:1].values)
)
"""
# add the regression line to our x/y scatter plot
trace2 = Scatter(
    x = mouse.weight,
    y = est2.predict(X2)
)


# Plot
data = [trace0,trace2]

layout = Layout(
    showlegend=True,
    height=600,
    width=600,
)

fig = dict( data=data, layout=layout )
plot(fig)  

"""
mouse.data # print the data to the screen in a nice format

## plot a x/y scatter plot with the data
plot(mouse.data$weight, mouse.data$size)

## create a "linear model" - that is, do the regression
mouse.regression <- lm(size ~ weight, data=mouse.data)
## generate a summary of the regression
summary(mouse.regression)

## add the regression line to our x/y scatter plot
abline(mouse.regression, col="blue")
"""