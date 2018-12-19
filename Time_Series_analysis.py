#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:02:45 2018

@author: partha
"""
## Import dependency
import warnings    # do not distrub mood
import itertools   # some useful function
import pandas as pd  # table and data manipulation
import numpy as np   # vectors and matrics
import matplotlib.pyplot as plt  # plots
import statsmodels.api as sm     # Statistics and econometrics
import matplotlib
%matplotlib inline
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k' 

## Data import
df = pd.read_csv("/home/partha/Desktop/Link3/Sylhettop10olt.csv")
df.head()
df.shape
#(3905,90)
df.dtypes

# check null values
df.isnull().sum()
df.drop('Unnamed: 90',inplace = True, axis = 1) # drop unnecessary column

## Convertaed date time & columns manipulation 
#df['rocorddate'] = sorted(df.recorddate)
df['recorddate'] = pd.to_datetime(df.recorddate )
df['usage'] = df.usage/(1024*1024*1024)

# check individual olt (optical line termination)
individual_olt = df['olt'].unique()

specific_olt = df[df.olt == individual_olt[2]]
specific_olt['recorddate'] = sorted(specific_olt['recorddate'])
specific_olt.drop_duplicates(['recorddate'],inplace=True)
specific_olt.isnull().sum()
olt_usage = specific_olt[['recorddate','usage']]
olt_usage.dtypes
olt_usage = olt_usage.set_index('recorddate')
olt_usage.index
olt_usage['2017':]
olt_usage = olt_usage.drop(olt_usage.index[[0]])
olt_usage = olt_usage.drop(olt_usage.index[[-1]]) #drop last row
#olt_usage = olt_usage[olt_usage.index != '2018-08-09']


olt_usage.plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(olt_usage, model='additive', freq = 30)
fig = decomposition.plot()
plt.show()


## ================ Arima modelling ========================================================================
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(olt_usage,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(olt_usage,
                                order=(0, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ma.L1         -0.2935   3132.327  -9.37e-05      1.000   -6139.542    6138.955
ar.S.L12      -0.0685      2.445     -0.028      0.978      -4.860       4.723
ma.S.L12      -0.9015   1788.519     -0.001      1.000   -3506.335    3504.532
sigma2      6.993e+04      0.043   1.62e+06      0.000    6.99e+04    6.99e+04
==============================================================================

print(results.summary().tables[0])

                                 Statespace Model Results                                 
==========================================================================================
Dep. Variable:                              usage   No. Observations:                  353
Model:             SARIMAX(0, 1, 1)x(1, 1, 1, 12)   Log Likelihood               -2287.315
Date:                            Wed, 19 Dec 2018   AIC                           4582.630
Time:                                    16:14:25   BIC                           4598.096
Sample:                                08-03-2017   HQIC                          4588.784
                                     - 08-08-2018                                         
Covariance Type:                              opg                                         
==========================================================================================

# visulaisation
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Prediction & visulisation
pred = results.get_prediction(start=pd.to_datetime('2018-07-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = olt_usage['2017':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Olt Usage')
plt.legend()
plt.show()

olt_usage_forecasted = pred.predicted_mean
#olt_usage_forecasted = olt_usage_forecasted.to_frame()
olt_usage_truth = olt_usage['2018-07-01':]

olt_usage_truth = pd.Series(olt_usage_truth.usage)
mse = ((olt_usage_forecasted - olt_usage_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# The Mean Squared Error of our forecasts is 127387.04
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

The Root Mean Squared Error of our forecasts is 356.91

## Producing and visualizing forecasts
#pred_uc = results.forecast(steps=48, full_output=True)
from math import factorial as fact
pred_uc = results.get_forecast(steps= 60 )
pred_ci = pred_uc.conf_int()
ax = olt_usage.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()
