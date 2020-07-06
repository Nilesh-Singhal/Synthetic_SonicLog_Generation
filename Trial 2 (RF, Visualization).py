# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:20:28 2020

@author: niles
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

"""Reading the whole dataset CAL, CNC, GR HRD, HRM, PE, ZDEN, DTC, DTS"""
df = pd.read_csv('SWPLA_Dataset/train.csv')
# remove all rows that contains missing value
df.replace(['-999', -999], np.nan, inplace=True)
df.dropna(axis=0, inplace=True)

"""Outlier Removal"""
"""**Outliers removal based on thresholding**"""
df = df[(df.GR > 0) & (df.GR  <= 250)]
df = df[(df.PE > 0) & (df.PE  <= 8)]
df = df[(df.ZDEN > 1.95) & (df.ZDEN  <= 2.95)]
df = df[(df.CNC > 0) & (df.CNC  <= 1)]
df.shape

"""**Z-Score**"""
# Z-Score
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df))
print(z)
threshold = 3
print(np.where(z > 3))
df = df[(z < 3).all(axis=1)]
df.shape

"""Function for plotting the dataset in a log format, part of Visualization"""
def log_plot(logs):
    logs = logs.sort_values(by='Depth')
    top = logs.Depth.min()
    bot = logs.Depth.max()
    
    f, ax = plt.subplots(nrows=1, ncols=9, figsize=(12,8))
    ax[0].plot(logs.CAL, logs.Depth, color='green')
    ax[1].plot(logs.CNC, logs.Depth, color='red')
    ax[2].plot(logs.GR, logs.Depth, color='black')
    ax[3].plot(logs.HRD, logs.Depth, color='blue')
    ax[4].plot(logs.HRM, logs.Depth, color='c')
    ax[5].plot(logs.PE, logs.Depth, color='g')
    ax[6].plot(logs.ZDEN, logs.Depth, color='r')
    ax[7].plot(logs.DTC, logs.Depth, color='b')
    ax[8].plot(logs.DTS, logs.Depth, color='m')
    
    for i in range(len(ax)):
        ax[i].set_ylim(top,bot)
        ax[i].invert_yaxis()
        ax[i].grid()
        
    ax[0].set_xlabel("CAL")
    ax[0].set_xlim(logs.CAL.min(),logs.CAL.max())
    ax[0].set_ylabel("Depth(ft)")
    ax[1].set_xlabel("CNC")
    ax[1].set_xlim(logs.CNC.min(),logs.CNC.max())
    ax[2].set_xlabel("GR")
    ax[2].set_xlim(logs.GR.min(),logs.GR.max())
    ax[3].set_xlabel("HRD")
    ax[3].set_xlim(logs.HRD.min(),logs.HRD.max())
    ax[4].set_xlabel("HRM")
    ax[4].set_xlim(logs.HRM.min(),logs.HRM.max())
    ax[5].set_xlabel("PE")
    ax[5].set_xlim(logs.PE.min(),logs.PE.max())
    ax[6].set_xlabel("ZDEN")
    ax[6].set_xlim(logs.ZDEN.min(),logs.ZDEN.max())
    ax[7].set_xlabel("DTC")
    ax[7].set_xlim(logs.DTC.min(),logs.DTC.max())
    ax[8].set_xlabel("DTS")
    ax[8].set_xlim(logs.DTS.min(),logs.DTS.max())
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[6].set_yticklabels([])
    ax[7].set_yticklabels([]); ax[8].set_yticklabels([])
    
    f.suptitle('Representational Log View', fontsize=14,y=0.94)

#Making Representational Log View
Depth = np.array(list(range(1,len(df.PE)+1)))
Depth = Depth.reshape(len(df.PE),1)
df_depth = np.column_stack((Depth,df.iloc[:,0:9]))
df_depth = pd.DataFrame(df_depth, columns=['Depth','CAL','CNC','GR','HRD','HRM','PE','ZDEN','DTC','DTS' ])
log_plot(df_depth)

# Heat Map
import seaborn as sb
C_mat = df.corr()
fig = plt.figure(figsize = (15,15))
sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

#Box Plot
# Cut the window with 3 rows and 3 columns:
plt.subplot(331)
sns.boxplot(x=df['CAL'])
plt.subplot(332)
sns.boxplot(x=df['CNC'])
plt.subplot(333)
sns.boxplot(x=df['GR'])
plt.subplot(334)
sns.boxplot(x=df['HRD'])
plt.subplot(335)
sns.boxplot(x=df['HRM'])
plt.subplot(336)
sns.boxplot(x=df['PE'])
plt.subplot(337)
sns.boxplot(x=df['ZDEN'])
plt.subplot(338)
sns.boxplot(x=df['DTC'])
plt.subplot(339)
sns.boxplot(x=df['DTS'])
plt.show()

ax = sns.boxplot(data=df, orient="h", palette="Set2")

import seaborn as sns# Create the default pairplot (correlogram)
#sns.pairplot(df.iloc[:,0:9], kind = "reg")
sns.pairplot(df.iloc[:,0:9])
plt.show()

# Histogram
import matplotlib.pyplot as plt
df.hist(figsize = (12,10))
plt.show()

plt.plot(df,'-')

"""High Correlation between CNC, GR, ZDEN, DTC, DTS"""
"""Feature Generation"""
"""Vs according to the Gardner relation"""
Vsgardner = (df.ZDEN/0.37)**(1/0.22)
Vsdts = 1/(df.DTS)
Vpdtc = 1/(df.DTC)
Vsbrocher = 0.7858 - 1.2344*Vpdtc + 0.7949*Vpdtc**2 - 0.1238*Vpdtc**3 + 0.0064*Vpdtc**4
dtsbrocher = 1/Vsbrocher
Vpbrocher = 39.128*df.ZDEN - 63.064*(df.ZDEN)**2 + 37.083*(df.ZDEN)**3 - 9.1819*(df.ZDEN)**4 + 0.8228*df.ZDEN**5

# Random forest model
#RF = RandomForestRegressor(n_estimators=400, max_depth = 4, max_leaf_nodes = 20, random_state=100)
#RF_fitted = RF.fit(X_train, np.squeeze(Y_train))
#Y_pred = RF_fitted.predict(X_test)
#ET Regressor
#reg_fitted = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(X_train, np.squeeze(Y_train))
#Y_pred = reg_fitted.predict(X_test)
#RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
#RMSE_final1 = np.sqrt(0.5*(RMSE**2 + RMSE1**2))

plt.plot(Depth, Vsdts)
plt.title("Vs(DTS) vs Depth")
plt.show()
plt.plot(Depth, Vsgardner)
plt.title("Vs(ZDEN) vs Depth")
plt.show()
plt.plot(df.ZDEN, Vsgardner)
plt.title("Vs vs density")
plt.show()
plt.scatter(Vsdts, Vsgardner)
plt.title("Vs(ZDEN) vs Vs(DTS)")
axes = plt.gca()
axes.set_xlim([min(Vsdts),max(Vsdts)])
plt.show()
plt.scatter(Vsdts, Vsbrocher)
plt.title("Vs(from Vp as obtained from Vpdtc) vs Vs(DTS)")
axes = plt.gca()
axes.set_xlim([min(Vsdts),max(Vsdts)])
plt.show()
plt.scatter(dtsbrocher, df.DTS)
plt.title("DTS(from Vsbrocher as obtained from DTC) vs Given DTS")
axes = plt.gca()
axes.set_xlim([min(dtsbrocher),max(dtsbrocher)])
plt.show()

"""Training Data CAL, CNC, GR, HRD, HRM, PE, ZDEN"""
df_train_DTC_X = df.iloc[:, 0:7]
df_train_DTC_X = StandardScaler().fit_transform(df_train_DTC_X)
df_train_DTC_Y = df.iloc[:, 7:8]
df_train_DTS_X = np.column_stack((df.CNC, dtsbrocher))
df_train_DTS_X = StandardScaler().fit_transform(df_train_DTS_X)
df_train_DTS_Y = df.iloc[:, 8:9]

#df1 refers to hidden 20percent data (testing data)
df1 = pd.read_csv('C:/Users/niles/Desktop/Python_Projects/SWPLA_Contest/SWPLA_Dataset/real_result_20perc.csv') 

"""Outlier Removal"""
"""**Outliers removal based on thresholding**"""
df1 = df1[(df1.GR > 0) & (df1.GR  <= 250)]
df1 = df1[(df1.PE > 0) & (df1.PE  <= 8)]
df1 = df1[(df1.ZDEN > 1.95) & (df1.ZDEN  <= 2.95)]
df1= df1[(df1.CNC > 0) & (df1.CNC  <= 1)]
df1.shape

"""**Z-Score**"""
# Z-Score
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df1))
print(z)
threshold = 3
print(np.where(z > 3))
df1 = df1[(z < 3).all(axis=1)]
df1.shape


df1_test_DTC_X = df1.iloc[:,0:7]
df1_test_DTC_X = StandardScaler().fit_transform(df1_test_DTC_X)
df1_test_DTC_Y = df1.iloc[:, 7:8]
df1_test_DTS_Y = df1.iloc[:, 8:9]

""" Random forest model for fitting and predicting DTC"""
"""For tuning"""
#RF = RandomForestRegressor(n_estimators=400, max_depth = 4, max_leaf_nodes = 20, random_state=100)
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [80, 90, 100, 110],
#    'max_features': [2, 3],
#    'min_samples_leaf': [3, 4, 5],
#    'min_samples_split': [8, 10, 12],
#    'n_estimators': [100, 200, 300, 1000]
#}
#RF_best_fit = grid_search(RF, param_grid, df_train_DTC_X, np.squeeze(df_train_DTC_Y))
"""Tuned Model"""
RF_best = RandomForestRegressor(bootstrap=True, max_features=4, n_estimators=400, max_depth = 4, max_leaf_nodes = 20, min_samples_split = 8, random_state=100)
RF_best_fit = RF_best.fit(df_train_DTC_X, np.squeeze(df_train_DTC_Y))
pred_DTC = RF_best_fit.predict(df1_test_DTC_X)
RMSE_DTC = np.sqrt(mean_squared_error(df1_test_DTC_Y, pred_DTC))

""" Creating test data for hidden dataset from the predicted DTC from above model (we will determine dtsbrocher)"""
Vpdtc = 1/(pred_DTC)
Vsbrocher = 0.7858 - 1.2344*Vpdtc + 0.7949*Vpdtc**2 - 0.1238*Vpdtc**3 + 0.0064*Vpdtc**4
dtsbrocher = 1/Vsbrocher
df1_test_DTS_X = np.column_stack((df1.CNC, dtsbrocher))
df1_test_DTS_X = StandardScaler().fit_transform(df1_test_DTS_X)

""" Random forest model for fitting and predicting DTS"""
"""For tuning"""
#RF_best_fit1 = grid_search(RF, param_grid, df_train_DTS_X, np.squeeze(df_train_DTS_Y))
"""Tuned Model"""
RF_best = RandomForestRegressor(bootstrap=True, max_features=2, n_estimators=400, max_depth = 4, max_leaf_nodes = 20, min_samples_split = 8, random_state=100)
RF_best_fit1 = RF_best.fit(df_train_DTS_X, np.squeeze(df_train_DTS_Y))
pred_DTS = RF_best_fit1.predict(df1_test_DTS_X)
RMSE_DTS = np.sqrt(mean_squared_error(df1_test_DTS_Y, pred_DTS))

RMSE_final = np.sqrt(0.5*(RMSE_DTC**2 + RMSE_DTS**2))
print('RMSE_DTC', RMSE_DTC)
print('RMSE_DTS', RMSE_DTS)
print('RMSE_final', RMSE_final)