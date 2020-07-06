"""First Outlier Removal
   Second Visualization for identifying parameter correlation
   Third feature generation with only training dataset 
   Fourth clustering for whole dataset
   Fifth Machine Learning Model"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

"""Function for tuning random forrest using grid search"""
def grid_search(clf, param_grid, X_train, y_train):
    """
    Fits a classifier to its training data and prints its ROC AUC score.
    
    INPUT:
    - clf (classifier): classifier to fit
    - param_grid (dict): classifier parameters used with GridSearchCV
    - X_train (DataFrame): training input
    - y_train (DataFrame): training output
            
    OUTPUT:
    - classifier: input classifier fitted to the training data
    """
    # cv uses StratifiedKFold
    # scoring r2 as parameter
    grid = GridSearchCV(estimator=clf, 
                        param_grid=param_grid, 
                        scoring='r2', 
                        cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    
    return grid.best_estimator_

 
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
    
"""Reading the whole dataset CAL, CNC, GR HRD, HRM, PE, ZDEN, DTC, DTS"""
df = pd.read_csv('SWPLA_Dataset/train.csv')
# remove all rows that contains missing value
df.replace(['-999', -999], np.nan, inplace=True)
df.dropna(axis=0, inplace=True)

"""Scatter plot before Outlier Detection for creating intution for how to go with the process"""
import seaborn as sns# Create the default pairplot
sns.pairplot(df)
plt.show()

"""Outlier Removal"""
"""**Outliers removal based on thresholding**"""
df = df[(df.GR > 0) & (df.GR  <= 250)]
df = df[(df.PE > 0) & (df.PE  <= 8)]
df = df[(df.ZDEN > 1.95) & (df.ZDEN  <= 2.95)]
df.shape

%matplotlib inline
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

"""Creating another varaible to store the DTC and DTS values before scaling to compare with the reverted predicted value"""
df_revert = df # this process is used as in isolation forest data is scaled

"""Outlier Detection and Removal using Isolation Forest (CNC vs DTS, ZDEN vs DTS, DTC vs DTS)"""
import pandas as pd
import numpy as np
import matplotlib.font_manager
import matplotlib.pyplot as plt

# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from scipy import stats

"""Outlier removal based on CNC VS DTS"""
df.plot.scatter('CNC','DTS')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df[['CNC','DTS']] = scaler.fit_transform(df[['CNC','DTS']])
df[['CNC','DTS']].head()

X1 = df['CNC'].values.reshape(-1,1)
X2 = df['DTS'].values.reshape(-1,1)

X = np.concatenate((X1,X2),axis=1)
random_state = np.random.RandomState(42)
outliers_fraction = 0.08
# Define seven outlier detection tools to be compared
classifiers = {'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state)}

xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))
    
    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    
    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 =  np.array(dfx['CNC'][dfx['outlier'] == 0]).reshape(-1,1)
    IX2 =  np.array(dfx['DTS'][dfx['outlier'] == 0]).reshape(-1,1)
    
    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 =  dfx['CNC'][dfx['outlier'] == 1].values.reshape(-1,1)
    OX2 =  dfx['DTS'][dfx['outlier'] == 1].values.reshape(-1,1)
         
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
        
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
        
    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
          
    # fill blue map colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        
    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        
    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        
    b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
    
    c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
       
    plt.axis('tight')  
    
    # loc=2 is used for the top left corner 
    plt.legend(
        [a.collections[0], b,c],
        ['learned decision function', 'inliers','outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)
      
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(clf_name)
    plt.show()

df = df.drop(df[df.outlier == 1].index)

"""Outlier removal based on ZDEN VS DTS"""
df.plot.scatter('ZDEN','DTS')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df[['ZDEN','DTS']] = scaler.fit_transform(df[['ZDEN','DTS']])
df[['ZDEN','DTS']].head()

X1 = df['ZDEN'].values.reshape(-1,1)
X2 = df['DTS'].values.reshape(-1,1)

X = np.concatenate((X1,X2),axis=1)
random_state = np.random.RandomState(42)
outliers_fraction = 0.08
# Define seven outlier detection tools to be compared
classifiers = {'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state)}

xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))
    
    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    
    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 =  np.array(dfx['ZDEN'][dfx['outlier'] == 0]).reshape(-1,1)
    IX2 =  np.array(dfx['DTS'][dfx['outlier'] == 0]).reshape(-1,1)
    
    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 =  dfx['ZDEN'][dfx['outlier'] == 1].values.reshape(-1,1)
    OX2 =  dfx['DTS'][dfx['outlier'] == 1].values.reshape(-1,1)
         
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
        
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
        
    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
          
    # fill blue map colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        
    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        
    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        
    b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
    
    c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
       
    plt.axis('tight')  
    
    # loc=2 is used for the top left corner 
    plt.legend(
        [a.collections[0], b,c],
        ['learned decision function', 'inliers','outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)
      
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(clf_name)
    plt.show()

df = df.drop(df[df.outlier == 1].index)

"""Outlier removal based on DTC VS DTS"""
df.plot.scatter('DTC','DTS')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df[['DTC','DTS']] = scaler.fit_transform(df[['DTC','DTS']])
df[['DTC','DTS']].head()

X1 = df['DTC'].values.reshape(-1,1)
X2 = df['DTS'].values.reshape(-1,1)

X = np.concatenate((X1,X2),axis=1)
random_state = np.random.RandomState(42)
outliers_fraction = 0.08
# Define seven outlier detection tools to be compared
classifiers = {'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state)}

xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))
    
    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    
    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 =  np.array(dfx['DTC'][dfx['outlier'] == 0]).reshape(-1,1)
    IX2 =  np.array(dfx['DTS'][dfx['outlier'] == 0]).reshape(-1,1)
    
    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 =  dfx['DTC'][dfx['outlier'] == 1].values.reshape(-1,1)
    OX2 =  dfx['DTS'][dfx['outlier'] == 1].values.reshape(-1,1)
         
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
        
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
        
    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
          
    # fill blue map colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        
    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        
    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        
    b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
    
    c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
       
    plt.axis('tight')  
    
    # loc=2 is used for the top left corner 
    plt.legend(
        [a.collections[0], b,c],
        ['learned decision function', 'inliers','outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)
      
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(clf_name)
    plt.show()

df = df.drop(df[df.outlier == 1].index)
    
"""Visualization for identifying correlation between parameters"""
"""Scatter plot after Outlier Detection for checking the effectiveness of the OD process"""
import seaborn as sns# Create the default pairplot
sns.pairplot(df.iloc[:,0:9])
plt.show()

# Histogram
import matplotlib.pyplot as plt
df.hist(figsize = (12,10))
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

# Heat Map
import seaborn as sb
C_mat = df.corr()
fig = plt.figure(figsize = (15,15))
sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

#Making Representational Log View
Depth = np.array(list(range(1,len(df.PE)+1)))
Depth = Depth.reshape(len(df.PE),1)
df_depth = np.column_stack((Depth,df.iloc[:,0:9]))
df_depth = pd.DataFrame(df_depth, columns=['Depth','CAL','CNC','GR','HRD','HRM','PE','ZDEN','DTC','DTS' ])
log_plot(df_depth)

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
df1.shape

%matplotlib inline
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

scaler = MinMaxScaler(feature_range=(0, 1))
df1[['CNC','ZDEN', 'DTC','DTS']] = scaler.fit_transform(df1[['CNC','ZDEN','DTC','DTS']])
    
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
RF_best = RandomForestRegressor(bootstrap=True, max_features=3, n_estimators=200, max_depth = 80, max_leaf_nodes = None, min_samples_split = 8, random_state=100)
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
RF_best = RandomForestRegressor(bootstrap=True, max_features=2, n_estimators=200, max_depth = 80, max_leaf_nodes = None, min_samples_split = 8, random_state=100)
RF_best_fit1 = RF_best.fit(df_train_DTS_X, np.squeeze(df_train_DTS_Y))
pred_DTS = RF_best_fit1.predict(df1_test_DTS_X)
RMSE_DTS = np.sqrt(mean_squared_error(df1_test_DTS_Y, pred_DTS))

RMSE_final = np.sqrt(0.5*(RMSE_DTC**2 + RMSE_DTS**2))

"""Neural Network"""
sns.pairplot(df1)


"""Setting up Environment in PyCaret"""
from pycaret.utils import version
version()

import pycaret
from pycaret.clustering import *

cluster1 = setup(df.iloc[:,0:9], normalize = True, 
                   ignore_features = [],
                   session_id = 123)

"""- **session_id :**  A pseduo-random number distributed as a seed in all functions for later reproducibility. If no `session_id` is passed, a random number is automatically generated that is distributed to all functions. In this experiment, the `session_id` is set as `123` for later reproducibility.<br/>
<br/>
- **Missing Values :**  When there are missing values in original data this will show as True. Notice that `Missing Values` in the information grid above is `True` as the data contains missing values which are automatically imputed using `mean` for numeric features and `constant` for categorical features. The method of imputation can be changed using the `numeric_imputation` and `categorical_imputation` parameters in `setup()`. <br/>
<br/>
- **Original Data :**  Displays the original shape of dataset. In this experiment (1026, 82) means 1026 samples and 82 features. <br/>
<br/>
- **Transformed Data :** Displays the shape of the transformed dataset. Notice that the shape of the original dataset (1026, 82) is transformed into (1026, 91). The number of features has increased due to encoding of categorical features in the dataset. <br/>
<br/>
- **Numeric Features :**  The number of features inferred as numeric. In this dataset, 77 out of 82 features are inferred as numeric. <br/>
<br/>
- **Categorical Features :**  The number of features inferred as categorical. In this dataset, 5 out of 82 features are inferred as categorical. Also notice that we have ignored one categorical feature `MouseID` using the `ignore_feature` parameter. <br/>
<br/>"""

# 7.0 Create a Model
kmeans = create_model('kmeans')
print(kmeans)
plot_model(kmeans)
plt.show()

"""We have created a kmeans model using `create_model()`. Notice the `n_clusters` parameter is set to `4` which is the default when you do not pass a value to the `num_clusters` parameter. In the below example we will create a `kmodes` model with 6 clusters."""

#kmodes = create_model('kmodes', num_clusters = 3)
#print(kmodes)

#dbscan = create_model('dbscan')
#dbscan.eps = 0.01
#dbscan.min_samples = 500
#print(dbscan)

hclust = create_model('hclust')
hclust.n_clusters =3
print(hclust)

birch = create_model('birch')
birch.n_clusters =3
birch

#optics = create_model('optics')
#optics.min_samples = 500
#optics.eps = 0.1

"""Simply replacing `kmeans` with `kmodes` inside `create_model()` has created a`kmodes` clustering model. There are 9 models available in the `pycaret.clustering` module. To see the complete list, please see the docstring. If you would like to read more about the use cases and limitations of different models, you may __[click here](https://scikit-learn.org/stable/modules/clustering.html)__ to read more.

# 8.0 Assign a Model

Now that we have created a model, we would like to assign the cluster labels to our dataset (1080 samples) to analyze the results. We will achieve this by using the `assign_model()` function. See an example below:
"""
kmean_results = assign_model(kmeans)
kmean_results.head()

import pandas as pd
# Encoding label data
kmean_results['Cluster'] = pd.Categorical(kmean_results['Cluster'])
kmean_resultsDummies = pd.get_dummies(kmean_results['Cluster'], prefix = '')
kmean_results = pd.concat([kmean_results, kmean_resultsDummies], axis=1)
kmean_results = kmean_results.iloc[:,np.r_[0:7, 8:12]]
#############################################################################
#############################################################################
# DTC
DTC_train = kmeans_results
DTC_test = data.iloc[:, 7:8]
DTC_train = pd.concat([DTC_train, DTC_test], axis=1)

from pycaret.regression import *
reg1 = setup(DTC_train, target = 'DTC', session_id = 123, silent = True) 
compare_models(blacklist = ['tr']) 

##############################################################################
##############################################################################
df1 = pd.read_csv('SWPLA_Dataset/real_result_20perc.csv')
df1_train = df1.iloc[:, 0:7]

exp_clu102 = setup(df1_train, normalize = True, 
                   ignore_features = [],
                   session_id = 123)
kmeans1 = create_model('kmeans')
print(kmeans1)

kmean_results1 = assign_model(kmeans1)
kmean_results1.head()

kmean_results1['Cluster'] = pd.Categorical(kmean_results1['Cluster'])
kmean_results1Dummies = pd.get_dummies(kmean_results1['Cluster'], prefix = '')
kmean_results1 = pd.concat([kmean_results1, kmean_results1Dummies], axis=1)
kmean_results1 = kmean_results1.iloc[:,np.r_[0:7, 8:12]]