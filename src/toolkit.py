import numpy as np
import pandas as pd
import sys

from sklearn.feature_selection import SelectPercentile as SP
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskLassoCV
from sklearn import preprocessing


def select_features(x, y, methods=('variance', 'correlation', 'l1', 'forest')):
    ''' methods = ('variance', 'correlation', 'l1', 'forest')
        - variance: use variance threshold to discard features that are mostly 0 or 1
        - correlation: use chi2 test to remove most very correlated features
        - l1: use l1 penalty to remove features that make solution sparse
        - forest: use ExtraTreesClassifier to point out importance of features
                    select important ones
    '''
    features = x.loc[:,'Feature_1':'Feature_2']

    if 'variance' in methods:
        vt = VT(threshold=(0.99*(1-0.99)))
        vt.fit(features)
        

    if 'correlation' in methods:
        cr = SP(f_regression, percentile=80)

    if 'l1' in methods:
        rgr = MultiTaskLassoCV(cv=5, n_jobs=-1)
        m = SFM(rgr)
        

    if 'forest' in methods:
        clf = RandomRorestRegressor(n_estimators=300, max_features=0.7,n_jobs=-1).fit(x,y)
        m = SFM(clf)
        m.fit(x.values, y.values)

    for indices in idx_list:
        x_indices = x_indices & indices
    print 'All: %s' % len(x_indices)

    return list(x_indices)
        

def kaggle_error(y_truth, y_preds, weights=None):
    n = y_truth.shape[0]
    intra_weights = weights.Weight_Intraday.values.reshape(n,1)
    daily_weights = weights.Weight_Daily.values.reshape(n,1)
    intra_error = 1./n * np.sum(intra_weights * np.abs(y_truth - y_preds))
    daily_error = 1./n * np.sum(daily_weights * np.abs(y_truth - y_preds))
    return intra_error + daily_error


def get_train_data(file_train, scale=False, size=None):
    data = pd.read_csv(file_train)
    data = data.fillna(data.mean())
    x = data.loc[:size, 'Feature_1':'Ret_120']
    y = data.loc[:size, 'Ret_121':'Ret_180']
    if scale:
        x = preprocessing.scale(x)
    weights = data.loc[:size, 'Weight_Intraday':'Weight_Daily']

    return x, y, weights, 


def get_test_data(file_test, scale=False):
    data = pd.read_csv(file_train)
    data = data.fillna(data.mean())
    x = data.loc[:size]
    if scale:
        x = preprocessing.scale(x)

    return x

    
