import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from ENCModelTreeClassifier4 import ENCModelTreeClassifier

class RFwENCMTClassifier(BaseEstimator):
    def __init__(self, random_state, trim=5, dist_type="L2", imp_type=4, min_samples_leaf=50, scaling='minmax', criterion='gini', n_estimators=10, max_leaf_nodes=None):
        self.random_state = random_state
        self.Sub_Models = []
        self.trim = trim
        self.dist_type = dist_type
        self.imp_type = imp_type
        self.min_samples_leaf = min_samples_leaf
        self.scaling = scaling
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.selected_columns = []
        
    def fit(self, X, y):
        random.seed(self.random_state)
        self.X_train = pd.DataFrame(X)
        self.y_train = pd.Series(y,name='y')
        for i in range(self.n_estimators):
            My_model = ENCModelTreeClassifier(random_state=random.randint(1, self.n_estimators*1000),
                                              trim=self.trim, dist_type=self.dist_type, imp_type=self.imp_type,
                                              min_samples_leaf=self.min_samples_leaf, scaling=self.scaling, 
                                              criterion=self.criterion, max_leaf_nodes=self.max_leaf_nodes)
            tmp = pd.merge(self.X_train,self.y_train,left_index=True,right_index=True)
            tmp = tmp.sample(frac=0.8)
            tmpX = tmp.loc[:,tmp.columns!='y'].sample(frac=0.9,axis=1)
            tmpy = tmp['y']
            self.selected_columns.append(list(tmpX.columns))
            My_model.fit(tmpX,tmpy)
            self.Sub_Models.append(My_model)
        return

    def __predictions(self, X):
        X_tmp = pd.DataFrame(X)
        X_tmp = X_tmp.reset_index(drop=True)
        My_result = pd.DataFrame()
        for i in range(self.n_estimators):
            My_model = self.Sub_Models[i]
            if i==0:
                My_result['score'] = My_model.predict_proba(X_tmp[self.selected_columns[i]])[:,1]/self.n_estimators
            else:
                My_result['score'] = My_result['score'] + My_model.predict_proba(X_tmp[self.selected_columns[i]])[:,1]/self.n_estimators
        My_result['0'] = 1 - My_result['score']
        My_result['1'] = My_result['score']
        My_result = My_result.drop('score',axis=1)
        return My_result.to_numpy(), np.where(My_result['1']>0.5,1,0)
        
    def predict(self, X):
        return self.__predictions(X)[1] 

    def predict_proba(self, X):
        return self.__predictions(X)[0]
