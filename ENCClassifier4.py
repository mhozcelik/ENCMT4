import time
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import variation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.base import BaseEstimator

# ------------------------------------------------------------------

def Trimming(X,trim):
    Xp = X
    top5 = []
    bottom5 = []
    for c in X.columns[(X.dtypes != object) & (X.dtypes != "category")]:
        # winsorization
        if X[c].nunique()>=50: 
            cut_off95 = np.percentile(X[c],(100-trim))
            cut_off05 = np.percentile(X[c],trim)
        else:
            cut_off95 = 9999999999.
            cut_off05 =-9999999999.
        top5.append(cut_off95)
        bottom5.append(cut_off05)
        Xp[c] = np.where(X[c]>cut_off95,cut_off95,X[c])
        Xp[c] = np.where(X[c]<cut_off05,cut_off05,X[c])
    return Xp, top5, bottom5

def Calculate_Relevance(data, target, bins=10):
    Rel_main_df = pd.DataFrame()
    Rel_detail_df = pd.DataFrame()
    cols = data.columns
    for ivars in cols[~cols.isin([target])]:
        My_y = data[target]
        if (data[ivars].dtype.kind in 'biufc') and (len(np.unique(data[ivars]))>bins):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            #binned_x = np.digitize(data[ivars], bins=np.quantile(data[ivars],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
            d0 = pd.DataFrame({'x': binned_x, 'y': My_y})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': My_y})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['%'] = d['Events'] / d['N']
        d['Lift'] = d['%'] / My_y.mean()
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['Importance'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        # print("Information value of " + ivars + " is " + str(round(d['IV Part'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "Importance" : [d['Importance'].sum()]}, columns = ["Variable", "Importance"])
        Rel_main_df=pd.concat([Rel_main_df,temp], axis=0)
        Rel_detail_df=pd.concat([Rel_detail_df,d], axis=0)
    Rel_main_df = Rel_main_df.set_index('Variable')
    Rel_main_df["Importance"]=np.where(Rel_main_df["Importance"]>1000,1000,Rel_main_df["Importance"])
    return Rel_main_df, Rel_detail_df

# ------------------------------------------------------------------

def Centroid(X_train, y_train, class_value):
    X_train = X_train.loc[y_train==class_value]
    X = X_train.to_numpy()
    X_Center = X.mean(axis=0)
    X_Var = X.var(ddof=1,axis=0)
    X_Var = np.where(X_Var<=X_Center*0.05,X_Center*0.05,X_Var)
    X_Var = np.where(X_Var<=0.05,0.05,X_Var)
    return X_Center, X_Var

def Calc_Distance (df, X_Var, X_Center, Importance, dist_type, imp_type):
    Hom = 1. / X_Var
    if imp_type == 1:
        ImpHom = Hom*(Importance['Importance'])
    elif imp_type == 2:
        ImpHom = Hom*(1+Importance['Importance'])
    elif imp_type == 3:
        ImpHom = Hom*((Importance['Importance'])**2)
    elif imp_type == 4:
        ImpHom = Hom*((1+Importance['Importance'])**2)
    elif imp_type == 5:
        ImpHom = Hom*(np.sqrt(Importance['Importance']))
    elif imp_type == 6:
        ImpHom = Hom*(np.sqrt(1+Importance['Importance']))
    elif imp_type == 7:
        ImpHom = Hom*(np.log(1+Importance['Importance']))
    else:
        ImpHom = Hom*((1+Importance['Importance'])**2)
    if dist_type == 'L1':
        # L1: Manhattan style
        Dif = np.absolute(df - X_Center)
    elif dist_type == 'L2':
        # L2: Euclidean style
        Dif = (df - X_Center)**2
    else:
        Dif = (df - X_Center)**2
    Distance = Dif.dot(ImpHom)
    Distance = np.where(Distance<0.000001,0.000001,Distance)
    Distance = pd.DataFrame(Distance)
    return Distance

# ------------------------------------------------------------------

def ENC(X, y, X0_Center, X0_Var, X1_Center, X1_Var, Importance, train_mode, cut_off, pred_min, pred_max, dist_type, imp_type):
    Distance0 = Calc_Distance(X, X0_Var, X0_Center, Importance, dist_type, imp_type )
    Distance1 = Calc_Distance(X, X1_Var, X1_Center, Importance, dist_type, imp_type )
    # conversion to probabilities (scores)
    DistanceSum = Distance0.sum(axis=1) + Distance1.sum(axis=1)
    Distance0 = Distance0.div(DistanceSum,axis=0)
    Distance1 = Distance1.div(DistanceSum,axis=0)
    pred_class_info = pd.concat([Distance0.sum(axis=1),Distance1.sum(axis=1)], axis=1)
    # NET Similarity to class ONE
    pred = pred_class_info.iloc[:,0] - pred_class_info.iloc[:,1]
    pred = (pred-pred_min)/(pred_max-pred_min)       
    pred = pd.DataFrame(pred,columns=['score'])
    pred = pred.fillna(0)
    pred_class = np.full((len(pred_class_info),1), False, dtype=bool)
    if train_mode==True:
        pred_class[pred.nlargest(y.sum(), ['score']).index] = True
        cut_off = float(pred[pred_class==True].min())
    else:
        pred_class[pred>=cut_off] = True
    pred_class = np.array(pred_class[:,0],dtype=bool)
    return pred, pred_class, cut_off

# ------------------------------------------------------------------

def ENC_Tuning(X_train, y_train, dist_type, imp_type):
    # LEARNING PHASE STEP 1 - Importance
    y_train = pd.Series(y_train,name='y')
    My_data = pd.concat([X_train,y_train],axis=1)
    Importance, Importance_detail = Calculate_Relevance(data=My_data, target='y')
    # LEARNING PHASE STEP 2 - Centroids
    X0_Center, X0_Var = Centroid(X_train, y_train, 0)
    X1_Center, X1_Var = Centroid(X_train, y_train, 1)
    pred_min = 0
    pred_max = 1
    # LEARNING PHASE STEP 3: ENC
    pred, pred_class, cut_off = ENC(X_train,y_train, X0_Center, X0_Var, X1_Center, X1_Var, Importance, True, 0, pred_min, pred_max, dist_type, imp_type)
    pred_min = float(pred.min())
    pred_max = float(pred.max())
    return Importance, X0_Center, X0_Var, X1_Center, X1_Var, cut_off, pred_min, pred_max

# ------------------------------------------------------------------

class ENCClassifier(BaseEstimator):
    def __init__(self, trim=5, dist_type="L2", imp_type=4):
        self.trim      = trim
        self.dist_type = dist_type
        self.imp_type  = imp_type
        
    def fit(self, X, y):
        self.X_train = pd.DataFrame(X)
        self.y_train = pd.Series(y)
        self.X_train, self.top5, self.bottom5 = Trimming(self.X_train,self.trim)
        self.Importance             = pd.DataFrame()
        self.X0_Center, self.X0_Var = pd.DataFrame(), pd.DataFrame()
        self.X1_Center, self.X1_Var = pd.DataFrame(), pd.DataFrame()
        self.Importance, self.X0_Center, self.X0_Var, self.X1_Center, self.X1_Var, self.cut_off, self.pred_min, self.pred_max \
        = ENC_Tuning(self.X_train, self.y_train, self.dist_type, self.imp_type)
        return
    
# --------------

    def __predictions(self, X):
        temp = ENC(X, y = np.zeros(X.shape[0]), 
                   X0_Center=self.X0_Center, X0_Var=self.X0_Var, 
                   X1_Center=self.X1_Center, X1_Var=self.X1_Var, 
                   Importance=self.Importance, 
                   train_mode=False,
                   cut_off=self.cut_off,
                   pred_min = self.pred_min,
                   pred_max = self.pred_max,
                   dist_type = self.dist_type,
                   imp_type = self.imp_type
                  )
        return temp

    def predict(self, X):
        return pd.DataFrame(self.__predictions(X)[1])

    def predict_proba(self, X):
        return self.__predictions(X)[0]


