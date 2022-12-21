import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
from copy import deepcopy
from random import randint

import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import boxcox as BxCx
from sklearn.neural_network import MLPRegressor

def open_data(path):
    X1path = pjoin(path, "X1.csv")
    X2path = pjoin(path, "X2.csv")
    Y1path = pjoin(path, "Y1.csv")

    X1 = pd.read_csv(X1path, na_values="\\N")
    X2 = pd.read_csv(X2path, na_values="\\N")
    Y1 = pd.read_csv(Y1path, na_values="\\N", header=None)
    
    return X1, X2, Y1

def strToFloatArray(df, *features):
    df_copy = df.copy()
    for feature in features:
        newFeature  = []
        for item, index in zip(df[feature], range(len(df[feature]))):
            newFeature.append(np.array(item[1:-1].split(",")).astype(float))
        df_copy[feature] = newFeature
    return df_copy

def Binarize(df, feature, n_min):
    df_copy = df.copy()
    df_copy[feature].fillna("Unknown", inplace=True)
    df_copy[feature] = df_copy[feature].str.split(",", expand = False)
    
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(df_copy[feature]), columns=[feature + '_' for i in range(len(mlb.classes_))] + mlb.classes_)   
    res = res[res.columns[res.sum()>n_min]]
    
    df_copy = df_copy.join(res)
    df_copy.drop(feature, axis=1, inplace=True)
    if feature + '_' + 'Unknown' in df_copy.columns :
        df_copy.drop(feature + '_' + 'Unknown', axis=1, inplace=True)
    return df_copy

# Compute the Root Mean Square Error
def compute_rmse(predict, target):
#     if target.flatten().shape[0] != 1:
    if len(target.shape) == 2:
        target = target.squeeze()
    if len(predict.shape) == 2:
        predict = predict.squeeze()
    diff = target - predict
    if len(diff.shape) == 1:
        diff = np.expand_dims(diff, axis=-1)
    rmse = np.sqrt(diff.T@diff / diff.shape[0])
#     else :
#         diff = target - predict
#         rmse = np.abs(diff)
    return float(rmse)


class Process:

    def __init__(self, X, Y) -> None:
        self.X, self.Y = X.copy(), Y.copy()
        self.X_test = None
        self.Y_test = None
        self.X_train = None
        self.Y_train = None
        self.train_index = None
        self.test_index = None
        self.seed = None
        self.testSize = None
        self.transforms = {}
        self.models = {}
        self.preds = {}

    def resetProcessing(self, keepPreds = True):
        if self.train_index:
            self.setTrainTest(train_index=self.train_index, test_index=self.test_index)
        else:
            self.setTrainTest(test_size=self.testSize, newSeed=False)
        self.transforms = {}
        self.models = {}
        if not keepPreds:
            self.preds = {}

    def setTrainTest(self, test_size = 1/3, train_index = None, test_index = None, newSeed = True):
        if train_index is not None and test_index is not None:
            self.X_train, self.X_test = self.X.iloc[train_index].copy(), self.X.iloc[test_index].copy()
            self.Y_train, self.Y_test = self.Y.iloc[train_index].copy(), self.Y.iloc[test_index].copy()
            self.train_index = train_index
            self.test_index = test_index
        else : 
            self.testSize = test_size
            if newSeed:
                self.seed = randint(0, 2**32 - 1)
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.testSize, random_state=self.seed)

    def PCA_embeddings(self, feature, N):
        temp = preprocessing.scale(np.vstack(self.X_train[feature]))
        pca = PCA(n_components = N)
        self.X_train = pd.concat([self.X_train, pd.DataFrame(pca.fit_transform(temp), columns=[feature + str(i) for i in range(N)]).set_index(self.X_train.index)], axis=1)

        temp = preprocessing.scale(np.vstack(self.X_test[feature]))
        self.X_test = pd.concat([self.X_test, pd.DataFrame(pca.transform(temp), columns=[feature + str(i) for i in range(N)]).set_index(self.X_test.index)], axis=1)
        
        self.X_train.drop(feature, axis=1, inplace=True)
        self.X_test.drop(feature, axis=1, inplace=True)

        self.transforms["PCA_" + feature] = pca

        #print(f"PCA of {feature} with {N} components explain {np.sum(pca.explained_variance_ratio_) * 100}% of the variance.")

    def emb_most_corr(self, *features): 
        for feature in features:
            newFeature_train = np.vstack(self.X_train[feature])
            newFeature_test = np.vstack(self.X_test[feature])
            
            temp_df_newFeature_train = pd.DataFrame(newFeature_train, columns=[feature + str(i) for i in range(len(newFeature_train[0]))]).set_index(self.X_train.index)
            temp_df_newFeature_test = pd.DataFrame(newFeature_test, columns=[feature + str(i) for i in range(len(newFeature_test[0]))]).set_index(self.X_test.index)
            
            corr = temp_df_newFeature_train.corrwith(self.Y[0])
            temp_df_newFeature_train = temp_df_newFeature_train[temp_df_newFeature_train.columns[np.abs(corr)>=0.08]]
            temp_df_newFeature_test = temp_df_newFeature_test[temp_df_newFeature_test.columns[np.abs(corr)>=0.08]]
            
            self.X_train = pd.concat([self.X_train, temp_df_newFeature_train], axis=1)
            self.X_test = pd.concat([self.X_test, temp_df_newFeature_test], axis=1)
            
            self.X_train.drop(feature, axis=1, inplace=True)
            self.X_test.drop(feature, axis=1, inplace=True)
            
    def standardize(self, features):
        features = [feature for feature in features if feature in self.X_train.columns]
        
        scaler = StandardScaler()
        self.X_train[features] = scaler.fit_transform(self.X_train[features])
        self.X_test[features] = scaler.transform(self.X_test[features])

        name = "standardize"
        for feature in features:
            name += "_" + feature
        self.transforms[name] = scaler

    def minmaxize(self, features = None, withOutliers = True):
        features = features if features else self.X_train.columns
        features = [feature for feature in features if feature in self.X_train.columns]
        
        scaler = MinMaxScaler()
        if withOutliers:
            self.X_train[features] = scaler.fit_transform(self.X_train[features])
            self.X_test[features] = scaler.transform(self.X_test[features])
            name = "minmaxize"
            for feature in features:
                name += "_" + feature
            self.transforms[name] = scaler
        else : 
            threshold = 1.5
            for feature in features:
                temp = self.X_train[feature].copy()
                qi, qf = np.quantile(temp, [0.25, 0.75])
                mean = np.mean(temp)
                IQR = abs(qf - qi)
                temp = temp[temp <= qf + threshold * IQR]
                temp = temp[temp >= qi - threshold * IQR]
                temp = temp.to_numpy().reshape(-1, 1)
                scaler = scaler.fit(temp)
                self.X_train[feature] = scaler.transform(self.X_train[feature].to_numpy().reshape(-1, 1))
                self.X_test[feature] = scaler.transform(self.X_test[feature].to_numpy().reshape(-1, 1))
                self.transforms["minmaxize_" + feature] = scaler


    def removeDuplicate(self, colmns):
        #df_dupl = self.X_train[self.X_train.duplicated() == True]
        df_dupl = self.X_train[self.X_train[colmns].duplicated() == True]
        self.X_train = self.X_train.drop(df_dupl.index, axis=0)
        self.Y_train = self.Y_train.drop(df_dupl.index, axis=0)

    def corrThreshold(self, threshold = 0.1):
        corr = self.X_train.corrwith(self.Y_train[0])
        self.X_train = self.X_train[self.X_train.columns[np.abs(corr)>=threshold]]
        self.X_test = self.X_test[self.X_test.columns[np.abs(corr)>=threshold]]
        
    
    def MI_selection(self, N, worst=True):
        MI = mutual_info_regression(self.X_train, np.concatenate(self.Y_train.to_numpy()))
        sorted_arg = np.flip(np.array(np.argsort(MI)))
        if worst:
            self.X_train = self.X_train[self.X_train.keys()[sorted_arg[:-N]]]
            self.X_test = self.X_test[self.X_test.keys()[sorted_arg[:-N]]]
        else:
            self.X_train = self.X_train[self.X_train.keys()[sorted_arg[:N]]]
            self.X_test = self.X_test[self.X_test.keys()[sorted_arg[:N]]]


    def removeRedundantFeatures(self, threshold = 0.75):
        cor_matrix = self.X_train.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        self.X_test.drop(to_drop, axis=1, inplace=True)
        self.X_train.drop(to_drop, axis=1, inplace=True)

    def removeOutliers(self, features, threshold = 1.5):
        tempX = self.X_train.copy()
        tempY = self.Y_train.copy()
        for feature in features:
            qi, qf = np.quantile(self.X_train[feature], [0.25, 0.75])
            mean = np.mean(self.X_train[feature])
            IQR = abs(qf - qi)
            tempY = tempY.loc[tempX[feature] <= qf + threshold * IQR]
            tempY = tempY.loc[tempX[feature] >= qi - threshold * IQR]
            tempX = tempX.loc[tempX[feature] <= qf + threshold * IQR]
            tempX = tempX.loc[tempX[feature] >= qi - threshold * IQR]
        self.X_train = tempX
        self.Y_train = tempY

    def addModel(self, modelType, name = None, **kwargs):
        if modelType == "linear":
            model = LinearRegression()
            model = model.fit(np.vstack(self.X_train.to_numpy()), np.vstack(self.Y_train.to_numpy())*1e-6)
        elif modelType == "knn":
            n_neighbors = 10
            p = 2
            weights = "uniform"
            for arg, val in zip(kwargs, kwargs.values()):
                if arg == "n_neighbors":
                    n_neighbors = val
                if arg == "p":
                    p = val
                if arg == "weights":
                    weights = val
            model = KNeighborsRegressor(n_neighbors=n_neighbors, p=p)
            model.fit(np.vstack(self.X_train.to_numpy()), np.vstack(self.Y_train.to_numpy())*1e-6)
        elif modelType == "randomForest":
            n_estimators = 100
            criterion = "squared_error"
            max_depth = None
            min_samples_split = 2
            min_samples_leaf = 1
            for arg, val in zip(kwargs, kwargs.values()):
                if arg == "n_estimators":
                    n_estimators = val
                if arg == "criterion":
                    criterion = val
                if arg == "max_depth":
                    max_depth = val
                if arg == "min_samples_split":
                    min_samples_split = val
                if arg == "min_samples_leaf":
                    min_samples_leaf = val
            model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            model.fit(np.vstack(self.X_train.to_numpy()), np.ravel(np.vstack(self.Y_train.to_numpy())*1e-6))
        elif modelType == "mlp":
            hidden_layer_sizes = np.array([100])
            activation = "relu"
            max_iter = 200
            for arg, val in zip(kwargs, kwargs.values()):
                if arg == "hidden_layer_sizes":
                    hidden_layer_sizes= val
                if arg == "activation":
                    activation = val
                if arg == "max_iter":
                    max_iter = val
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate='adaptive', learning_rate_init=1e-2)
            model.fit(np.vstack(self.X_train.to_numpy()), np.ravel(np.vstack(self.Y_train.to_numpy())*1e-6))
        name = name if name else modelType
        self.models[name] = model

    def useModel(self, modelType, X = None):
        if X is not None:
            return self.models[modelType].predict(np.vstack(X.to_numpy()))
        else:
            self.preds[modelType] = self.models[modelType].predict(np.vstack(self.X_test.to_numpy()))
            return self.preds[modelType]