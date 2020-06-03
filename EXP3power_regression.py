# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:23:47 2019

Target : 
    1. workflow for classic classification
    2. general functions for labeling, spliting, model collection, modeling, and ploting 
    3. dataset is divided to train (0.8) and test (0.2) by function split_with_scaling, then 
    in trainng process, estimators with k-fold cross validation to see accuracy
TBD : 
    importance or other indicator for parameters judgement after encording 
    (ex. random_classifier.feature_importances_)

@author: Will
"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# commom regressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RANSACRegressor # Linear models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor  # Ensemble methods
from xgboost import XGBRegressor, plot_importance # XGBoost
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression
from sklearn.neighbors import KNeighborsRegressor

# commom regressor metric
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def label_onehot_encoder (ds, Y_col_name) : 
    """
    divide original dataset (ds) to X and Y to encoder X only, then concate X and Y back to ds
    dataset (dataframe) : 
    Y_col_name : target column name    
    """
    # split to X and Y
    X = ds.drop(columns=[Y_col_name])
    Y = ds[Y_col_name]
    
    # encode X
    object_columns = X.columns[X.dtypes == 'object']
    number_columns = X.columns[X.dtypes != 'object']

    qualitative_1 = X[object_columns]
    quantitative_1 = X[number_columns]

    for k in qualitative_1.columns:  
        le=LabelEncoder()
        qualitative_1[k][qualitative_1[k].notnull()]=le.fit_transform(qualitative_1[k][qualitative_1[k].notnull()])

    encoder=OneHotEncoder(sparse=False,dtype=np.int) # dtype=np.int is not nessesary, sparse = false means sparse matrix
    qualitative_1 = pd.DataFrame(encoder.fit_transform(qualitative_1),index=qualitative_1.index)    
    # concat qualitative and quantitative for X
    labeled_X = pd.concat([qualitative_1,quantitative_1],axis=1) 
    
    # concat X and Y
    labeled_ds = pd.concat([labeled_X,Y],axis=1) 
    return labeled_ds, object_columns, number_columns

def split_with_scaling (labeled_ds, Y_col_name, scaler) : 
    """
    to split labeled dataset with scaler by test_size 0.2 and fixed random_state = 7
    labeled_ds : dataset W/ labeling process (categorical) or W/O labeling process (numerical)
    scaler : MinMaxScaler, StandardScaler or so
    """
    X = labeled_ds.drop(columns=[Y_col_name])
    Y = labeled_ds[Y_col_name]
    
    scaled_X = scaler.fit_transform(X)    
    X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = 0.2, random_state = 7)    
    return  X_train, X_test, Y_train, Y_test
   
def model_collection_score (X_train, Y_train, model_collection, kfolds = 5, 
                            metric = "roc_auc", type_name = "classification") : 
    """
    to get preliminary results (score) for each item in model assembly
    input (standarized) : X_train, Y_train processed after train and test split with scaler
    model assembly (dict) : {model name : model instance}
    kfolds (int) : for cross-validation
    metrics : example : classification : "roc_auc", regression : "r2"
    """
    model_results = []
    model_names   = []    
    
    for model_name in model_collection:
        model   = model_collection[model_name]
        k_fold  = KFold(n_splits = kfolds)
        results = cross_val_score(model, X_train, Y_train, cv = k_fold, scoring = metric)
        model_results.append(results)
        model_names.append(model_name)   
    
    figure = plt.figure()
    figure.suptitle('%s models comparison' % type_name, fontsize = 16)
    axis = figure.add_subplot(111)
    plt.boxplot(model_results)
    axis.set_xticklabels(model_names, rotation = 45, ha="right", fontsize = 16)
    axis.set_ylabel(metric, fontsize = 16)
    plt.margins(0.1, 0.1)
    plt.show()    
     
def regressor_model_build_results (objmodel, X_train, X_test, Y_train, Y_test, method_name) : 
    """
    to build regressor model for prediction and metric
    input (standarized) : X_train, Y_train processed after train and test split with scaler 
    input : metric method name
    output : prediction result Y_pred with metric score by metric name
    """
    objmodel.fit(X_train,Y_train)
    predictions = objmodel.predict(X_test)    
    metric_score = metric_selection (Y_test, predictions, method_name)               
    
    return metric_score, predictions

# need more common-used metric ?
def metric_selection (true, predictions, method_name) : 
    """
    bundle top 3 metric score stimation
    input : true value and predictions with method name
    output (dict) : method_name and score
    """
    
    if method_name == "r2" : 
        # from sklearn.metrics import r2_score
        score = r2_score(true, predictions)
    elif method_name == "MAPE" : 
        true_, predictions_ = np.array(true), np.array(predictions)
        score = np.mean(np.abs((true_ - predictions_) / true_)) * 100
    elif method_name == "RMSE" : 
        # from sklearn.metrics import mean_squared_error
        score = np.sqrt(mean_squared_error(true, predictions))
    metric_score = {"method_name" : method_name, "score" : score}
    
    return metric_score

#-----------------------------------------
# regression plot
# residual plot ?
def serial_true_pred_plot (Y_test, predictions, title_name = 'Y_test vs pred') : 
    x_axis = np.array(range(0, predictions.shape[0]))
    plt.plot(x_axis, predictions, linestyle="--", marker=".", color='r', label="pred")
    plt.plot(x_axis, Y_test, linestyle="--", marker=".", color='g', label="Y_test")
    plt.xlabel('Row number')
    plt.ylabel("A.U.")
    plt.title(title_name)
    plt.legend(loc='upper left')
    plt.show()    
    
if __name__ == '__main__' : 
       
    # model for online_moisture
    dataset = pd.read_excel('OnlineMoisture_original.xlsx', sheet_name = '工作表4')
    cols_to_process = ["定型機2-主機速度",
                       "布種", "流程",
                       "定型機2-01.風車速度", "定型機2-02.風車速度", "定型機2-03.風車速度", "定型機2-04.風車速度",
                       "定型機2-05.風車速度", "定型機2-06.風車速度", "定型機2-07.風車速度", "定型機2-08.風車速度",
                       "定型機2-01.風車溫度", "定型機2-02.風車溫度", "定型機2-03.風車溫度", "定型機2-04.風車溫度",
                       "定型機2-05.風車溫度", "定型機2-06.風車溫度", "定型機2-07.風車溫度", "定型機2-08.風車溫度",
                       "定型機2-.1定型機排風速度", "定型機2-.2定型機排風速度",
                       "實際含水率(Y)"]
    
    ds_to_process = dataset[cols_to_process]    
    
    
    name_transfering_list = {"定型機2-主機速度": "main speed", 
                             "布種" : "fabric type", "流程" : "process type",
                             "定型機2-01.風車速度": "No.1 CF speed", "定型機2-02.風車速度": "No.2 CF speed",
                             "定型機2-03.風車速度": "No.3 CF speed", "定型機2-04.風車速度": "No.4 CF speed",
                             "定型機2-05.風車速度": "No.5 CF speed", "定型機2-06.風車速度": "No.6 CF speed",
                             "定型機2-07.風車速度": "No.7 CF speed", "定型機2-08.風車速度": "No.8 CF speed",
                             "定型機2-01.風車溫度": "No.1 CF temp", "定型機2-02.風車溫度": "No.2 CF temp",
                             "定型機2-03.風車溫度": "No.3 CF temp", "定型機2-04.風車溫度": "No.4 CF temp",
                             "定型機2-05.風車溫度": "No.5 CF temp", "定型機2-06.風車溫度": "No.6 CF temp",
                             "定型機2-07.風車溫度": "No.7 CF temp", "定型機2-08.風車溫度": "No.8 CF temp",
                             "定型機2-.1定型機排風速度": "No.1 EF speed", "定型機2-.2定型機排風速度": "No.2 EF speed",
                             "實際含水率(Y)": "online_moisture"
                             }    
    ds_to_process.columns = ds_to_process.columns.map(name_transfering_list)
    ds_to_process.rename(columns = name_transfering_list, inplace=True)
    
    
    labeled_ds, object_columns, number_columns = label_onehot_encoder (ds_to_process, "online_moisture")
    scaler = MinMaxScaler()
    X_train, X_test, Y_train, Y_test = split_with_scaling (labeled_ds, "online_moisture", scaler) 

    # EDA plot
#    f, ax = plt.subplots()
#    ax.set_title('online moisture distribution for each fabric type')
#    sns.distplot(ds_to_process.loc[ds_to_process["fabric type"] == "P"]["online_moisture"], hist=False, color='pink', label = "P")
#    sns.distplot(ds_to_process.loc[ds_to_process["fabric type"] == "N"]["online_moisture"], hist=False, color='blue', label = "N")
#    sns.distplot(ds_to_process.loc[ds_to_process["fabric type"] == "D"]["online_moisture"], hist=False, color='g', label = "D")
#    sns.distplot(ds_to_process.loc[ds_to_process["fabric type"] == "K"]["online_moisture"], hist=False, color='k', label = "K")
#    ds_to_process[["No.1 CF temp", "No.2 CF temp", "No.3 CF temp", "No.4 CF temp", 
#                   "No.5 CF temp", "No.6 CF temp", "No.7 CF temp", "No.8 CF temp"]].plot(kind = "kde")

    
#    model_collection = {"Linear" : LinearRegression(),
#              "RF" : RandomForestRegressor(),
#              "GB" : GradientBoostingRegressor(),
#              "KNN" : KNeighborsRegressor(),
#              "DT" : DecisionTreeRegressor(),
#              "SVR" : SVR(),
#              "AdaBoost" : AdaBoostRegressor()}
#
#    model_collection_score (X_train, Y_train, model_collection, kfolds = 10, metric = "r2", type_name = "regression")
    
    # target model testing
    objmodel = RandomForestRegressor()
    metric_score, predictions =\
    regressor_model_build_results (objmodel, X_train, X_test, Y_train, Y_test, "MAPE")

    #plot
#    serial_true_pred_plot (Y_test, predictions.reshape(-1), title_name = 'Y_test vs prediction')
#    sns.residplot(Y_test, (predictions.reshape(-1) - Y_test), lowess=True, color="g")
#    plt.title("residual distribution")
#    sns.distplot((predictions.reshape(-1) - Y_test))    
    
    # model for electric power
    
#    dataset_power = pd.read_excel('OnlineMoisture_original.xlsx', sheet_name = '工作表4')

    # correlation check
#    cols_to_process = ["定型機2-主機速度", "定型機2-上漿機速度", "定型機2-入口上輪速度",
#                       "定型機2-入口餵布速度", "定型機2-入口下輪速度", "定型機2-入口架帶速度",
#                       "定型機2-出口上輪速度", "定型機2-出口下輪速度",
#                       "定型機2-01.風車速度", "定型機2-02.風車速度", "定型機2-03.風車速度", "定型機2-04.風車速度",
#                       "定型機2-05.風車速度", "定型機2-06.風車速度", "定型機2-07.風車速度", "定型機2-08.風車速度",
#                       "定型機2-.1定型機排風速度", "定型機2-.2定型機排風速度",
#                       "定型機E303_1 -KW", "定型機E303_2 -KW", "E303_3 -KW", "total_kw"] 
#    ds_to_process = dataset_power[cols_to_process]         
#    ds_to_process = ds_to_process.loc[(ds_to_process["total_kw"] > 40)]    
#    
#    C_mat = ds_to_process.corr()
#    fig = plt.figure(figsize = (15,15))
#    sns.heatmap(C_mat, square = True)
#    plt.show()        
    
#    cols_to_process = ["定型機2-主機速度", "定型機2-上漿機速度", "定型機2-入口上輪速度",
#                       "定型機2-入口餵布速度", "定型機2-入口下輪速度", "定型機2-入口架帶速度",
#                       "定型機2-出口上輪速度", "定型機2-出口下輪速度",
#                       "定型機2-01.風車速度", "定型機2-02.風車速度", "定型機2-03.風車速度", "定型機2-04.風車速度",
#                       "定型機2-05.風車速度", "定型機2-06.風車速度", "定型機2-07.風車速度", "定型機2-08.風車速度",
#                       "定型機2-.1定型機排風速度", "定型機2-.2定型機排風速度",
#                       "total_kw"] 
#    ds_to_process = dataset_power[cols_to_process]  
#    name_transfering_list = {"定型機2-主機速度" : "main speed", "定型機2-上漿機速度" : "M2",
#                             "定型機2-入口上輪速度" : "M3", "定型機2-入口餵布速度" : "M4",
#                             "定型機2-入口下輪速度" : "M5", "定型機2-入口架帶速度" : "M6",
#                             "定型機2-出口上輪速度" : "M7", "定型機2-出口下輪速度" : "M8",
#                             "定型機2-01.風車速度": "No.1 CF speed", "定型機2-02.風車速度": "No.2 CF speed",
#                             "定型機2-03.風車速度": "No.3 CF speed", "定型機2-04.風車速度": "No.4 CF speed",
#                             "定型機2-05.風車速度": "No.5 CF speed", "定型機2-06.風車速度": "No.6 CF speed",
#                             "定型機2-07.風車速度": "No.7 CF speed", "定型機2-08.風車速度": "No.8 CF speed",
#                             "定型機2-.1定型機排風速度": "No.1 EF speed", "定型機2-.2定型機排風速度": "No.2 EF speed",
#                             "total_kw": "motor_power"
#                             }    
#    
#    ds_to_process.columns = ds_to_process.columns.map(name_transfering_list)
#    ds_to_process.rename(columns = name_transfering_list, inplace=True)    
#    
#    ds_to_process = ds_to_process.loc[(ds_to_process["motor_power"] > 40)]
#    
#    scaler = MinMaxScaler()
#    X_train, X_test, Y_train, Y_test = split_with_scaling (ds_to_process, "motor_power", scaler)     
#    
#    
#    model_collection = {"Linear" : LinearRegression(),
#              "RF" : RandomForestRegressor(),
#              "GB" : GradientBoostingRegressor(),
#              "KNN" : KNeighborsRegressor(),
#              "DT" : DecisionTreeRegressor(),
#              "SVR" : SVR(),
#              "AdaBoost" : AdaBoostRegressor()}
#
#    model_collection_score (X_train, Y_train, model_collection, kfolds = 10, metric = "r2", type_name = "regression")
#
#    
#    # target model testing    
#    objmodel = GradientBoostingRegressor()
#    metric_score, predictions =\
#    regressor_model_build_results (objmodel, X_train, X_test, Y_train, Y_test, "RMSE")   
    
    # plot
#    serial_true_pred_plot (Y_test, predictions.reshape(-1), title_name = 'Y_test vs prediction')
#    sns.residplot(Y_test, (predictions.reshape(-1) - Y_test), lowess=True, color="g")
#    plt.title("residual distribution")
#    sns.distplot((predictions.reshape(-1) - Y_test))    
    
    # EXP simulation 
#    simulated_dataset = pd.read_excel('SimulatedEXP.xlsx', sheet_name = '工作表4')
#    cols_to_process = ["定型機2-主機速度",
#                       "定型機2-01.風車速度", "定型機2-02.風車速度", "定型機2-03.風車速度", "定型機2-04.風車速度",
#                       "定型機2-05.風車速度", "定型機2-06.風車速度", "定型機2-07.風車速度", "定型機2-08.風車速度",
#                       "定型機2-.1定型機排風速度", "定型機2-.2定型機排風速度",
#                       "total_kw"] 
#    simulated_dataset_to_process = simulated_dataset[cols_to_process]  
#    name_transfering_list = {"定型機2-主機速度": "main speed", 
#                             "定型機2-01.風車速度": "No.1 CF speed", "定型機2-02.風車速度": "No.2 CF speed",
#                             "定型機2-03.風車速度": "No.3 CF speed", "定型機2-04.風車速度": "No.4 CF speed",
#                             "定型機2-05.風車速度": "No.5 CF speed", "定型機2-06.風車速度": "No.6 CF speed",
#                             "定型機2-07.風車速度": "No.7 CF speed", "定型機2-08.風車速度": "No.8 CF speed",
#                             "定型機2-.1定型機排風速度": "No.1 EF speed", "定型機2-.2定型機排風速度": "No.2 EF speed",
#                             "total_kw": "motor_power"
#                             }  
#    simulated_dataset_to_process.columns = simulated_dataset_to_process.columns.map(name_transfering_list)
#    simulated_dataset_to_process.rename(columns = name_transfering_list, inplace=True)      
#    
#    scaler = MinMaxScaler()
#    
#    X_exp = simulated_dataset_to_process.drop(columns=["motor_power"])
#    Y_exp = simulated_dataset_to_process["motor_power"]
#    scaled_X_exp = scaler.fit_transform(X_exp)
#    predictions_exp = objmodel.predict(scaled_X_exp)
    