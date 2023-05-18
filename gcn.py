# -*- codeing = utf-8 -*-
# @Time :2023/4/25 8:39
# @Author :lqLi
# @Site :
# @File :gnn.py
# @Software :PyCharm
import os
import pandas as pd
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')
import numpy as np
import deepchem as dc
from sklearn.model_selection import train_test_split
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from metric_ import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

kfold = 10
EPOCH = 20
global random_seed_
random_seed_= 12546
np.random.seed(random_seed_)
tf.random.set_seed(random_seed_)
#param_file
hyperparams_result = []
ten_fold_result_all = []
result_test = []
#import data
df_file = pd.read_csv('Original.csv')
x = np.array(df_file.iloc[:, 0] ).reshape(-1,1)
y = np.array(df_file.iloc[:, 1]).reshape(-1,1)
# train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2, random_state=2)
#combined_train
train_combined = np.c_[xtrain,ytrain]
df_train = pd.DataFrame(train_combined,columns=['smiles','label'])
df_train.to_csv('./df_train.csv',index=False)
test_combined = np.c_[xtest,ytest]
df_test = pd.DataFrame(test_combined,columns=['smiles','label'])
df_test.to_csv('./df_test.csv', index=False)
#ROS_sampling
ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(xtrain, ytrain)
train_ros_combined = np.c_[x_ros, y_ros]
df_ros_train = pd.DataFrame(train_ros_combined,columns=['smiles','label'])
df_ros_train.to_csv('./df_ros_train.csv',index=False)
#RUS_sampling
rus = RandomUnderSampler(random_state=42)
x_rus, y_rus = rus.fit_resample(xtrain, ytrain)
train_rus_combined = np.c_[x_rus, y_rus]
df_rus_train = pd.DataFrame(train_rus_combined,columns=['smiles','label'])
df_rus_train.to_csv('./df_rus_train.csv',index=False)
#task
task = ['label']
FEATURIZERS = ['GraphConv']
df_test = 'df_test.csv'
train_files = ['df_train.csv','df_ros_train.csv','df_rus_train.csv']
for df_train in train_files:
    featurizer = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(tasks=task, feature_field='smiles', featurizer=featurizer)
    train_set = loader.create_dataset(df_train)
    test_set = loader.create_dataset(df_test)
    print("Number of compounds in train set: %d" % len(train_set))
    print("Number of compounds in test set: %d" % len(test_set))
   
    splitter = dc.splits.RandomStratifiedSplitter()
    train_dataset, valid_dataset = splitter.train_test_split(dataset=train_set,frac_train=0.9)
    # Metric
    metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    # Model
    def model_bulider(**model_params):
        batch_size = model_params['batch_size']
        learning_rate = model_params["learning_rate"]
        dropout = model_params["drop_out"]
        return dc.models.GraphConvModel(n_tasks=1, mode='classification', learning_rate=learning_rate,batch_size=batch_size,
                                        dropout =dropout)

     # param_
    params = {
          'learning_rate': [0.005,0.01,0.1],
          'batch_size': [16,32,64,128],
            'drop_out' : [0,0.1,0.25,0.5],
            'uncertainty':[True]}


    optimizer = dc.hyper.GridHyperparamOpt(model_bulider)
    best_model, best_hyperparams, all_results =optimizer.hyperparam_search(params, train_dataset, valid_dataset,
                                                                               metric,logdir=f'./models/{os.path.splitext(df_train)[0]}.m')

    columns_hyper = ['learning_rate', 'batch_size', 'drop_out', 'uncertainty']
    hyperparams_best = pd.DataFrame(best_hyperparams).T
    index_hyper = 'param_' + ''.join(os.path.splitext(df_train)[0])
    hyperparams_best.index=[index_hyper]
    hyperparams_best.columns = [columns_hyper]
    batch_size_best = np.array(hyperparams_best)[0][1]
    learning_rate_best = np.array(hyperparams_best)[0][0]
    drop_out_best = np.array(hyperparams_best)[0][2]
    
    hyperparams_result.append(hyperparams_best)
    #cv_split
    splitter = dc.splits.RandomStratifiedSplitter()
    cv_sets = splitter.k_fold_split(train_set, kfold)
    cv_result = pd.DataFrame()

    # 10-fold cv
    for i in range(kfold):
        cv_y_true = []
        cv_fold_train = cv_sets[i][0]  # 
        cv_fold_test = cv_sets[i][1]  # 
        fold_x_train = cv_fold_train.X
        fold_y_train = cv_fold_train.y
        fold_x_test = cv_fold_test.X
        fold_y_test = cv_fold_test.y
        print(f'**************************正在计算10_fold_交叉验证的第{i}折************************',
                  pd.DataFrame(cv_fold_test.y)[0].value_counts())
        model = dc.models.GraphConvModel(n_tasks=1, mode='classification',learning_rate=learning_rate_best,
                                             batch_size=batch_size_best,drop_out=drop_out_best,model_dir=f'./models/{os.path.splitext(df_train)[0]}.m')
        model.fit(cv_fold_train, nb_epoch=EPOCH)
        cv_y_true.extend(fold_y_test)
        cv_y_pred = model.predict(cv_fold_test)[:, :, 1].ravel()
        cv_y_prob = np.where(cv_y_pred >= 0.5, 1, 0)

        # statistic result
        columns = ['TN', 'FP', 'FN', 'TP', 'SE', 'SP', 'ACC', 'MCC', 'AUC_PRC', 'AUC_ROC']
        fold_result = pd.DataFrame(np.array(statistical(cv_y_true, cv_y_prob, cv_y_pred)),index=[columns],
                                       columns=[f'{os.path.splitext(df_train)[0]}_fold{i}']).T
        ten_fold_result_all.append(fold_result)

        #test set
    print(f'**************************开始预测{os.path.splitext(df_train)[0]}测试集中的结果************************')
    model = dc.models.GraphConvModel(n_tasks=1, mode='classification', learning_rate=learning_rate_best,
                                             batch_size=batch_size_best, drop_out=drop_out_best,
                                             model_dir=f'./models/{os.path.splitext(df_train)[0]}.m')
    model.fit(train_set, nb_epoch=EPOCH)
    test_y_true = test_set.y
    test_y_pred = model.predict(test_set)[:, :, 1].ravel()
    test_y_prob = np.where(test_y_pred >= 0.5, 1, 0)
    columns_test = ['TN', 'FP', 'FN', 'TP', 'SE', 'SP', 'ACC', 'MCC', 'AUC_PRC', 'AUC_ROC']
    test_result = pd.DataFrame(np.array(statistical(test_y_true, test_y_prob, test_y_pred)),index=[columns_test],
                                   columns = [f'{os.path.splitext(df_train)[0]}_GCN_test_result']).T
    result_test.append(test_result)
    print(f'**************************预测END***********************************************')

#out_result
test_result_all = pd.concat(result_test, axis=0)
columns2 = ['TN', 'FP', 'FN', 'TP', 'SE', 'SP', 'ACC', 'MCC', 'AUC_PRC', 'AUC_ROC']
#param_result
hyperparams_result_all = pd.concat(hyperparams_result,axis=0)
hyperparams_result_all.to_csv('./hyperparams/hyperparams_result.csv')
#cv_result
ten_fold_result_all = pd.concat(ten_fold_result_all, axis=0)
ten_fold_result_all.to_csv(f'./10_fold_cv/inh_10_fold_result.csv')
#test_result
test_result_all.to_csv(f'./test_result/GCN_test_result.csv')





