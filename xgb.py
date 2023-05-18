# -*- codeing = utf-8 -*-
# @Time : 2022/6/210 14:29
# @Author :LqLi
# @File :xgb_learn.py
# @software: PyCharm
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,auc, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score, \
    matthews_corrcoef, cohen_kappa_score
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score


result_cv = []
result_test = []
param_result = []
xgb_cv_10_fold_result_all = []
test_prob_all =[]
test_pred_all = []
#load data
file_names = ['RDKit_2D_filter', 'MACCS',  'ECFP4',  'ExtFP', 'KRFP']
for file in file_names:
    df = pd.read_csv(file+'.csv',encoding='gbk')
    # data
    x=df.iloc[:,1:]
    y=np.array(df.iloc[:,0]) 
  
    # train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(x, y, test_size=0.2,random_state=2)


# param_
    parameters = {
        'learning_rate': [0.005,0.01,0.1],
        'n_estimators': range(50,1000,50),
        'max_depth': range(3,10,1),
        'seed': [0],
        'objective': ["binary:logistic"],
         'eval_metric':['logloss']}
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    clf = XGBClassifier()

    #grid search
    score = []
    model = []
    param = []
    i = 1
    for i in range(1, 11):
       i = i + 1
       grid = GridSearchCV(clf, parameters, scoring="balanced_accuracy", cv=cv,n_jobs=-1).fit(xtrain, ytrain)
       best_param = grid.best_params_
       best_model = grid.best_estimator_
       best_score = grid.best_score_
       score.append(best_score)
       param.append(best_param)
       model.append(best_model)

    score_ = pd.DataFrame(score, columns=['score'])
    n = np.array(model)
    model_ = pd.DataFrame(n, columns=['model'])
    param_ = pd.DataFrame(param)
    param_score_model = pd.concat([score_, model_, param_], axis=1)
    score_sort = param_score_model.sort_values(by="score", ascending=False)
    bestmodel = score_sort.iloc[0, 1]
    param_best = score_sort.iloc[0, 2:]
    bestparam = pd.DataFrame(param_best).T

    #cv_validation
    cv_10_fold_auc = []
    cv_10_fold_acc = []
    cv_10_fold_bacc = []
    cv_10_fold_recall = []
    cv_10_fold_sp = []
    cv_10_fold_precision = []
    cv_10_fold_f1 = []
    cv_10_fold_mcc = []
    for fold_train_id, fold_test_id in cv.split(xtrain, ytrain):
        cv_y_true = []
        cv_y_pred = []
        cv_y_prob = [] 
        fold_x_train, fold_x_test = np.array(xtrain)[fold_train_id], np.array(xtrain)[fold_test_id]
        fold_y_train, fold_y_test = ytrain[fold_train_id], ytrain[fold_test_id]
        model = bestmodel.fit(fold_x_train, fold_y_train)
        cv_y_true.extend(fold_y_test)
        cv_y_pred.extend(model.predict(fold_x_test))
        cv_y_prob.extend(model.predict_proba(fold_x_test)[:, 1]) 
        #metric
        cv_cm = confusion_matrix(cv_y_true, cv_y_pred, labels=[0, 1])
        cv_auc = roc_auc_score(cv_y_true, cv_y_prob)
        cv_acc = accuracy_score(cv_y_true, cv_y_pred)
        cv_bacc = balanced_accuracy_score(cv_y_true, cv_y_pred)
        cv_recall = recall_score(cv_y_true, cv_y_pred, pos_label=1, average="binary")
        cv_specifity = cv_cm[0, 0] / (cv_cm[0, 0] + cv_cm[0, 1])
        cv_precision = precision_score(cv_y_true, cv_y_pred, pos_label=1, average="binary")
        cv_f1 = f1_score(cv_y_true, cv_y_pred)
        cv_matthews_corrcoef = matthews_corrcoef(cv_y_true, cv_y_pred)
       #
        cv_10_fold_auc.append(cv_auc)
        cv_10_fold_acc.append(cv_acc)
        cv_10_fold_bacc.append(cv_bacc)
        cv_10_fold_recall.append(cv_recall)
        cv_10_fold_sp.append(cv_specifity)
        cv_10_fold_precision.append(cv_precision)
        cv_10_fold_f1.append(cv_f1)
        cv_10_fold_mcc.append(cv_matthews_corrcoef)
    #
    cv_result = [np.mean(cv_10_fold_auc), np.mean(cv_10_fold_acc), np.mean(cv_10_fold_bacc),
                 np.mean(cv_10_fold_recall),np.mean(cv_10_fold_sp), np.mean(cv_10_fold_precision),
                 np.mean(cv_10_fold_f1),np.mean(cv_10_fold_mcc)]

    # test_validation
    model2 = bestmodel.fit(xtrain, ytrain)
    test_y_pred = model2.predict(xtest)
    test_y_true = ytest
    test_y_prob = model2.predict_proba(xtest)[:, 1]

    test_cm = confusion_matrix(test_y_true, test_y_pred, labels=[0, 1])
    test_auc = roc_auc_score(test_y_true, test_y_prob)
    test_acc = accuracy_score(test_y_true, test_y_pred)
    test_bacc = balanced_accuracy_score(test_y_true, test_y_pred)
    test_recall = recall_score(test_y_true, test_y_pred, pos_label=1, average="binary")
    test_specifity = test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1])
    test_precision = precision_score(test_y_true, test_y_pred, pos_label=1, average="binary")
    test_f1 = f1_score(test_y_true, test_y_pred)
    test_matthews_corrcoef = matthews_corrcoef(test_y_true, test_y_pred)

    test_result = [test_auc, test_acc, test_bacc, test_recall, test_specifity, test_precision, test_f1,
                   test_matthews_corrcoef]

    # result_output
    columns = ["AUC", "ACC", "B_ACC", "SE", "SP", "Precision", "f1", "MCC"]
    index1 = 'cv_' + ''.join(file)
    index2 = 'test_' + ''.join(file)
    xgb_cv_10_fold_result =pd.DataFrame( [cv_10_fold_auc, cv_10_fold_acc, cv_10_fold_bacc, cv_10_fold_recall, cv_10_fold_sp,
                        cv_10_fold_precision,cv_10_fold_f1, cv_10_fold_mcc],index=[columns],
                    columns=[file+'_fold0',file+'_fold1',file+'_fold2',file+'_fold3',file+'_fold4',file+'_fold5',
                             file+'_fold6',file+'_fold7',file+'_fold8',file+'_fold9']).T

    xgb_cv_10_fold_result_all.append(xgb_cv_10_fold_result)
    xgb_cv = pd.DataFrame([cv_result], index=[index1], columns=columns)
    xgb_test = pd.DataFrame([test_result], index=[index2], columns=columns)
    result_cv.append(xgb_cv)
    result_test.append(xgb_test)
    index3 = 'param_' + ''.join(file)
    bestparam.index = [index3]
    param_result.append(bestparam)
    test_pred=pd.DataFrame(test_y_pred,columns=[file+'_test_pred'],index=None)
    test_pred_all.append(test_pred)
    test_prob = pd.DataFrame(test_y_prob,columns=[file+'_test_prob'])
    test_prob_all.append(test_prob)

# merge data
result_all_cv = pd.concat(result_cv, axis=0)
result_all_test = pd.concat(result_test, axis=0)
result_all_param = pd.concat(param_result, axis=0)
result_all_10_fold_result = pd.concat(xgb_cv_10_fold_result_all,axis=0)
result_all_test_pred = pd.concat(test_pred_all,axis=1)
result_all_test_prob  =pd.concat(test_prob_all,axis=1)
result_all_cv.to_csv("xgb_result/" + 'xgb_result_cv.csv')
result_all_test.to_csv("xgb_result/" + 'xgb_result_test.csv')
result_all_param.to_csv("xgb_result/" + 'xgb_result_param.csv')
result_all_10_fold_result.to_csv("xgb_result/"+'xgb_cv_10_fold_result.csv')
result_all_test_pred.to_csv("xgb_result/" + 'xgb_result_test_pred.csv',index=None)
result_all_test_prob.to_csv("xgb_result/" + 'xgb_result_test_prob.csv',index=None)

