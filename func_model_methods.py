import matplotlib.pyplot as plt
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,det_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from func_data import*
from func_sampling_methods import*
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearnex import patch_sklearn 

from sklearn.svm import SVC
from sys import displayhook

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def get_metrics_plot(Y_test, pos_prob, pred, sampling_method_name: str, model_name: str) -> list:
    roc_auc = roc_auc_score(Y_test, pos_prob)
    recall = recall_score(Y_test, pred)
    class_0_precision = precision_score(Y_test, pred, pos_label=0)
    class_1_precision = precision_score(Y_test, pred, pos_label=1)
    class_0_recall = recall_score(Y_test, pred, pos_label=0)
    class_1_recall = recall_score(Y_test, pred, pos_label=1)
    precision = precision_score(Y_test, pred)
    f1 = f1_score(Y_test, pred)
    accuracy = accuracy_score(Y_test, pred)
    ndf = [sampling_method_name, class_0_recall, class_0_precision, class_1_recall, class_1_precision, f1, accuracy,
           roc_auc]
    cm = confusion_matrix(Y_test, pred)

    precision, recall, thresholds = precision_recall_curve(Y_test, pred)

    # Calculate f-Score
    fscore = (2 * precision * recall) / (precision + recall)

    # locate the index of the largest g-mean
    ix = np.argmax(fscore)

    print('Best Threshold=%f, G-mean=%.3f' % (thresholds[ix], fscore[ix]))

    

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # CM plot
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='g', ax=axes[0])
    axes[0].set_title(f'Confusion matrix of {model_name}+ {sampling_method_name}')
    axes[0].xaxis.set_label_position("top")
    axes[0].set_xlabel('y prediction')
    axes[0].set_ylabel('y actual')

    # ROC curve plot
    fpr, tpr, _ = roc_curve(Y_test, pos_prob)
    axes[1].plot(fpr, tpr, marker='.', label=f'{model_name}+ {sampling_method_name}')
    axes[1].scatter(fpr[ix], tpr[ix], marker='o', color='black')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return ndf


def random_forest(func_list: list,params: dict,data_split:list) -> None: 
    
    X_train,X_test,Y_train,Y_test=data_split

    X_train,X_test=one_hot_encode(X_train,X_test)
    X_train,X_test=standard_scaling(X_train,X_test)

    #ramka danych do przechowywania wyników
    df_score = pd.DataFrame(columns=['Random Forest','Recall 0','Precision 0','Recall 1','Precision 1','F1 Score', 'Accuracy','Roc Auc'])
    
    # iterujemy po wektorze z funkcjami
    for func in func_list:
        #tworzymy pipeline
         
        pipeline = Pipeline(steps=[
            ('preprocess', func()[0]),
            ('randomforestclassifier', RandomForestClassifier(criterion='gini',n_estimators=100, random_state=42))
        ])
        

        #scoring={"AUC":'roc_auc',"Precision":'precision'}
        #,refit="Precision"
    
        #werfyikacja kryżowa
        new_params = {'randomforestclassifier__' + key: params[key] for key in params}
        grid_search = GridSearchCV(pipeline, param_grid=new_params, cv=3, scoring='precision',
                                return_train_score=True,n_jobs=-1,verbose=3)
        grid_search.fit(X_train,Y_train)
        print(grid_search.best_params_)
        random_forest_best=grid_search.best_estimator_
        pos_prob=random_forest_best.predict_proba(X_test)[:,1]
        pred=random_forest_best.predict(X_test)
        sampling_method_name=func()[1]
        
        ndf=get_metrics_plot(Y_test,pos_prob,pred,sampling_method_name,model_name="Random Forest Clasifier")
        df_score=df_score.append(pd.Series(ndf, index=df_score.columns[:len(ndf)]), ignore_index=True)
        
    displayhook(df_score)

def svc(func_list: list, params: dict,data_split) -> None:

    X_train,X_test,Y_train,Y_test=data_split

    X_train,X_test=one_hot_encode(X_train,X_test)
    X_train,X_test=standard_scaling(X_train,X_test)
    
    df_score = pd.DataFrame(columns=['SVC with','Recall 0','Precision 0','Recall 1','Precision 1','F1 Score', 'Accuracy','Roc Auc'])

    for func in func_list:
        
        pipeline = Pipeline(steps=[
            ('preprocess', func()[0]),
            ('svcmodel', SVC(random_state=42, probability=True))
        ])

        new_params = {'svcmodel__' + key: params[key] for key in params}
        grid_search = GridSearchCV(pipeline, param_grid=new_params, cv=2, scoring='f1', return_train_score=True, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        svc_best = grid_search.best_estimator_
        pos_prob=svc_best.predict_proba(X_test)[:,1]
        pred=svc_best.predict(X_test)

        sampling_method_name=func()[1]
        
        #zapisujemy dane do df
        ndf=get_metrics_plot(Y_test,pos_prob,pred,sampling_method_name,model_name="SVC")
        

        df_score=df_score.append(pd.Series(ndf, index=df_score.columns[:len(ndf)]), ignore_index=True)


    displayhook(df_score)

def xgb_boost_classifier(func_list: list,params: dict,data_split) -> None: 
    
    
    X_train,X_test,Y_train,Y_test=data_split
    X_train,X_test=one_hot_encode(X_train,X_test)
    X_train,X_test=standard_scaling(X_train,X_test)

    

  
    #df do przechowywania wyników
    df_score = pd.DataFrame(columns=['XGB Boost','Recall 0','Precision 0','Recall 1','Precision 1','F1 Score', 'Accuracy','Roc Auc'])
    
    #konieczny encoding w xgb_boost
    le=LabelEncoder()
    Y_train_enc=le.fit_transform(Y_train)
    xgb_model=xgb.XGBClassifier(learning_rate=0.1,n_estimators=100,objective='binary:logistic')

    # iterujemy po wektorze z funkcjami
    for func in func_list:
        #tworzymy pipeline
        
        
        pipeline = Pipeline(steps=[
            ('preprocess', func()[0]),
            ('xgbmodel', xgb_model)
        ])

        scoring={"F1":'f1',"AUC":'roc_auc',"Precision":'precision'}
        

        #werfyikacja kryżowa
        new_params = {'xgbmodel__' + key: params[key] for key in params}
        grid_search = GridSearchCV(pipeline, param_grid=new_params, cv=3, scoring=scoring, 
                                return_train_score=True,refit="Precision",n_jobs=-1)
        grid_search.fit(X_train,Y_train_enc)
        print(grid_search.best_params_)
        xgb_boost_best=grid_search.best_estimator_
        pos_prob=xgb_boost_best.predict_proba(X_test)[:,1]
        pred=xgb_boost_best.predict(X_test)

        sampling_method_name=func()[1]
        
        #zapisujemy dane do df
        ndf=get_metrics_plot(Y_test,pos_prob,pred,sampling_method_name,model_name="XGB Boost")
        
        df_score=df_score.append(pd.Series(ndf, index=df_score.columns[:len(ndf)]), ignore_index=True)
       
        
        
    displayhook(df_score)
 
def balanced_random_forest_classifier(params: dict,data_split): 
    
    X_train,X_test,Y_train,Y_test=data_split

    X_train,X_test=one_hot_encode(X_train,X_test)
    X_train,X_test=standard_scaling(X_train,X_test)

    df_score = pd.DataFrame(columns=['Balanced Random Forest','Recall 0','Precision 0','Recall 1','Precision 1','F1 Score', 'Accuracy','Roc Auc'])
    
    balanced_bagging_model=BalancedRandomForestClassifier(class_weight='balanced_subsample',replacement=True,random_state=42,n_jobs=-1)

    
    #tworzymy pipeline
    pipeline = Pipeline(steps=[
        ('balanced_bagging_model', balanced_bagging_model)
    ])

    
    #werfyikacja kryżowa
    new_params = {'balanced_bagging_model__' + key: params[key] for key in params}
    grid_search = GridSearchCV(pipeline, param_grid=new_params, cv=3, scoring='roc_auc',
                            return_train_score=True,n_jobs=-1)
    grid_search.fit(X_train,Y_train)
    print(grid_search.best_params_)
    xgb_boost_best=grid_search.best_estimator_
    pos_prob=xgb_boost_best.predict_proba(X_test)[:,1]
    pred=xgb_boost_best.predict(X_test)
    
    sampling_method_name=""
        
    #zapisujemy dane do df
    ndf=get_metrics_plot(Y_test,pos_prob,pred,sampling_method_name,model_name="Balanced Random Forest")
        
    df_score=df_score.append(pd.Series(ndf, index=df_score.columns[:len(ndf)]), ignore_index=True)
  
    
    displayhook(df_score)

def rus_boost_classifier(params: dict,data_split): 
    
    X_train,X_test,Y_train,Y_test=data_split

    X_train,X_test=one_hot_encode(X_train,X_test)
    X_train,X_test=standard_scaling(X_train,X_test)

    df_score = pd.DataFrame(columns=['RusBoost','Recall 0','Precision 0','Recall 1','Precision 1','F1 Score', 'Accuracy','Roc Auc'])
    
    estimator=RandomForestClassifier(n_estimators=500, random_state=42)
    rus_model=RUSBoostClassifier(estimator=estimator,replacement=True,random_state=42)
    
    #tworzymy pipeline
    pipeline = Pipeline(steps=[
        ('rus_model', rus_model)
    ])

    
    #werfyikacja kryżowa
    new_params = {'rus_model__' + key: params[key] for key in params}
    grid_search = GridSearchCV(pipeline, param_grid=new_params, cv=3, scoring='roc_auc',
                            return_train_score=True,n_jobs=-1)
    grid_search.fit(X_train,Y_train)
    print(grid_search.best_params_)
    xgb_boost_best=grid_search.best_estimator_
    pos_prob=xgb_boost_best.predict_proba(X_test)[:,1]
    pred=xgb_boost_best.predict(X_test)
    
    sampling_method_name=""
        
    #zapisujemy dane do df
    ndf=get_metrics_plot(Y_test,pos_prob,pred,sampling_method_name,model_name="Rus Boost")
        
    df_score=df_score.append(pd.Series(ndf, index=df_score.columns[:len(ndf)]), ignore_index=True)
  
    
    displayhook(df_score)