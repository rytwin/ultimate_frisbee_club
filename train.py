#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:39:46 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib
import pickle
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_recall_curve, \
    accuracy_score, precision_score, recall_score, f1_score, log_loss
from scipy.stats import binned_statistic
from scipy.special import expit


# change model_final to True when ready to save the model
# this will run metrics and make plots for the test set
# then run the script and the model details, predictions, and object itself will be saved
# BE CAREFUL ABOUT SAVING OVER PREVIOUS MODELS
# model_name is commented out to prevent accidental saving/overwriting
# uncomment model_name and add a unique name to append to the file name
# create folder: models/{model_name} to save files

model_final = False
#model_name = ''

# read in csv
df = pd.read_csv('data/model_data.csv')
label_df = df['action_id']

# update to choose regression type, target variable, hyperparameters
target_var = 'scored_on_poss'
regression_type = 'logistic' # update with 'linear' or 'logistic'
regularization_type = 'lasso' # update with 'lasso' or 'ridge'
lam = [1]

player_columns = ['dist_from_ez', 'wind_speed', 'upwind', 'downwind']
for c in df.columns:
    if c.startswith('player'):
        player_columns.append(c)

model_options = [['dist_from_ez', 'wind_speed', 'o_pt', 'decayed_rtg', 'num_passes', 'upwind_dist', 'dist_from_middle']]

# scale variables for logistic regression (for variables that aren't binary)
# if you add non-binary predictors to the model, add them to this list too
scale_cols = ['dist_from_ez', 'wind_speed', 'opp_rtg', 'decayed_rtg', 'num_passes',
              'dist_from_middle', 'opp_rank', 'upwind_dist', 'downwind_dist']
all_models = pd.DataFrame()

####################################################################################
################# You shouldn't need to edit anything below this! ##################
####################################################################################

def plot_and_df_with_preds(model, labels_df, target, pred_vars, X, y=np.nan, eval_type='', model_type='linear'):
    ''' model: model used for predictions
        labels_df: dataframe or series with labels for data points
        target: string with name of target variable
        pred_vars: list of strings with names of predictor variables
        X: dataframe with input variables
        y: series with corresponding response variables. if none given, it is assumed
            that this is a prediction on a set without a known response, so all
            references to y are skipped. This includes skipping the plot function
        eval_type: string, used for title of plot
        model_type: string, for type of regression ('linear' or 'logistic')
        
        returns dataframe with labels, input variables, response, and predicted response
    '''  
    num_predictors = len(pred_vars)
    if isinstance(y, pd.Series):
        df = pd.merge(X, y, left_index=True, right_index=True)
    else:
        df = X
    df = pd.merge(labels_df, df, left_index=True, right_index=True, how='right')
    y_pred = pd.DataFrame(model.predict(np.array(X).reshape(-1,num_predictors)))
    y_pred.rename(columns={0: f'Pred_{target}'}, inplace=True)
    y_pred.index = df.index
    df_pred = pd.merge(df, y_pred, left_index=True, right_index=True)
    if isinstance(y, pd.Series):
        df_pred['pred_error'] = df_pred[f'Pred_{target}'] - df_pred[target]
    
    if model_type == 'linear':
        plot_actual_vs_pred(y, y_pred, target, pred_vars, eval_type)
    elif model_type == 'logistic':
        y_pred_prob = pd.DataFrame(model.predict_proba(np.array(X).reshape(-1,num_predictors)))
        y_pred_prob.drop([0], axis=1, inplace=True)
        y_pred_prob.rename(columns={1: f'Pred_prob_{target}'}, inplace=True)
        y_pred_prob.index = df.index
        df_pred = pd.merge(df_pred, y_pred_prob, left_index=True, right_index=True)
        if isinstance(y, pd.Series):
            df_pred['pred_prob_error'] = df_pred[f'Pred_prob_{target}'] - df_pred[target]
            plot_logistic_preds(df_pred, target, pred_vars)
        
    return df_pred

def plot_actual_vs_pred(y, y_pred, target, pred_vars, eval_type):
    ''' y: dataframe or array with actual results
        y_pred: dataframe or array with predicted results
        target: string with name of target variable
        pred_vars: list of strings with names of predictor variables
        eval_type: string, used for title of plot
        
        returns plot of predicted vs. actual results
    '''
    model_vars = '-'.join(pred_vars)
    line = np.linspace(min(y), max(y), 100)
    plt.scatter(y, y_pred, color='black', alpha=0.5)
    plt.plot(line, line, color='blue')
    plt.title(f'{eval_type} predictions vs. actual\n{model_vars}')
    plt.grid(alpha=0.4, ls='--')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.xlim([min(y)-0.02*max(y), max(y) + 0.02*max(y)])
    plt.ylim([min(y)-0.02*max(y), max(y) + 0.02*max(y)])
    plt.show()

def plot_logistic_preds(df, target, pred_vars):
    ''' df_pred: dataframe with predictions, results, and predicted probs
        target: string with name of target variable
        pred_vars: list of strings with names of predictor variables
        
        makes plots for evalution of logistic regression model
    '''
    model_vars = '-'.join(pred_vars)
    
    ConfusionMatrixDisplay.from_predictions(df[target], df[f'Pred_{target}'])
    logit_roc_auc = roc_auc_score(df[target], df[f'Pred_{target}'])
    fpr, tpr, thresholds = roc_curve(df[target], df[f'Pred_prob_{target}'])
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(fpr, tpr, label='AUROC = %0.3f' % logit_roc_auc, color='blue')
    ax[0].plot([0, 1], [0, 1],'r--')
    ax[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'Receiver operating characteristic\n{model_vars}',
              xlim=[-0.02, 1.02], ylim=[0.0, 1.05])
    ax[0].grid(alpha=0.4, ls='--')
    ax[0].legend()
    
    precision, recall, t = precision_recall_curve(df[target], df[f'Pred_prob_{target}'])
    ax[1].plot(recall, precision, color='purple')
    ax[1].set(ylabel='Precision', xlabel='Recall', title=f'Precision-Recall Curve\n{model_vars}',
              xlim=[-0.02, 1.02], ylim=[0.0, 1.05])
    ax[1].grid(alpha=0.4, ls='--')
    fig.tight_layout()
    plt.show()
    
    df_cal = df.dropna(subset=[f'Pred_prob_{target}'])
    df_cal['prob_group'] = df_cal[f'Pred_prob_{target}'].apply(lambda x: math.floor(x * 10) / 10) + 0.05
    df_cal = df_cal.groupby('prob_group')[target].mean().reset_index(name='actual_prob')
    xline = np.linspace(0, 1, 100)
    yline = np.linspace(0, 1, 100)
    plt.plot(df_cal['prob_group'], df_cal['actual_prob'], color='blue', marker='o')
    plt.plot(xline, yline, ls = '--', color='black')
    plt.grid(alpha=0.4, ls='--')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual probability')
    plt.title(model_vars)
    plt.show()

# train the model and plot results
for list_of_vars in model_options:
    
    if (regularization_type != 'ridge') & (regularization_type != 'lasso'):
        print(f"ERROR: regularization_type must be 'lasso' or 'ridge', NOT '{regularization_type}'")
        break
          
    predictors = list_of_vars
    num_predictors = len(predictors)
    df_model = df.dropna(subset=predictors)

    # split into train/test and create linear regression model based on chosen predictors
    X = df_model[predictors]
    y = df_model[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=303)
    
    # scale variables for logistic regression (for variables that aren't binary)
    if regression_type == 'logistic':
        scaling_dict_mean = {}
        scaling_dict_std = {}
        for c in scale_cols:
            if c in X_train.columns.values:
                scaling_dict_mean[c] = np.mean(X_train[c])
                scaling_dict_std[c] = np.std(X_train[c])
                X_train[c] = (X_train[c] - np.mean(X_train[c])) / np.std(X_train[c])
    
    for a in lam:
        if regression_type == 'linear':
            if regularization_type == 'ridge':
                lm = Ridge(alpha=a)
            else:
                lm = Lasso(alpha=a)
            lm.fit(np.array(X_train).reshape(-1,num_predictors), y_train)
            model_params = pd.DataFrame({'Variable': predictors, 'Coefficient': lm.coef_})
            model_params = pd.concat([model_params, pd.DataFrame({'Variable': ['Intercept', 'alpha'],
                                                                  'Coefficient': [lm.intercept_, a]})],
                                     ignore_index=True)

            # evaluate model using MSE and R^2 from Kfold cross-validation
            results = cross_validate(lm, X_train, y_train,  scoring=['neg_mean_squared_error', 'r2'], cv=10)
            r2 = results['test_r2'].mean()
            mse = abs(results['test_neg_mean_squared_error'].mean())
            model_eval = pd.DataFrame({'Predictors': ['-'.join(predictors)], 'alpha': [a], 'R2': [r2], 'MSE': [mse]})

            df_train_pred = plot_and_df_with_preds(lm, label_df, target_var, predictors,
                                                   X_train, y_train, 'Training', regression_type)
            all_models = pd.concat([model_eval, all_models])
            
            if model_final:
                df_test_pred = plot_and_df_with_preds(lm, label_df, target_var, predictors,
                                                      X_test, y_test, 'Test', regression_type)
                test_acc = accuracy_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_prec = precision_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_recall = recall_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_f1 = f1_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_auc = roc_auc_score(df_test_pred[target_var], df_test_pred[f'Pred_prob_{target_var}'])
                test_ll = log_loss(df_test_pred[target_var], df_test_pred[f'Pred_prob_{target_var}'])
                test_metrics = pd.DataFrame({'Metric': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'logloss'],
                                             'Value': [test_acc, test_prec, test_recall, test_f1, test_auc, test_ll]})
                print(test_metrics)
            else:
                print('-'.join(predictors), '\n', model_params)
                print(f'r2: {round(r2, 3)}, mse: {round(mse, 3)}\n')
            
            #X_pred = df_pred[predictors]
            #df_pred = plot_and_df_with_preds(lm, tmyr_df, X_pred)
            
        elif regression_type == 'logistic':
            if regularization_type == 'ridge':
                lm = LogisticRegression(penalty='l2', C=1/a)
            elif regularization_type == 'lasso':
                lm = LogisticRegression(penalty='l1', solver='liblinear', C=1/a, random_state=100)
            lm.fit(np.array(X_train).reshape(-1,num_predictors), y_train)
            model_params = pd.DataFrame({'Variable': predictors, 'Coefficient': lm.coef_[0]})
            model_params = pd.concat([model_params, pd.DataFrame({'Variable': ['Intercept', 'alpha'],
                                                                  'Coefficient': [lm.intercept_[0], a]})],
                                     ignore_index=True)

            # evaluate model using MSE and R^2 from Kfold cross-validation
            results = cross_validate(lm, X_train, y_train,
                                     scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'neg_log_loss'],
                                     cv=10)
            accuracy = results['test_accuracy'].mean()
            precision = results['test_precision'].mean()
            recall = results['test_recall'].mean()
            roc_auc = results['test_roc_auc'].mean()
            f1 = results['test_f1'].mean()
            logloss = abs(results['test_neg_log_loss'].mean())
            model_eval = pd.DataFrame({'Predictors': ['-'.join(predictors)], 'alpha': [a],
                                       'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'roc_auc': [roc_auc],
                                       'f1_score': [f1], 'log_loss': [logloss]})
            
            df_train_pred = plot_and_df_with_preds(lm, label_df, target_var, predictors,
                                                   X_train, y_train, 'Training', regression_type)
            all_models = pd.concat([model_eval, all_models])
            
            if model_final:
                
                for c in scale_cols:
                    if c in X_test.columns.values:
                        X_test[c] = (X_test[c] - scaling_dict_mean[c]) / scaling_dict_std[c]
                
                df_test_pred = plot_and_df_with_preds(lm, label_df, target_var, predictors,
                                                      X_test, y_test, 'Test', regression_type)
                test_acc = accuracy_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_prec = precision_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_recall = recall_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_f1 = f1_score(df_test_pred[target_var], df_test_pred[f'Pred_{target_var}'])
                test_auc = roc_auc_score(df_test_pred[target_var], df_test_pred[f'Pred_prob_{target_var}'])
                test_ll = log_loss(df_test_pred[target_var], df_test_pred[f'Pred_prob_{target_var}'])
                test_metrics = pd.DataFrame({'Metric': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'logloss'],
                                             'Value': [test_acc, test_prec, test_recall, test_f1, test_auc, test_ll]})
                print(test_metrics)
            else:
                print('-'.join(predictors), '\n', model_params)
                print(f'accuracy: {round(accuracy, 3)}, precision: {round(precision, 3)}, recall: {round(recall, 3)}')
                print(f'roc_auc: {round(roc_auc, 3)}, f1: {round(f1, 3)}, log_loss: {round(logloss, 3)}\n')
            
            #X_pred = df_pred[predictors]
            #df_pred = plot_and_df_with_preds(lm, tmyr_df, X_pred)
            
        else:
            print(f"regression_type must be 'linear' or 'logistic', NOT '{regression_type}'")

# plot logistic regression with sigmoid curve
if (regression_type == 'logistic') & (model_final):        
    X_train_arr = np.array(X_train).reshape(-1,num_predictors)
    y_probs = lm.predict_proba(X_train_arr)[:, 1]
    y_preds = lm.predict(X_train_arr)

    lintransform = np.array(lm.coef_) @ X_train_arr.T + lm.intercept_
    lintransform = lintransform[0]

    bin_means, bin_edges, binnumber = binned_statistic(lintransform, y_train.values,
        statistic='mean', bins=10)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    plt.figure(1, figsize=(10, 5))
    plt.clf()

    plt.scatter(lintransform, y_train.values, label="train data", color="black", zorder=20, marker = "|", alpha=0.2)
    X_linspace = np.linspace(-5, 5, 300)

    plt.hist(lintransform[y_train.values == 1], density=False, bins=20, fc=(0, 1, 0, 0.4), \
              weights=0.03*np.ones_like(lintransform[y_train.values == 1]), label="histogram win")

    plt.hist(lintransform[y_train.values == 0], density=False, bins=20, fc=(1, 0, 0, 0.4), \
              weights=0.03*np.ones_like(lintransform[y_train.values == 0]), label='histogram loss')


    loss = expit(X_linspace).ravel()
    plt.plot(X_linspace, loss, label="Logistic Regression Model", color="blue", linewidth=1)

    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
                label='binned empirical score probability')

    plt.axvline(x = 0, color = "purple", label = 'decision boundary')


    plt.xlim(-5, 5)
    plt.xlabel("Transformed X")
    plt.ylabel("Scored on Possession")
    plt.legend(loc = 6)
    plt.show()

# save final model outputs
if model_final:
    print('Saving model details:')
    joblib.dump(lm, f'models/{model_name}/{regression_type}_model_{target_var}.pkl')
    model_params.to_csv(f'models/{model_name}/{regression_type}_model_{target_var}_details.csv', index=False)
    test_metrics.to_csv(f'models/{model_name}/{regression_type}_model_{target_var}_metrics.csv', index=False)
    
    if regression_type == 'logistic':
        with open(f'models/{model_name}/{regression_type}_model_{target_var}_scaling_mean_dict.pkl', 'wb') as f:
            pickle.dump(scaling_dict_mean, f)
        with open(f'models/{model_name}/{regression_type}_model_{target_var}_scaling_std_dict.pkl', 'wb') as f:
            pickle.dump(scaling_dict_std, f)
    
    df_train_pred['pred_set'] = 'train'
    df_test_pred['pred_set'] = 'test'
    all_preds = pd.concat([df_test_pred, df_train_pred], axis=0)#.sort_values(by=label_df)
    all_preds.to_csv(f'models/{model_name}/{regression_type}_model_{target_var}_predictions.csv', index=False)
    print('Complete!')

