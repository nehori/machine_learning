import os
import gc
import time
import pickle
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from contextlib import contextmanager
from nehori import tilib

# [How to install]
# /c/Python38/Scripts/pip install lightgbm
# /c/Python38/Scripts/pip install optuna
# /c/Python38/Scripts/pip install seaborn
# /c/Python38/Scripts/pip install TA_Lib-0.4.19-cp38-cp38-win_amd64.whl
# /c/Python38/Scripts/pip install pandas-profiling
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# [Run]
# /c/Python38/python new2021.py

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# CSVの読み込み
def read_csv(stock_id, skiprows, skipfooter):
    file = "tosho/" + str(stock_id) + ".csv"
    if not os.path.exists(file):
        print("[Error] " + file + "　does not exist.")
        return None, False
    else:
        return pd.read_csv(file, skiprows=skiprows,
                           skipfooter=skipfooter, engine="python",
                           names=("date", "open", "high", "low", "close", "volume"),
                           # For "ValueError: DataFrame.dtypes for data must be int, float or bool."
                           dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
                           ), True

# 概要出力
def display_overview(df):
    # それぞれのデータのサイズを確認
    print("The size of df is : "+str(df.shape))
    # 列名を表示
    print(df.columns)
    # 表の一部分表示
    print(df.head().append(df.tail()))

# 予測値（*日後の始値の上昇値）
def get_target_value(df):
    df['target'] = (df['open'].shift(-3) - df['open'].shift(-1)) / df['open'].shift(-1)
    df.loc[(df['target'] > 0.03), 'target'] = 1
    df.loc[(-0.03 > df['target']), 'target'] = 0
    return df

# データ前処理
def pre_processing(df):
    # 目的変数（*日後の始値の上昇値）
    df = get_target_value(df)
    # 曜日追加
    df['day'] = pd.to_datetime(df['date']).dt.dayofweek
    # 新特徴データ
    df = tilib.add_new_features(df)
    # 欠損値を列の1つ手前の値で埋める
    df = df.fillna(method='ffill')
    return df

# feature importanceをプロット
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize = (8, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

# ROC曲線をプロット
def display_roc(list_label, list_score):
    fpr, tpr, thresholds = roc_curve(list_label, list_score)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

# Optuna(ハイパーパラメータ自動最適化ツール)
class Objective:
    def __init__(self, x, y, excluded_feats, num_folds = 4, stratified = False):
        self.x = x
        self.y = y
        self.excluded_feats = excluded_feats
        self.stratified = stratified
        self.num_folds = num_folds

    def __call__(self, trial):
        df_train = self.x
        y = self.y
        excluded_feats = self.excluded_feats
        stratified = self.stratified
        num_folds = self.num_folds
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
        else:
            folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)
        oof_preds = np.zeros(df_train.shape[0])
        feats = [f for f in df_train.columns if f not in excluded_feats] 
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], y)):
            X_train, y_train = df_train[feats].iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = df_train[feats].iloc[valid_idx], y.iloc[valid_idx]
            clf = LGBMClassifier(objective = 'binary',
                                    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 100.0),
                                    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-4, 100.0),
                                    num_leaves = trial.suggest_int('num_leaves', 10, 40),
                                    silent = True)
            # trainとvalidを指定し学習
            clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], 
                    eval_metric = 'auc', verbose = 0, early_stopping_rounds = 200)
            oof_preds[valid_idx] = clf.predict_proba(X_valid, num_iteration = clf.best_iteration_)[:, 1]
        accuracy = roc_auc_score(y, oof_preds)
        return 1.0 - accuracy

import lightgbm as lgb

# 決定木を可視化
def display_tree(clf):
    print('Plotting tree with graphviz...')
    graph = lgb.create_tree_digraph(clf, tree_index=1, format='png', name='Tree',
                                    show_info=['split_gain','internal_weight','leaf_weight','internal_value','leaf_count'])
    graph.render(view=True)
    
def load_model(num):
    clf = None
    file = "model" + str(num) + ".pickle"
    if os.path.exists(file):
       with open(file, mode='rb') as fp:
           clf = pickle.load(fp)
    return clf

def save_model(num, clf):
    with open("model" + str(num) + ".pickle", mode='wb') as fp:
          pickle.dump(clf, fp, protocol=2)

# Cross validation with KFold
def cross_validation(df_train, y, df_test, excluded_feats, num_folds = 4, stratified = False, debug = False):
    print("Starting cross_validation. Train shape: {}, test shape: {}".format(df_train.shape, df_test.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    df_feature_importance = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in excluded_feats] 
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], y)):
        X_train, y_train = df_train[feats].iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = df_train[feats].iloc[valid_idx], y.iloc[valid_idx]
        # LightGBM
        clf = LGBMClassifier(max_depth=8,
                             num_leaves = 29)
        
        # trainとvalidを指定し学習
        clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], 
                eval_metric = "auc", verbose = 0, early_stopping_rounds = 200)

        oof_preds[valid_idx] = clf.predict_proba(X_valid, num_iteration = clf.best_iteration_)[:, 1]
        sub_preds = clf.predict_proba(df_test[feats], num_iteration = clf.best_iteration_)[:, 1]
        df_fold_importance = pd.DataFrame()
        df_fold_importance["feature"] = feats
        df_fold_importance["importance"] = clf.feature_importances_
        df_fold_importance["fold"] = n_fold + 1
        df_feature_importance = pd.concat([df_feature_importance, df_fold_importance], axis=0)
        save_model(n_fold, clf)
#        display_tree(clf)
        del clf, X_train, y_train, X_valid, y_valid
        gc.collect()
    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
    #display_roc(y, oof_preds)
    display_importances(df_feature_importance)
    return sub_preds

def pred_load_model(clfs, df, stock_id, excluded_feats):
    n_splits = len(clfs)
    sub_preds = np.zeros(df.shape[0])
    feats = [f for f in df.columns if f not in excluded_feats]
    for clf in clfs:
        sub_preds += clf.predict_proba(df[feats], num_iteration = clf.best_iteration_)[:, 1] / n_splits
    s = tilib.create_protra_dataset(stock_id, df["date"], sub_preds, 0.6)
    return s

# 時価総額ランキングTop20
stock_names = [
    "7203", "6861", "6758", "9432", "9984", "6098", "8306", "9983", "9433", "6367",
    "6594", "4063", "8035", "9434", "7974", "4519", "7267", "7741", "6902", "6501"
             ]

def main(df_train, df_test, stock_id):
    # 概要出力
    #display_overview(df_train)
    # 学習モデル構築
    df_test = df_test.drop("target", axis=1)
    df_train = df_train.dropna(subset=["target"])
    # 正解データ・失敗データだけ利用する
    df_train = df_train[(df_train['target'] == 1) | (df_train['target'] == 0)]
    excluded_feats = ['target', 'date']
    s = ""
    # 学習データが存在する場合
    if (len(df_train)):
       if True:
          # 交差検証
          y_pred = cross_validation(df_train, df_train['target'], df_test, excluded_feats, 2, True, True)
          print(y_pred)
          s = tilib.create_protra_dataset(stock_id, df_test["date"], y_pred, 0.8)
       else:
          # ハイパーパラメータ探索
          objective = Objective(x=df_train, y=df_train['target'],
                                excluded_feats=excluded_feats, num_folds = 5, stratified = True)
          study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))
          study.optimize(objective, n_trials = 50)
    return s

DIR = ".\\"

# 結合版
if __name__ == '__main__':
    with timer("Cross validation"):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()        
        for stock_id in stock_names:
            print(str(stock_id))
            # CVを使っているのでTest用に一定数を未知のデータとする
            df, val = read_csv(stock_id, 500, 500)
            #display_overview(df)
            if not val:
                continue
            df = pre_processing(df)
            df_train = pd.concat([df_train, df])
            # CVを使っているのでTest用に一定数を未知のデータとする
            df_test, val = read_csv(stock_id, 0, 500)
            #df_test = pd.concat([df_test, df])
            # データ前処理
        df_test = pre_processing(df_test)
        #display_overview(df_train)
        # closeの欠損値が含まれている行を削除
        df_train = df_train.dropna(subset=["close"])
        main(df_train, df_test, stock_id)
    s = ""
    with timer("start back test"):
        clf = []
        for i in range(2):
            clf.append(load_model(i))
        excluded_feats = ['target', 'date']
        for stock_id in stock_names:
            df_test, val = read_csv(stock_id, 0, 0)
            df_test = pre_processing(df_test)
            s += pred_load_model(clf, df_test, stock_id, excluded_feats)
    with open(DIR + "LightGBM.pt", mode='w') as f:
        f.write(tilib.merge_protra_dataset(s))

