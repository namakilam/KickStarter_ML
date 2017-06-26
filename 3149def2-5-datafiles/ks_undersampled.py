from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
def undersampled(data, ratio=1):
    conv = data[data.final_status == 1]
    conv = conv.sample(n=len(conv))
    othr = data[data.final_status == 0].sample(n=ratio*len(conv))
    return pd.concat([conv, othr]).sample(frac=1)

train = pd.read_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/train_clean2.csv')
test = pd.read_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/test_clean2.csv')

undersampled_train = undersampled(train)

y_train = undersampled_train['final_status']
test_id = test['project_id']
undersampled_train.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline','final_status','backers_count'], inplace=True, axis=1)
test.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline'], inplace=True, axis=1)
#min_max_scaler = preprocessing.MinMaxScaler()
#train_scaled = min_max_scaler.fit_transform(train)
#train = pd.DataFrame(train_scaled)

#test_scaled = min_max_scaler.fit_transform(test)
#test = pd.DataFrame(test_scaled)

dtrain = xgb.DMatrix(data=undersampled_train, label = y_train)
dtest = xgb.DMatrix(data=test)

params = {
    'objective':'binary:logistic',
    'eval_metric':'error',
    'eta':0.025,
    'max_depth':6,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':5}

bst = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=40,nfold=5L,verbose_eval=10)

bst_train = xgb.train(params, dtrain, num_boost_round=1000)

p_test = bst_train.predict(dtest)

sub = pd.DataFrame()
sub['project_id'] = test_id
sub['final_status'] = p_test

sub['final_status'] = [1 if x > 0.5 else 0 for x in sub['final_status']]

sub.to_csv("/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/xgb_with_python_feats.csv",index=False)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import auc,roc_curve

undersampled_train.shape
test.head()
#clf = DecisionTreeClassifier(max_features=None, max_depth=11)
#clf = clf = RandomForestClassifier(n_estimators=500, max_features=26,oob_score=True, random_state=0)
#clf = SVC()


from sklearn.linear_model import LogisticRegression
C_s = np.logspace(-10, 1, 11)
scores = list()
scores_std = list()
lr = LogisticRegression(penalty = 'l1')

for C in C_s:
    lr.C = C
    this_scores = cross_val_score(lr, undersampled_train, y_train, cv=4,scoring='roc_auc')
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

lr_results = pd.DataFrame({'score':scores, 'C':C_s})
lr_results



from sklearn.ensemble import RandomForestClassifier
msl_s = [6, 7, 8, 9, 10]
scores = list()
scores_std = list()
rf = GradientBoostingClassifier(n_estimators=100, max_features=31, max_depth=3)

for msl in msl_s:
    rf.min_samples_split = msl
    this_scores = cross_val_score(rf, undersampled_train, y_train, cv=4,scoring='roc_auc')
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
rf_results = pd.DataFrame({'score':scores, 'Max Depth': msl_s})
rf_results

msl_s = [500, 750, 1000]
scores = list()
scores_std = list()
rf = RandomForestClassifier()

for msl in msl_s:
    rf.n_estimators = msl
    this_scores = cross_val_score(rf, undersampled_train, y_train, cv=4,scoring='roc_auc')
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
rf_results = pd.DataFrame({'score':scores, 'Max Depth': msl_s})
rf_results

#clf = GradientBoostingClassifier(learning_rate=0.04, n_estimators=500, subsample=0.93, max_depth=10, max_features=None)
#clf = AdaBoostClassifier(learning_rate=0.5, n_estimators=500)
#clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, max_iter=500,hidden_layer_sizes=(44, 2), learning_rate='invscaling')
#clf = KNeighborsClassifier(weights='distance',algorithm='kd_tree', leaf_size=1000, n_neighbors=2)
clf = GradientBoostingClassifier(n_estimators=100,max_features=31, max_depth=3, min_samples_split=6)
clf.fit(undersampled_train, y_train)
pred = clf.predict(test)

out_column = ['final_status']
sub = pd.DataFrame(data=pred, columns=out_column)
sub['project_id'] = test_id
sub = sub[['project_id', 'final_status']]
sub.head()
sub.to_csv("/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/gb14Starter.csv",index = False)
