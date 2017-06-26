import pandas as pd

def split_count(row, col):
    cnt = 0
    try:
        cnt = len(row[col].split())
    except AttributeError as e:
        cnt = 0
    return cnt

def encode_country(row, country):
    if row['country'] == country:
        return 1
    else:
        return 0

def encode_currency(row, currency):
    if row['currency'] == currency:
        return 1
    else:
        return 0

def currency_convert(row, data):
    if row['currency'] == 'USD':
        return row['goal']
    else:
        return row['goal'] * data['rates'][row['currency']]

train = pd.read_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/train.csv')
test = pd.read_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/test.csv')

train.head()
train['country'].unique()
train['currency'].unique()
test['country'].unique()
test['currency'].unique()
train['disable_communication'].unique()
train['goal'].describe()
train_success = train.ix[(train['final_status'] == 1)]
train_failure = train.ix[(train['final_status'] == 0)]
train_success['goal'].describe()
train_failure['goal'].describe()

train['disable_communication'] = train['disable_communication'].astype(int)
test['disable_communication'] = test['disable_communication'].astype(int)

import requests

url = 'https://v3.exchangerate-api.com/bulk/2c4f52b308a96a049fa5a275/USD'

response = requests.get(url)
data = response.json()

train['goal'] = train.apply(lambda row: currency_convert(row, data), axis=1)
test['goal'] = test.apply(lambda row: currency_convert(row, data), axis=1)
c1 = set(train['country'].values)
c2 = set(test['country'].values)
c1 = c1.union(c2)

for country in list(c1):
    train[country] = train.apply(lambda row: encode_country(row, country), axis=1)

for country in list(c1):
    test[country] = test.apply(lambda row: encode_country(row, country), axis=1)

train.head()
test.head()
train['d_c'] = train['deadline'] - train['created_at']
train['d_l'] = train['deadline'] - train['launched_at']
train['s_c'] = train['state_changed_at'] - train['created_at']
train['s_l'] = train['state_changed_at'] - train['launched_at']
train['l_c'] = train['launched_at'] - train['created_at']

test['d_c'] = test['deadline'] - test['created_at']
test['d_l'] = test['deadline'] - test['launched_at']
test['s_c'] = test['state_changed_at'] - test['created_at']
test['s_l'] = test['state_changed_at'] - test['launched_at']
test['l_c'] = test['launched_at'] - test['created_at']



train['keyword_len'] = train.apply(lambda row: len(row['keywords'].split('-')), axis=1)
train['name_len'] = train.apply(lambda row: split_count(row, 'name'), axis=1)
train['desc_len'] = train.apply(lambda row: split_count(row, 'desc'), axis=1)

test['keyword_len'] = test.apply(lambda row: len(row['keywords'].split('-')), axis=1)
test['name_len'] = test.apply(lambda row: split_count(row, 'name'), axis=1)
test['desc_len'] = test.apply(lambda row: split_count(row, 'desc'), axis=1)

train.head()


test.head()

train.to_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/train_clean2.csv', index = False)
test.to_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/test_clean2.csv', index = False)
y_train = train['final_status']
test_id = test['project_id']



train.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline','final_status','backers_count'], inplace=True, axis=1)
test.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline'], inplace=True, axis=1)
train.head()
test.head()
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



train = pd.read_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/train_clean2.csv')
test = pd.read_csv('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/test_clean2.csv')

y_train = train['final_status']
test_id = test['project_id']
train.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline','final_status','backers_count'], inplace=True, axis=1)
test.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline'], inplace=True, axis=1)

dtrain = xgb.DMatrix(data=train, label = y_train)
dtest = xgb.DMatrix(data=test)

params = {
    'objective':'binary:logistic',
    'eval_metric':'error',
    'eta':0.04,
    'max_depth':10,
    'subsample':0.93,
    'colsample_bytree':0.7,
    'min_child_weight':5}

bst = xgb.cv(params, dtrain, num_boost_round=1000,nfold=5L,verbose_eval=10)

bst_train = xgb.train(params, dtrain, num_boost_round=1000)

p_test = bst_train.predict(dtest)

sub = pd.DataFrame()
sub['project_id'] = test_id
sub['final_status'] = p_test

sub['final_status'] = [1 if x > 0.5 else 0 for x in sub['final_status']]

sub.to_csv("/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/xgb_with_python_feats.csv",index=False)

#min_max_scaler = preprocessing.MinMaxScaler()
#train_scaled = min_max_scaler.fit_transform(train)
#train = pd.DataFrame(train_scaled)

#test_scaled = min_max_scaler.fit_transform(test)
#test = pd.DataFrame(test_scaled)

train.head()
test.head()
#clf = DecisionTreeClassifier(max_features=None, max_depth=11)
#clf = clf = RandomForestClassifier(n_estimators=500, max_features=26,oob_score=True, random_state=0)
#clf = SVC()
clf = GradientBoostingClassifier(learning_rate=0.04, n_estimators=500, subsample=0.93, max_depth=10, max_features=None)
#clf = AdaBoostClassifier(learning_rate=0.5, n_estimators=500)
#clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, max_iter=500,hidden_layer_sizes=(44, 2), learning_rate='invscaling')
#clf = KNeighborsClassifier(weights='distance',algorithm='kd_tree', leaf_size=1000, n_neighbors=2)
clf.fit(train, y_train)
pred = clf.predict(test)

out_column = ['final_status']
sub = pd.DataFrame(data=pred, columns=out_column)
sub['project_id'] = test_id
sub = sub[['project_id', 'final_status']]
sub.head()
sub.to_csv("/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/gb11Starter.csv",index = False)
