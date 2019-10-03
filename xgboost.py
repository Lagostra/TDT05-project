import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_select import train_test_split
from sklearn.metrics import accuracy_score

from dataset import load_data
from preprocessing import preprocess

# Pre-Processing Parameters
BALANCE_TARGETS = True
ENHANCED_FEATURES = ['sum', 'mean', 'min', 'max', 'std', 'median', 'skew', 'kurt']

train_set, test_set = load_data()

test_set_ids = test_set['ID_code']
test_set = test_set.drop('ID_code')
train_set = train_set.drop('ID_code')

train_preprocessed = preprocess(train_set, balance_targets=BALANCE_TARGETS, add_features=ENHANCED_FEATURES)
test_set = preprocess(test_set, balance_targets=False, add_features=ENHANCED_FEATURES)

X, y = train_data.drop('target'), train_data['target']
train_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(bojective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)

print(f'Accuracy over validation set: {acc}')