import urllib

train_url = 'https://www.kaggle.com/c/santander-customer-transaction-prediction/download/train.csv'
test_url = 'https://www.kaggle.com/c/santander-customer-transaction-prediction/download/train.csv'

urllib.request.urlretrieve(train_url, 'data/test.csv')