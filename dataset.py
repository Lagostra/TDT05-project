import urllib.request
import pandas as pd
import os
import kaggle

train_url = 'https://www.kaggle.com/c/santander-customer-transaction-prediction/download/train.csv'
test_url = 'https://www.kaggle.com/c/santander-customer-transaction-prediction/download/train.csv'

def load_data():
    """if not os.path.isfile('data/train.csv') or not os.path.isfile('data/train.csv'):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        kaggle.api.authenticate()
        
        os.chdir('data')
        kaggle.api.competitions_data_download_file('santander-customer-transaction-prediction', 'test.csv')
        os.chdir('..')"""
    
    train_set = pd.read_csv('data/train.csv')
    test_set = pd.read_csv('data/test.csv')

    return train_set, test_set

if __name__ == '__main__':
    get_data()