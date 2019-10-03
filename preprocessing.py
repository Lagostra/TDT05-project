import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

def preprocess(data_set, reduce_size=-1, balance_targets=True, target_col='target', add_features=['sum', 'mean', 'min', 'max', 'std', 'median', 'skew', 'kurt']):
    '''
        Preprocesses the data set contained in the DataFRame data_set.
    '''
    print('\n\nPRE-PROCESSING')

    if reduce_size > 0:
        print('Reducing data set size')
        reduce(data_set, reduce_size)
    
    if balance_targets:
        print('Balancing distribution of target values')
        balance(data_set, target_col)
    
    if add_features:
        print('Adding enhanced features')
        feature_enhancement(data_set, add_features, target_col)


def reduce(data_set, reduce_size):
    sample_frac = reduce_size / data_set.shape[0]
    reduced = train_set.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_frac))
    return reduced

def balance(data_set, target_col):
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    oversampled_x, oversampled_y = sm.fit_sample(data_set.drop(target_col, axis=1), data_set[target_col])
    oversampled = pd.concat([pd.DataFrame(oversampled_x), pd.DataFrame(oversampled_y)], axis=1)
    oversampled.columns = data_set.columns
    return oversampled

def feature_enhancement(data_set, enhanced_features, target_col):
    features = data_set.drop(target_col)
    
    if 'sum' in enhanced_features:
        data_set['sum'] = features.sum(axis=1)
    if 'mean' in enhanced_features:
        data_set['mean'] = features.mean(axis=1)
    if 'min' in enhanced_features:
        data_set['min'] = features.min(axis=1)
    if 'max' in enhanced_features:
        data_set['max'] = features.max(axis=1)
    if 'std' in enhanced_features:
        data_set['std'] = features.std(axis=1)
    if 'median' in enhanced_features:
        data_set['median'] = features.median(axis=1)
    if 'skew' in enhanced_features:
        data_set['skew'] = features.skew(axis=1)
    if 'kurt' in enhanced_features:
        data_set['kurt'] = features.kurtn(axis=1)