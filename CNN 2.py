#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('dark')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm_notebook as tqdm
import tensorflow as tf


# ## Load the data
# We set a few flags for whether we are running locally or on Kaggle, or whether we want the reduced or full dataset. For the reduced dataset, the test set consists of samples from the full train set. For use during model evaluation, this has been augmented with reverse-engineered targets. This is simply to allow us to evaluate the AUC score on this set as well - it is never used for training.

# In[2]:


IS_LOCAL = True  # Sets whether we are running locally or on kaggle
USE_REDUCED = True  # Sets whether we should use the smaller dataset
data_index = 2*int(IS_LOCAL) + int(USE_REDUCED)
train_path = ('../input/santander-customer-transaction-prediction/train.csv',
             '../input/santandersmall/train_small.csv',
             'train.csv',
             'train_small.csv')[data_index]
test_path = ('../input/santander-customer-transaction-prediction/test.csv',
             '../input/santandersmall/test_small_with_targets.csv',
             'test.csv',
             'test_small.csv')[data_index]

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# In[3]:


features = [col for col in train_df.columns if col not in ['target', 'ID_code']]
if not 'target' in test_df:
    test_df['target'] = -1

all_df = pd.concat([train_df, test_df], sort=False)


# ## Feature Engineering

# ### Removing fake test samples
# It was discovered during the competition that some of the test samples were synthetic. Moreover, it was stated that not all of the data in the test set was used for evaluation, and the synthetic data corresponds to this. In order to achieve the best scores possible, the fake samples should be removed before calculating features such as counts. We calculate the indices of the fake rows below using the method provided by the Kaggle user YaG320. When using the reduced dataset, we have no synthetic samples. This is handled correctly by the algorithm below.

# In[4]:


unique_count = np.zeros((test_df.shape[0], len(features)))

for f, feature in tqdm(enumerate(features), total=len(features)):
    _, i, c = np.unique(test_df[feature], return_counts=True, return_index=True)
    unique_count[i[c == 1], f] += 1

real_sample_indices = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_sample_indices = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
print('Real:', len(real_sample_indices))
print('Synthetic:', len(synthetic_sample_indices))

del unique_count


# ### Calculate counts
# Counts of the values of the different features is often a powerful feature in itself. We calculate these below.

# In[5]:


all_real_df = pd.concat([train_df, test_df.iloc[real_sample_indices, :]], sort=False)

for feature in tqdm(features):
    real_series = all_real_df[feature]
    
    # We only use the real samples to produce the count
    counts = real_series.groupby(real_series).count()
    
    full_series = all_df[feature]
    all_df[f'{feature}_count'] = full_series.map(counts)

del all_real_df
del real_series
del full_series


# ### Normalization
# Normalizing features to have zero mean, unit variance can often speed up training.

# In[6]:


for feature in tqdm(features):
    all_df[feature] = StandardScaler().fit_transform(all_df[feature].values.reshape(-1, 1))
    all_df[f'{feature}_count'] = MinMaxScaler().fit_transform(all_df[f'{feature}_count'].values.reshape(-1, 1))


# In[7]:


for f in range(len(features)):
    features.append(f'{features[f]}_count')


# ### Splitting datasets back up
# We are now done with our feature engineering, so we can split the data back into train and test sets.

# In[8]:


train_df = all_df.iloc[:train_df.shape[0], :]
test_df = all_df.iloc[train_df.shape[0]:, :]

del all_df


# ## Model

# We use a neural network model to make predictions. 10-fold stratified cross validation is used, and we rely on early stopping, setting a large default number of epochs of 100. Binary cross-entropy is used for the loss function, which is a good fit for a binary classification task such as this one. The output of the network is a single output from a sigmoid function, which can be interpreted as a probability.

# In[9]:


N_SPLITS = 5
BATCH_SIZE = 256
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 3

OPTIMIZER = 'adam'
LOSS='binary_crossentropy'
METRICS=[tf.keras.metrics.AUC()]


# In[10]:


def get_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((len(features) * 1, 1), input_shape=(len(features) * 1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Conv1D(64, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Conv1D(128, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Conv1D(256, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        #tf.keras.layers.Conv1D(512, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        #tf.keras.layers.Conv1D(1024, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dense(2048, activation='relu'),
        #tf.keras.layers.Conv1D(2048, 1, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    return model


# In[ ]:


kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

models = []
histories = []

for fold_num, (train_index, val_index) in tqdm(enumerate(kfold.split(train_df[features].values, train_df['target'].values)), total=N_SPLITS):
    print(f'Fold {fold_num+1}/{N_SPLITS}:')
    
    X_train = train_df.loc[train_index, features].values
    y_train = train_df.loc[train_index, 'target'].values.reshape(-1, 1)
    X_val = train_df.loc[val_index, features].values
    y_val = train_df.loc[val_index, 'target'].values.reshape(-1, 1)
    
    model = get_cnn_model()
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='auc', mode='max', patience=EARLY_STOPPING_PATIENCE)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping_callback])
    histories.append(history)
    
    val_preds = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    print(f'Fold validation AUC: {val_auc}')
    print()
    models.append(model)


# ### Training plots

# In[ ]:


plt.figure(figsize=(20, 5 * (len(histories) // 5 + 1)))
for i, h in enumerate(histories):
    plt.subplot(len(histories) // 5 + 1, 5, i+1)
    plt.plot(h.history['loss'], label='Train loss')
    plt.plot(h.history['val_auc'], label='Val AUC')
    plt.legend()


# ## Create predictions
# 

# In[ ]:


train_preds = np.zeros(train_df.shape)
test_preds = np.zeros(test_df.shape)

for model in models:
    pred_train = model.predict(train_df[features])
    pred_test = model.predict(test_df[features])
    
    train_preds += pred_train
    test_preds += pred_test

train_preds /= len(models)
test_preds /= len(models)


# In[ ]:


train_preds = train_preds[:, 0]
test_preds = test_preds[:, 0]


# In[ ]:


train_auc = roc_auc_score(train_df['target'], train_preds)
print(f'Train AUC: {train_auc}')

if test_df['target'][0] != -1:
    test_auc = roc_auc_score(test_df['target'], test_preds)
    print(f'Test AUC: {test_auc}')


# ### Create submission

# In[ ]:


sub = pd.DataFrame({'ID_code': test_df['ID_code'], 'target': test_preds})
sub.to_csv('submission.csv', index=False)


# In[ ]:


from IPython.display import FileLink
FileLink('submission.csv')

