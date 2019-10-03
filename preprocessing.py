import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

train_set = pd.read_csv('/data/')