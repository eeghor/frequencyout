import pandas as pd
from pyod.models.copod import COPOD
from sklearn.ensemble import IsolationForest

categorical_columns = ['entry_key', 'user_key', 'location_id', 'department_id', 'account_name']
d = pd.read_csv('data/d_.csv.gz')
print(d.describe())