import pandas as pd
from pyod.models.copod import COPOD
d = pd.read_csv('data/d_.csv.gz')
print(d.describe())