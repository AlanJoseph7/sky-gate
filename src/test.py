import pandas as pd
import numpy as np
from utils import get_data_path, FEATURE_COLUMNS

df = pd.read_csv(get_data_path('adsb_features.csv'))
print('Rows:', len(df))
print('Aircraft:', df['icao'].nunique())
print('Min rows per aircraft:', df.groupby('icao').size().min())
print('Label counts:', df['label'].value_counts().to_dict())
print('NaN count:', df[FEATURE_COLUMNS].isna().sum().sum())
print('Inf count:', np.isinf(df[FEATURE_COLUMNS].values).sum())
print('Feature ranges:')
print(df[FEATURE_COLUMNS].describe().loc[['min','max']].T)