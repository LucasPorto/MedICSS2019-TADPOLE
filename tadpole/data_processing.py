import numpy as np
import pandas as pd
from io_local import get_age_at_exam

FEATURE_SET_1 = [
    'AGE',
    'Hippocampus_bl',
    'Entorhinal_bl',
    'Fusiform_bl',
    'MidTemp_bl',
    'WholeBrain_bl',
    'Ventricles_bl',
    'Ventricles'
]

LABELS = [
    'DX',
    'ADAS13'
]

class Dataset(object):
    def __init__(self, data_path, features=FEATURE_SET_1, labels=LABELS):
        df = pd.read_csv(data_path, low_memory=False)
        variables_to_check = features + labels[1:]
        for var_name in variables_to_check:
            var0 = df[var_name].iloc[0]
            if isinstance(var0, str):
                df[var_name] = df[var_name].astype(np.int)

        df['EXAMDATE'] = pd.to_datetime(df.EXAMDATE)
        age = get_age_at_exam(df)

        data = df[['RID'] + features]
        for col in data.columns[2:]:
            data[col] = data[col] / df['ICV_bl']

        data = pd.concat([data, df[labels]], axis=1)
        data = data.set_index('RID', append=True).swaplevel()

        self.data = pd.concat([data, age['AGE_AT_EXAM']], axis=1).reset_index(level=0)
        self.data['AGE_AT_EXAM'] = self.data['AGE_AT_EXAM'] - self.data['AGE']
        self.data = self.data.rename(index=str, columns={'AGE_AT_EXAM': 'YEARS_FROM_BL'})









