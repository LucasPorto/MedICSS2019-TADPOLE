from os.path import join
from data_processing import Dataset
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMResultsWrapper

from sklearn.preprocessing import MinMaxScaler


class MixedModel(object):
    def __init__(self):
        self.scaler = MinMaxScaler()

        self.fe_features = [
            'AGE',
            'YEARS_FROM_BL'
        ]

        self.re_features = [
            'YEARS_FROM_BL',
        ]

        self.labels = ['Ventricles']
        self.model_vent, self.model_adas = self.train()

    def get_dataset(self):
        dataLocationLB1LB2 = '../data/'  # current directory
        tadpoleLB1LB2_file = join(dataLocationLB1LB2, 'TADPOLE_LB1_LB2.csv')
        dataset = Dataset(tadpoleLB1LB2_file)
        X_vent = dataset.data
        X_vent = X_vent[['RID'] + list(set(self.fe_features + self.re_features)) + self.labels]
        X_vent = X_vent.dropna(axis=0)
        y_vent = X_vent[self.labels]
        rid_vent = X_vent['RID']
        X_vent = X_vent.drop(labels=['RID'] + self.labels, axis=1)

        X_adas = dataset.data
        X_adas = X_adas[['RID'] + list(set(self.fe_features + self.re_features)) + ['ADAS13']]
        X_adas = X_adas.dropna(axis=0)
        y_adas = X_adas[['ADAS13']]
        rid_adas = X_adas['RID']
        X_adas = X_adas.drop(labels=['RID'] + ['ADAS13'], axis=1)

        return (rid_vent, X_vent, y_vent), (rid_adas, X_adas, y_adas)

    def train(self):
        vent_train_data, adas_train_data = self.get_dataset()

        #---- Ventricle model
        rid, X, y = vent_train_data
        intercepts = np.ones((X.shape[0], 1))
        intercepts_df = pd.DataFrame({'INT': intercepts.reshape(-1)})
        squared_term = pd.DataFrame({'YBL_SQ': X['YEARS_FROM_BL'].apply(np.square).values.reshape(-1)})
        X_int = X.reset_index(drop=True).join(intercepts_df).join(squared_term)
        # X_int[self.fe_features + self.re_features] = self.scaler.fit_transform(
        #     X_int[self.fe_features + self.re_features])
        model = MixedLM(endog=y.values,
                        exog=X_int[['INT'] + self.fe_features + ['YBL_SQ']],
                        groups=rid,
                        exog_re=X_int[['INT'] + self.re_features]
                        )

        results_vent = model.fit()

        #----- Adas model
        rid, X, y = adas_train_data
        intercepts = np.ones((X.shape[0], 1))
        intercepts_df = pd.DataFrame({'INT': intercepts.reshape(-1)})
        squared_term = pd.DataFrame({'YBL_SQ': X['YEARS_FROM_BL'].apply(np.square).values.reshape(-1)})

        X_int = X.reset_index(drop=True).join(intercepts_df).join(squared_term)
        # X_int[self.fe_features + self.re_features] = self.scaler.fit_transform(
        #     X_int[self.fe_features + self.re_features])

        model = MixedLM(endog=y.values,
                        exog=X_int[['INT'] + self.fe_features + ['YBL_SQ']],
                        groups=rid,
                        exog_re=X_int[['INT'] + self.re_features]
                        )

        results_adas = model.fit()

        return results_vent, results_adas

    def predict(self, model, fe_mat, re_mat, rid):
        fe_mat = fe_mat.values
        fe_mat = np.concatenate((fe_mat, np.square(fe_mat[:,-1]).reshape(-1,1)), axis=1)
        re_mat = re_mat.values
        # scaled = self.scaler.transform(np.concatenate((fe_mat[:, 1:], re_mat[:, 1:]), axis=1))
        #
        # fe_mat[:, 1:] = scaled[:, 0:fe_mat[:, 1:].shape[1]]
        # re_mat[:, 1:] = scaled[:, fe_mat[:, 1:].shape[1]:]

        fe_vector = model.fe_params.values
        re_vector = model.random_effects[rid].values
        pred = fe_mat @ fe_vector + re_mat @ re_vector

        c_interval = model.conf_int(alpha=0.5)

        fe_vector_low = np.asarray(c_interval[0:4][0])
        fe_vector_high = np.asarray(c_interval[0:4][1])

        pred_low = fe_mat @ fe_vector_low + re_mat @ re_vector
        pred_high = fe_mat @ fe_vector_high + re_mat @ re_vector

        return pred, (pred_low, pred_high)

    def create_prediction(self, train_data, train_targets, data_forecast, rid):
        """Create a linear regression prediction that does a first order
        extrapolation in time of ADAS13 and ventricles.

        :param train_data: Features in training data.
        :type train_data: pd.DataFrame
        :param train_targets: Target in trainign data.
        :param pd.DataFrame
        :param data_forecast: Empty data to insert predictions into
        :type data_forecast: pd.DataFrame
        :return: Data frame in same format as data_forecast.
        :rtype: pd.DataFrame
        """
        # * Clinical status forecast: predefined likelihoods per current status
        most_recent_data = pd.concat(
            (train_targets, train_data[['EXAMDATE', 'AGE_AT_EXAM', 'ICV_bl'] + self.re_features[1:]]),
            axis=1).sort_values(by='EXAMDATE')

        most_recent_CLIN_STAT = most_recent_data['CLIN_STAT'].dropna().tail(1).iloc[0]
        if most_recent_CLIN_STAT == 'NL':
            CNp, MCIp, ADp = 0.3, 0.4, 0.3
        elif most_recent_CLIN_STAT == 'MCI':
            CNp, MCIp, ADp = 0.1, 0.5, 0.4
        elif most_recent_CLIN_STAT == 'Dementia':
            CNp, MCIp, ADp = 0.15, 0.15, 0.7
        else:
            CNp, MCIp, ADp = 0.33, 0.33, 0.34

        # Use the same clinical status probabilities for all months
        data_forecast.loc[:, 'CN relative probability'] = CNp
        data_forecast.loc[:, 'MCI relative probability'] = MCIp
        data_forecast.loc[:, 'AD relative probability'] = ADp

        # * Ventricles volume forecast: = most recent measurement, default confidence interval

        age_bl = train_data['AGE'].reset_index(drop=True)
        age_at_exam = most_recent_data['AGE_AT_EXAM'].dropna().iloc[-1] + data_forecast['Forecast Month'] / 12
        years_from_bl = age_at_exam - age_bl.iloc[0]
        intercepts = np.ones((years_from_bl.shape[0], 1))
        age_bl = intercepts * age_bl.iloc[0]

        fe_mat = pd.DataFrame({'INT': intercepts.reshape(-1),
                               'AGE': age_bl.reshape(-1),
                               'YEARS_FROM_BL': years_from_bl.values.reshape(-1)
                               })

        re_dict = {'INT': intercepts.reshape(-1),
                   'YEARS_FROM_BL': years_from_bl.values.reshape(-1)
                    }

        if len(self.re_features) > 1:
            for feature in self.re_features[1:]:
                re_dict[feature] = most_recent_data[feature].iloc[0] / most_recent_data['ICV_bl'].iloc[0]
                re_dict[feature] = re_dict[feature] * intercepts.reshape(-1)

        re_mat = pd.DataFrame(re_dict)

        vent_prediction, cint = self.predict(self.model_vent, fe_mat, re_mat, rid)
        adas_prediction, cint_adas = self.predict(self.model_adas, fe_mat, re_mat, rid)

        data_forecast.loc[:, 'ADAS13'] = adas_prediction
        data_forecast.loc[:, 'ADAS13 50% CI lower'] = cint_adas[0]
        data_forecast.loc[:, 'ADAS13 50% CI upper'] = cint_adas[1]

        data_forecast.loc[:, 'Ventricles_ICV'] = vent_prediction
        data_forecast.loc[:, 'Ventricles_ICV 50% CI lower'] = np.maximum(0, cint[0])
        data_forecast.loc[:, 'Ventricles_ICV 50% CI upper'] = np.minimum(1, cint[1])
        return data_forecast
