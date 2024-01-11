import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import holidays
from workalendar.europe import Romania
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from imblearn.under_sampling import NearMiss, RandomUnderSampler


class DateTransformer(TransformerMixin):
    def __init__(self, day='dayofweek', month=False, holiday=False):
        self.day = day
        self.month = month
        self.holiday = holiday

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        X = df.copy()
        X['Date'] = pd.to_datetime(X['Date']).dt.date

        if self.day == 'dayofweek':
            X['Day'] = X['Date'].apply(lambda x: x.weekday())
        elif self.day == 'weekend':
            X['Weekend'] = X['Date'].apply(lambda x: 1 if x.weekday() >= 4 else 0)
        if self.month:
            X['Month'] = X['Date'].apply(lambda x: x.month)
        if self.holiday:
            min_year = X['Date'].min().year
            max_year = X['Date'].max().year
            holiday = list(holidays.Romania(years=range(min_year, max_year + 1)).keys())
            holiday += [date + pd.Timedelta(days=2) for date in holiday] + [date - pd.Timedelta(days=2) for date in holiday]
            X['Holiday'] = X['Date'].apply(lambda x: 1 if x in holiday else 0)

        X.drop(columns=['Date'], inplace=True)
        return X
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

geolocator = Nominatim(user_agent="hospital")
def get_coordinates(city):
        location = geolocator.geocode(city)
        if location:
            return round(location.latitude, 2), round(location.longitude, 2)
        else:
            return None, None

class CityTransformer(TransformerMixin):
    def __init__(self, encoding='ohe', population=False, coordinates=False):
        self.encoding = encoding
        self.population = population
        self.coordinates = coordinates
        self.populations = {
            'Cluj Napoca': 322_108,
            'Timisoara': 333_613,
            'Iasi': 357_192,
            'Constanta': 319_168,
            'Bucuresti': 2_103_346,
        }
        self.city_coordinates = {city: get_coordinates(city) for city in self.populations.keys()}
    
    def fit(self, X, y=None):
        if self.encoding == 'le':
            self.city_encoder = LabelEncoder()
            self.city_encoder.fit(X['City'])
        elif self.encoding == 'ohe':
            self.city_encoder = OneHotEncoder(sparse=False, dtype=np.int64)
            self.city_encoder.fit(X[['City']])
        return self

    def transform(self, X):     
        if self.population:
            X['Population'] = X['City'].apply(lambda x: self.populations[x] if x in self.populations else None)
        if self.coordinates:
            X['Latitude'], X['Longitude'] = zip(*X['City'].apply(lambda x: self.city_coordinates[x]))

        if self.encoding == 'ohe':
            categ = self.city_encoder.get_feature_names_out()
            X[categ] = self.city_encoder.transform(X[['City']])
            X.drop(columns=['City'], inplace=True)
        elif self.encoding == 'le':
            X['City'] = self.city_encoder.transform(X['City'])
        else:
            X.drop(columns=['City'], inplace=True)
        return X
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

class BasicExperiments(TransformerMixin):
    def __init__(self, avg=False, scaler=False, scale_columns=[]):
        self.avg = avg
        self.scaler = scaler
        self.scale_columns = scale_columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.avg:
            X['Avg'] = X['Min'] + X['Max'] / 2
            X.drop(columns=['Min', 'Max'], inplace=True)
        if self.scaler:
            scaler = StandardScaler()
            for column in self.scale_columns:
                if column in X.columns:
                    X[column] = scaler.fit_transform(X[[column]])

        return X
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

class BasicTransformations(TransformerMixin):
    def fit(self, X, y=None):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(X['Sex'])
        return self

    def transform(self, df):
        X = df.copy()
        X['Sex'] = self.label_encoder.transform(X['Sex'])
        return X
    
class MultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi target classifier that uses one classifier per target and undersamples the majority class for each target
    """
    def __init__(self, undersampler=RandomUnderSampler, n_estimators=100, max_depth=5, learning_rate=0.1, gamma=0.1):
        self.undersampler = undersampler
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.classes_ = [0, 1]

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        self.estimators_ = [XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, gamma=self.gamma) 
                                for _ in range(y.shape[1])]

        for i, estimator in enumerate(self.estimators_):
            if self.undersampler:
                sampler = self.undersampler()
                mask = (y[:, i] == 1) | (y.sum(axis=1) == 0)
                X_res, y_res = sampler.fit_resample(X[mask], y[mask][:, i])
                estimator.fit(X_res, y_res)
            else:
                estimator.fit(X, y[:, i])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
        return predictions
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return predictions