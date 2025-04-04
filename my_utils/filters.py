import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures, DropDuplicateFeatures
from feature_engine.selection import ProbeFeatureSelection
from sklearn.linear_model import LinearRegression


class SimpleFilter:

    def __init__(self, variance_threshold=0.0, corr_threshold=0.9):
        self.lowVarianceFilter = VarianceThreshold()
        self.filter_duplicates = DropDuplicateFeatures()
        self.correlated_filter = DropCorrelatedFeatures(threshold=corr_threshold)
        self.advanced_filtered = ProbeFeatureSelection(
            estimator=LinearRegression(),
            scoring="neg_mean_absolute_percentage_error",
            n_probes=3,
            distribution="normal",
        )

    def fit(self, X_data, y_data):
        print(X_data.shape)
        self.lowVarianceFilter.fit(X_data)
        lv = self.lowVarianceFilter.transform(X_data)

        lv_df = pd.DataFrame(
            data=lv,
            columns=self.lowVarianceFilter.get_feature_names_out(),
            index=X_data.index,
        )
        lv_df = lv_df.loc[:,~lv_df.columns.duplicated()].copy()
        print(lv_df.shape)
        self.filter_duplicates.fit(lv_df)
        no_dup = self.filter_duplicates.transform(lv_df)
        print(no_dup.shape)
        self.correlated_filter.fit(no_dup)
        not_corr = self.correlated_filter.transform(no_dup)
        print(not_corr.shape)
        self.advanced_filtered.fit(not_corr, y_data)

    def transform(self, X_data, y_data):
        X_data = X_data.copy()
        X_data_low = self.lowVarianceFilter.transform(X_data)
        X_data_low_df = pd.DataFrame(
            data=X_data_low,
            columns=self.lowVarianceFilter.get_feature_names_out(),
            index=X_data.index,
        )   
        X_data_low_df = X_data_low_df.loc[:,~X_data_low_df.columns.duplicated()].copy()
        print(X_data_low_df.shape)
        no_dup = self.filter_duplicates.transform(X_data_low_df)
        print(no_dup.shape)
        not_corr = self.correlated_filter.transform(no_dup)
        print(not_corr.shape)
        X_transformed = self.advanced_filtered.transform(not_corr)
        return X_transformed, y_data


# Filtro que identifica valores que se desvian mucho del rango tipico de los datos 
# y crea un nuevo dataset en el que ya no estan estos valores atipicos

class MyFilter:

    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier
        self.iqr_limites = {}  # Almacenar los límites IQR para cada columna

    def fit(self, X_data, y_data=None):
        for column in X_data.columns:

            # Calcular el primer cuartil (Q1) y el tercer cuartil (Q3) de la columna
            Q1 = X_data[column].quantile(0.25)
            Q3 = X_data[column].quantile(0.75)

            # Calcular el rango intercuartílico (IQR)
            IQR = Q3 - Q1

            # Calcular los límites inferior y superior utilizando el multiplicador IQR
            limite_inf = Q1 - self.iqr_multiplier * IQR
            limite_sup= Q3 + self.iqr_multiplier * IQR

            # Almacenar los límites en un diccionario
            self.iqr_limites[column] = (limite_inf, limite_sup)
        return self

    def transform(self, X_data, y_data=None):

        X_filtered = X_data.copy()
        y_filtered = y_data.copy() if y_data is not None else None

        # Marca todos los datos como válidos
        outliers_mask = pd.Series([True] * len(X_filtered)) 


        for column in X_filtered.columns:
            # Obtener los límites inferior y superior para cada columna
            limite_inf, limite_sup = self.iqr_limites[column]

            # Identifica los valores atipicos
            column_outliers_mask = (X_filtered[column] < limite_inf) | (X_filtered[column] > limite_sup)
            
            # Actualiza y elimina los atipicos
            outliers_mask = outliers_mask & ~column_outliers_mask  

        # Filtra los datos sin atipicos
        X_filtered = X_filtered[outliers_mask]

        # Filtra la variable objetivo si está presente
        if y_filtered is not None:
            y_filtered = y_filtered[outliers_mask]

        return X_filtered, y_filtered