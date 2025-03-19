from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_csv("src/semi_clean_data.csv")
feature_list = ['DEPARTURE_DELAY', 'TAXI_OUT', 'ELAPSED_TIME', 'TAXI_IN', 'ARRIVAL_DELAY']


class DataPreprocessor(TransformerMixin):  # inheritance
    def __init__(self, scale_cols=None):
        self.scale_cols = scale_cols
        self.scaler = None
        
    def show_me_the_plot(self, feature_name):
        sns.boxplot(x=df[feature_name])
        plt.title(f"{feature_name} Boxplot")
        plt.show()

    def outlier_analysis_single_var(self, feature_list):
        for feature in feature_list:
            df_feature = df[feature]
            Q1 = df_feature.quantile(0.25)
            Q3 = df_feature.quantile(0.75)
            IQR = 2.5 * (Q3 - Q1)
            lower_limit = Q1 - IQR 
            upper_limit = Q3 + IQR 
            outliers = (df_feature > upper_limit) | (df_feature < lower_limit)
            df_feature.loc[outliers > upper_limit] = upper_limit
            df_feature.loc[outliers < lower_limit] = lower_limit

    def outlier_analysis_multi_vars(self):
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        clf.predict()


    

processor = DataPreprocessor()
processor.outlier_analysis_single_var(feature_list=feature_list)
        