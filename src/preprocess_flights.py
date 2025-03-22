import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA 
from sklearn.base import TransformerMixin


feature_list = ['DEPARTURE_DELAY', 'TAXI_OUT', 'ELAPSED_TIME', 'TAXI_IN', 'ARRIVAL_DELAY']

class DataPreprocessor(TransformerMixin):  # inheritance
    def __init__(self, scale_cols=None):
        self.scale_cols = scale_cols
        self.scaler = None


    def show_me_the_plot(self, df, feature_name):
        """
        Creates and displays a boxplot for the given feature.
        Parameters:
            feature_name (str): The name of the feature to plot as a boxplot.
        """
        sns.boxplot(x=df[feature_name])
        plt.title(f"{feature_name} Boxplot")
        plt.show()


    def outlier_analysis_single_var(self, df, feature_list):
        """
        Filters the outlier observations for each features which are continuous
        and equalizes upper and lower limits. 
        Parameters:
            df (pandas.DataFrame): The input data frame.
            feature_list (list): A list of continuous feature names to be processed.

        Returns:
            pandas.DataFrame: The data frame with outliers adjusted.
        """
        for feature in feature_list:
            df_feature = df[feature].copy()
        
            Q1 = df_feature.quantile(0.25)
            Q3 = df_feature.quantile(0.75)
            IQR = 4 * (Q3 - Q1)
            lower_limit = Q1 - IQR
            upper_limit = Q3 + IQR

            df_feature.loc[df_feature > upper_limit] = upper_limit
            df_feature.loc[df_feature < lower_limit] = lower_limit

            df.loc[:, feature] = df_feature
        
        return df


    def outlier_analysis_multi_vars(self, df, feature_list):
        df_continuous = df[feature_list]
        percentile = 0.03

        pca = PCA(n_components=5)
        reduced_df = pca.fit_transform(df_continuous)

        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        clf.fit_predict(reduced_df)

        df_scores = clf.negative_outlier_factor_
        threshold = np.percentile(df_scores, percentile*100)
        non_outliers = df_scores > threshold
        filtered_df = df[non_outliers]

        return filtered_df
        

    def missing_value_analysis(self, df):
        """
        Although there are no missing values in the dataset for now,
        this function removes missing values to ensure control and consistency.
        """
        df.dropna(inplace=True)
        return df
        

        