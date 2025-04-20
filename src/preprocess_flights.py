import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        """
        The function first reduces dimensionality to handle multiple features efficiently,
        then identifies and removes outliers based on the LOF scores.

        Parameters:
            df (pandas.DataFrame): The input data frame.
            feature_list (list): A list of continuous feature names to be processed.

        Returns:
            pandas.DataFrame: The data frame with outliers deleted.
        """
        df_continuous = df[feature_list]
        percentile = 0.03

        pca = PCA(n_components=2)
        reduced_df = pca.fit_transform(df_continuous)

        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
        clf.fit_predict(reduced_df)

        df_scores = clf.negative_outlier_factor_
        threshold = np.percentile(df_scores, percentile*100)
        non_outliers = df_scores > threshold
        filtered_df = df[non_outliers]
        print(f'Toplam Veri: {len(df)}, Aykırı Olmayan Veri: {len(filtered_df)}')
        return filtered_df
            

    def missing_value_analysis(self, df):
        """
        Although there are no missing values in the dataset for now,
        this function removes missing values to ensure control and consistency.
        """
        df.dropna(inplace=True)
        return df
    
    
    def drop_items(self, df):
        """
        Drops columns that are not available at prediction time or are not useful for modeling.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame to clean.
            
        Returns:
            pd.DataFrame: The DataFrame after dropping specified columns.
        """
        df.drop(["FLIGHT_NUMBER", "DEPARTURE_TIME", "DEPARTURE_DELAY", 
                "TAXI_OUT", "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME", 
                "AIR_TIME", "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME"], axis=1, inplace=True)
        return df


    def encode_categories(self, df):
        """
        Encodes categorical variables using frequency encoding.
        
        - 'AIRLINE', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', and 'DESTINATION_AIRPORT' 
        are encoded based on their frequency of occurrence in the dataset.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame containing categorical features.
            
        Returns:
            pd.DataFrame: The DataFrame with encoded categorical features.
        """
        airline_freq = df['AIRLINE'].value_counts(normalize=True)
        df['AIRLINE'] = df['AIRLINE'].map(airline_freq)
        
        tail_freq = df['TAIL_NUMBER'].value_counts()
        df['TAIL_NUMBER'] = df['TAIL_NUMBER'].map(tail_freq)

        origin_freq = df['ORIGIN_AIRPORT'].value_counts(normalize=True)
        df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].map(origin_freq)

        dest_freq = df['DESTINATION_AIRPORT'].value_counts(normalize=True)
        df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].map(dest_freq)

        return df

    
    def fast_process(self, df, feature_list, drop_cols=True, encode_cats=True, remove_outliers=True, drop_missing=True):
        """
        Applies a complete preprocessing pipeline on the dataset including:
        - Column dropping
        - Categorical encoding
        - Outlier analysis (single and multi-variate)
        - Missing value removal

        Parameters:
            df (pd.DataFrame): The raw input DataFrame.
            feature_list (list): List of continuous features for outlier analysis.
            drop_cols (bool): Whether to drop irrelevant columns.
            encode_cats (bool): Whether to encode categorical variables.
            remove_outliers (bool): Whether to apply outlier detection.
            drop_missing (bool): Whether to drop rows with missing values.

        Returns:
            pd.DataFrame: The fully preprocessed dataset ready for modeling.
        """
        df = df.copy()
        steps = []

        if drop_cols:
            steps.append(("Dropping unnecessary features...", self.drop_items))
        if encode_cats:
            steps.append(("Encoding categorical variables...", self.encode_categories))
        if remove_outliers:
            steps.append(("Analyzing single-variable outliers...", lambda d: self.outlier_analysis_single_var(d, feature_list)))
            steps.append(("Analyzing multi-variable outliers...", lambda d: self.outlier_analysis_multi_vars(d, feature_list)))
        if drop_missing:
            steps.append(("Dropping missing values...", self.missing_value_analysis))

        for desc, func in tqdm(steps, desc="Fast Processing", ncols=80):
            print(desc)
            df = func(df)

        print("Fast processing completed!")
        return df



        # Veri Özeti Fonksiyonu: Kategorik ve sayısal değişkenlerin özet istatistiklerini tek bir tabloda gösteren fonksiyon.
        # scale fonksiyonu
        # pipelines