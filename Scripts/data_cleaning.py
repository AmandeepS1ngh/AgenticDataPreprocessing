import pandas as pd
import numpy as np

class DataCleaning:
    def handle_missing_values(self, df, strategy="mean"):
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # 1. Impute numeric columns
        if len(numeric_cols) > 0:
            if strategy == "mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == "median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif strategy == "mode":
                for col in numeric_cols:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
            elif strategy == "drop":
                pass
                
        # 2. Impute categorical/string columns (mode fallback since mean/median are numeric only)
        if len(categorical_cols) > 0:
            if strategy != "drop":
                for col in categorical_cols:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                    else:
                        df[col] = df[col].fillna("Unknown")
                        
        # 3. Handle drop strategy for the entire dataframe
        if strategy == "drop":
            df.dropna(inplace=True)
            
        return df
    
    def remove_duplicates(self, df):
        return df.drop_duplicates()
    
    def fix_data_types(self, df):
        for col in df.columns:
            try:
                df[col]= pd.to_numeric(df[col])
            except ValueError:
                pass
        return df
    
    def normalize_data(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        return df
    
    def clean_data(self, df, missing_value_strategy="mean", normalize=False):
        df = self.handle_missing_values(df, strategy=missing_value_strategy)
        df = self.remove_duplicates(df)
        df = self.fix_data_types(df)
        if normalize:
            df = self.normalize_data(df)
        print("Data cleaning completed successfully.")
        return df
