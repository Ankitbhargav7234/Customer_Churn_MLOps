from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import get_logger
from src.utils.exception import CustomException
logger = get_logger(__name__)

@dataclass
class EDAConfig:
    output_dir: str
    target_column: str
    train_data_path: str

@dataclass
class EDAArtifact:
    report_dir: str
    
class data_eda_utils:
    
    def __init__(self):
        pass

    def save_data(self, df: pd.DataFrame, path: str):
        try:
            df.to_csv(path, index=False)
            logger.info(f"Data saved at {path}")
        except Exception as e:
            raise CustomException(e)
        
    def basic_info(self, df):
        logger.info("Getting basic dataset info")
        logger.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")


    def missing_values(self, df, train_data_path):
        logger.info("Checking missing values")
        null_percentages = df.isnull().sum() / len(df) * 100
        logger.info("Percentage of null values per column:\n", null_percentages)
        # Identify columns to drop (more than 40% nulls)
        columns_to_drop = null_percentages[null_percentages > 40].index
        df.drop(columns=columns_to_drop, inplace=True)
        logger.info(f"\nDropped columns with more than 40% null values: {list(columns_to_drop)}")

        # Impute remaining null values
        for col in df.columns:
            if df[col].isnull().any() and null_percentages[col] <= 40:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled nulls in numerical column '{col}' with median: {median_val}")
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    mode_val = df[col].mode()[0] # mode() can return multiple values, take the first
                    df[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled nulls in categorical column '{col}' with mode: {mode_val}")

        logger.info(f"\nNull values after processing:\n{df.isnull().sum()}")

        self.save_data(df, train_data_path)

    def dropping_useless_columns(self, df, threshold=0.95):
        logger.info("Dropping useless columns")
        

    def plot_target_distribution(self, df, target_col, output_dir):
        logger.info("Plotting target distribution")
        plt.figure()
        sns.countplot(x=target_col, data=df)
        plt.title("Target Distribution")
        plt.savefig(os.path.join(output_dir, "target_distribution.png"))
        plt.close()


    def plot_numeric_distributions(self, df, output_dir):
        logger.info("Plotting numeric distributions")
        num_cols = df.select_dtypes(include=np.number).columns

        for col in num_cols:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"{col} Distribution")
            plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
            plt.close()


    def plot_categorical_vs_target(self, df, target_col, output_dir):
        logger.info("Plotting categorical vs target")
        cat_cols = df.select_dtypes(include="object").columns

        for col in cat_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=col, hue=target_col, data=df)
            plt.xticks(rotation=45)
            plt.title(f"{col} vs {target_col}")
            plt.savefig(os.path.join(output_dir, f"{col}_vs_target.png"))
            plt.close()


    def correlation_heatmap(self, df, output_dir):
        logger.info("Generating correlation heatmap")
        plt.figure(figsize=(10, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()