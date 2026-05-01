from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EDAConfig:
    output_dir: str
    target_column: str

@dataclass
class EDAArtifact:
    report_dir: str
    
class data_eda_utils:
    
    def __init__(self):
        pass

    def basic_info(self, df):
        logger.info("Getting basic dataset info")
        logger.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")


    def missing_values(self, df):
        logger.info("Checking missing values")
        missing = df.isnull().sum()
        logger.info(f"Missing values: {missing[missing > 0]}")


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