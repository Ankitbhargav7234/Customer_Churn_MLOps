import os
import pandas as pd
from src.eda.eda_utils import data_eda_utils
from src.eda.eda_utils import (
    EDAConfig,
    EDAArtifact,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

class EDA:
    def __init__(self, config: EDAConfig):
        self.config = config

    def initiate_eda(self, train_path: str) -> EDAArtifact:
        logger.info("Starting EDA stage")
        obj = data_eda_utils()
        df = pd.read_csv(train_path)

        os.makedirs(self.config.output_dir, exist_ok=True)
        obj.basic_info(df)
        obj.missing_values(df)
        obj.plot_target_distribution(df, self.config.target_column, self.config.output_dir)
        obj.plot_numeric_distributions(df, self.config.output_dir)
        obj.plot_categorical_vs_target(df, self.config.target_column, self.config.output_dir)
        obj.correlation_heatmap(df, self.config.output_dir)

        logger.info("EDA completed")

        return EDAArtifact(report_dir=self.config.output_dir)