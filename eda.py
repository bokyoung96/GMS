import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader import DataHelper


class EDA(DataHelper):
    def __init__(self):
        super().__init__()
        mirae_data = self.run_mirae()

    @staticmethod
    def get_histogram(df: pd.DataFrame) -> plt.plot:
        num_cols = df.shape[1]
        num_rows = ((num_cols - 1) // 3) + 1

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))
        for i, col in enumerate(df.columns):
            ax = axes[i // 3, i % 3]
            df[col].plot.hist(ax=ax, bins=100, alpha=0.5, kde=True)
            ax.set_title(col)

        plt.tight_layout()
        plt.show()
