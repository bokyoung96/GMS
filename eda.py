import pandas as pd
import matplotlib.pyplot as plt

from loader import DataHelper


class EDA(DataHelper):
    def __init__(self):
        super().__init__()
        self.mirae_data = self.run_mirae()

    @staticmethod
    def get_histogram(df: pd.DataFrame) -> plt.plot:
        num_columns = len(df.columns)

        fig, axes = plt.subplots(nrows=num_columns,
                                 ncols=1,
                                 figsize=(8, num_columns*4))
        for i, column in enumerate(df.columns):
            axes[i].hist(df[column], bins=50,
                         color='skyblue', edgecolor='black')
            axes[i].set_title(column)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()
