import numpy as np
import pandas as pd
from enum import Enum, unique

MIRAE_NAME = "mirae-%s.csv"
INDIV_NAME = "indiv-%s.csv"


@unique
class MiraeItem(Enum):
    apy_itm = "apy_itm_hist"
    cs = "cs_data"
    mkt_idx = "mkt_idx"

    def as_mirae(self):
        return MIRAE_NAME % self.value


@unique
class IndivItem(Enum):
    pass

    def as_indiv(self):
        return INDIV_NAME % self.value


@unique
class DataPool(Enum):
    mirae_apy_itm = MiraeItem.apy_itm.as_mirae()
    mirae_cs = MiraeItem.cs.as_mirae()
    mirae_mkt_idx = MiraeItem.mkt_idx.as_mirae()


class DataHelper:
    def __init__(self):
        pass

    def get_data(self, data_lists: list) -> pd.DataFrame:
        return [pd.read_csv("./{}".format(data_list.value)) for data_list in data_lists]

    def run_mirae(self) -> list:
        mirae_data_lists = [DataPool.mirae_apy_itm,
                            DataPool.mirae_cs,
                            DataPool.mirae_mkt_idx]

        mirae_data = self.get_data(mirae_data_lists)
        return mirae_data
