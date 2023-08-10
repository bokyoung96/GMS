"""
TEAM 고객의 미래를 새롭게

코드 목적: 데이터 로딩
"""

import numpy as np
import pandas as pd
from enum import Enum, unique

MIRAE_NAME = "mirae-%s.csv"
INDIV_NAME = "indiv-%s.csv"


@unique
class MiraeItem(Enum):
    """
    <DESCRIPTION>
    미래에셋증권에서 제공한 데이터를 3가지 분류(apy_itm, cs, mkt_idx)로 구분합니다.
    """
    apy_itm = "apy_itm_hist"
    cs = "cs_data"
    mkt_idx = "mkt_idx"

    def as_mirae(self):
        return MIRAE_NAME % self.value


@unique
class IndivItem(Enum):
    """
    <DESCRIPTION>
    연구를 위해 필요한 추가 데이터를 N가지 분류로 구분합니다.
    """
    pass

    def as_indiv(self):
        return INDIV_NAME % self.value


@unique
class DataPool(Enum):
    """
    <DESCRIPTION>
    미래에셋증권에서 제공하는 데이터와 추가 데이터를 하나의 Data-pool에 군집합니다.
    """
    mirae_apy_itm = MiraeItem.apy_itm.as_mirae()
    mirae_cs = MiraeItem.cs.as_mirae()
    mirae_mkt_idx = MiraeItem.mkt_idx.as_mirae()


class DataHelper:
    """
    <DESCRIPTION>
    필요한 데이터를 로드합니다.
    """

    def __init__(self):
        pass

    def get_data(self, data_lists: list) -> pd.DataFrame:
        """
        <DESCRIPTION>
        data_lists에 속한 데이터를 읽습니다.
        """
        return [pd.read_csv("./{}".format(data_list.value)) for data_list in data_lists]

    def run_mirae(self) -> list:
        """
        <DESCRIPTION>
        미래에셋증권에서 제공하는 데이터를 mirae_data에 저장합니다.
        """
        mirae_data_lists = [DataPool.mirae_apy_itm,
                            DataPool.mirae_cs,
                            DataPool.mirae_mkt_idx]

        mirae_data = self.get_data(mirae_data_lists)
        return mirae_data
