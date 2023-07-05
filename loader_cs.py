import numpy as np
import pandas as pd
from enum import Enum, unique

from loader import DataHelper

AST = "AST%s"
M = "_M%s"

CASH = "CASH_%s"
DMST = "DMST_%s"
OVST = "OVST_%s"


@unique
class AssetItem(Enum):
    eval = "_EVAL"
    pchs = "_PCHS"
    itm = "_ITM"
    mkt = "_MKT"

    @classmethod
    def as_asset_rank(cls):
        return [AST % str(rank) for rank in range(1, 4)]

    @classmethod
    def as_asset_month(cls):
        return [M % str(month) for month in range(1, 5)]

    def as_asset_dmst(self):
        return DMST % AST % self.value

    def as_asset_ovst(self):
        return OVST % AST % self.value

    def as_asset_dmst_rank(self):
        return [DMST % rank + self.value for rank in AssetItem.as_asset_rank()]

    def as_asset_ovst_rank(self):
        return [OVST % rank + self.value for rank in AssetItem.as_asset_rank()]


@unique
class CustomerPool(Enum):
    # ASSET ITEM
    asset_rank = AssetItem.as_asset_rank()
    asset_month = AssetItem.as_asset_month()

    dmst_ast_eval = [AssetItem.eval.as_asset_dmst() +
                     month for month in asset_month]
    ovst_ast_eval = [AssetItem.eval.as_asset_ovst() +
                     month for month in asset_month]

    dmst_ast_pchs = [AssetItem.pchs.as_asset_dmst() +
                     month for month in asset_month]
    ovst_ast_pchs = [AssetItem.pchs.as_asset_ovst() +
                     month for month in asset_month]

    dmst_ast_pchs_rank = [
        dmst_rank + month for dmst_rank in AssetItem.pchs.as_asset_dmst_rank() for month in AssetItem.as_asset_month()]
    ovst_ast_pchs_rank = [
        ovst_rank + month for ovst_rank in AssetItem.pchs.as_asset_ovst_rank() for month in AssetItem.as_asset_month()]

    # TRANSACTION ITEM


class CustomerHelper(DataHelper):
    def __init__(self, item_type: str):
        """
        <item_type>

        ast: ASSET ITEM
        trans: TRANSACTION ITEM
        """
        super().__init__()
        self.customer = self.run_mirae()[1]
        self._customer_info = ['AGE_TCD', 'MT_EP_EXIST_YN']
        self.customer_info = self.customer[self._customer_info]
        self.item_type = item_type

    def filter_name(self) -> list:
        return list(filter(lambda name: self.item_type in name,
                           CustomerPool.__dict__.keys()))

    def filter_data(self) -> pd.DataFrame:
        filter_name = self.filter_name()
        length = len(filter_name)
        filter_attr = np.concatenate(
            [getattr(CustomerPool, filter_name[i]).value for i in range(length)])
        res = pd.concat(
            [self.customer_info, self.customer[filter_attr]], axis=1)
        return res

    def filter_data_category(self, item_category: str) -> pd.DataFrame:
        """
        <item_category>

        eval
        pchs
        pchs_rank
        """
        df = self.filter_data()
        filter_category = [col for col in df.columns if item_category in col]
        res = df[self._customer_info + filter_category]
        return res

    def filter_data_month(self):
        return

    # TODO: customer_info의 기본 데이터 + filter_data 동시 불러오기
    # TODO: 데이터 수 CHECK
