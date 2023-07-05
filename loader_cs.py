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
    def __init__(self):
        super().__init__()
        self.customer = self.run_mirae[1]
        self.customer_info = self.customer[['AGE_TCD',
                                            'MT_EP_EXIST_YN']]

    @staticmethod
    def filter_data(item_type: str) -> list:
        return list(filter(lambda name: item_type in name, CustomerPool.__dict__.keys()))

    def filter_data_month(self):
        return
