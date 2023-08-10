"""
TEAM 고객의 미래를 새롭게

코드 목적: 미래에셋증권에서 제공하는 데이터 중 cs 데이터를 세부 구분하여 로드합니다.
"""

import numpy as np
import pandas as pd
from enum import Enum, unique

from loader import DataHelper

"""
<DESCRIPTION>
cs 데이터를 ASSET, TRANSACTION, ACCESS, SNAPSHOT, RATIO 5가지 카테고리로 구분합니다.
"""

# COMMON
M = "_M%s"

# ASSET, SNAPSHOT
AST = "AST%s"
DMST = "DMST_%s"
OVST = "OVST_%s"

# TRANSACTION
BUY = "BUY%s"
SEL = "SEL%s"

# ACCESS
DYS = "DYS"

# SNAPSHOT
DMETF = "DMETF_%s"
OVETF = "OVETF_%s"
APY = "APY_%s"

# RATIO
TR_RATIO = "TR_RATIO"


class MonthItem:
    """
    <DESCRIPTION>
    cs 데이터를 분기 단위로 구분합니다.
    """
    @staticmethod
    def as_month():
        """
        <DESCRIPTION>
        월 말일 기준으로 데이터를 구분합니다.
        """
        return [M % str(month) for month in range(1, 5)]

    @staticmethod
    def as_sub_month():
        """
        <DESCRIPTION>
        월별 초순, 중순, 하순 기준으로 데이터를 구분합니다.
        """
        return [
            M % f"{month}_{sub_month}" for month in range(1, 5) for sub_month in range(1, 4)]


@unique
class AssetItem(Enum):
    """
    <DESCRIPTION>
    ASSET 데이터를 dmst, ovst로 구분합니다.
    dmst, ovst는 eval, pchs, itm, mkt로 세부 구분되며, 이 중 [1,2,3]위 데이터가 존재하는 경우, rank로 각 순위가 구분됩니다.
    구분되지 않은 기타 데이터는 etc에 저장됩니다.
    """
    eval = "_EVAL"
    pchs = "_PCHS"
    itm = "_ITM"
    mkt = "_MKT"

    @classmethod
    def as_asset_rank(cls):
        return [AST % str(rank) for rank in range(1, 4)]

    def as_asset_dmst(self):
        return DMST % AST % self.value

    def as_asset_ovst(self):
        return OVST % AST % self.value

    def as_asset_dmst_rank(self):
        return [DMST % rank + self.value for rank in AssetItem.as_asset_rank()]

    def as_asset_ovst_rank(self):
        return [OVST % rank + self.value for rank in AssetItem.as_asset_rank()]

    @classmethod
    def as_asset_etc(cls):
        return ["CASH_AST", "DMST_ITM_CNT", "OVST_ITM_CNT"]


@unique
class TransactionItem(Enum):
    """
    <DESCRIPTION>
    TRANSACTION 데이터를 buy, sel로 구분합니다.
    buy, sel은 cnt, itm, mkt, amt로 세부 구분되며, 이 중 [1,2,3]위 데이터가 존재하는 경우, rank로 각 순위가 구분됩니다.
    구분되지 않은 기타 데이터는 etc에 저장됩니다.
    """
    cnt = "_CNT"
    itm = "_ITM"
    mkt = "_MKT"
    amt = "_AMT"

    @classmethod
    def as_trans_rank(cls):
        return [str(rank) for rank in range(1, 4)]

    def as_trans_buy(self):
        return BUY % self.value

    def as_trans_sel(self):
        return SEL % self.value

    def as_trans_buy_rank(self):
        return [BUY % rank + self.value for rank in TransactionItem.as_trans_rank()]

    def as_trans_sel_rank(self):
        return [SEL % rank + self.value for rank in TransactionItem.as_trans_rank()]

    @classmethod
    def as_trans_etc(cls):
        return ["BUY_ITM_CNT", "SEL_ITM_CNT",
                "STK_IN", "STK_OUT",
                "MNY_IN", "MNY_OUT"]


@unique
class AccessItem(Enum):
    """
    <DESCRIPTION>
    ACCESS 데이터를 conn, mts, hts로 구분합니다.
    """
    conn = "CONN_%s"
    mts = "MTS_%s"
    hts = "HTS_%s"

    def as_access_dys(self):
        return self.value % DYS


@unique
class SnapshotItem(Enum):
    """
    <DESCRIPTION>
    SNAPSHOT 데이터를 dmst, ovst, dmetf, ovetf로 구분합니다.
    dmst, ovst, dmetf, ovetf는 fst_buy_ym, fin_buy_ym, tr_months_cnt로 세부 구분됩니다.
    구분되지 않은 기타 데이터는 apy에 저장됩니다.
    """
    fst_buy_ym = "FST_BUY_YM"
    fin_buy_ym = "FIN_BUY_YM"
    tr_months_cnt = "TR_MONTHS_CNT"

    def as_snapshot_dmst(self):
        return DMST % self.value

    def as_snapshot_ovst(self):
        return OVST % self.value

    def as_snapshot_dmetf(self):
        return DMETF % self.value

    def as_snapshot_ovetf(self):
        return OVETF % self.value

    @classmethod
    def as_snapshot_apy(cls):
        return [APY % item for item in ["ITM_CNT", "FIN_YM"]]


@unique
class RatioItem(Enum):
    """
    <DESCRIPTION>
    RATIO 데이터를 day, swing, months, mid, hld, years로 구분합니다.
    """
    day = "DAY_%s"
    swing = "SWING_%s"
    months = "MONTHS_%s"
    mid = "MID_%s"
    hld = "HLD_%s"
    years = "YEARS_%s"

    def as_ratio_tr(self):
        return self.value % TR_RATIO


@unique
class CustomerPool(Enum):
    """
    <DESCRIPTION>
    5가지 카테고리로 구분한 cs 데이터를 하나의 Customer-pool에 군집합니다.
    """
    # MONTH ITEM

    as_month = MonthItem.as_month()
    sub_month = MonthItem.as_sub_month()

    # ASSET ITEM

    dmst_ast_eval = [AssetItem.eval.as_asset_dmst() +
                     month for month in as_month]
    ovst_ast_eval = [AssetItem.eval.as_asset_ovst() +
                     month for month in as_month]

    dmst_ast_pchs = [AssetItem.pchs.as_asset_dmst() +
                     month for month in as_month]
    ovst_ast_pchs = [AssetItem.pchs.as_asset_ovst() +
                     month for month in as_month]

    dmst_ast_itm_rank = [
        dmst_rank + month for dmst_rank in AssetItem.itm.as_asset_dmst_rank() for month in MonthItem.as_month()]
    ovst_ast_itm_rank = [
        ovst_rank + month for ovst_rank in AssetItem.itm.as_asset_ovst_rank() for month in MonthItem.as_month()]

    dmst_ast_mkt_rank = [
        dmst_rank + month for dmst_rank in AssetItem.mkt.as_asset_dmst_rank() for month in MonthItem.as_month()]
    ovst_ast_mkt_rank = [
        ovst_rank + month for ovst_rank in AssetItem.mkt.as_asset_ovst_rank() for month in MonthItem.as_month()]

    dmst_ast_eval_rank = [
        eval_rank + month for eval_rank in AssetItem.eval.as_asset_dmst_rank() for month in MonthItem.as_month()]
    ovst_ast_eval_rank = [
        eval_rank + month for eval_rank in AssetItem.eval.as_asset_ovst_rank() for month in MonthItem.as_month()]

    dmst_ast_pchs_rank = [
        dmst_rank + month for dmst_rank in AssetItem.pchs.as_asset_dmst_rank() for month in MonthItem.as_month()]
    ovst_ast_pchs_rank = [
        ovst_rank + month for ovst_rank in AssetItem.pchs.as_asset_ovst_rank() for month in MonthItem.as_month()]

    ast_etc = [etc + month for etc in AssetItem.as_asset_etc()
               for month in MonthItem.as_month()]

    # TRANSACTION ITEM

    buy_trs_cnt = [TransactionItem.cnt.as_trans_buy() +
                   month for month in sub_month]
    sel_trs_cnt = [TransactionItem.cnt.as_trans_sel() +
                   month for month in sub_month]

    buy_trs_amt = [TransactionItem.amt.as_trans_buy() +
                   month for month in sub_month]
    sel_trs_amt = [TransactionItem.amt.as_trans_sel() +
                   month for month in sub_month]

    buy_trs_itm_rank = [
        itm_rank + month for itm_rank in TransactionItem.itm.as_trans_buy_rank() for month in MonthItem.as_sub_month()]
    sel_trs_itm_rank = [
        itm_rank + month for itm_rank in TransactionItem.itm.as_trans_sel_rank() for month in MonthItem.as_sub_month()]

    buy_trs_mkt_rank = [
        mkt_rank + month for mkt_rank in TransactionItem.mkt.as_trans_buy_rank() for month in MonthItem.as_sub_month()]
    sel_trs_mkt_rank = [
        mkt_rank + month for mkt_rank in TransactionItem.mkt.as_trans_sel_rank() for month in MonthItem.as_sub_month()]

    buy_trs_amt_rank = [
        amt_rank + month for amt_rank in TransactionItem.amt.as_trans_buy_rank() for month in MonthItem.as_sub_month()]
    sel_trs_amt_rank = [
        amt_rank + month for amt_rank in TransactionItem.amt.as_trans_sel_rank() for month in MonthItem.as_sub_month()]

    trs_etc = [
        etc + month for etc in TransactionItem.as_trans_etc() for month in MonthItem.as_sub_month()]

    # ACCESS ITEM

    dys_acs_conn = [AccessItem.conn.as_access_dys() +
                    month for month in sub_month]

    dys_acs_mts = [AccessItem.mts.as_access_dys() +
                   month for month in sub_month]

    dys_acs_hts = [AccessItem.hts.as_access_dys() +
                   month for month in sub_month]

    # SNAPSHOT ITEM

    dmst_temp_fst_buy_ym = SnapshotItem.fst_buy_ym.as_snapshot_dmst()
    ovst_temp_fst_buy_ym = SnapshotItem.fst_buy_ym.as_snapshot_ovst()
    dmetf_temp_fst_buy_ym = SnapshotItem.fst_buy_ym.as_snapshot_dmetf()
    ovetf_temp_fst_buy_ym = SnapshotItem.fst_buy_ym.as_snapshot_ovetf()

    snp_fst_buy_ym = [dmst_temp_fst_buy_ym,
                      ovst_temp_fst_buy_ym,
                      dmetf_temp_fst_buy_ym,
                      ovetf_temp_fst_buy_ym]

    dmst_temp_fin_buy_ym = SnapshotItem.fin_buy_ym.as_snapshot_dmst()
    ovst_temp_fin_buy_ym = SnapshotItem.fin_buy_ym.as_snapshot_ovst()
    dmetf_temp_fin_buy_ym = SnapshotItem.fin_buy_ym.as_snapshot_dmetf()
    ovetf_temp_fin_buy_ym = SnapshotItem.fin_buy_ym.as_snapshot_ovetf()

    snp_fin_buy_ym = [dmst_temp_fin_buy_ym,
                      ovst_temp_fin_buy_ym,
                      dmetf_temp_fin_buy_ym,
                      ovetf_temp_fin_buy_ym]

    dmst_temp_tr_months_cnt = SnapshotItem.tr_months_cnt.as_snapshot_dmst()
    ovst_temp_tr_months_cnt = SnapshotItem.tr_months_cnt.as_snapshot_ovst()
    dmetf_temp_tr_months_cnt = SnapshotItem.tr_months_cnt.as_snapshot_dmetf()
    ovetf_temp_tr_months_cnt = SnapshotItem.tr_months_cnt.as_snapshot_ovetf()

    snp_tr_months_cnt = [dmst_temp_tr_months_cnt,
                         ovst_temp_tr_months_cnt,
                         dmetf_temp_tr_months_cnt,
                         ovetf_temp_tr_months_cnt]

    snp_apy = SnapshotItem.as_snapshot_apy()

    # RATIO ITEM

    tr_temp_day = RatioItem.day.as_ratio_tr()
    tr_temp_swing = RatioItem.swing.as_ratio_tr()
    tr_temp_months = RatioItem.months.as_ratio_tr()
    tr_temp_mid = RatioItem.mid.as_ratio_tr()
    tr_temp_hld = RatioItem.hld.as_ratio_tr()
    tr_temp_years = RatioItem.years.as_ratio_tr()

    rto_item_calendar = [tr_temp_day,
                         tr_temp_months,
                         tr_temp_years]

    rto_item_etc = [tr_temp_swing,
                    tr_temp_mid,
                    tr_temp_hld]


class CustomerHelper(DataHelper):
    """
    <DESCRIPTION>
    5가지 카테고리로 구분한 cs 데이터를 사용자의 요청에 따라 카테고리 별로 로드합니다.
    """

    def __init__(self, item_type: str):
        """
        <item_type>

        ast: ASSET ITEM
        trs: TRANSACTION ITEM
        acs: ACCESS ITEM
        snp: SNAPSHOT ITEM
        rto: RATIO ITEM
        """
        super().__init__()
        self.item_type = item_type

        self.customer = self.run_mirae()[1]
        self.customer = self.customer.replace(
            {'MT_EP_EXIST_YN': {'Y': 1, 'N': 0}})
        self._customer_info = ['AGE_TCD', 'MT_EP_EXIST_YN',
                               'LST_BEST_EA', 'LST_BEST_YM']
        self.customer_info = self.customer[self._customer_info]

    def filter_name(self) -> list:
        """
        <DESCRIPTION>
        cs 데이터에서 item_type에 속한 데이터의 컬럼명을 추출합니다.
        """
        return list(filter(lambda name: self.item_type in name,
                           CustomerPool.__members__.keys()))

    def filter_data(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터에서 item_type에 속한 데이터를 추출합니다.
        """
        filter_name = self.filter_name()
        length = len(filter_name)
        filter_attr = np.concatenate(
            [getattr(CustomerPool, filter_name[i]).value for i in range(length)])
        res = pd.concat(
            [self.customer_info, self.customer[filter_attr]], axis=1)
        return res

    def filter_data_category(self, item_category: str) -> pd.DataFrame:
        """
        <DESCRIPTION>
        filter_data로 추출된 데이터 중, 세부 구분 기준에 부합하는 데이터를 추출합니다.

        <item_category>
        REMINDER: 하단 세부 구분은 대표 예시이며, 사용자의 편의에 따라 조정할 수 있습니다.

        [ast]
        eval, pchs, itm, mkt

        [trs]
        cnt, itm, mkt, amt

        [acs]
        conn, mts, hts

        [snp]
        fst_buy_ym, fin_buy_ym, tr_months_cnt

        [rto]
        day, swing, months, mid, hld, years
        """
        item_category = item_category.upper()
        df = self.filter_data()
        filter_category = [col for col in df.columns if item_category in col]
        res = df[self._customer_info + filter_category]
        return res

    def filter_data_month(self, month: str) -> pd.DataFrame:
        """
        <DESCRIPTION>
        filter_data로 추출된 데이터 중, 세부 월 구분 기준에 부합하는 데이터를 추출합니다.

        <month>

        [ast]
        m1, m2, m3, m4

        [trs, acs]
        m1_1 ~ 4
        m2_1 ~ 4
        m3_1 ~ 4
        m4_1 ~ 4
        """
        month = month.upper()
        df = self.filter_data()
        filter_month = [col for col in df.columns if month in col]
        res = df[self._customer_info + filter_month]
        return res


"""
<DESCRIPTION>
코드 실행 예시입니다.
본 파일(loader_cs.py)를 import하는 코드가 존재하므로, 주석 처리해두었습니다.
"""

# if __name__ == "__main__":
#     # REMINDER: TAKES LONG TIME (3 MIN)

#     helpers = ['ast', 'trs', 'acs', 'snp', 'rto']
#     customer_helpers = [CustomerHelper(helper) for helper in helpers]

#     res = []

#     for helper in customer_helpers:
#         res.append(helper.filter_data())
