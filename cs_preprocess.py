"""
TEAM 고객의 미래를 새롭게

코드 목적: 전처리가 완료된 cs 데이터를 AutoEncoder로 분석하기 위해 추가 전처리를 진행합니다.
"""
import pickle
from sklearn.preprocessing import MinMaxScaler

from cs_analysis import *


def temp_strs(item_type: str = 'ast'):
    # TODO: temp_strs should be deleted after ticker is adjusted.
    """
    <DESCRIPTION>
    cs 데이터 TICKER 업데이트 전, string 형태 데이터를 제외하고 분석을 진행하기 위한 temporary func입니다.
    CustomerAnalysis() 클래스 호출 비용이 큰 것을 고려해 동일 함수(finder_cols)를 재구현했습니다.
    ast, trs 카테고리에 활용됩니다.
    """
    if item_type == 'ast':
        strs_itm = {'dmst_ast_itm_rank': CustomerPool.dmst_ast_itm_rank.value,
                    'ovst_ast_itm_rank': CustomerPool.ovst_ast_itm_rank.value}
    elif item_type == 'trs':
        strs_itm = {'buy_trs_itm_rank': CustomerPool.buy_trs_itm_rank.value,
                    'sel_trs_itm_rank': CustomerPool.sel_trs_itm_rank.value}

    def finder_cols(strs: dict):
        res = list(strs.values())
        res = list(chain(*res))
        return res
    res = finder_cols(strs_itm)
    return res


class CustomerPreprocess:
    def __init__(self, item_type: str = 'ast'):
        self.item_type = item_type

    def data_loader(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 카테고리 분류(item_type)에 기반해 로드합니다.
        """
        with open('./res_categories/res_{}.pkl'.format(self.item_type), 'rb') as f:
            data = pickle.load(f)
            print("\n ***** DATA LOADED ***** \n")
        return data

    @staticmethod
    def pp_nans(df: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        데이터의 NaN값을 해당 컬럼의 평균값으로 변환합니다.
        """
        # TODO: Consider data with NaNs considering 0 transactions.
        return df.fillna(df.mean())

    @staticmethod
    def pp_scaling(df: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        데이터에 스케일링을 적용합니다.
        """
        # TODO: Consider whether scaling should be applied in every columns.
        return MinMaxScaler().fit_transform(df)

    @property
    def pp_load(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 카테고리 분류에 기반해 로드하고 전처리를 진행합니다.
        """
        data = self.data_loader()
        data.drop(columns=temp_strs(self.item_type), inplace=True)
        # TODO: temp_strs should be deleted after ticker is adjusted.

        res = self.pp_scaling(self.pp_nans(data))
        return res


"""
<DESCRIPTION>
코드 실행 예시입니다.
본 파일(cs_preprocess.py)를 import하는 코드가 존재하므로, 주석 처리해두었습니다.
"""

# if __name__ == "__main__":
#     pp = CustomerPreprocess(item_type='ast')
#     data = pp.pp_load
