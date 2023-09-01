"""
TEAM 고객의 미래를 새롭게

코드 목적: 전처리가 완료된 cs 데이터를 AutoEncoder로 분석하기 위해 추가 전처리를 진행합니다.
"""
import pickle
from sklearn.preprocessing import MinMaxScaler

from cs_analysis import *


class CustomerPreprocess:
    def __init__(self, item_type: str = 'ast'):
        self.item_type = item_type

    def data_loader(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 카테고리 분류(item_type)에 기반해 로드합니다.
        CustomerTicker와 같은 함수명을 공유하나, res_{}_adj.pkl을 포함하는 차이가 존재합니다.
        """
        state = any(item == self.item_type for item in ['ast', 'trs'])

        if state:
            with open('./res_categories/res_{}_adj.pkl'.format(self.item_type), 'rb') as f:
                data = pickle.load(f)
        else:
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
