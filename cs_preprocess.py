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
        One-Hot Encoding이 적용된 데이터는 스케일링이 진행되지 않습니다.
        세부 사항은 하단 pp_load()에서 다루어집니다.
        """
        return MinMaxScaler().fit_transform(df)

    @property
    def pp_load(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 카테고리 분류에 기반해 로드하고 전처리를 진행합니다.
        """
        data = self.data_loader()

        customer_analysis = CustomerAnalysis()
        encoding_data = customer_analysis.convert_string_mkt()
        encoding_cols = encoding_data.columns.tolist()

        state = any(item == self.item_type for item in ['ast', 'trs'])
        if state:
            drop_cols = [col for col in encoding_cols if col in data.columns]
            data.drop(drop_cols, axis=1, inplace=True)
            data_cols = data.columns
            temp = encoding_data[drop_cols]

            res = self.pp_scaling(self.pp_nans(data))
            res = pd.DataFrame(res, columns=data_cols)
            res = pd.concat([res, temp], axis=1)
        else:
            data_cols = data.columns
            res = self.pp_scaling(self.pp_nans(data))
            res = pd.DataFrame(res, columns=data_cols)
        return res


"""
<DESCRIPTION>
코드 실행 예시입니다.
본 파일(cs_preprocess.py)를 import하는 코드가 존재하므로, 주석 처리해두었습니다.
"""

if __name__ == "__main__":
    item_type = ['ast', 'trs', 'acs', 'snp', 'rto']

    for item in item_type:
        pp = CustomerPreprocess(item_type=item)
        data = pp.pp_load
        data.to_pickle('./res_pp_categories/res_pp_{}.pkl'.format(item))
        print("\n ***** res_pp_{}.pkl SAVED *****".format(item))
    print("***** TASK COMPLETED *****")
