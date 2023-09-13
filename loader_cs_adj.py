"""
TEAM 고객의 미래를 새롭게

코드 목적: 전처리가 완료된 cs 데이터를 AutoEncoder에 대입하기 위해 5가지 카테고리로 재분류합니다.

세부 사항:
cs 데이터는 기존 520개 컬럼에서 전처리 이후 1666개 컬럼으로 증가했습니다. 
추가/제거된 컬럼들을 고려해 재분류할 것입니다.
"""

import pickle
import os

from cs_analysis import *


dir_name = "./res_categories/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


class CustomerLoader(CustomerAnalysis):
    """
    <DESCRIPTION>
    cs 데이터의 카테고리 재분류를 진행합니다.
    """

    def __init__(self, item_type: str = 'ast'):
        super().__init__(item_type)
        self.cols_mkt = self.convert_string_mkt().columns.tolist()
        with open('res.pkl', 'rb') as f:
            self.data = pickle.load(f)
            print("\n ***** DATA LOADED *****")

    def get_category(self, item_type: str):
        """
        <DESCRIPTION>
        cs 데이터를 카테고리별로 구분한 다음, 각 카테고리에 해당하는 컬럼들을 list로 추출합니다.
        """
        category_values = [val.value for name,
                           val in CustomerPool.__members__.items() if item_type in name]
        res = np.concatenate(category_values).tolist()
        return res

    def get_ast(self):
        """
        <DESCRIPTION>
        ast 카테고리에 해당하는 컬럼들을 list로 추출합니다.
        기존 124개의 컬럼에서 296개의 One-Hot Encoding mkt 컬럼을 추가합니다.
        이후, One-Hot Encoding에 활용된 24개의 string 형태 mkt 컬럼을 제거합니다.
        최종 개수: 396개입니다.
        추가 사항: Ticker 변동(Ticker에서 재무지표 데이터로) 전입니다. cs_ticker.py에서 변동됩니다.
        """
        cols_mkt = [col for col in self.cols_mkt if 'AST' in col]
        cols_pool = self.get_category('ast')
        cols_pool_adj = list(
            set(cols_pool) - set(self.finder_cols(self.strs_mkts)))
        res = cols_mkt + cols_pool_adj
        return res

    def get_trs(self):
        """
        <DESCRIPTION>
        trs 카테고리에 해당하는 컬럼들을 list로 추출합니다.
        기존 336개의 컬럼에서 950개의 One-Hot Encoding mkt 컬럼을 추가합니다.
        이후, One-Hot Encoding에서 활용된 72개의 string 형태 mkt 컬럼을 제거합니다.
        최종 개수: 1214개입니다.
        추가 사항: Ticker 변동(Ticker에서 재무지표 데이터로) 전입니다. cs_ticker.py에서 변동됩니다.
        """
        cols_mkt = [col for col in self.cols_mkt if any(
            keyword in col for keyword in ['BUY', 'SEL'])]
        cols_pool = self.get_category('trs')
        cols_pool_adj = list(
            set(cols_pool) - set(self.finder_cols(self.strs_mkts)))
        res = cols_mkt + cols_pool_adj
        return res

    def get_acs(self):
        """
        <DESCRIPTION>
        acs 카테고리에 해당하는 컬럼들을 list로 추출합니다.
        기존의 컬럼과 변동 없습니다.
        최종 개수: 36개입니다.
        """
        res = self.get_category('acs')
        return res

    def get_snp(self):
        """
        <DESCRIPTION>
        snp 카테고리에 해당하는 컬럼들을 list로 추출합니다.
        기존 17개(18개 - 1개, cols_info)의 컬럼에서 6개의 날짜 차이를 계산한 컬럼을 추가합니다.
        이후, 날짜 차이를 계산한 8개(9개 - 1개, cols_info)의 string 형태 컬럼을 제거합니다.
        최종 개수: 14개입니다.

        * Computing time을 감축시키기 위해 컬럼명을 직접 작성했습니다.
        * cols_info는 CustomerHelper에서 모든 카테고리에 공통으로 포함한 컬럼입니다.
        * cols_info에서 LST_BEST_YM은 사전 제외하였습니다.
        """
        cols_date = ['DMST_BUY_YM_DIFF', 'OVST_BUY_YM_DIFF',
                     'DMETF_BUY_YM_DIFF', 'OVETF_BUY_YM_DIFF']
        cols_etc = ['LST_BEST_YM_DIFF', 'APY_FIN_YM_DIFF']
        cols_info = ['AGE_TCD', 'MT_EP_EXIST_YN', 'LST_BEST_EA']
        cols_pool = self.get_category('snp')
        cols_pool_adj = list(
            set(cols_pool) - set(self.finder_cols(self.date_snp) + self.etc_snp))
        res = cols_date + cols_etc + cols_info + cols_pool_adj
        return res

    def get_rto(self):
        """
        <DESCRIPTION>
        rto 카테고리에 해당하는 컬럼들을 list로 추출합니다.
        기존의 컬럼과 변동 없습니다.
        최종 개수: 6개입니다.
        """
        res = self.get_category('rto')
        return res

    def get_all(self, save_pkl: str = 'N'):
        categories = ['ast', 'trs', 'acs', 'snp', 'rto']
        res = []
        res_flatten = []

        for category in categories:
            method_name = f'get_{category}'
            method = getattr(self, method_name, None)
            if method:
                res.append(method())
                res_flatten.extend(method())
                if save_pkl == 'Y':
                    self.data[method()].to_pickle(
                        './res_categories/res_{}.pkl'.format(category))
                    print("\n ***** res_{}.pkl SAVED *****".format(category))
                else:
                    pass
        return res, res_flatten


"""
<DESCRIPTION>
코드 실행 예시입니다.
하단 코드는 pickle 파일을 생성하므로, 주석 처리해두었습니다.
"""

# if __name__ == "__main__":
#     loader = CustomerLoader()
#     res = loader.get_all(save_pkl='N')[0]
#     print("\n ***** CATEGORY DIVIDED ***** \n")
#     print(res)
