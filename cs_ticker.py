"""
TEAM 고객의 미래를 새롭게

코드 목적: cs 데이터의 ticker를 숫자 데이터로 변환합니다.

세부 사항:
cs 데이터는 기존 1666개 컬럼에서 Ticker 전처리 이후 2242개 컬럼으로 증가했습니다. 
"""
import re
import pickle
import warnings
import pandas as pd
from tqdm import tqdm
from itertools import chain

from loader_cs import CustomerPool


class CustomerTicker:
    def __init__(self, item_type: str = 'ast'):
        self.item_type = item_type
        self.data_eikon_tkrs = pd.read_csv('./cs_ticker_eikon.csv')

    def data_loader(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 카테고리 분류(item_type)에 기반해 로드합니다.
        """
        with open('./res_categories/res_{}.pkl'.format(self.item_type), 'rb') as f:
            data = pickle.load(f)
            print("\n ***** DATA LOADED ***** \n")
        return data

    def data_eikon_loader(self, *args) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Eikon Refinitiv API를 활용해 각 ticker의 재무지표 데이터를 로드합니다.

        <* args>
        데이터 중 필요한 컬럼만 로드합니다.
        Ticker, Period는 필수로 요구됩니다.
        """
        df = self.data_eikon_tkrs[list(args)]
        df['Ticker'] = df[['Ticker', 'Period']].apply(
            lambda x: '_'.join(x), axis=1)
        df.drop('Period', axis=1, inplace=True)
        df.drop_duplicates(subset='Ticker', keep='first', inplace=True)
        return df

    def temp_strs(self):
        """
        <DESCRIPTION>
        cs 데이터 TICKER 업데이트 전, string 형태 데이터를 제외하고 분석을 진행하기 위한 temporary func입니다.
        CustomerAnalysis() 클래스 호출 비용이 큰 것을 고려해 동일 함수(finder_cols)를 재구현했습니다.
        ast, trs 카테고리에 활용됩니다.
        """
        if self.item_type == 'ast':
            strs_itm = {'dmst_ast_itm_rank': CustomerPool.dmst_ast_itm_rank.value,
                        'ovst_ast_itm_rank': CustomerPool.ovst_ast_itm_rank.value}
        elif self.item_type == 'trs':
            strs_itm = {'buy_trs_itm_rank': CustomerPool.buy_trs_itm_rank.value,
                        'sel_trs_itm_rank': CustomerPool.sel_trs_itm_rank.value}

        def finder_cols(strs: dict):
            res = list(strs.values())
            res = list(chain(*res))
            return res
        res = finder_cols(strs_itm)
        return res

    @staticmethod
    def tkr_split(df: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        xxx.xx의 형태를 갖는 데이터를 xxx.xx_Mx의 형태로 변환합니다.
        trs 데이터의 경우, Mx_x 컬럼에 속한 xxx.xx 형태의 데이터를 xxx.xx_Mx의 형태로 변환합니다.
        """
        pattern = r'_M\d+'
        for col in df.columns:
            period = re.search(pattern, col).group()
            df[col] = df[col].apply(lambda x: str(
                x) + period if not pd.isna(x) else x)
        return df

    def tkr_chg(self, arg: str = 'Mktcap') -> pd.DataFrame:
        """
        <DESCRIPTION>
        ast, trs 데이터의 ticker와 Eikon Refinitiv API에서 다운로드한 ticker를 매치합니다.
        즉, ast, trs 데이터의 ticker를 재무지표 데이터로 변환합니다.
        """
        temp_tkrs = self.data_loader()
        data_tkrs = temp_tkrs[self.temp_strs()]
        data_tkrs = self.tkr_split(data_tkrs)

        eikon_tkrs = self.data_eikon_loader('Ticker', 'Period', arg)

        for col in data_tkrs.columns:
            res = pd.merge(data_tkrs, eikon_tkrs, left_on=col,
                           right_on='Ticker', how='left')
            data_tkrs[col] = res[arg]
        data_tkrs.columns = [col + '_' + arg for col in data_tkrs.columns]
        return data_tkrs

    def tkr_unite(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        tkr_chg를 모든 재무지표 데이터에 적용합니다.
        """
        temp = []
        args = self.data_eikon_tkrs.columns.drop(
            ['Ticker', 'Ret', 'Date', 'Period'])

        # NOTE: HIGH COST IN trs.
        warnings.filterwarnings("ignore")
        for arg in tqdm(args):
            data_tkrs = self.tkr_chg(arg=arg)
            temp.append(data_tkrs)
            print("\n ***** {} chg finished. Moving on... ***** \n".format(arg))
        warnings.filterwarnings("default")

        print("\n ***** UNITING... ***** \n")
        res = pd.concat(temp, axis=1)
        print("***** TASK COMPLETED *****")
        return res

    def data_transfer(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        ast, trs 데이터에서 ticker 데이터를 제거하고, tkr_unite으로 변환된 각 ticker별 재무지표 데이터를 추가합니다.
        ast 최종 개수: 540개입니다.
        trs 최종 개수: 1646개입니다.
        """
        temp_tkrs = self.data_loader().drop(self.temp_strs(), axis=1)
        res = pd.concat([temp_tkrs, self.tkr_unite()], axis=1)
        return res


"""
<DESCRIPTION>
코드 실행 예시입니다.
하단 코드는 pickle 파일을 생성하므로, 주석 처리해두었습니다.
"""

# if __name__ == "__main__":
#     item_type = ['ast', 'trs']

#     for item in item_type:
#         customer_ticker = CustomerTicker(item_type=item)
#         res = customer_ticker.data_transfer()
#         res.to_pickle('./res_categories/res_{}_adj.pkl'.format(item))
#         print("\n ***** res_{}_adj.pkl SAVED *****".format(item))
