"""
TEAM 고객의 미래를 새롭게

코드 목적: cs 데이터의 전처리를 진행합니다. 이와 관련한 자세한 내용은 개발기획서에 기록되어 있습니다.
"""

import pickle
from itertools import chain
from datetime import datetime
from dateutil.relativedelta import relativedelta

from loader_cs import *


def customer_classificer(param):
    """
    <DESCRIPTION>
    하단 CustomerAnalysis 클래스에서 cs 데이터의 5가지 카테고리 선택 여부에 따라 코드 실행 유무를 결정할 수 있도록 구현한 데코레이터입니다.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if args and args[0] == param:
                return func(self, *args, **kwargs)
            else:
                return None
        return wrapper
    return decorator


class CustomerAnalysis(CustomerHelper):
    """
    <DESCRIPTION>
    cs 데이터의 전처리를 진행합니다.
    """

    def __init__(self, item_type: str = 'ast'):
        super().__init__(item_type)

        self.customer_filter = self.filter_data()
        self.customer_filter = self.customer_filter.replace(
            {'MT_EP_EXIST_YN': {'Y': 1, 'N': 0}})

        self.mkts = ['런던', '스위스', '노르웨이', '싱가포르', '런던(USD)', '핀란드', '코스피', '인도네시아', '토론토(USD)', '미국기타', 'OTC', '나스닥', '토론토', '*', '상해A', 'UPCOM',
                     '뉴욕', '코스닥', '심천A', '독일', '홍콩(USD)', 'CSE(바이오큐어)', '호주', '호치민', '코넥스', '동경', '홍콩', '프랑스', '아멕스', '토론토벤처', '그리스', '네덜란드']

        self.tkrs = ['.L', '.S', '.OL', '.SI', '.L', '.H', '.KS', '.JK', '.TO', '.PK^L22', '.PK', '.O', '.TO', '',
                     '.SS', '.HNO', '', '.KQ', '.SZ', '.DE', '.HK', '.K', '.AX', '.HM', '.KN', '.T', '.HK',
                     '.PA', '.K', '.V', '.AT^L14', '.AS']

        self.strs_mkts = {'dmst_ast_mkt_rank': CustomerPool.dmst_ast_mkt_rank.value,
                          'ovst_ast_mkt_rank': CustomerPool.ovst_ast_mkt_rank.value,
                          'buy_trs_mkt_rank': CustomerPool.buy_trs_mkt_rank.value,
                          'sel_trs_mkt_rank': CustomerPool.sel_trs_mkt_rank.value}

        self.strs_trs = {'buy_trs_itm_rank': CustomerPool.buy_trs_itm_rank.value,
                         'sel_trs_itm_rank': CustomerPool.sel_trs_itm_rank.value,
                         'buy_trs_mkt_rank': CustomerPool.buy_trs_mkt_rank.value,
                         'sel_trs_mkt_rank': CustomerPool.sel_trs_mkt_rank.value}

        self.strs_ast = {'dmst_ast_itm_rank': CustomerPool.dmst_ast_itm_rank.value,
                         'ovst_ast_itm_rank': CustomerPool.ovst_ast_itm_rank.value,
                         'dmst_ast_mkt_rank': CustomerPool.dmst_ast_mkt_rank.value,
                         'ovst_ast_mkt_rank': CustomerPool.ovst_ast_mkt_rank.value}

        self.date_snp = {'snp_fst_buy_ym': CustomerPool.snp_fst_buy_ym.value,
                         'snp_fin_buy_ym': CustomerPool.snp_fin_buy_ym.value}

        self.etc_snp = ['LST_BEST_YM', 'APY_FIN_YM']

    @staticmethod
    def get_nan(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        <DESCRIPTION>
        데이터의 컬럼 중 NaN을 threshold 이상으로 보유하고 있는 컬럼은 이상치로 판단하여 제거합니다.
        """
        nans = df.isna().sum() / len(df)
        res = nans[nans > threshold].index
        return res

    @staticmethod
    def get_string(df: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        데이터의 컬럼 중 string 데이터를 보유한 컬럼명을 추출합니다.
        """
        # NOTE: AFTER FILLNA() PROCESS IN get_preprocess.
        strings = df.apply(lambda x: any(isinstance(val, str) for val in x))
        res = df.columns[strings].tolist()
        return res

    @staticmethod
    def extract_string(df: pd.DataFrame) -> pd.DataFrame:
        """
        <DESCRIPTION>
        데이터의 컬럼 중 string 데이터를 보유한 컬럼을 추출합니다.
        컬럼명(Val), 컬럼에 속한 데이터(Elements) 형태로 추출됩니다.
        """
        strings = CustomerAnalysis.get_string(df)

        temp = []
        for val in strings:
            elements = df[val].dropna().astype(str).unique()
            temp.append([val, elements])

        res = pd.DataFrame(temp, columns=['Val', 'Elements'])
        return res

    @staticmethod
    def format_float_kr(num_str: str):
        """
        <DESCRIPTION>
        KR 종목번호 중 .0으로 오기입된 종목번호를 변경 표기합니다.
        EX) 005935 & 5935.0: 5935.0에서 005935로 변경됩니다.
        """
        try:
            num = float(num_str)
            num_str = str(num)
            ints, decs = num_str.split(".")
            ints = ints.zfill(6)
            return ints
        except ValueError:
            return num_str

    @staticmethod
    def format_float_hk(num_str: str):
        """
        <DESCRIPTION>
        HK 종목번호 중 총 글자수 5개 & 0으로 시작하는 오기입된 종목번호를 변경 표기합니다.
        EX) 00700 & 002594: 00700에서 0700로 변경됩니다..
        """
        try:
            if len(num_str) == 5 and (num_str.startswith('00') or num_str.startswith('0')):
                return num_str[1:]
            else:
                return num_str
        except (TypeError, AttributeError):
            return num_str

    @customer_classificer(param='ast')
    def convert_float_kr(self, item_type: str):
        """
        <DESCRIPTION>
        format_float_kr을 ast에 속한 특정 컬럼에 적용합니다.
        """
        cols = CustomerPool.dmst_ast_itm_rank.value
        res = self.customer[cols].applymap(self.format_float_kr)
        return res

    @customer_classificer(param='ast')
    def convert_float_hk(self, item_type: str):
        """
        <DESCRIPTION>
        format_float_hk을 ast에 속한 특정 컬럼에 적용합니다.
        """
        cols = CustomerPool.ovst_ast_itm_rank.value
        res = self.customer[cols].applymap(self.format_float_hk)
        return res

    @customer_classificer(param='trs')
    def convert_float_trs(self, item_type: str):
        """
        <DESCRIPTION>
        format_float_hk을 trs에 속한 특정 컬럼에 적용합니다.
        """
        flts = {'buy_trs_itm_rank': CustomerPool.buy_trs_itm_rank.value,
                'sel_trs_itm_rank': CustomerPool.sel_trs_itm_rank.value}
        cols = self.finder_cols(flts)

        res = self.customer[cols].applymap(self.format_float_kr)
        res = res.applymap(self.format_float_hk)
        return res

    def convert_string_mkt(self):
        """
        <DESCRIPTION>
        시장 정보 데이터에 One-Hot Encoding을 적용하여 정수형 데이터로 변환합니다.
        """
        strs = self.strs_mkts
        strs = self.finder_cols(strs)

        customer = self.customer.filter(strs)
        dummies = pd.get_dummies(customer[strs])
        return dummies

    @customer_classificer(param='snp')
    def convert_date_snp(self, item_type: str):
        """
        <DESCRIPTION>
        타임스탬프형 데이터 중 DMST, OVST, DMETF, OVETF의 fst_buy_ym, fin_buy_ym간 날짜 차이(월)를 계산해 정수형 데이터로 변환합니다.
        """
        date = self.finder_cols(self.date_snp)
        df = self.customer.filter(date)

        matched_df = self.split_dfs(df)
        for col in matched_df.columns:
            matched_df[col] = pd.to_datetime(matched_df[col], format="%Y%m")

        def calculate_month_difference(row):
            if pd.isnull(row[0]) or pd.isnull(row[1]):
                return None
            diff = relativedelta(row[1], row[0])
            return diff.years * 12 + diff.months

        date_diff = []
        for col in range(0, len(matched_df.columns)-1, 2):
            temp = matched_df.iloc[:, col:col+2]
            diff = temp.apply(calculate_month_difference, axis=1)
            date_diff.append(diff)

        date_diff_df = pd.DataFrame(date_diff).T
        date_diff_df.columns = ['DMST_BUY_YM_DIFF', 'OVST_BUY_YM_DIFF',
                                'DMETF_BUY_YM_DIFF', 'OVETF_BUY_YM_DIFF']
        return date_diff_df

    @customer_classificer(param='snp')
    def convert_etc_snp(self, itme_type: str):
        """
        <DESCRIPTION>
        타임스탬프형 데이터 중 lst_best_ym과 apy_fin_ym을 특정 조건에 따라 정수형 데이터로 변환합니다.
        lst_best_ym: 202212까지 날짜 차이(월) 계산
        apy_fin_ym: 201912부터 날짜 차이(월) 계산
        """
        df = self.customer.filter(self.etc_snp)
        for col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y%m')

        base_1 = datetime.strptime('202212', '%Y%m')
        base_2 = datetime.strptime('201912', '%Y%m')

        def calculate_month_difference(df_parsed: pd.Series, base: datetime):
            res = []
            for date in df_parsed:
                if pd.notnull(date):
                    diff = relativedelta(base, date)
                    res.append(diff.years * 12 + diff.months)
                else:
                    res.append(None)
            return res

        base_1_diff = calculate_month_difference(df.iloc[:, 0], base_1)
        base_2_diff = calculate_month_difference(df.iloc[:, 1], base_2)

        etc_diff_df = pd.DataFrame([base_1_diff, base_2_diff]).T
        etc_diff_df.columns = ['LST_BEST_YM_DIFF', 'APY_FIN_YM_DIFF']

        etc_diff_df['APY_FIN_YM_DIFF'] = etc_diff_df['APY_FIN_YM_DIFF'] * -1
        etc_diff_df.replace(-0, 0, inplace=True)
        return etc_diff_df

    @staticmethod
    def finder_cols(strs: dict):
        """
        dict 형태의 데이터를 list로 변환합니다.
        """
        res = list(strs.values())
        res = list(chain(*res))
        return res

    def finder_tkrs(self, item_type: str):
        """
        시장 정보 데이터와 이를 대표하는 티커를 산출합니다.
        Eikon Refinitiv, Bloomberg를 사용하기 위한 선행 코드입니다.
        """
        if item_type == 'trs':
            strs = self.strs_trs
        elif item_type == 'ast':
            strs = self.strs_ast
        strs = self.finder_cols(strs)

        df = self.customer.filter(strs)
        df_itm = df[df.columns[df.columns.str.contains(r'_ITM_', regex=True)]]
        df_mkt = df[df.columns[df.columns.str.contains(r'_MKT_', regex=True)]]

        res = pd.concat([df_itm.iloc[:, i] for i in range(
            len(df_itm.columns))], axis=0).reset_index(drop=True)
        res = pd.concat([res, pd.concat([df_mkt.iloc[:, i] for i in range(
            len(df_mkt.columns))], axis=0).reset_index(drop=True)], axis=1).dropna()
        res.columns = ['Ticker', 'Mkt']

        exs = []
        for mkt in self.mkts:
            try:
                exs.append(res[res['Mkt'] == mkt].iloc[0])
            except IndexError:
                pass
        return res, exs

    @staticmethod
    def split_dfs(df: pd.DataFrame):
        """
        하나의 데이터프레임을 두개로 구분, 이후 열 단위(1열:1열, 2열:2열 등)로 매칭하여 새로운 데이터프레임으로 추출합니다.
        """
        pd.options.mode.chained_assignment = None
        middle_col = len(df.columns) // 2
        df1 = df.iloc[:, :middle_col]
        df2 = df.iloc[:, middle_col:]

        matched_df = pd.DataFrame()
        for i in range(middle_col):
            matched_df[df1.columns[i]] = df1.iloc[:, i]
            matched_df[df2.columns[i]] = df2.iloc[:, i]
        return matched_df

    def get_convert(self):
        """
        cs 데이터를 전처리 과정을 통해 분석 가능한 정수형 데이터로 변환합니다.
        종목 번호 데이터는 Eikon Refinitiv와 Bloomberg에서 간편히 검색할 수 있도록 변환합니다.
        """
        res = self.customer.copy()

        convert_float_kr = self.convert_float_kr('ast')
        convert_float_hk = self.convert_float_hk('ast')
        convert_float_trs = self.convert_float_trs('trs')

        res[convert_float_kr.columns] = convert_float_kr
        res[convert_float_hk.columns] = convert_float_hk
        res[convert_float_trs.columns] = convert_float_trs
        res.replace({'nan': np.nan}, inplace=True)

        mkts_tkrs = dict(zip(self.mkts, self.tkrs))

        def post_tkrs(item_type: str):
            # NOTE: HIGH COST
            if item_type == 'trs':
                strs = self.strs_trs
            elif item_type == 'ast':
                strs = self.strs_ast

            strs = self.finder_cols(strs)
            df = res.filter(strs)

            matched_df = self.split_dfs(df)
            matched_df.replace(mkts_tkrs, inplace=True)
            mathced_cols = list(range(0, len(matched_df.columns), 2))

            for col in mathced_cols:
                col1 = matched_df.columns[col]
                col2 = matched_df.columns[col+1]
                matched_df[col1] = matched_df[col1] + matched_df[col2]

            res_df = matched_df[matched_df.columns[matched_df.columns.str.contains(
                r'_ITM_', regex=True)]]
            return res_df

        trs = post_tkrs('trs')
        ast = post_tkrs('ast')

        res[trs.columns] = trs
        res[ast.columns] = ast

        one_hot_encoding = self.convert_string_mkt()
        date_diff_df = self.convert_date_snp('snp')
        etc_diff_df = self.convert_etc_snp('snp')

        res.drop(self.finder_cols(self.strs_mkts), axis=1, inplace=True)
        res.drop(self.finder_cols(self.date_snp) +
                 self.etc_snp, axis=1, inplace=True)

        res = pd.concat(
            [res, one_hot_encoding, date_diff_df, etc_diff_df], axis=1)
        return res


"""
<DESCRIPTION>
코드 실행 예시입니다.
본 파일(cs_analysis.py)를 import하는 코드가 존재하므로, 주석 처리해두었습니다.
"""

# if __name__ == "__main__":
#     # REMINDER: TAKES LONG TIME (10 MIN+)

#     customer_analysis = CustomerAnalysis()

#     res = customer_analysis.get_convert()
#     res.to_pickle('res.pkl')

#     print(res)
