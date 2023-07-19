from sklearn.preprocessing import RobustScaler

from loader_cs import *


# AST DECORATOR


def customer_classificer(param):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if args and args[0] == param:
                return func(self, *args, **kwargs)
            else:
                return None
        return wrapper
    return decorator


# CUSTOMER ANALYSIS


class CustomerAnalysis(CustomerHelper):
    def __init__(self, item_type: str = 'ast'):
        super().__init__(item_type)

        self.customer_filter = self.filter_data()
        self.customer_filter = self.customer_filter.replace(
            {'MT_EP_EXIST_YN': {'Y': 1, 'N': 0}})

    @staticmethod
    def get_nan(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        nans = df.isna().sum() / len(df)
        res = nans[nans > threshold].index
        return res

    @staticmethod
    def get_string(df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: AFTER FILLNA() PROCESS IN get_preprocess.
        strings = df.apply(lambda x: any(isinstance(val, str) for val in x))
        res = df.columns[strings].tolist()
        return res

    @staticmethod
    def extract_string(df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: ast / ITM, MKT (str type)
        strings = CustomerAnalysis.get_string(df)
        temp = []
        for val in strings:
            elements = df[val].dropna().astype(str).unique()
            temp.append([val, elements])
        res = pd.DataFrame(temp, columns=['Val', 'Elements'])
        return res

    @customer_classificer(param='ast')
    def convert_string(self, item_type: str):
        # strs = {'dmst_ast_itm_rank': CustomerPool.dmst_ast_itm_rank.value,
        #         'ovst_ast_itm_rank': CustomerPool.ovst_ast_itm_rank.value,
        #         'dmst_ast_mkt_rank': CustomerPool.dmst_ast_mkt_rank.value,
        #         'ovst_ast_mkt_rank': CustomerPool.ovst_ast_mkt_rank.value}
        # NOTE: itm / high cost to replace; change to integer.

        strs = {'dmst_ast_mkt_rank': CustomerPool.dmst_ast_mkt_rank.value,
                'ovst_ast_mkt_rank': CustomerPool.ovst_ast_mkt_rank.value}

        def converter(name: str) -> pd.DataFrame:
            temp = self.customer_filter.filter(strs[name])
            unique_df = pd.DataFrame(
                set(temp.stack().values), columns=['unique_df'])
            unique_df['unique_id'] = pd.factorize(
                unique_df['unique_df'])[0] + 1
            convert_log = dict(
                zip(unique_df['unique_df'], unique_df['unique_id']))
            res = temp.replace(convert_log)
            return res, convert_log

        log = []
        for key in strs:
            temp, convert_log = converter(key)
            cols = strs[key]
            self.customer_filter[cols] = temp[cols]
            log.append(convert_log)
        return log

    def get_preprocess(self):
        # TODO: Add static defs.
        scaler = RobustScaler()
        df = self.customer_filter.fillna(0)
        pass
