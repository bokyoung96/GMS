from sklearn.preprocessing import RobustScaler

from loader_cs import *


# AST DECORATOR


def run_ast(param):
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
        # NOTE: AFTER FILLNA() PROCESS IN get_process.
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

    @run_ast(param='ast')
    def convert_string(self, item_type: str):
        ovst_ast_mkt_rank = CustomerPool.ovst_ast_mkt_rank.value
        df = self.customer_filter.filter(ovst_ast_mkt_rank).apply(
            lambda x: pd.factorize(x)[0] + 1)
        # convert_log = {}
        # for col in df.columns:
        #     convert_log[col] = dict(zip(convert_log[col], convert_log[col]))
        return df

    def get_preprocess(self):
        # TODO: Add static defs.
        scaler = RobustScaler()
        df = self.customer_filter.fillna(0)
        pass
