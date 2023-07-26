from sklearn.preprocessing import RobustScaler
from itertools import chain

from loader_cs import *
from cs_learning import *


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

        self.mkts = ['런던', '스위스', '노르웨이', '싱가포르', '런던(USD)', '핀란드', '코스피', '인도네시아', '토론토(USD)', '미국기타', 'OTC', '나스닥', '토론토', '*', '상해A', 'UPCOM',
                     '뉴욕', '코스닥', '심천A', '독일', '홍콩(USD)', 'CSE(바이오큐어)', '호주', '호치민', '코넥스', '동경', '홍콩', '프랑스', '아멕스', '토론토벤처', '그리스', '네덜란드']

        self.tkrs = ['.L', '.S', '.OL', '.SI', '.L', '.H', '.KS', '.JK', '.TO', '.PK^L22', '.PK', '', '.TO', '',
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

    @staticmethod
    def format_float_kr(num_str: str):
        """
        Change strings in KR with .0.
        EX) 005935 & 5935.0: 5935.0 changes to 005935.
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
        Change strings in HK with 5 length & 0 in front.
        EX) 00700 & 002594: 00700 changes to 0700.
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
        cols = CustomerPool.dmst_ast_itm_rank.value
        res = self.customer[cols].applymap(self.format_float_kr)
        return res

    @customer_classificer(param='ast')
    def convert_float_hk(self, item_type: str):
        cols = CustomerPool.ovst_ast_itm_rank.value
        res = self.customer[cols].applymap(self.format_float_hk)
        return res

    @customer_classificer(param='trs')
    def convert_float_trs(self, item_type: str):
        flts = {'buy_trs_itm_rank': CustomerPool.buy_trs_itm_rank.value,
                'sel_trs_itm_rank': CustomerPool.sel_trs_itm_rank.value}
        cols = list(flts.values())
        cols = list(chain(*cols))

        res = self.customer[cols].applymap(self.format_float_kr)
        res = res.applymap(self.format_float_hk)
        return res

    @staticmethod
    def finder_cols(strs: dict):
        res = list(strs.values())
        res = list(chain(*res))
        return res

    def finder_tkrs(self, item_type: str):
        """
        Find mkts and its representative tckrs.
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
        Convert original customer data to analyzable data.
        Change mkt tkrs to its representative form in Eikon API.
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
        return trs, ast, res

    def convert_string_mkt(self):
        """
        One-hot Encoding.
        """
        strs = self.strs_mkts
        strs = list(strs.values())
        strs = list(chain(*strs))

        customer = self.customer.filter(strs)
        dummies = pd.get_dummies(customer[strs])
        return dummies

    # def convert_string_mkt(self):
    #     strs = {'dmst_ast_mkt_rank': CustomerPool.dmst_ast_mkt_rank.value,
    #             'ovst_ast_mkt_rank': CustomerPool.ovst_ast_mkt_rank.value,
    #             'buy_trs_mkt_rank': CustomerPool.buy_trs_mkt_rank.value,
    #             'sel_trs_mkt_rank': CustomerPool.sel_trs_mkt_rank.value}
    #     strs = list(strs.values())
    #     strs = list(chain(*strs))

    #     customer = self.customer.filter(strs)
    #     unique_df = pd.DataFrame(set(customer.stack().values),
    #                              columns=['unique_df']).sort_values(by='unique_df')
    #     unique_df['unique_id'] = pd.factorize(unique_df['unique_df'])[0] + 1
    #     one_hot = to_categorical(unique_df['unique_id'])
    #     unique_df['one_hot'] = pd.DataFrame({'one_hot': one_hot.tolist()})
    #     # unique_df['one_hot'] = unique_df['one_hot'].apply(lambda x: [x])
    #     # convert_log = dict(zip(unique_df['unique_df'], unique_df['one_hot']))

    #     convert_log = dict(zip(unique_df['unique_df'], unique_df['unique_id']))
    #     res = customer.replace(convert_log)
    #     return res, convert_log

    # def get_convert(self):
    #     # NOTE: HIGH TIME-COST
    #     res = self.customer.copy()

    #     convert_float_kr = self.convert_float_kr('ast')
    #     convert_float_hk = self.convert_float_hk('ast')
    #     convert_float_trs = self.convert_float_trs('trs')
    #     convert_string_mkt = self.convert_string_mkt()

    #     res[convert_float_kr.columns] = convert_float_kr
    #     res[convert_float_hk.columns] = convert_float_hk
    #     res[convert_float_trs.columns] = convert_float_trs
    #     # res[convert_string_mkt.columns] = convert_string_mkt
    #     pd.concat([res, convert_string_mkt()])
    #     res.replace({'nan': np.nan}, inplace=True)
    #     return res

    # @customer_classificer(param='ast')
    # def convert_string(self, item_type: str) -> dict:
    #     # NOTE: ITM / high cost to replace.
    #     # NOTE: Different space / vector & index cannot be used together.

    #     strs = {'dmst_ast_mkt_rank': CustomerPool.dmst_ast_mkt_rank.value,
    #             'ovst_ast_mkt_rank': CustomerPool.ovst_ast_mkt_rank.value, }
    #     # 'buy_trs_mkt_rank': CustomerPool.buy_trs_mkt_rank.value,
    #     # 'sel_trs_mkt_rank': CustomerPool.sel_trs_mkt_rank.value}

    #     def converter(name: str) -> pd.DataFrame:
    #         temp = self.customer_filter.filter(strs[name])
    #         unique_df = pd.DataFrame(
    #             set(temp.stack().values), columns=['unique_df'])
    #         unique_df['unique_id'] = pd.factorize(
    #             unique_df['unique_df'])[0] + 1
    #         convert_log = dict(
    #             zip(unique_df['unique_df'], unique_df['unique_id']))
    #         res = temp.replace(convert_log)
    #         return res, convert_log

    #     log = []
    #     for key in strs:
    #         temp, convert_log = converter(key)
    #         cols = strs[key]
    #         self.customer_filter[cols] = temp[cols]
    #         log.append(convert_log)
    #     return log

    @staticmethod
    def get_autoencoder(df: pd.DataFrame, latent_dim: int, num_epochs: int):
        data = torch.Tensor(df.to_numpy())

        input_dim = data.shape[1]
        model = AutoEncoder(input_dim, latent_dim)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            encoded_data, decoded_data = model(data)
            loss = criterion(decoded_data, data)
            loss.backward()
            optimizer.step()

        encoded_data, decoded_data = model(data)
        encoded_data = torch.Tensor(encoded_data).detach().numpy()
        decoded_data = torch.Tensor(decoded_data).detach().numpy()
        return encoded_data, decoded_data

    def get_preprocess(self):
        scaler = RobustScaler()
        nans = self.get_nan(self.customer_filter)
        df = self.customer_filter.drop(nans, axis=1).fillna(0)
        strs = self.get_string(df)

        temp_strs = df.filter(strs, axis=1)
        temp_ints = df.drop(strs, axis=1)

        df_scaled = pd.DataFrame(
            scaler.fit_transform(temp_ints), columns=temp_ints.columns)

        df_encoded = self.get_autoencoder(df_scaled, 3, 100)[0]
        # TODO: K-means by KMeansClustering in cs_analysis.py
        # NOTE: Index does not change. Masking will be in progress by index.
        # NOTE: Use index to reform df including string datas.
        return df_encoded
