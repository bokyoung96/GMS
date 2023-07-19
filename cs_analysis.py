from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

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
    def convert_string(self, item_type: str) -> dict:
        # NOTE: ITM / high cost to replace.
        # NOTE: Different space / vector & index cannot be used together.

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
        # TODO: Is train, test required in autoencoder? Unsupervised learning?
        # df_train, df_test = train_test_split(
        #     df_scaled, test_size=0.2, random_state=42)

        df_encoded = self.get_autoencoder(df_scaled, 3, 100)[0]
        # TODO: K-means by KMeansClustering in cs_analysis.py
        # NOTE: Index does not change. Masking will be in progress by index.
        # NOTE: Use index to reform df including string datas.
        return
