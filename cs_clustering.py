"""
TEAM 고객의 미래를 새롭게

코드 목적: AutoEncoder로 차원이 축소된 cs 데이터에 군집을 부여합니다.
"""
from tqdm import tqdm
from sklearn.cluster import KMeans

from cs_autoencoder import *


class AELoader(AutoEncoder):
    def __init__(self,
                 item_type: str = 'ast',
                 test_size: int = 0.2,
                 epochs: int = 100):
        super().__init__(item_type, test_size, epochs)

    @property
    def model_loader(self):
        """
        <DESCRIPTION>
        AutoEncoder로 차원이 축소된 cs 데이터를 카테고리 분류(item_type)에 기반해 로드합니다.
        """
        return load_model('./cs_ae_callback/cs_autoencoder_callback_{}.hdf5'.format(self.item_type))

    @property
    def latent_layer_loader(self):
        """
        <DESCRIPTION>
        저장된 AutoEncoder의 latent layer를 로드합니다.
        """
        model = self.model_loader
        min_output_shape = float('inf')
        for layer in model.layers:
            if isinstance(layer, Dense):
                output_shape = layer.output_shape
                if output_shape[1] < min_output_shape:
                    min_output_shape = output_shape[1]
                    min_output_layer = layer
        latent_layer_model = Model(inputs=model.input,
                                   outputs=min_output_layer.output)
        return latent_layer_model

    def dim_reduction_loader(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        Latent layer를 바탕으로 차원이 축소된 cs 데이터를 로드합니다.
        """
        data = self.data_loader
        latent_layer_model = self.latent_layer_loader
        return latent_layer_model.predict(data)


class KMLoader:
    def __init__(self):
        pass

    @staticmethod
    def dim_combination_loader() -> pd.DataFrame:
        """
        <DESCRIPTION>
        차원이 축소된 각 카테고리별 cs 데이터를 하나의 데이터프레임으로 결합 및 로드합니다.
        trs는 용량이 커, 두번에 나누어 본 과정을 진행합니다.
        """
        items = ['ast', 'acs', 'snp', 'rto']

        dfs = []
        for item in items:
            loader = AELoader(item_type=item)
            df = pd.DataFrame(loader.dim_reduction_loader())
            df.columns = [str(col) + '_{}'.format(item) for col in df.columns]
            dfs.append(df)
        res = pd.concat(dfs, axis=1)

        # FOR TRS CATEGORY
        loader = AELoader(item_type='trs')
        latent_layer_model = loader.latent_layer_loader

        data = pd.read_pickle("./res_pp_categories/res_pp_trs.pkl")
        print("\n ***** DATA CATEGORY trs LOADED ***** \n")
        split_idx = len(data) // 2
        data_1 = data[:split_idx]
        data_2 = data[split_idx:]

        df_1 = pd.DataFrame(latent_layer_model.predict(data_1))
        df_2 = pd.DataFrame(latent_layer_model.predict(data_2))
        temp = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
        temp.columns = [str(col) + '_{}'.format('trs') for col in df_1.columns]

        res = pd.concat([res, temp], axis=1)
        return res

    def km_n_clusters(self) -> list:
        """
        <DESCRIPTION>
        K-Means Inertia method를 기반으로 최적 군집 개수를 찾습니다.
        """
        df = self.dim_combination_loader()

        score = []
        for n_clusters in tqdm(range(1, 11)):
            kmeans = KMeans(n_clusters=n_clusters,
                            init="k-means++",
                            random_state=42)
            kmeans.fit(df)
            score.append(kmeans.inertia_)

        plt.figure(figsize=(15, 5))
        plt.plot(range(1, 11), score)
        plt.title('K-Means: The Elbow Method (Inertia)')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        plt.grid()
        plt.plot()
        return score

    def km_run(self, n_clusters: int = 5):
        """
        <DESCRIPTION>
        최적 군집 개수에 기반해 K-Means 군집 분석을 진행합니다.
        """
        df = self.dim_combination_loader()
        model = KMeans(n_clusters=n_clusters, init='k-means++',
                       random_state=42).fit(df)
        df['LABEL'] = model.labels_
        return df, model

    def km_example_plot(self, n_clusters: int = 5):
        """
        <DESCRIPTION>
        본 데이터로 진행하는 K-Means 군집 분석은 15차원으로, 그래프로 표현할 수 없습니다.
        이에 따라, 예시로 카테고리 중 ast 항목에 대한 K-Means를 진행하고 그래프로 나타냅니다.
        """
        item = 'ast'
        loader = AELoader(item_type=item)
        df = pd.DataFrame(loader.dim_reduction_loader())
        model = KMeans(n_clusters=n_clusters, init='k-means++',
                       random_state=42).fit(df)
        df['LABEL'] = model.labels_

        fig = plt.figure(figsize=(10, 10))
        plt.title("K-Means: ast category example")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0],
                   df.iloc[:, 1],
                   df.iloc[:, 2],
                   c=df.LABEL,
                   s=10,
                   cmap='rainbow',
                   alpha=1)
        plt.show()
        return

    def km_label(self):
        """
        <DESCRIPTION>
        K-Means로 확인한 label을 전처리가 완료된 원 데이터에 표기하여 고객 데이터 군집을 역추적합니다.
        """
        data = self.km_run()[0]
        label = data['LABEL']
        temp = []

        item_type = ['ast', 'trs', 'acs', 'snp', 'rto']
        for item in item_type:
            df = pd.read_pickle(
                './res_pp_categories/res_pp_{}.pkl'.format(item))
            temp.append(df)

        res = pd.concat(temp, axis=1)
        res['LABEL'] = label
        data.to_pickle('./res_clustering.pkl')
        res.to_pickle('./res_pp_categories/res_pp_label.pkl')
        return data, res


"""
<DESCRIPTION>
코드 실행 예시입니다.
본 파일(cs_clutering.py)를 import하는 코드가 존재하므로, 주석 처리해두었습니다.
"""

# if __name__ == "__main__":
#     loader = KMLoader()
#     res, model = loader.km_run(n_clusters=5)
