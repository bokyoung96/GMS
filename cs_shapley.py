"""
TEAM 고객의 미래를 새롭게

코드 목적: AutoEncoder에서 중요 요소로 작용한 특성을 Shapley Value로 찾습니다.
"""
import shap

from cs_autoencoder import *

tf.compat.v1.disable_v2_behavior()


class Shapley:
    def __init__(self, item_type: str = 'ast', test_size: int = 0.2,  num_choice: int = 20):
        self.item_type = item_type
        self.test_size = test_size
        self.num_choice = num_choice

    @property
    def data_loader(self):
        """
        <DESCRIPTION>
        각 군집별 대표 고객을 각 카테고리에서 랜덤하게 추출합니다.
        """
        df = pd.read_pickle(
            './res_pp_categories/res_pp_{}.pkl'.format(self.item_type))
        res = pd.read_pickle('./res_clustering.pkl')
        df['LABEL'] = res.LABEL

        background = pd.DataFrame()
        for label in range(5):
            label_df = df[df['LABEL'] == label]
            background = pd.concat(
                [background, label_df.sample(n=self.num_choice, random_state=42)])
        return background

    @property
    def data_split(self):
        """
        <DESCRIPTION>
        cs 데이터를 train / validation set으로 구분합니다.
        이후, 각 군집별 대표 고객을 랜덤하게 추출합니다.
        """
        df = pd.read_pickle(
            './res_pp_categories/res_pp_{}.pkl'.format(self.item_type))
        res = pd.read_pickle('./res_clustering.pkl')
        df['LABEL'] = res.LABEL
        x_train, x_val = train_test_split(df,
                                          test_size=self.test_size,
                                          random_state=42)

        x_train_label = pd.DataFrame()
        for label in range(5):
            label_df = x_train[x_train['LABEL'] == label]
            x_train_label = pd.concat(
                [x_train_label, label_df.sample(n=1, random_state=42)])
        return x_train, x_val, x_train_label

    def shapley(self):
        """
        <DESCRIPTION>
        Shapley Value를 계산합니다.
        """
        model = load_model(
            './cs_ae_callback/cs_autoencoder_callback_{}.hdf5'.format(self.item_type))
        background = self.data_loader
        x_train, x_val, x_train_label = self.data_split
        x_train.drop(['LABEL'], axis=1, inplace=True)
        x_train_label.drop(['LABEL'], axis=1, inplace=True)
        background.drop(['LABEL'], axis=1, inplace=True)

        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(x_train_label.values)

        shap.initjs()
        shap.force_plot(explainer.expected_value[0],
                        shap_values[0][0],
                        feature_names=x_train.columns)
        shap.summary_plot(shap_values[0], x_train, plot_type='bar')
        return explainer, shap_values
