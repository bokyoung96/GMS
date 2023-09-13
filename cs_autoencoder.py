"""
TEAM 고객의 미래를 새롭게

코드 목적: cs 데이터를 AutoEncoder로 분석합니다.
"""
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

dir_name = "./cs_ae_callback/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


class AutoEncoder:
    """
    <DESCRIPTION>
    Tensorflow keras에 기반하여 AutoEncoder를 구현합니다.
    """

    def __init__(self,
                 item_type: str = 'ast',
                 test_size: int = 0.2,
                 epochs: int = 100):
        self.item_type = item_type
        self.test_size = test_size
        self.epochs = epochs

    @property
    def data_loader(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 카테고리 분류(item_type)에 기반해 로드합니다.
        """
        with open('./res_pp_categories/res_pp_{}.pkl'.format(self.item_type), 'rb') as f:
            data = pickle.load(f)
        print("\n ***** DATA CATEGORY {} LOADED ***** \n".format(self.item_type))
        return data

    @property
    def data_split(self) -> pd.DataFrame:
        """
        <DESCRIPTION>
        cs 데이터를 train / validation set으로 구분합니다.
        """
        data = self.data_loader
        x_train, x_val = train_test_split(data,
                                          test_size=self.test_size,
                                          random_state=42)
        return x_train, x_val

    def auto_encoder(self, output_dim: int = 3):
        """
        <DESCRIPTION>
        AutoEncoder를 구현합니다.
        trs 카테고리 데이터는 학습 오류로 제외하였습니다.
        """
        # DATA
        x_train, x_val = self.data_split

        # SHAPES & LAYERS
        input_dim = x_train.shape[1]
        layers = round(input_dim / 2)

        if self.item_type == 'rto':
            input_layer = Input((input_dim, ))

            encode_layer = Dense(5, activation='relu')(input_layer)
            encode_layer = Dense(4, activation='relu')(encode_layer)

            latent_layer = Dense(3, activation='sigmoid')(encode_layer)

            decode_layer = Dense(4, activation='relu')(latent_layer)
            decode_layer = Dense(5, activation='relu')(decode_layer)
            decode_layer = Dense(6, activation='relu')(decode_layer)

        elif self.item_type == 'trs':
            input_layer = Input((input_dim, ))

            encode_layer = Dense(
                1024, activation=LeakyReLU(alpha=0.3))(input_layer)
            encode_layer = Dense(
                512, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                256, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                128, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                64, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                32, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                16, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                8, activation=LeakyReLU(alpha=0.3))(encode_layer)
            encode_layer = Dense(
                4, activation=LeakyReLU(alpha=0.3))(encode_layer)

            latent_layer = Dense(3, activation='sigmoid')(encode_layer)

            decode_layer = Dense(
                4, activation=LeakyReLU(alpha=0.3))(latent_layer)
            decode_layer = Dense(
                8, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                16, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                32, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                64, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                128, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                256, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                512, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(
                1024, activation=LeakyReLU(alpha=0.3))(decode_layer)
            decode_layer = Dense(1646, activation='sigmoid')(decode_layer)

        else:
            layer_sizes = []
            for layer in range(layers):
                layer_size = max(output_dim, int(input_dim / (2**layer)))
                if layer_size <= output_dim:
                    break
                layer_sizes.append(layer_size)

            # INPUT LAYER
            input_layer = Input((input_dim, ))

            # ENCODE LAYER
            encode_layer = Dense(
                layer_sizes[1], activation='relu')(input_layer)
            for size in layer_sizes[2:]:
                encode_layer = Dense(size, activation='relu')(encode_layer)

            # LATENT LAYER
            latent_layer = Dense(
                output_dim, activation='sigmoid')(encode_layer)

            # DECODE LAYER
            layer_sizes.reverse()
            decode_layer = Dense(
                layer_sizes[0], activation='relu')(latent_layer)
            for size in layer_sizes[1:-1]:
                decode_layer = Dense(size, activation='relu')(decode_layer)
            decode_layer = Dense(
                layer_sizes[-1], activation='relu')(decode_layer)

        # MODEL
        model = Model(inputs=input_layer,
                      outputs=decode_layer)
        adam = Adam(
            learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='mse')
        model.summary()

        # CALLBACKS
        checkpoints = ModelCheckpoint(filepath='./cs_ae_callback/cs_autoencoder_callback_{}.hdf5'.format(self.item_type),
                                      monitor='val_loss',
                                      save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.00001,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
        callbacks = [checkpoints, early_stopping]

        # FITTING
        res = model.fit(x_train,
                        x_train,
                        epochs=self.epochs,
                        batch_size=32,
                        validation_data=(x_val, x_val),
                        callbacks=callbacks)

        stopped_epoch = early_stopping.stopped_epoch
        return model, res, stopped_epoch

    def auto_encoder_plot(self, res, stopped_epoch):
        """
        <DESCRIPTION>
        각 카테고리별 AutoEncoder의 training, validation loss를 그래프로 나타냅니다.
        """
        epoch = np.arange(1, len(res.history['loss'])+1)
        plt.figure(figsize=(15, 5))
        plt.plot(epoch, res.history['loss'],
                 linewidth=2, label='Training loss')
        plt.plot(epoch, res.history['val_loss'],
                 linewidth=2, label='Validation loss')
        plt.axvline(x=stopped_epoch, color='r', linestyle='--',
                    label='Early stopping epoch')
        plt.title('AutoEncoder performance in {} category'.format(self.item_type))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.savefig('./loss_plot_{}.png'.format(self.item_type))


"""
<DESCRIPTION>
코드 실행 예시입니다.
본 파일(cs_autoencoder.py)를 import하는 코드가 존재하므로, 주석 처리해두었습니다.
"""

if __name__ == "__main__":
    items = ['ast', 'trs', 'acs', 'snp', 'rto']
    items = ['trs']
    for item in items:
        loader = AutoEncoder(item_type=item)
        model, res, stopped_epoch = loader.auto_encoder(output_dim=3)
        loader.auto_encoder_plot(res, stopped_epoch)
