import shap
import keras
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class AutoEncoder:
    def __init__(self,
                 data_selection: str = 'category',
                 test_size: int = 0.2):
        self.data_selection = data_selection
        if data_selection == 'category':
            self.item_type = input(
                "Select category [ast, trs, acs, snp, rto]: ")
        else:
            pass
        self.test_size = test_size

    @property
    def data_loader(self):
        # NOTE: CANNOT BE USED. MEMORY LOSS ISSUE.
        assert self.data_selection != 'category', "ERROR: data_selection == category."

        dfs = []
        items = ['ast', 'trs', 'acs', 'snp', 'rto']
        for item in items:
            df = pd.read_pickle(
                './res_pp_categories/res_pp_{}.pkl'.format(item))
            dfs.append(df)
            print("\n ***** DATA CATEGORY {} LOADED ***** \n".format(item))

        res = pd.concat(dfs, axis=1)
        print("\n ***** DATA TOTAL LOADED ***** \n")
        return res

    @property
    def data_category_loader(self):
        assert self.data_selection == 'category', "ERROR: data_selection != category."

        with open('./res_pp_categories/res_pp_{}.pkl'.format(self.item_type), 'rb') as f:
            data = pickle.load(f)
        print("\n ***** DATA CATEGORY {} LOADED ***** \n".format(self.item_type))
        return data

    def auto_encoder(self, output_dim: int = 3):
        # DATA
        if self.data_selection == 'category':
            data = self.data_category_loader
        else:
            data = self.data_loader
        x_train, x_val = train_test_split(data,
                                          test_size=self.test_size,
                                          random_state=42)

        # SHAPES & LAYERS
        input_dim = x_train.shape[1]
        layers = round(input_dim / 2)

        layer_sizes = []
        for layer in range(layers):
            layer_size = max(output_dim, int(input_dim / (2**layer)))
            if layer_size <= output_dim:
                break
            layer_sizes.append(layer_size)

        # INPUT LAYER
        input_layer = Input((input_dim, ))

        # ENCODE LAYER
        encode_layer = Dense(layer_sizes[1], activation='relu')(input_layer)
        for size in layer_sizes[2:]:
            encode_layer = Dense(size, activation='relu')(encode_layer)

        # LATENT LAYER
        latent_layer = Dense(output_dim, activation='sigmoid')(encode_layer)

        # DECODE LAYER
        layer_sizes.reverse()
        decode_layer = Dense(layer_sizes[0], activation='relu')(latent_layer)
        for size in layer_sizes[1:]:
            decode_layer = Dense(size, activation='relu')(decode_layer)

        # MODEL
        model = Model(inputs=input_layer,
                      outputs=decode_layer)
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # CALLBACKS
        checkpoints = ModelCheckpoint(filepath='cs_autoencoder_callback.hdf5',
                                      monitor='val_loss',
                                      save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0001,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
        callbacks = [checkpoints, early_stopping]

        # FITTING
        res = model.fit(x_train,
                        x_train,
                        epochs=100,
                        batch_size=128,
                        validation_data=(x_val, x_val),
                        callbacks=callbacks)
        return res

    @staticmethod
    def auto_encoder_plot(res):
        epoch = np.arange(1, len(res.history['loss'])+1)
        plt.plot(epoch, res.history['loss'], label='training loss')
        plt.plot(epoch, res.history['val_loss'], label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    loader = AutoEncoder('category')
    # loader = AutoEncoder('total')

    res = loader.auto_encoder(output_dim=3)
