from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import cv2
from scipy.stats import stats
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier


def train_id_to_path(x):
    return 'train/' + x + ".jpg"


def test_id_to_path(x):
    return 'test/' + x + ".jpg"


image_height = 128
image_width = 128


# define a function that accepts an image url and outputs an eager tensor
def path_to_eagertensor(image_path):
    raw = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    # image = tf.image.resize_with_pad(image, image_height, image_width) #optional with padding to retain original dimensions
    image = tf.image.resize(image, (image_height, image_width))
    return image


def main():
    tf.config.get_visible_devices()
    if 'GPU' in str(device_lib.list_local_devices()):
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        sess = tf.compat.v1.Session(config=config)

    np.random.seed(1)
    # Training settings
    use_cuda = True  # Switch to False if you only want to use your CPU


    # get the data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    meta = train.drop(['Id', 'Pawpularity'], axis=1)
    train["img_path"] = train["Id"].apply(train_id_to_path)
    test["img_path"] = test["Id"].apply(test_id_to_path)
    train['ten_bin_pawp'] = pd.qcut(train['Pawpularity'], q=10, labels=False)
    train = train.astype({"ten_bin_pawp": str})
    train10bin_stats = train.groupby('ten_bin_pawp')
    # print(train10bin_stats.describe())
    # print(train.head())

    X = []
    B = []
    for img in train['img_path']:
        new_img_tensor = path_to_eagertensor(img)

        #pca for rgb image, reference from Iqbal Hussain
        image = cv2.cvtColor(new_img_tensor.numpy(), cv2.COLOR_BGR2RGB)
        blue, green, red = cv2.split(image)
        df_blue = blue / 255
        df_green = green / 255
        df_red = red / 255
        pca_b = PCA(n_components=75)
        pca_b.fit(df_blue)
        trans_pca_b = pca_b.transform(df_blue)
        pca_g = PCA(n_components=75)
        pca_g.fit(df_green)
        trans_pca_g = pca_g.transform(df_green)
        pca_r = PCA(n_components=75)
        pca_r.fit(df_red)
        trans_pca_r = pca_r.transform(df_red)
        b_arr = pca_b.inverse_transform(trans_pca_b)
        g_arr = pca_g.inverse_transform(trans_pca_g)
        r_arr = pca_r.inverse_transform(trans_pca_r)
        img_reduced = (cv2.merge((b_arr, g_arr, r_arr)))

        # l=new_img_tensor.numpy().flatten()
        # B.append(l)
        X.append(img_reduced)

    print(type(X), len(X))
    X = np.array(X)
    print(type(X), X.shape)

    y = train['Pawpularity']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
    # x_train, x_test, y_train, y_test = train_test_split(meta, y, test_size=0.1, random_state=7)

    X_submission = []
    for img in test['img_path']:
        new_img_tensor = path_to_eagertensor(img)
        X_submission.append(new_img_tensor)

    #print(type(X_submission), len(X_submission))
    X_submission = np.array(X_submission)

    # for logistic regression with image data
    # clf = LogisticRegression(random_state=1,max_iter=200).fit(x_train, y_train)
    # c=clf.predict(x_test)

    # for logistic regression with meta data
    # clf = LogisticRegression(random_state=1,max_iter=1000).fit(x_train, y_train)
    # c=clf.predict(x_test)

    # for random forest ensemble
    # regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=20, max_features="sqrt",max_samples=5000)
    # regr.fit(x_train, y_train)
    # c=regr.predict(x_test)

    # for random forest ensemble with only metadata
    """regr = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=100,max_features="sqrt",max_samples=5000)
    regr.fit(x_train, y_train)
    c=regr.predict(x_test)

    rms = mean_squared_error(y_test, c, squared=False)
    print(rms)
    scores = cross_val_score(regr,meta, y, cv=5,scoring="neg_root_mean_squared_error")
    print(scores)"""

    x_traint, x_testt, y_traint, y_testt = train_test_split(x_train, y_train, test_size=0.11, random_state=7)

    inputs = tf.keras.Input(shape=(image_height, image_width, 3))

    x = inputs

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7),  padding='valid',
                               kernel_initializer='he_normal',  activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  padding='same',
                               kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), activation='relu')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                               kernel_regularizer=l2(0.0003), activation='relu')(x)
    """x = tf.keras.layers.BatchNormalization()(x)"""

    #x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),kernel_regularizer=l2(0.0002), activation='relu')(x)
    #x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    print(model.summary())

    model.compile(
        loss='mse',
        optimizer='Adam',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"])

    data_augmentation = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")
    cnnmodel = model.fit(
        data_augmentation.flow(x_traint, y_traint, batch_size=32),
        validation_data=(x_testt, y_testt),
        steps_per_epoch=len(x_traint) // 32,
        epochs=30
    )

    #print(cnnmodel.history["val_rmse"])

    cnn_pred = model.predict(x_test)
    rms = mean_squared_error(y_test, cnn_pred, squared=False)
    print(rms)

    cnn_pred = model.predict(X_submission)
    cnn = pd.DataFrame()
    cnn['Id'] = test['Id']
    cnn['Pawpularity'] = cnn_pred
    cnn.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
