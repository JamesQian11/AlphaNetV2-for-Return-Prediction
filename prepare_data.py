import argparse
import datetime
import glob
import os
import numpy
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from alphanet import AlphaNetV2, AlphaNetV3, AlphaNetV4, load_model
from src.alphanet.data import TrainValData, TimeSeriesData
from alphanet.metrics import UpDownAccuracy
import matplotlib.pyplot as plt


def get_rolling_training_set(train_length, validate_length):
    stock_data_list = np.load('zz500_training_data_combined.npy', allow_pickle=True).tolist()
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=train_length,
                                  validate_length=validate_length,
                                  history_length=30,
                                  sample_step=2,
                                  train_val_gap=10)
    return train_val_data


def get_rolling_testing_set(train_length, validate_length):
    stock_data_list = np.load('zz500_testing_data_combined.npy', allow_pickle=True).tolist()
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=train_length,
                                  validate_length=validate_length,
                                  history_length=30,
                                  sample_step=1,
                                  train_val_gap=10)
    return train_val_data


if __name__ == "__main__":
    print('----------------------run------------------------------------------')
    # stock_data_list = np.load('data/zz500_testing_data_combined_old.npy', allow_pickle=True).tolist()[0]
    # stock_data_list_1 = np.load('zz500_testing_data_combined.npy', allow_pickle=True).tolist()[0]
    # a = np.load('lugutong_factor_20180102_20211230.npy', allow_pickle=True)

    time_index_path = 'time_index.csv'
    time_index = pd.read_csv(time_index_path).values
    time_step = [20160118, 20160531, 20161230, 20170531, 20171229, 20180531, 20181228, 20190531, 20191231, 20200529, 20201231, 20210531, 20211231, 20220228]
    ALPHA_table = []
    for i in range(0, 8):
        r_train = np.where(time_index == time_step[i + 4])[0].tolist()[0] - np.where(time_index == time_step[i])[0].tolist()[0]
        r_test = np.where(time_index == time_step[i + 5])[0].tolist()[0] - np.where(time_index == time_step[i])[0].tolist()[0]
        if i < 8:
            test_gap = r_test - r_train
        # elif i == 8:
        #     test_gap = r_test - r_train
        print(i, len(time_step) - 5, time_step[i])
        print(r_train, r_train-10, test_gap)
        # train_0, val_0, dates_info_0 = get_rolling_training_set(r_train, 10).get(time_step[i], order="by_date")
        train_1, val_1, dates_info_1 = get_rolling_testing_set(r_train - 10, test_gap).get(time_step[i], order="by_date")
        pred_time = dates_info_1['validation']['dates_list'][0]
        # for i, elem in enumerate(val_1):
        #     print(i, '-->', elem[1].numpy())
        #     print(elem)

        model = AlphaNetV2(l2=0.001, dropout=0.0)
        model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(), UpDownAccuracy()])
        predict = []

        for j in range(0, 5):
            # history = model.fit(train_0.batch(500).cache(), validation_data=val_0.batch(500).cache(), epochs=100)
            # model.save_weights('AlphaNetV2_lugutong_%d_%d' % (pred_time, j))

            model.load_weights('zz500/AlphaNetV2_zz500_%d_%d' % (pred_time, j))
            y_predict = model.predict(val_1.batch(500).cache())
            _y_predict = y_predict.flatten().tolist()
            predict.append(_y_predict)
            print('-----predicting----')

        predict_all = np.array(predict)
        y_predict_mean = predict_all.mean(axis=0)
        _y_predict_ = y_predict_mean.flatten()

        not_nan = np.load('not_nan.npy', allow_pickle=True).tolist()
        trans_val = []
        pointer = 0
        for x in not_nan:
            if x:
                trans_val.append(_y_predict_[pointer])
                pointer += 1
            else:
                trans_val.append(np.nan)
        A = numpy.array(trans_val).reshape((test_gap, 500))
        ALPHA_table.extend(A.tolist())
    numpy.save('zz500_factor_2011229_20211230.npy', ALPHA_table)

    # #     A_rank = pd.DataFrame(A).rank(axis=1, pct=True).values
    # #     alpha_dict = numpy.zeros((test_gap, 1485))
    # #     alpha_dict[A_rank > 0.95] = 1
    # #     ALPHA_table.extend(alpha_dict.tolist())
    # # C = pd.DataFrame(ALPHA_table)
    # # alpah_MA = C.rolling(3).mean().to_numpy()
    # # alpah_MA[alpah_MA > 0] = 1
    # numpy.save('lugutong_alpha_20180102_20220214.npy', ALPHA_table)

print('--------------------------finish-------------------------------')

