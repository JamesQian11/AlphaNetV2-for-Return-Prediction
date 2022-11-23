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
from alphanet.data import TrainValData, TimeSeriesData
from alphanet.metrics import UpDownAccuracy
import matplotlib.pyplot as plt


def file_preprocess(filename):
    input_csv = pd.read_csv(filename)
    input_csv.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    input_csv.index = input_csv['date']
    input_csv = input_csv.drop(columns='date')
    return input_csv.replace(0, np.nan)


def prepare_stock():
    """
        close/free_turn
        open/turn
        volume/low
        vwap/high
        low/high
        vwap/close
    """
    Open = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJOPEN.csv'
    high = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJHIGH.csv'
    low = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJLOW.csv'
    close = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJCLOSE.csv'
    vwap = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_AVGPRICE.csv'
    returns1 = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJFACTOR.csv'
    turn = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_TURN.csv'
    free_turn = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_FREETURNOVER.csv'
    volume = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_VOLUME.csv'

    volume_df = file_preprocess(volume)
    Open_df = file_preprocess(Open)
    high_df = file_preprocess(high)
    low_df = file_preprocess(low)
    close_df = file_preprocess(close)
    vwap_df = file_preprocess(vwap)
    returns1_df = close_df.pct_change(1)
    turn_df = file_preprocess(turn)
    free_turn_df = file_preprocess(free_turn)
    close_free_turn_df = close_df / free_turn_df
    open_turn_df = Open_df / turn_df
    volume_low_df = volume_df / low_df
    vwap_high_df = vwap_df / high_df
    low_high_df = low_df / high_df
    vwap_close_df = vwap_df / close_df

    Stocks = close_df.columns[1:]
    for i in Stocks:
        print(i)
        open_data = Open_df[i]
        high_data = high_df[i]
        low_data = low_df[i]
        close_data = close_df[i]
        volume_data = volume_df[i]
        vwap_data = vwap_df[i]
        return1_data = returns1_df[i]
        turn_data = turn_df[i]
        free_turn_data = free_turn_df[i]
        close_free_turn_data = close_free_turn_df[i]
        open_turn_data = open_turn_df[i]
        volume_low_data = volume_low_df[i]
        vwap_high_data = vwap_high_df[i]
        low_high_data = low_high_df[i]
        vwap_close_data = vwap_close_df[i]

        _open_data = open_data.rename('open_' + i)
        _high_data = high_data.rename('high_' + i)
        _low_data = low_data.rename('low_' + i)
        _close_data = close_data.rename('close_' + i)
        _vwap_data = vwap_data.rename('vwap_' + i)
        _volume_data = volume_data.rename('volume_' + i)
        _return1_data = return1_data.rename('return1_' + i)
        _turn_data = turn_data.rename('turn_' + i)
        _free_turn_data = free_turn_data.rename('free_turn_' + i)
        _close_free_turn_data = close_free_turn_data.rename('CFT_' + i)
        _open_turn_data = open_turn_data.rename('OT_' + i)
        _volume_low_data = volume_low_data.rename('VL_' + i)
        _vwap_high_data = vwap_high_data.rename('VH_' + i)
        _low_high_data = low_high_data.rename('LH_' + i)
        _vwap_close_data = vwap_close_data.rename('VC_' + i)

        stock = pd.DataFrame(
            [_open_data, _high_data, _low_data, _close_data, _vwap_data, _volume_data, _return1_data, _turn_data,
             _free_turn_data,
             _close_free_turn_data, _open_turn_data, _volume_low_data, _vwap_high_data, _low_high_data,
             _vwap_close_data])
        stock_name = 'stocks/' + i + '.csv'
        stock.to_csv(stock_name, header=True)
        print(stock_name)


def filtering_ZT_DT():
    high_path = 'Data_XXX/S_DQ_ADJHIGH.csv'
    low_path = 'Data_XXX/S_DQ_ADJLOW.csv'
    close_path = 'Data_XXX/S_DQ_ADJCLOSE.csv'
    tradestatus_path = 'Data_XXX/S_DQ_TRADESTATUS_SIG.csv'
    adjfactor_path = 'Data_XXX/S_DQ_ADJFACTOR.csv'
    high_df = file_preprocess(high_path)
    low_df = file_preprocess(low_path)
    close_adj = file_preprocess(close_path)
    pre_close_adj = close_adj.shift(1)
    tradestatus = file_preprocess(tradestatus_path)
    adjfactor = file_preprocess(adjfactor_path)
    buyvalid = tradestatus.copy()  # tradestatus 停牌的股票 不能交易
    sellvalid = tradestatus.copy()
    low_n = low_df / adjfactor
    high_n = high_df / adjfactor
    buyvalid[(low_n == high_n) & (close_adj > pre_close_adj)] = 0  # 一字涨停不能买 low = adjlow/adjfactor (复权后的价格/复权因子）
    sellvalid[(low_n == high_n) & (close_adj < pre_close_adj)] = 0  # 一字跌停不能卖
    a = buyvalid.apply(lambda x: x.sum(), axis=1)
    b = tradestatus.apply(lambda x: x.sum(), axis=1)
    total = (buyvalid * sellvalid).replace(0.0, np.nan)
    c = total.apply(lambda x: x.sum(), axis=1)
    total_name = "Data_XXX/trade_stocks_index.csv"
    total.to_csv(total_name, header=True)


def cal_returns():
    close = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_ADJCLOSE.csv'
    close_df = file_preprocess(close)
    returns = close_df.pct_change(10)
    returns_name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns.csv'
    returns.to_csv(returns_name, header=True)


def get_stock_pools():
    index = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/lugutong_20220219.csv'
    returns = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns.csv'
    trade_index = 'Data_XXX/trade_stocks_index.csv'
    returns_data = pd.read_csv(returns)
    returns_data.index = returns_data['date']
    returns_data = returns_data.drop(columns='date')
    zz500 = []
    for i in pd.read_csv(index, header=None).values:
        print(i[0])
        zz500.append(i[0])
    zz500_returns = returns_data[zz500]
    name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_lugutong.csv'
    zz500_returns.to_csv(name, header=True)

    trade_index_data = pd.read_csv(trade_index)
    trade_index_data.index = trade_index_data['date']
    trade_index_data = trade_index_data.drop(columns='date')
    zz500_trade_index_data = trade_index_data[zz500]
    name1 = 'Data_XXX/lugutong_trade_stocks_index.csv'
    zz500_trade_index_data.to_csv(name1, header=True)


def get_file_names_list(path):
    file_names = glob.glob(os.path.join(path, '*.csv'))
    data_list = []
    for i in file_names:
        data = i.split('/')[-1]
        data_list.append(data)
    data = sorted(data_list)
    return data


def combine_training_data():
    stocks = 'stocks'
    returns = 'Data_XXX/S_DQ_returns_lugutong_trade.csv'

    returns_data = pd.read_csv(returns)[10:]

    index = returns_data.columns[1:]
    stock_data_list = []
    for i in index:  # 时间索引
        for d in get_file_names_list(stocks):  # 股票索引
            _stock_name = d.split('.')[0] + '.' + d.split('.')[1]
            if i == _stock_name:
                print('stock_name', _stock_name)
                stock_name = f'{stocks}/{d}'
                _stock = d.split('.')[0] + '.' + d.split('.')[1]

                date_table = returns_data['date'][:1441]  # 取returns的时间作为 时间的索引
                stock_table = pd.read_csv(stock_name).iloc[:, 11: 1452]
                ret = returns_data[_stock][10:]  # 取未来10天内的收益率做标签
                DSR = TimeSeriesData(dates=date_table.values,
                                     data=stock_table.values.T,
                                     labels=ret.values)
                stock_data_list.append(DSR)
    np.save('lugutong_returns_trade_combined.npy', stock_data_list)


def get_rolling_training_set(train_length, validate_length):
    stock_data_list = np.load('lugutong_training_data_combined.npy', allow_pickle=True).tolist()
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=train_length,
                                  validate_length=validate_length,
                                  history_length=30,
                                  sample_step=2,
                                  train_val_gap=10)
    return train_val_data


def combine_testing_data():
    stocks = 'stocks'
    returns = 'Data_XXX/S_DQ_returns_lugutong_trade.csv'
    returns_data = pd.read_csv(returns)[10:]
    index = returns_data.columns[1:]
    stock_data_list = []
    for i in index:
        for d in get_file_names_list(stocks):
            _stock_name = d.split('.')[0] + '.' + d.split('.')[1]
            if i == _stock_name:
                print('stock_name', _stock_name)
                stock_name = f'{stocks}/{d}'
                _stock = d.split('.')[0] + '.' + d.split('.')[1]
                date_table = returns_data['date'][:1441]
                stock_table = pd.read_csv(stock_name).iloc[:, 11: 1452]
                ret = returns_data[_stock][10:]
                DSR = TimeSeriesData(dates=date_table.values,
                                     data=stock_table.values.T,
                                     labels=ret.values)
                stock_data_list.append(DSR)
    np.save('lugutong_returns_trade_combined.npy', stock_data_list)


def get_rolling_testing_set(train_length, validate_length):
    stock_data_list = np.load('lugutong_testing_data_combined.npy', allow_pickle=True).tolist()
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=train_length,
                                  validate_length=validate_length,
                                  history_length=30,
                                  sample_step=1,
                                  train_val_gap=10)
    return train_val_data


if __name__ == "__main__":
    print('----------------------run------------------------------------------')
    time_index_path = 'time_index.csv'
    time_index = pd.read_csv(time_index_path).values
    time_step = [20160118, 20160531, 20161230, 20170531, 20171229, 20180531, 20181228, 20190531, 20191231, 20200529, 20201231, 20210531, 20211231, 20220228]
    ALPHA_table = []
    for i in range(0, 9):
        print(i, len(time_step) - 5, time_step[i])
        r_train = np.where(time_index == time_step[i + 4])[0].tolist()[0] - np.where(time_index == time_step[i])[0].tolist()[0]
        r_test = np.where(time_index == time_step[i + 5])[0].tolist()[0] - np.where(time_index == time_step[i])[0].tolist()[0]
        if i < 8:
            test_gap = r_test - r_train
        elif i == 8:
            test_gap = r_test - r_train - 10
        train_0, val_0, dates_info_0 = get_rolling_training_set(r_train, 10).get(time_step[i], order="by_date")
        train_1, val_1, dates_info_1 = get_rolling_testing_set(r_train - 10, test_gap).get(time_step[i], order="by_date")
        for i, elem in enumerate(val_1):
            print(i, '-->', elem[1].numpy())
            print(elem)

        model = AlphaNetV2(l2=0.001, dropout=0.0)
        # model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(), UpDownAccuracy()])
        # predict = []
        #
        # for j in range(0, 5):
        #     history = model.fit(train_0.batch(500).cache(), validation_data=val_0.batch(500).cache(), epochs=100)
        #     model.save_weights('checkpoints/AlphaNetV2_lugutong_%d_%d' % (i, j))
        #
        #     model.load_weights('checkpoints/AlphaNetV2_lugutong_%d_%d' % (i, j))
        #     y_predict = model.predict(val_1.batch(500).cache())
        #     _y_predict = y_predict.flatten().tolist()
        #     predict.append(_y_predict)
        #
        # predict_all = np.array(predict)
        # y_predict_mean = predict_all.mean(axis=0)
        # _y_predict_ = y_predict_mean.flatten()
        #
        # not_nan = np.load('data/not_nan.npy', allow_pickle=True).tolist()
        # trans_val = []
    #     pointer = 0
    #     for x in not_nan:
    #         if x:
    #             trans_val.append(_y_predict_[pointer])
    #             pointer += 1
    #         else:
    #             trans_val.append(np.nan)
    #     A = numpy.array(trans_val).reshape((test_gap, 500))
    #     A_rank = pd.DataFrame(A).rank(axis=1, pct=True).values
    #     alpha_dict = numpy.zeros((test_gap, 500))
    #     alpha_dict[A_rank > 0.8] = 1
    #     ALPHA_table.extend(alpha_dict.tolist())
    # numpy.save('npy_data/alpha_returns_debug_21.npy', ALPHA_table)

print('--------------------------finish-------------------------------')

