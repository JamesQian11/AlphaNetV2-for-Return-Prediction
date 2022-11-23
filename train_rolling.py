import argparse
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

import talib as ta


# read data
def file_preprocess(filename):
    input_csv = pd.read_csv(filename)
    input_csv.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    input_csv.index = input_csv['date']
    input_csv = input_csv.drop(columns='date')
    return input_csv.replace(0, np.nan)


def prepare_stock():
    Open = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJOPEN.csv'
    high = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJHIGH.csv'
    low = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJLOW.csv'
    close = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJCLOSE.csv'
    vwap = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_AVGPRICE.csv'
    returns1 = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_ADJFACTOR.csv'
    turn = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_TURN.csv'
    free_turn = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_FREETURNOVER.csv'
    volume = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3-master/Data_XXX/S_DQ_VOLUME.csv'
    '''
    close/free_turn
    open/turn
    volume/low
    vwap/high
    low/high
    vwap/close  
    '''
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


# def get_zz500():

def ZTDT():
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


def get_zz500_zscore_returns():
    returns = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500-ret_dropnan.csv'
    returns_data = pd.read_csv(returns)
    returns_data.index = returns_data['date']
    returns_data = returns_data.drop(columns='date')
    mean = returns_data.mean()
    std = returns_data.std()
    returns_zscore = (returns_data - mean) / std
    zz500_returns_zscore_name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500-ret_dropnan_zscore.csv'
    returns_zscore.to_csv(zz500_returns_zscore_name, header=True)


def get_zz500():
    index = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/zz500_20211210.csv'
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
    name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500.csv'
    zz500_returns.to_csv(name, header=True)

    trade_index_data = pd.read_csv(trade_index)
    trade_index_data.index = trade_index_data['date']
    trade_index_data = trade_index_data.drop(columns='date')
    zz500_trade_index_data = trade_index_data[zz500]
    name1 = 'Data_XXX/zz500_trade_stocks_index.csv'
    zz500_trade_index_data.to_csv(name1, header=True)

    # Ret = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/常用指数收益率.csv'
    # Ret_data = pd.read_csv(Ret)
    # zz500_ret = Ret_data['000905.SH']
    # _zz500_returns = pd.DataFrame(zz500_returns.values - zz500_ret.values[:, np.newaxis], index=zz500_returns.index,
    #                               columns=zz500_returns.columns)
    # __zz500_name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500-ret.csv'
    # _zz500_returns.to_csv(__zz500_name, header=True)
    # label_all = pd.read_csv(__zz500_name)
    # label_all.index = label_all['date']
    # label_all = label_all.drop(columns='date')
    # zz500_label = label_all.dropna(axis=0, how='all').dropna(axis=1, how='any')
    # zz500_label_name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500-ret_dropnan.csv '
    # zz500_label.to_csv(zz500_label_name, header=True)


def get_file_names_list(path):
    file_names = glob.glob(os.path.join(path, '*.csv'))
    data_list = []
    for i in file_names:
        data = i.split('/')[-1]
        data_list.append(data)
    data = sorted(data_list)
    return data


def get_returns_rank():
    zz500_returns_org = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500.csv'
    zz500_returns_org = pd.read_csv(zz500_returns_org)
    zz500_returns_org.index = zz500_returns_org['date']
    zz500_returns_org = zz500_returns_org.drop(columns='date')
    # zz500_returns_label = zz500_returns_org.dropna(axis=0, how='all').dropna(axis=1, how='any')
    zz500_returns_label_rank = zz500_returns_org.rank(axis=1, pct=True)
    zz500_returns_label_rank_name = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_returns_zz500_rank.csv'
    zz500_returns_label_rank.to_csv(zz500_returns_label_rank_name, header=True)


def combine_data():
    stocks = 'stocks'
    returns = 'Data_XXX/S_DQ_returns_zz500_trade.csv'
    # zz500_trade_stocks = 'Data_XXX/zz500_trade_stocks_index.csv'
    returns_data = pd.read_csv(returns)[10:]

    # create an empty list
    index = returns_data.columns[1:]
    stock_data_list = []
    # compute label (future return)
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
                """ 
                所取的时间范围概括：input 时间-----开始的数据时间 20160118， 结束的数据时间 20211217
                                 label 时间-----begin：20160201 结束：20211231
                """
                DSR = TimeSeriesData(dates=date_table.values,  # date column
                                     data=stock_table.values.T,  # data columns
                                     labels=ret.values)  # label column
                print('ret\n', ret)
                print('date_table\n', date_table)
                print('stock_table\n', stock_table)
                stock_data_list.append(DSR)
    np.save('zz500_returns_trade_combined.npy', stock_data_list)


def get_training_set():
    stock_data_list = np.load('zz500_returns_trade_combined.npy', allow_pickle=True).tolist()
    # put stock list into TrainValData() class, specify dataset lengths
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=1000,  # 1200 trading days for training 1197
                                  validate_length=197,  # 150 trading days for validation  233
                                  history_length=30,  # each input contains 30 days of history
                                  sample_step=2,  # jump to days forward for each sampling
                                  train_val_gap=10)  # leave a 10-day gap between training and validation 1400 1197
    ''' 
    {'start_date': 20160118, 'end_date': 20200227}, 'validation': {'start_date': 20200313, 'end_date': 20201230}}
    '''
    # get one training period that start from
    _train, _val, _dates_info = train_val_data.get(20160118, order="by_date")
    '''
    输入数据时间段的： input： 20160118--20160304 （30天的数据）
                    label： 区间 20160304--20160318 （后10天的收益率）
                    data_info: 20160304
    '''
    # print(_dates_info)
    return _train, _val, _dates_info


def get_test_set():
    stock_data_list = np.load('zz500_returns_data_combined.npy', allow_pickle=True).tolist()
    # put stock list into TrainValData() class, specify dataset lengths
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=223,  # 1200 trading days for training 1197
                                  validate_length=1200,  # 150 trading days for validation  233
                                  history_length=30,  # each input contains 30 days of history
                                  sample_step=1,  # jump to days forward for each sampling
                                  train_val_gap=10)  # leave a 10-day gap between training and validation

    # get one training period that start from
    _train, _val, _dates_info = train_val_data.get(20160104, order="by_date")
    # print(_dates_info)
    '''
     {'training':{'start_date': 20160118, 'end_date': 20161215}, 'validation': {'start_date': 20161230, 'end_date': 20211207}}
    '''
    return _train, _val, _dates_info


def plot():
    plt.title('Loss')
    plt.plot(epochs, loss, 'blue', label='loss')
    plt.legend()
    plt.show()

    plt.title('rms')
    plt.plot(epochs, rms, 'blue', label='rms')
    plt.legend()
    plt.show()

    plt.title('acc')
    plt.plot(epochs, acc, 'blue', label='acc')
    plt.legend()
    plt.show()

    plt.title('val_loss')
    plt.plot(epochs, val_loss, 'blue', label='val_loss')
    plt.legend()
    plt.show()

    plt.title('val_rms')
    plt.plot(epochs, val_rms, 'blue', label='val_rms')
    plt.legend()
    plt.show()

    plt.title('val_acc')
    plt.plot(epochs, val_acc, 'blue', label='val_acc')
    plt.legend()
    plt.show()


def double2line():
    result_val = np.array(trans_val).reshape((117, 500))
    result_val_df = pd.DataFrame(result_val).rank(axis=1).values
    alpha_dict = numpy.zeros((117, 500))
    alpha_dict[result_val_df > 450] = 1
    alpha = []
    for i in range(0, 117):
        # print(i)
        s = alpha_dict[i, :]
        # print(s)
        c = np.row_stack((s, s)).tolist()
        alpha.append(c)


''' 
1。 rooling 半年: 做训练，
2。 10个 av
3。 重点参数debug
4。 dataprepress
'''


def rolling_train():
    for i in range(0, 10):
        model = AlphaNetV2(l2=0.001, dropout=0.0)
        model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(), UpDownAccuracy()])
        train, val, dates_info = get_rolling_training_set().get(20160118, order="by_date")
        model.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        model.save('model/AlphaNetV2_Weights_zz500_returns_trade_1')

        print("----------------------------finish one step----------------------------")

        train, val, dates_info = get_rolling_training_set().get(20160718, order="by_date")
        A = load_model('model/AlphaNetV2_Weights_zz500_returns_trade_1')
        A.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        A.save('model/AlphaNetV2_Weights_zz500_returns_trade_2')

        print("----------------------------finish two step----------------------------")

        train, val, dates_info = get_rolling_training_set().get(20170118, order="by_date")
        model = load_model('model/AlphaNetV2_Weights_zz500_returns_trade_2')
        model.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        model.save('model/AlphaNetV2_Weights_zz500_returns_trade_3')

        print("----------------------------finish three step----------------------------")

        train, val, dates_info = get_rolling_training_set().get(20170718, order="by_date")
        model = load_model('model/AlphaNetV2_Weights_zz500_returns_trade_3')
        model.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        model.save('model/AlphaNetV2_Weights_zz500_returns_trade_4')

        print("----------------------------finish four step----------------------------")

        train, val, dates_info = get_rolling_training_set().get(20180118, order="by_date")
        model = load_model('model/AlphaNetV2_Weights_zz500_returns_trade_4')
        model.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        model.save('model/AlphaNetV2_Weights_zz500_returns_trade_5')

        print("----------------------------finish five step----------------------------")

        train, val, dates_info = get_rolling_training_set().get(20180718, order="by_date")
        model = load_model('model/AlphaNetV2_Weights_zz500_returns_trade_5')
        model.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        model.save('model/AlphaNetV2_Weights_zz500_returns_trade_6')
        model.save_weights('model/AlphaNetV2_Weights_zz500_returns_trade_6_%d' % i)
        print(model.summary())


#  {'training': {'start_date': 20160701, 'end_date': 20180214}, 'validation': {'start_date': 20180308, 'end_date': 20180731}}

def get_rolling_training_set(train_length, validate_length):
    stock_data_list = np.load('lugutong_returns_trade_combined.npy', allow_pickle=True).tolist()
    # put stock list into TrainValData() class, specify dataset lengths
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=train_length,  # 1200 trading days for training 1197
                                  validate_length=validate_length,  # 150 trading days for validation  233
                                  history_length=30,  # each input contains 30 days of history
                                  sample_step=2,  # jump to days forward for each sampling
                                  train_val_gap=10)  # leave a 10-day gap between training and validation 1400 1197
    ''' 
    {'start_date': 20160118, 'end_date': 20200227}, 'validation': {'start_date': 20200313, 'end_date': 20201230}}
    '''
    # get one training period that start from
    return train_val_data


def get_rolling_testing_set(train_length, validate_length):
    stock_data_list = np.load('lugutong_returns_data_combined.npy', allow_pickle=True).tolist()
    # put stock list into TrainValData() class, specify dataset lengths
    train_val_data = TrainValData(time_series_list=stock_data_list,
                                  train_length=train_length,  # 1200 trading days for training 1197
                                  validate_length=validate_length,  # 150 trading days for validation  233
                                  history_length=30,  # each input contains 30 days of history
                                  sample_step=1,  # jump to days forward for each sampling
                                  train_val_gap=10)  # leave a 10-day gap between training and validation 1400 1197
    ''' 
    {'start_date': 20160118, 'end_date': 20200227}, 'validation': {'start_date': 20200313, 'end_date': 20201230}}
    '''
    # get one training period that start from
    return train_val_data


if __name__ == "__main__":
    print('----------------------run------------------------------------------')
    ''' 
    time_index_path = '/home/vision/Projects/01_Quantification/AlphaNetV3/time_index.csv'
    time_index = pd.read_csv(time_index_path).values
    time_step = [20160118, 20160531, 20161230, 20170531, 20171229, 20180531, 20181228, 20190531, 20191231, 20200529, 20201231, 20210531, 20211231]

    ALPHA_table = []
    for i in range(len(time_step) - 5):
        print(i, len(time_step) - 5)

        r_train = np.where(time_index == time_step[i + 4])[0].tolist()[0] - np.where(time_index == time_step[i])[0].tolist()[0]
        r_test = np.where(time_index == time_step[i + 5])[0].tolist()[0] - np.where(time_index == time_step[i])[0].tolist()[0]
        if i < 7:
            test_gap = r_test - r_train
        elif i == 7:
            test_gap = r_test - r_train - 10

        train_0, val_0, dates_info_0 = get_rolling_training_set(r_train, 10).get(time_step[i], order="by_date")
        train_1, val_1, dates_info_1 = get_rolling_testing_set(r_train - 10, test_gap).get(time_step[i], order="by_date")

        model = AlphaNetV2(l2=0.001, dropout=0.0)
        model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(), UpDownAccuracy()])
        predict = []
        for j in range(0, 5):
            print(i, j, "traning_%d  <----------------------------" % j)
            # history = model.fit(train_0.batch(500).cache(), validation_data=val_0.batch(500).cache(), epochs=100)
            # model.save_weights('model/AlphaNetV2_Weights_lugutong_returns_trade_%d_%d' % (i, j))

            model.load_weights('model/AlphaNetV2_Weights_lugutong_returns_trade_%d_%d' % (i, j))

            y_predict = model.predict(val_1.batch(500).cache())
            _y_predict = y_predict.flatten().tolist()
            predict.append(_y_predict)
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
        A = numpy.array(trans_val).reshape((test_gap, 1485))
        A_rank = pd.DataFrame(A).rank(axis=1, pct=True).values
        alpha_dict = numpy.zeros((test_gap, 1485))
        alpha_dict[A_rank > 0.95] = 1
        ALPHA_table.extend(alpha_dict.tolist())
'''
    ALPHA_table = np.load('lugutong_alpha_20180102_20211231.npy', allow_pickle=True).tolist()
    C = pd.DataFrame(ALPHA_table)
    alpah_MA = C.rolling(7).mean().to_numpy()
    alpah_MA[alpah_MA > 0] = 1
    numpy.save('lugutong_alpha_20180102_202112317.npy', alpah_MA)

    # B = np.load('npy_data/alpha_returns_debug_21.npy')
    # C = pd.DataFrame(B)
    # alpah_MA = C.rolling(10).mean().to_numpy()
    # alpah_MA[alpah_MA > 0] = 1
    # numpy.save('npy_data/alpha_returns_debug_24.npy', alpah_MA)




    # print(dates_info_0["training"][0:1])
    # print(dates_info_1["validation"][0:1])
    # model = AlphaNetV2(l2=0.001, dropout=0.0)
    # model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(), UpDownAccuracy()])
    # history = model.fit(train_1_0.batch(500).cache(), validation_data=val_1_0.batch(500).cache(), epochs=10)
    # model.save_weights('model/AlphaNetV2_Weights_zz500_returns_trade_%d' % r1)
    #
    # model.load_weights('model/AlphaNetV2_Weights_zz500_returns_trade_%d' % r1)
    # y_predict = model.predict(val_1_1.batch(500).cache())
    # _y_predict = y_predict.flatten()
    # not_nan = np.load('not_nan.npy', allow_pickle=True).tolist()
    # trans_val = []
    # pointer = 0
    # for x in not_nan:
    #     if x:
    #         trans_val.append(_y_predict[pointer])
    #         pointer += 1
    #     else:
    #         trans_val.append(np.nan)
    # A = numpy.array(trans_val).reshape((gap, 500))
    # A_rank = pd.DataFrame(A).rank(axis=1, pct=True).values
    # alpha_dict = numpy.zeros((gap, 500))
    # alpha_dict[A_rank > 0.8] = 1
    #
    # numpy.save('npy_data/alpha_returns_debug_%d.npy' % r1, alpha_dict)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default="test", type=str,
                        dest='phase', help='The phase of module.')
    args = parser.parse_args()
    
    if args.phase == 'train':
        # train
        history = model.fit(train.batch(500).cache(), validation_data=val.batch(500).cache(), epochs=100)
        model.save_weights('model/AlphaNetV2_Weights_zz500_returns_trade_1000_197_30_2_10_100')

        loss = history.history['loss']
        rms = history.history['root_mean_squared_error']
        acc = history.history['up_down_accuracy']
        val_loss = history.history['val_loss']
        val_rms = history.history['val_root_mean_squared_error']
        val_acc = history.history['val_up_down_accuracy']
        epochs = range(1, len(loss) + 1)
        plot()

    elif args.phase == 'test':
        predict = []
        for i in range(0, 10):
            model.load_weights('model/AlphaNetV2_Weights_zz500_returns_trade_6_%d' % i)
            y_predict = model.predict(val.batch(500).cache())
            _y_predict = y_predict.flatten().tolist()
            predict.append(_y_predict)
        predict_all = np.array(predict)
        y_predict_mean = predict_all.mean(axis=0)

        _y_predict = y_predict_mean.flatten()
        GT = []
        for i, elem in enumerate(val):
            # print(i, '-->', elem[1].numpy())
            # print(elem)
            GT.append(elem[1].numpy())

        not_nan = np.load('not_nan.npy', allow_pickle=True).tolist()
        trans_val = []
        pointer = 0
        for x in not_nan:
            if x:
                trans_val.append(_y_predict[pointer])
                pointer += 1
            else:
                trans_val.append(np.nan)
        
        A = numpy.array(trans_val).reshape((1200, 500))
        A_rank = pd.DataFrame(A).rank(axis=1, pct=True).values
        alpha_dict = numpy.zeros((1200, 500))
        alpha_dict[A_rank > 0.8] = 1
        numpy.save('npy_data/alpha_returns_debug_11.npy', alpha_dict)
        '''
    # result_val = np.array(trans_val).reshape((600, 500))
    # result_val_df = pd.DataFrame(result_val).rank(axis=1, pct=True).values  # 倒序排， 取0-1的前百分之20
    # alpha_dict = numpy.zeros((600, 500))
    # alpha_dict[result_val_df > 0.8] = 1
    # alpha = []
    # for i in range(0, 600):
    #     s = alpha_dict[i, :]
    #     c = np.row_stack((s, s)).tolist()
    #     alpha.append(c)
    # A = numpy.array(alpha).reshape((1200, 500))
    # numpy.save('npy_data/alpha_returns_debug3.npy', A)

    # print(GT)
    # print(y_predict)
    # plt.figure(1)
    # plt.subplots(figsize=(500, 5))
    # plt.plot(GT, label='real_values')
    # plt.plot(y_predict, label='prediction')
    # plt.legend()
    # plt.title('Results')
    # plt.show()
    # combine_data()'''
print('--------------------------finish-------------------------------')

''' 
Conclusion: 1). 直接预测收益率 优于 排序后的预测
            2). 隔一天采样一次的收益率 优于 每天都采样的收益
to do list
            1). 对数据进行预处理， 排除涨停的股票 再进行预测     
            
            
            回测期为 20110131～20200731，调仓周期为 10 个交易日
            4：1，验证集更关注近期样本的表现
            
            
                 
'''

# train, val, dates_info = get_training_set()
# train, val, dates_info = get_test_set()

# for i, elem in enumerate(val):
# print(i, '-->', elem[1].numpy())
# print(elem)
