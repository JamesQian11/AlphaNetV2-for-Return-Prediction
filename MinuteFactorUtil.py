#!/usr/bin/env python
# coding: utf-8

import gc
import warnings

import numpy as np
import pandas as pd

from WMBYPlatform import StrategyPlatform

warnings.filterwarnings('ignore')


class MinuteFactorTest(object):
    def __init__(self, start_date, end_date, stock_list_file, sampling_step, return_periods):
        """

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_list_file: 股票池文件
            sampling_step (int): 几分钟采样一次
            return_periods (list): 取未来几分钟的收益率
        """
        self.start_date = start_date
        self.end_date = end_date
        self.stock_list_file = stock_list_file
        self.set_factor_step = sampling_step
        if 240 % self.set_factor_step != 0:
            raise ValueError(f"set_factor_step {sampling_step} does not divide 240")
        self.return_list = return_periods.copy()

        self.platform = StrategyPlatform()
        self.platform.SetDate(self.start_date, self.end_date)
        self.platform.SetStockListByCSVFile(self.stock_list_file)
        self.date_range = self.platform.date_num_list_trading

        self.__date_returns = None

    def get_return(self):
        """获取收益率数据"""
        if self.__date_returns is None:
            self.__date_returns = {}
            for i in self.date_range:
                daily_close = self.platform.LoadMinute('ashare', 'close', date_num=i).replace(0, np.nan)
                daily_returns = {}
                for step in self.return_list:
                    daily_return = daily_close.pct_change(step).shift(periods=-step)
                    daily_return_choose = daily_return.loc[daily_return.index[::self.set_factor_step]]
                    daily_returns.update({step: daily_return_choose})
                self.__date_returns.update({i: daily_returns})
        return self.__date_returns

    def clear_returns(self):
        """清空收益率数据"""
        self.__date_returns = None
        gc.collect()

    def get_table(self, factor, num_group=10):
        """给定分组并采样好的因子值，计算每天的收益率矩阵"""
        returns = self.get_return()

        required_shape = (240 / self.set_factor_step, self.platform.code_size)
        if any([f.shape != required_shape for f in factor.values()]):
            print([f.shape for f in factor.values()])
            raise ValueError("Bad shape")

        tables = {}
        for date in self.date_range:
            daily_factor = factor[date]
            table_all = pd.DataFrame()
            for key in self.return_list:
                daily_return = returns[date][key]
                all_in = []
                all_nu = []
                for i in range(1, num_group + 1):
                    returns_mean = daily_return[daily_factor == float(i)].mean(1).mean()
                    number_mean = (daily_factor == float(i)).sum(1).mean()

                    all_in.append(returns_mean)
                    all_nu.append(number_mean)

                index = [str(i) for i in range(1, num_group + 1)]
                table_all['number'] = pd.Series(all_nu, index=index)
                table_all[str(key)] = pd.Series(all_in, index=index)
            tables.update({date: table_all})
        return tables

    def get_avg_table(self, factors: dict, num_group: int = 10) -> pd.DataFrame:
        """给定因子，计算每组平均收益率矩阵

        Args:
            factors: 从日期八位数字到因子矩阵的映射。矩阵形状应为 (240, 股票数)
            num_group: 分成几组

        Returns:
            平均收益率矩阵
        """
        kf = set(factors.keys())
        gi = set(self.date_range)
        if kf == gi:
            rank_factors = self.rank_factor(factors, num_group)
            tables = self.get_table(rank_factors, num_group)

            table = tables[self.date_range[0]]
            for date in self.date_range[1:]:
                table = table + tables[date]
            return table / (len(self.date_range))
        else:
            raise ValueError("Your data doesn't match the platform's")

    def rank_factor(self, factors, num_group=10):
        """对因子进行分组和采样"""
        rank_factors = {}
        for date, value in factors.items():
            daily_factor = factors[date].replace(0, np.nan)
            daily_factor_choosed = daily_factor.loc[daily_factor.index[::self.set_factor_step]]
            daily_factor_rank = (daily_factor_choosed.rank(axis=1, pct=True) * num_group).apply(np.ceil)

            rank_factors.update({date: daily_factor_rank})
        return rank_factors

    def get_sample_factor(self):
        """获取示例因子"""
        factors = {}
        for i in self.date_range:
            daily_factor = self.platform.LoadMinute('ashare', 'high', date_num=i).replace(0, np.nan)
            factors.update({i: daily_factor})
        return factors


if __name__ == '__main__':
    start = 20221011
    end = 20221021
    stock_list = '/data/home/wmbytrader/stockPools/ashare_20211130.csv'
    set_factor_step_in = 15
    return_list_in = [1, 5, 10, 30, 60, 120]
    a = MinuteFactorTest(start, end, stock_list, set_factor_step_in, return_list_in)
    date_factors = a.get_sample_factor()  # 提取close数据作为例子
    a.get_avg_table(date_factors, 5)