# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame, Series
from typing import Dict, Optional, Union, Tuple
import logging

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MomentumCCITrendStrategy(IStrategy):
    """
    基于商品通道指数(CCI)的动量趋势跟踪策略
    
    策略说明：
    - 使用长期CCI(50周期)判断整体趋势方向
    - 使用短期CCI(5周期)的零轴穿越作为入场信号
    - 通过状态追踪确保同一趋势周期内只交易一次
    - 当长期CCI反向穿越零轴时平仓
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # 策略基础配置
    can_short: bool = True
    timeframe = "1d"
    
    # 策略参数
    cci_long_period = IntParameter(30, 100, default=50, space="buy")
    cci_short_period = IntParameter(3, 15, default=5, space="buy")
    enable_short_trading = BooleanParameter(default=True, space="buy")
    
    # 风险管理
    stoploss = -0.10  # 10% 止损
    
    # ROI (取利) 设置
    minimal_roi = {
        "0": 0.15,   # 15% 收益立即卖出
        "60": 0.10,  # 1小时后10%收益卖出
        "120": 0.05, # 2小时后5%收益卖出
        "240": 0.02  # 4小时后2%收益卖出
    }
    
    # 订单类型配置
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }
    
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }
    
    # 存储完整数据帧和状态信息
    _full_dataframes = {}
    _pair_states = {}
    
    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "CCI": {
                    "cci_long": {"color": "green", "type": "line"},
                    "cci_short": {"color": "yellow", "type": "line"},
                    "cci_zero": {"color": "gray", "type": "line", "value": 0}
                }
            }
        }
    
    def informative_pairs(self):
        """
        定义需要额外缓存的交易对/时间间隔组合
        """
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict):
        """
        计算策略所需的技术指标
        """
        pair = metadata['pair']
        
        # 计算长期和短期 CCI
        dataframe['cci_long'] = ta.CCI(
            dataframe['high'], 
            dataframe['low'], 
            dataframe['close'], 
            timeperiod=self.cci_long_period.value
        )
        
        dataframe['cci_short'] = ta.CCI(
            dataframe['high'], 
            dataframe['low'], 
            dataframe['close'], 
            timeperiod=self.cci_short_period.value
        )
        
        # 计算CCI穿越零轴的信号
        dataframe['cci_long_cross_below_zero'] = qtpylib.crossed_below(dataframe['cci_long'], 0)
        dataframe['cci_short_cross_below_zero'] = qtpylib.crossed_below(dataframe['cci_short'], 0)
        
        # 初始化状态追踪列
        dataframe['first_crossover_occurred'] = False
        dataframe['first_crossunder_occurred'] = False
        dataframe['buy_signal'] = False
        dataframe['sell_signal'] = False
        dataframe['exit_long_signal'] = False
        dataframe['exit_short_signal'] = False
        
        # 实现状态追踪逻辑
        if pair not in self._pair_states:
            self._pair_states[pair] = {
                'first_crossover_occurred': False,
                'first_crossunder_occurred': False,
                'last_cci_long_positive': False,
                'last_cci_long_negative': False
            }
        
        state = self._pair_states[pair]
        
        for i in range(len(dataframe)):
            current_cci_long = dataframe.iloc[i]['cci_long']
            current_cci_short = dataframe.iloc[i]['cci_short']
            prev_cci_long = dataframe.iloc[i-1]['cci_long'] if i > 0 else 0
            
            # 重置状态当CCI Long改变符号时
            if i > 0:
                if prev_cci_long <= 0 and current_cci_long > 0:
                    # CCI Long从负转正，重置crossunder状态
                    state['first_crossunder_occurred'] = False
                    state['last_cci_long_positive'] = True
                    state['last_cci_long_negative'] = False
                elif prev_cci_long >= 0 and current_cci_long < 0:
                    # CCI Long从正转负，重置crossover状态
                    state['first_crossover_occurred'] = False
                    state['last_cci_long_positive'] = False
                    state['last_cci_long_negative'] = True
            
            # 更新dataframe中的状态
            dataframe.iloc[i, dataframe.columns.get_loc('first_crossover_occurred')] = state['first_crossover_occurred']
            dataframe.iloc[i, dataframe.columns.get_loc('first_crossunder_occurred')] = state['first_crossunder_occurred']
            
            # 买入信号逻辑 (做多)
            buy_signal = (
                current_cci_long > 0 and 
                prev_cci_long > 0 and 
                dataframe.iloc[i]['cci_short_cross_above_zero'] and 
                not state['first_crossover_occurred']
            )
            
            if buy_signal:
                state['first_crossover_occurred'] = True
                dataframe.iloc[i, dataframe.columns.get_loc('buy_signal')] = True
            
            # 卖出信号逻辑 (做空)
            sell_signal = (
                current_cci_long < 0 and 
                prev_cci_long < 0 and 
                dataframe.iloc[i]['cci_short_cross_below_zero'] and 
                not state['first_crossunder_occurred']
            )
            
            if sell_signal:
                state['first_crossunder_occurred'] = True
                dataframe.iloc[i, dataframe.columns.get_loc('sell_signal')] = True
            
            # 平多仓信号
            if dataframe.iloc[i]['cci_long_cross_below_zero']:
                dataframe.iloc[i, dataframe.columns.get_loc('exit_long_signal')] = True
            
            # 平空仓信号
            if dataframe.iloc[i]['cci_long_cross_above_zero']:
                dataframe.iloc[i, dataframe.columns.get_loc('exit_short_signal')] = True
        
        # 保存完整数据帧
        self._full_dataframes[pair] = dataframe.copy()
        
        logger.info(f"{pair}: CCI指标计算完成，长期CCI周期={self.cci_long_period.value}，短期CCI周期={self.cci_short_period.value}")
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict):
        """
        基于技术指标填充入场信号
        """
        # 做多入场条件
        dataframe.loc[
            (dataframe['buy_signal'] == True),
            'enter_long'
        ] = 1
        
        # 做空入场条件（如果启用）
        if self.enable_short_trading.value:
            dataframe.loc[
                (dataframe['sell_signal'] == True),
                'enter_short'
            ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict):
        """
        基于技术指标填充出场信号
        """
        # 做多出场条件
        dataframe.loc[
            (dataframe['exit_long_signal'] == True),
            'exit_long'
        ] = 1
        
        # 做空出场条件（如果启用）
        if self.enable_short_trading.value:
            dataframe.loc[
                (dataframe['exit_short_signal'] == True),
                'exit_short'
            ] = 1
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, 
                          rate: float, time_in_force: str, current_time: datetime, 
                          entry_tag: str | None, side: str, **kwargs):
        """
        确认交易入场
        """
        try:
            logger.info(f"{pair}: 确认{side}交易入场 - 价格: {rate}, 数量: {amount}, 时间: {current_time}")
            return True
        except Exception as e:
            logger.error(f"{pair}: 确认交易入场时出错: {e}")
            return False
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs):
        """
        自定义退出逻辑
        """
        try:
            # 获取当前数据
            dataframe = self._full_dataframes.get(pair)
            if dataframe is None:
                return None
            
            # 找到当前时间对应的数据行
            current_data = dataframe[dataframe['date'] <= current_time]
            if current_data.empty:
                return None
            
            current_row = current_data.iloc[-1]
            
            # 检查是否触发退出条件
            if trade.is_short:
                # 空头仓位检查
                if current_row['exit_short_signal']:
                    logger.info(f"{pair}: CCI信号触发空头退出")
                    return "CCI Exit Short"
            else:
                # 多头仓位检查
                if current_row['exit_long_signal']:
                    logger.info(f"{pair}: CCI信号触发多头退出")
                    return "CCI Exit Long"
            
            return None
            
        except Exception as e:
            logger.error(f"{pair}: 自定义退出逻辑出错: {e}")
            return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str, 
                side: str, **kwargs):
        """
        调整杠杆率
        """
        return 3  # 使用3倍杠杆 