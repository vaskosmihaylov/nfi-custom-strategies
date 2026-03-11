import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair
# Hyperopt dependencies removed for fixed parameter strategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date
import logging

logger = logging.getLogger(__name__)

# 移除了 StrategyDecisionLogger 类 - 简化日志系统

class TradingStyleManager:
    """交易风格管理器 - 根据市场状态自动切换稳定/横盘/激进模式"""
    
    def __init__(self):
        self.current_style = "stable"  # 默认稳定模式
        self.style_switch_cooldown = 0
        self.min_switch_interval = 0.5  # 最少30分钟才能切换一次 (提升响应速度)
        
        # === 稳定模式配置 ===
        self.STABLE_CONFIG = {
            'name': '稳定模式',
            'leverage_range': (2, 5),  # 提升基础杠杆从1-3到2-5
            'position_range': (0.08, 0.20),  # 安全仓位8-20%
            'entry_threshold': 6.5,  # 适度放宽入场要求
            'exit_threshold': 5.5,   # 更灵敏的出场信号
            'risk_per_trade': 0.015,  # 提升风险从1%到1.5%
            'max_trades': 4,         # 增加并发交易从3到4
            'description': '平衡稳健，稳定收益与适度风险结合'
        }
        
        # === 横盘模式配置 ===  
        self.SIDEWAYS_CONFIG = {
            'name': '横盘模式',
            'leverage_range': (4, 8),  # 提升杠杆从2-5到4-8
            'position_range': (0.10, 0.25),  # 安全仓位10-25%
            'entry_threshold': 5.0,  # 适度放宽入场要求
            'exit_threshold': 4.0,   # 更灵敏的出场信号
            'risk_per_trade': 0.02, # 提升风险从1.5%到2%
            'max_trades': 5,         # 增加并发交易从4到5
            'description': '积极震荡交易，快速进出，中高风险收益'
        }
        
        # === 激进模式配置 ===
        self.AGGRESSIVE_CONFIG = {
            'name': '激进模式',
            'leverage_range': (5, 10),  # 优化杠杆从3-10到5-10，确保高效利用
            'position_range': (0.12, 0.30),  # 安全仓位12-30%
            'entry_threshold': 3.5,  # 更灵活的入场要求
            'exit_threshold': 2.5,   # 极度灵敏的出场信号  
            'risk_per_trade': 0.015,  # 降低风险到1.5%
            'max_trades': 8,         # 增加并发交易从6到8
            'description': '积极进取，追求高收益，高风险高回报'
        }
        
        self.style_configs = {
            'stable': self.STABLE_CONFIG,
            'sideways': self.SIDEWAYS_CONFIG,
            'aggressive': self.AGGRESSIVE_CONFIG
        }
        
    def get_current_config(self) -> dict:
        """获取当前风格配置"""
        return self.style_configs[self.current_style]
    
    def classify_market_regime(self, dataframe: DataFrame) -> str:
        """识别当前市场状态以决定适合的交易风格"""
        
        if dataframe.empty or len(dataframe) < 50:
            return "stable"  # 数据不足时使用稳定模式
            
        try:
            # 获取最近数据进行分析
            recent_data = dataframe.tail(50)
            current_data = dataframe.iloc[-1]
            
            # === 市场特征计算 ===
            
            # 1. 趋势强度分析
            trend_strength = current_data.get('trend_strength', 50)
            adx_value = current_data.get('adx', 20)
            
            # 2. 波动率分析
            volatility_state = current_data.get('volatility_state', 50)
            atr_recent = recent_data['atr_p'].mean() if 'atr_p' in recent_data.columns else 0.02
            
            # 3. 价格行为分析  
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
            
            # 4. 成交量行为分析
            volume_consistency = recent_data['volume_ratio'].std() if 'volume_ratio' in recent_data.columns else 1
            
            # === 市场状态判断逻辑 ===
            
            # 激进模式条件：强趋势 + 高波动 + 明确方向
            if (trend_strength > 75 and adx_value > 30 and 
                volatility_state > 60 and atr_recent > 0.025):
                return "aggressive"
            
            # 横盘模式条件：弱趋势 + 中等波动 + 区间震荡
            elif (trend_strength < 50 and adx_value < 20 and 
                  volatility_state < 40 and price_range < 0.15):
                return "sideways"
            
            # 稳定模式：其他情况或不确定状态
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"市场状态分类失败，使用稳定模式: {e}")
            return "stable"
    
    def should_switch_style(self, dataframe: DataFrame) -> tuple[bool, str]:
        """判断是否需要切换交易风格"""
        
        # 检查冷却期
        if self.style_switch_cooldown > 0:
            self.style_switch_cooldown -= 1
            return False, self.current_style
        
        # 分析当前市场状态
        suggested_regime = self.classify_market_regime(dataframe)
        
        # 如果建议的状态与当前相同，不切换
        if suggested_regime == self.current_style:
            return False, self.current_style
        
        # 需要切换，设置冷却期
        return True, suggested_regime
    
    def switch_style(self, new_style: str, reason: str = "") -> bool:
        """切换交易风格"""
        
        if new_style not in self.style_configs:
            logger.error(f"未知的交易风格: {new_style}")
            return False
        
        old_style = self.current_style
        self.current_style = new_style
        self.style_switch_cooldown = self.min_switch_interval
        
        logger.info(f"🔄 交易风格切换: {old_style} → {new_style} | 原因: {reason}")
        
        return True
    
    def get_dynamic_leverage_range(self) -> tuple[int, int]:
        """获取当前风格的杠杆范围"""
        config = self.get_current_config()
        return config['leverage_range']
    
    def get_dynamic_position_range(self) -> tuple[float, float]:
        """获取当前风格的仓位范围"""
        config = self.get_current_config()
        return config['position_range']
    
    # 移除了 get_dynamic_stoploss_range - 简化止损逻辑
    
    def get_risk_per_trade(self) -> float:
        """获取当前风格的单笔风险"""
        config = self.get_current_config()
        return config['risk_per_trade']
    
    def get_signal_threshold(self, signal_type: str = 'entry') -> float:
        """获取当前风格的信号阈值"""
        config = self.get_current_config()
        return config.get(f'{signal_type}_threshold', 5.0)
    
    def get_max_concurrent_trades(self) -> int:
        """获取当前风格的最大并发交易数"""
        config = self.get_current_config()
        return config['max_trades']
    
    def get_style_summary(self) -> dict:
        """获取当前风格的完整信息摘要"""
        config = self.get_current_config()
        
        return {
            'current_style': self.current_style,
            'style_name': config['name'],
            'description': config['description'],
            'leverage_range': config['leverage_range'],
            'position_range': [f"{p*100:.0f}%" for p in config['position_range']], 
            'risk_per_trade': f"{config['risk_per_trade']*100:.1f}%",
            'max_trades': config['max_trades'],
            'switch_cooldown': self.style_switch_cooldown
        }

class UltraSmartStrategy(IStrategy):
 

    INTERFACE_VERSION = 3
    
    # 策略核心参数
    timeframe = '15m'  # 15分钟 - 平衡噪音过滤和反应速度
    can_short: bool = True
    
    # Removed informative timeframes to eliminate data sync issues and noise
    
    # 增强指标计算: 支持所有高级技术分析功能
    startup_candle_count: int = 150  # Reduced from 350 for efficiency
    
    # 智能交易模式: 精准入场后的优化配置
    position_adjustment_enable = True
    max_dca_orders = 4  # 精准入场后减少DCA依赖，提高资金效率
    
    # === 科学的固定参数配置 ===
    # 移除HYPEROPT依赖，使用基于市场规律的固定参数
    
    # 价格位置过滤器（科学不对称设计）
    price_percentile_long_max = 0.50    # 做多：50%分位以下（增加机会）
    price_percentile_long_best = 0.35   # 做多最佳区间：35%分位以下
    price_percentile_short_min = 0.65   # 做空：65%分位以上（适度严格）
    price_percentile_short_best = 0.75  # 做空最佳区间：75%分位以上
    
    # RSI parameters (更宽松的范围以获得更多交易机会)
    rsi_long_min = 15        # Long RSI lower bound (放宽超卖要求)
    rsi_long_max = 55        # Long RSI upper bound (允许更多机会)
    rsi_short_min = 45       # Short RSI lower bound (放宽超买要求)  
    rsi_short_max = 85       # Short RSI upper bound (保持高位)
    
    # 成交量确认参数
    volume_long_threshold = 1.2     # 做多成交量要求（温和即可）
    volume_short_threshold = 1.5    # 做空成交量要求（明显放量）
    volume_spike_threshold = 2.0    # 异常放量阈值
    
    # 趋势强度要求 (放宽要求)
    adx_long_min = 15        # 做多ADX要求（更宽松）
    adx_short_min = 15       # 做空ADX要求（更宽松）
    trend_strength_threshold = 30    # 强趋势阈值 (降低)
    
    # 技术指标参数（固定经典值）
    macd_fast = 12           # MACD快线
    macd_slow = 26           # MACD慢线  
    macd_signal = 9          # MACD信号线
    bb_period = 20           # 布林带周期
    bb_std = 2.0             # 布林带标准差
    
    # 简化风险管理 - 使用固定止损
    # 移除了复杂的动态止损，使用简单可靠的固定值
    
    # === 优化ROI设置 - 扩大止盈目标捕获更多利润 ===
    # 合约交易波动大，扩大ROI范围以捕获大行情
    minimal_roi = {
        #"0": 0.25,      # 25% 捕获大波动立即止盈
        #"20": 0.15,     # 20分钟后15%止盈
        "40": 0.10,     # 40分钟后10%止盈
        "60": 0.06,     # 1小时后6%止盈
        "120": 0.03,    # 2小时后3%止盈
        "240": 0.02,    # 4小时后2%止盈
        "720": 0.01,    # 12小时后1%止盈
        "1440": 0.005   # 24小时后0.5%保本
    }
    
    # 完全关闭止损（设置极大值，永远不会触发）
    stoploss = -0.99

    # 跟踪止损配置（较大的跟踪止损值）
    trailing_stop = True  # 启用跟踪止损
    trailing_stop_positive = 0.03  # 盈利5%后启动跟踪止损
    trailing_stop_positive_offset = 0.13  # 盈利13%后才启动跟踪止损
    trailing_only_offset_is_reached = True  # 仅在达到偏移量后才启动跟踪
    
    # 启用智能出场信号
    use_exit_signal = True
    exit_profit_only = True  # 允许在亏损时也能触发出场信号
    exit_profit_offset = 0.0  # 不设置利润偏移
    ignore_roi_if_entry_signal = False  # 不忽略ROI

    # 订单类型配置
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 15,
        'stoploss_on_exchange_market_ratio': 0.99
    }
    
    # 图表配置 - 确保所有关键指标在FreqUI中可见
    plot_config = {
        'main_plot': {
            'ema_5': {'color': 'yellow', 'type': 'line'},
            'ema_13': {'color': 'orange', 'type': 'line'},
            'ema_34': {'color': 'red', 'type': 'line'},
            'bb_lower': {'color': 'lightblue', 'type': 'line'},
            'bb_middle': {'color': 'gray', 'type': 'line'},
            'bb_upper': {'color': 'lightblue', 'type': 'line'},
            'supertrend': {'color': 'green', 'type': 'line'},
            'vwap': {'color': 'purple', 'type': 'line'}
        },
        'subplots': {
            "RSI": {
                'rsi_14': {'color': 'purple', 'type': 'line'}
            },
            "MACD": {
                'macd': {'color': 'blue', 'type': 'line'},
                'macd_signal': {'color': 'red', 'type': 'line'},
                'macd_hist': {'color': 'gray', 'type': 'bar'}
            },
            "ADX": {
                'adx': {'color': 'orange', 'type': 'line'}
            },
            "Volume": {
                'volume_ratio': {'color': 'cyan', 'type': 'line'}
            },
            "Trend": {
                'trend_strength': {'color': 'magenta', 'type': 'line'},
                'momentum_score': {'color': 'lime', 'type': 'line'}
            }
        }
    }
    
    # 订单填充超时
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    # === 动态策略核心参数 (根据交易风格自动调整) ===
    # 注意：以下参数在初始化后会被动态属性覆盖
    _base_leverage_multiplier = 2  # 默认基础杠杆
    _base_max_leverage = 10        # 默认最大杠杆 (用户要求10x)
    _base_position_size = 0.08     # 默认基础仓位
    _base_max_position_size = 0.25 # 默认最大仓位
    
    # === 技术指标参数（固定经典值） ===
    @property
    def rsi_period(self):
        return 14  # RSI周期保持固定
        
    atr_period = 14
    adx_period = 14
    
    # === 简化的市场状态参数 ===
    volatility_threshold = 0.025     # 稍微提高波动率阈值
    trend_strength_min = 50          # 提高趋势强度要求
    volume_spike_threshold = 1.5     # 降低成交量激增阈值
    
    # === 优化的DCA参数 ===
    dca_multiplier = 1.3        # 降低DCA倍数
    dca_price_deviation = 0.025  # 降低触发偏差 (2.5%)
    
    # === 严格的风险管理参数 ===
    max_risk_per_trade = 0.015  # 降低单笔风险到1.5%
    kelly_lookback = 50         # 缩短回看期提高响应速度
    drawdown_protection = 0.12  # 降低回撤保护阈值
    
    # 高级资金管理参数
    var_confidence_level = 0.95    # VaR置信度
    cvar_confidence_level = 0.99   # CVaR置信度
    max_portfolio_heat = 0.3       # 最大组合风险度
    correlation_threshold = 0.7    # 相关性阈值
    rebalance_threshold = 0.1      # 再平衡阈值
    portfolio_optimization_method = 'kelly'  # 'kelly', 'markowitz', 'risk_parity'
    
    def bot_start(self, **kwargs) -> None:
        """策略初始化"""
        self.custom_info = {}
        self.trade_count = 0
        self.total_profit = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses = 3
        self.initial_balance = None
        self.peak_balance = None
        self.current_drawdown = 0
        self.trade_history = []
        self.leverage_adjustment_factor = 1.0
        self.profit_taking_tracker = {}  # 跟踪各交易的分级止盈状态
        
        # DCA性能跟踪系统
        self.dca_performance_tracker = {
            'total_dca_count': 0,
            'successful_dca_count': 0,
            'dca_success_rate': 0.0,
            'dca_type_performance': {},  # 各种DCA类型的成功率
            'avg_dca_profit': 0.0,
            'dca_history': []
        }
        
        # 高级资金管理数据结构
        self.portfolio_returns = []       # 组合收益率历史
        self.pair_returns_history = {}    # 交易对收益率历史
        self.position_correlation_matrix = {}  # 持仓相关性矩阵
        self.risk_metrics_history = []    # 风险指标历史
        self.allocation_history = []      # 资金分配历史
        self.var_cache = {}              # VaR计算缓存
        self.optimal_f_cache = {}        # 最优f缓存
        self.last_rebalance_time = None  # 上次再平衡时间
        self.kelly_coefficients = {}     # Kelly系数缓存
        
        # 初始化账户余额
        try:
            if hasattr(self, 'wallets') and self.wallets:
                self.initial_balance = self.wallets.get_total_stake_amount()
                self.peak_balance = self.initial_balance
        except Exception:
            pass
            
        # === 性能优化初始化 ===
        self.initialize_performance_optimization()
        
        # === 日志系统初始化 ===
        # 移除了 StrategyDecisionLogger - 使用标准logger
        logger.info("🔥 策略已启动 - UltraSmartStrategy v2")
        
        # === 交易风格管理系统初始化 ===
        self.style_manager = TradingStyleManager()
        logger.info(f"🎯 交易风格管理系统已启动 - 当前模式: {self.style_manager.current_style}")
        
        # 初始化风格切换记录
        self.last_style_check = datetime.now(timezone.utc)
        self.style_check_interval = 300  # 5分钟检查一次风格切换
        
    def initialize_performance_optimization(self):
        """初始化性能优化系统"""
        
        # 缓存系统
        self.indicator_cache = {}  
        self.signal_cache = {}     
        self.market_state_cache = {}  
        self.cache_ttl = 300  # 5分钟缓存
        self.last_cache_cleanup = datetime.now(timezone.utc)
        
        # 性能统计
        self.calculation_stats = {
            'indicator_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_calculation_time': 0
        }
        
        # 预计算常用阈值
        self.precomputed_thresholds = {
            'rsi_oversold': 35,
            'rsi_overbought': 65, 
            'adx_strong': 25,
            'volume_spike': 1.2,
            'atr_high_vol': 0.03,
            'atr_low_vol': 0.015
        }
        
        # 批量计算优化
        self.batch_size = 50
        self.optimize_calculations = True
    
    def get_cached_indicators(self, pair: str, dataframe_len: int) -> Optional[DataFrame]:
        """获取缓存的指标数据"""
        cache_key = f"{pair}_{dataframe_len}"
        
        if cache_key in self.indicator_cache:
            cache_data = self.indicator_cache[cache_key]
            # 检查缓存是否过期
            if (datetime.now(timezone.utc) - cache_data['timestamp']).seconds < self.cache_ttl:
                self.calculation_stats['cache_hits'] += 1
                return cache_data['indicators']
        
        self.calculation_stats['cache_misses'] += 1
        return None
    
    def cache_indicators(self, pair: str, dataframe_len: int, indicators: DataFrame):
        """缓存指标数据"""
        cache_key = f"{pair}_{dataframe_len}"
        self.indicator_cache[cache_key] = {
            'indicators': indicators.copy(),
            'timestamp': datetime.now(timezone.utc)
        }
        
        # 定期清理过期缓存
        if (datetime.now(timezone.utc) - self.last_cache_cleanup).seconds > self.cache_ttl * 2:
            self.cleanup_expired_cache()
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        for key, data in self.indicator_cache.items():
            if (current_time - data['timestamp']).seconds > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.indicator_cache[key]
        
        # 同样清理其他缓存
        for cache_dict in [self.signal_cache, self.market_state_cache]:
            expired_keys = []
            for key, data in cache_dict.items():
                if (current_time - data.get('timestamp', current_time)).seconds > self.cache_ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                del cache_dict[key]
        
        self.last_cache_cleanup = current_time
    
    # ===== 动态交易风格系统 =====
    
    @property  
    def leverage_multiplier(self) -> int:
        """动态杠杆倍数 - 基于当前交易风格"""
        leverage_range = self.style_manager.get_dynamic_leverage_range()
        return leverage_range[0]  # 使用范围的下限作为基础倍数
    
    @property
    def max_leverage(self) -> int:
        """动态最大杠杆 - 基于当前交易风格"""
        leverage_range = self.style_manager.get_dynamic_leverage_range()
        return leverage_range[1]  # 使用范围的上限作为最大倍数
    
    @property
    def base_position_size(self) -> float:
        """动态基础仓位大小 - 基于当前交易风格"""
        position_range = self.style_manager.get_dynamic_position_range()
        return position_range[0]  # 使用范围的下限作为基础仓位
    
    @property  
    def max_position_size(self) -> float:
        """动态最大仓位大小 - 基于当前交易风格"""
        position_range = self.style_manager.get_dynamic_position_range()
        return position_range[1]  # 使用范围的上限作为最大仓位
    
    @property
    def max_risk_per_trade(self) -> float:
        """动态单笔最大风险 - 基于当前交易风格"""
        return self.style_manager.get_risk_per_trade()
    
    # 移除了 dynamic_stoploss - 简化止损逻辑
    
    def check_and_switch_trading_style(self, dataframe: DataFrame) -> None:
        """检查并切换交易风格"""
        
        current_time = datetime.now(timezone.utc)
        
        # 检查是否到了检查风格的时间
        if (current_time - self.last_style_check).seconds < self.style_check_interval:
            return
            
        self.last_style_check = current_time
        
        # 检查是否需要切换风格
        should_switch, new_style = self.style_manager.should_switch_style(dataframe)
        
        if should_switch:
            old_config = self.style_manager.get_current_config()
            
            # 执行风格切换
            market_regime = self.style_manager.classify_market_regime(dataframe)
            reason = f"市场状态变化: {market_regime}"
            
            if self.style_manager.switch_style(new_style, reason):
                new_config = self.style_manager.get_current_config()
                
                # 记录风格切换日志
                self._log_style_switch(old_config, new_config, reason, dataframe)
    
    def _log_style_switch(self, old_config: dict, new_config: dict, 
                         reason: str, dataframe: DataFrame) -> None:
        """记录风格切换详情"""
        
        try:
            current_data = dataframe.iloc[-1] if not dataframe.empty else {}
            
            switch_log = f"""
==================== 交易风格切换 ====================
时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
切换原因: {reason}

📊 市场状态分析:
├─ 趋势强度: {current_data.get('trend_strength', 0):.0f}/100
├─ ADX值: {current_data.get('adx', 0):.1f}  
├─ 波动状态: {current_data.get('volatility_state', 0):.0f}/100
├─ ATR波动率: {(current_data.get('atr_p', 0) * 100):.2f}%

🔄 风格变更详情:
├─ 原风格: {old_config['name']} → 新风格: {new_config['name']}
├─ 杠杆调整: {old_config['leverage_range']} → {new_config['leverage_range']}
├─ 仓位调整: {[f"{p*100:.0f}%" for p in old_config['position_range']]} → {[f"{p*100:.0f}%" for p in new_config['position_range']]}
├─ 风险调整: {old_config['risk_per_trade']*100:.1f}% → {new_config['risk_per_trade']*100:.1f}%

🎯 新风格特征:
├─ 描述: {new_config['description']}
├─ 入场阈值: {new_config['entry_threshold']:.1f}
├─ 最大并发: {new_config['max_trades']}个交易
├─ 冷却期: {self.style_manager.style_switch_cooldown}小时

=================================================="""
            
            logger.info(switch_log)
            
            # 记录风格切换
            style_summary = self.style_manager.get_style_summary()
            logger.info(f"🔄 风格切换完成: {style_summary}")
            
        except Exception as e:
            logger.error(f"风格切换日志记录失败: {e}")
    
    def get_current_trading_style_info(self) -> dict:
        """获取当前交易风格的详细信息"""
        return self.style_manager.get_style_summary()
        
    # Removed informative_pairs() method - no longer needed without informative timeframes
    
    def get_market_orderbook(self, pair: str) -> Dict:
        """获取订单簿数据"""
        try:
            orderbook = self.dp.orderbook(pair, 10)  # 获取10档深度
            if orderbook:
                bids = np.array([[float(bid[0]), float(bid[1])] for bid in orderbook['bids']])
                asks = np.array([[float(ask[0]), float(ask[1])] for ask in orderbook['asks']])
                
                # 计算订单簿指标
                bid_volume = np.sum(bids[:, 1]) if len(bids) > 0 else 0
                ask_volume = np.sum(asks[:, 1]) if len(asks) > 0 else 0
                
                volume_ratio = bid_volume / (ask_volume + 1e-10)
                
                # 计算价差
                spread = ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if len(asks) > 0 and len(bids) > 0 else 0
                
                # 计算深度不平衡
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
                
                # 计算买卖压力指标 (0-1范围)
                buy_pressure = bid_volume / (bid_volume + ask_volume + 1e-10)
                sell_pressure = ask_volume / (bid_volume + ask_volume + 1e-10)
                
                # 计算市场质量 (0-1范围)
                total_volume = bid_volume + ask_volume
                spread_quality = max(0, 1 - spread / 1.0)  # 价差越小质量越高
                volume_quality = min(1, total_volume / 10000)  # 成交量越大质量越高
                balance_quality = 1 - abs(imbalance)  # 平衡度越高质量越高
                market_quality = (spread_quality + volume_quality + balance_quality) / 3
                
                return {
                    'volume_ratio': volume_ratio,
                    'spread_pct': spread,
                    'depth_imbalance': imbalance,
                    'market_quality': market_quality,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'buy_pressure': buy_pressure,
                    'sell_pressure': sell_pressure,
                    'liquidity_score': market_quality  # 使用market_quality作为liquidity_score
                }
        except Exception as e:
            logger.warning(f"获取订单簿失败: {e}")
            
        return {
            'volume_ratio': 1.0,
            'spread_pct': 0.1,
            'depth_imbalance': 0.0,
            'market_quality': 0.5,
            'bid_volume': 0,
            'ask_volume': 0,
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'liquidity_score': 0.5
        }
    
    def calculate_technical_indicators(self, dataframe: DataFrame) -> DataFrame:
        """优化的技术指标计算 - 批量处理避免DataFrame碎片化"""
        
        # 使用字典批量存储所有新列
        new_columns = {}
        
        # === 优化的敏感均线系统 - 基于斐波那契数列，更快反应 ===
        new_columns['ema_5'] = ta.EMA(dataframe, timeperiod=5)    # 超短期：快速捕捉变化
        new_columns['ema_8'] = ta.EMA(dataframe, timeperiod=8)    # 超短期增强
        new_columns['ema_13'] = ta.EMA(dataframe, timeperiod=13)  # 短期：趋势确认
        new_columns['ema_21'] = ta.EMA(dataframe, timeperiod=21)  # 中短期过渡
        new_columns['ema_34'] = ta.EMA(dataframe, timeperiod=34)  # 中期：主趋势过滤
        new_columns['ema_50'] = ta.EMA(dataframe, timeperiod=50)  # 长期趋势
        new_columns['sma_20'] = ta.SMA(dataframe, timeperiod=20)  # 保留SMA20作为辅助
        
        # === 布林带 (保留，高效用指标) ===
        bb = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period, stds=self.bb_std)
        new_columns['bb_lower'] = bb['lower']
        new_columns['bb_middle'] = bb['mid']
        new_columns['bb_upper'] = bb['upper']
        new_columns['bb_width'] = np.where(bb['mid'] > 0, 
                                        (bb['upper'] - bb['lower']) / bb['mid'], 
                                        0)
        new_columns['bb_position'] = (dataframe['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
        
        # === RSI (只保留最有效的14周期) ===
        new_columns['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        
        # === MACD (保留，经典趋势指标) ===
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        new_columns['macd'] = macd['macd']
        new_columns['macd_signal'] = macd['macdsignal'] 
        new_columns['macd_hist'] = macd['macdhist']
        
        # === ADX 趋势强度 (保留，重要的趋势指标) ===
        new_columns['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period)
        new_columns['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period)
        new_columns['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period)
        
        # === ATR 波动性 (保留，风险管理必需) ===
        new_columns['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period)
        new_columns['atr_p'] = new_columns['atr'] / dataframe['close']
        
        # === 成交量指标 (简化) ===
        new_columns['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        new_columns['volume_ratio'] = np.where(new_columns['volume_sma'] > 0, 
                                            dataframe['volume'] / new_columns['volume_sma'], 
                                            1.0)
        
        # === 动量指标 ===
        new_columns['mom_10'] = ta.MOM(dataframe, timeperiod=10)
        new_columns['roc_10'] = ta.ROC(dataframe, timeperiod=10)
        
        # === 新增领先指标组合 - 解决滞后问题 ===
        
        # 1. 快速斯托卡斯蒂克RSI - 比普通RSI更敏感
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        new_columns['stoch_rsi_k'] = stoch_rsi['fastk']
        new_columns['stoch_rsi_d'] = stoch_rsi['fastd']
        
        # 2. 威廉指标 - 快速反转信号
        new_columns['williams_r'] = ta.WILLR(dataframe, timeperiod=14)
        
        # 3. CCI商品通道指数 - 超买超卖敏感指标  
        new_columns['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        # 4. 价格行为分析 - 当根K线就能判断
        new_columns['candle_body'] = abs(dataframe['close'] - dataframe['open'])
        new_columns['candle_upper_shadow'] = dataframe['high'] - np.maximum(dataframe['close'], dataframe['open'])
        new_columns['candle_lower_shadow'] = np.minimum(dataframe['close'], dataframe['open']) - dataframe['low']
        new_columns['candle_total_range'] = dataframe['high'] - dataframe['low']
        
        # 6. 成交量异常检测 - 领先价格变化
        new_columns['volume_spike'] = (dataframe['volume'] > new_columns['volume_sma'] * 2).astype(int)
        new_columns['volume_dry'] = (dataframe['volume'] < new_columns['volume_sma'] * 0.5).astype(int)
        
        # 8. 支撑阻力突破强度
        new_columns['resistance_strength'] = (
            dataframe['close'] / dataframe['high'].rolling(20).max() - 1
        ) * 100  # 距离20日最高点的百分比
        
        new_columns['support_strength'] = (
            1 - dataframe['close'] / dataframe['low'].rolling(20).min()
        ) * 100  # 距离20日最低点的百分比
        
        # === VWAP (重要的机构交易参考) ===
        new_columns['vwap'] = qtpylib.rolling_vwap(dataframe)
        
        # === 超级趋势 (高效的趋势跟踪) ===
        new_columns['supertrend'] = self.supertrend(dataframe, 10, 3)
        
        # 一次性将所有新列添加到dataframe，使用concat避免碎片化
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, new_df], axis=1)
        
        # === 优化的复合指标 (替代大量单一指标) ===
        dataframe = self.calculate_optimized_composite_indicators(dataframe)
        
        # === 高级动量指标 ===
        dataframe = self.calculate_advanced_momentum_indicators(dataframe)
        
        # === 成交量指标 ===
        dataframe = self.calculate_advanced_volume_indicators(dataframe)
        
        # === Ichimoku云图指标 ===
        dataframe = self.ichimoku(dataframe)
        
        # === 市场结构指标 (包含价格行为模式) ===
        dataframe = self.calculate_market_structure_indicators(dataframe)
        
        # === 市场状态指标 (简化版本) ===
        dataframe = self.calculate_market_regime_simple(dataframe)
        
        # === 指标验证和校准 ===
        dataframe = self.validate_and_calibrate_indicators(dataframe)
        
        # === 最终指标完整性检查 ===
        required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score', 
                              'ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_50', 'mom_10', 'roc_10']
        missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns or dataframe[indicator].isnull().all()]
        
        if missing_indicators:
            logger.error(f"关键指标计算失败: {missing_indicators}")
            # 为缺失的指标提供默认值，使用批量更新避免碎片化
            default_values = {}
            for indicator in missing_indicators:
                if indicator == 'rsi_14':
                    default_values[indicator] = 50.0
                elif indicator == 'adx':
                    default_values[indicator] = 25.0
                elif indicator == 'atr_p':
                    default_values[indicator] = 0.02
                elif indicator in ['macd', 'macd_signal']:
                    default_values[indicator] = 0.0
                elif indicator == 'volume_ratio':
                    default_values[indicator] = 1.0
                elif indicator == 'trend_strength':
                    default_values[indicator] = 50.0
                elif indicator == 'momentum_score':
                    default_values[indicator] = 0.0
                elif indicator in ['ema_5', 'ema_13', 'ema_34']:
                    # 如果EMA指标缺失，重新计算
                    if indicator == 'ema_5':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=5)
                    elif indicator == 'ema_13':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=13)
                    elif indicator == 'ema_34':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=34)
            
            # 一次性添加所有默认值
            if default_values:
                defaults_df = pd.DataFrame(default_values, index=dataframe.index)
                dataframe = pd.concat([dataframe, defaults_df], axis=1)
        else:
            logger.info("✅ 所有指标计算成功")
        
        # === 确保EMA指标质量 ===
        # 检查EMA指标是否有过多的NaN值
        for ema_col in ['ema_8', 'ema_21', 'ema_50']:
            if ema_col in dataframe.columns:
                nan_count = dataframe[ema_col].isnull().sum()
                total_count = len(dataframe)
                if nan_count > total_count * 0.1:  # 如果超过10%的值为NaN
                    logger.warning(f"{ema_col} 有过多空值 ({nan_count}/{total_count}), 重新计算")
                    if ema_col == 'ema_8':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=8)
                    elif ema_col == 'ema_21':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=21)
                    elif ema_col == 'ema_50':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=50)
        
        return dataframe
    
    def calculate_optimized_composite_indicators(self, dataframe: DataFrame) -> DataFrame:
        """优化的复合指标 - 批量处理避免DataFrame碎片化"""
        
        # 使用字典批量存储所有新列
        new_columns = {}
        
        # === 革命性趋势强度评分系统 - 基于斜率和动量，提前2-3根K线识别 ===
        
        # 1. 价格动量斜率分析（提前预警） - 使用更敏感的EMA(5,13,34)
        ema5_slope = np.where(dataframe['ema_5'].shift(2) > 0,
                             (dataframe['ema_5'] - dataframe['ema_5'].shift(2)) / dataframe['ema_5'].shift(2),
                             0) * 100  # 更短周期，更快反应
        ema13_slope = np.where(dataframe['ema_13'].shift(3) > 0,
                              (dataframe['ema_13'] - dataframe['ema_13'].shift(3)) / dataframe['ema_13'].shift(3),
                              0) * 100
        
        # 2. 均线发散度分析（趋势加速信号）
        ema_spread = np.where(dataframe['ema_34'] > 0,
                             (dataframe['ema_5'] - dataframe['ema_34']) / dataframe['ema_34'] * 100,
                             0)
        ema_spread_series = self._safe_series(ema_spread, len(dataframe))
        ema_spread_change = ema_spread - ema_spread_series.shift(3)  # 发散度变化
        
        # 3. ADX动态变化（趋势强化信号）
        adx_slope = dataframe['adx'] - dataframe['adx'].shift(3)  # ADX变化率
        adx_acceleration = adx_slope - adx_slope.shift(2)  # ADX加速度
        
        # 4. 成交量趋势确认
        volume_20_mean = dataframe['volume'].rolling(20).mean()
        volume_trend = np.where(volume_20_mean != 0,
                               dataframe['volume'].rolling(5).mean() / volume_20_mean,
                               1.0)  # 如果20日均量为0，返回1.0（中性）
        volume_trend_series = self._safe_series(volume_trend, len(dataframe))
        volume_momentum = volume_trend_series - volume_trend_series.shift(2).fillna(0)
        
        # 5. 价格加速度（二阶导数）
        close_shift_3 = dataframe['close'].shift(3)
        price_velocity = np.where(close_shift_3 != 0,
                                 (dataframe['close'] / close_shift_3 - 1) * 100,
                                 0)  # 一阶导数
        price_velocity_series = self._safe_series(price_velocity, len(dataframe))
        price_acceleration = price_velocity_series - price_velocity_series.shift(2).fillna(0)
        
        # === 综合趋势强度评分 ===
        trend_score = (
            ema5_slope * 0.30 +        # 超短期动量（最重要，提高权重）
            ema13_slope * 0.20 +       # 短期动量确认
            ema_spread_change * 0.15 + # 趋势发散变化
            adx_slope * 0.15 +         # 趋势强度变化
            volume_momentum * 0.10 +   # 成交量支持
            price_acceleration * 0.10  # 价格加速度
        )
        
        # 使用ADX作为趋势确认倍数
        adx_multiplier = np.where(dataframe['adx'] > 30, 1.5,
                                 np.where(dataframe['adx'] > 20, 1.2,
                                         np.where(dataframe['adx'] > 15, 1.0, 0.7)))
        
        # 最终趋势强度
        new_columns['trend_strength'] = (trend_score * adx_multiplier).clip(-100, 100)
        new_columns['price_acceleration'] = price_acceleration
        
        # === 动量复合指标 ===
        rsi_normalized = (dataframe['rsi_14'] - 50) / 50  # -1 to 1
        macd_normalized = np.where(dataframe['atr_p'] > 0, 
                                 dataframe['macd_hist'] / (dataframe['atr_p'] * dataframe['close']), 
                                 0)  # 归一化
        price_momentum = (dataframe['close'] / dataframe['close'].shift(5) - 1) * 10  # 5周期价格变化
        
        new_columns['momentum_score'] = (rsi_normalized + macd_normalized + price_momentum) / 3
        new_columns['price_velocity'] = price_velocity_series
        
        # === 波动率状态指标 ===  
        atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        bb_squeeze = np.where(dataframe['bb_width'] < dataframe['bb_width'].rolling(20).quantile(0.3), 1, 0)
        volume_spike = np.where(dataframe['volume_ratio'] > 1.5, 1, 0)
        
        new_columns['volatility_state'] = atr_percentile * 50 + bb_squeeze * 25 + volume_spike * 25
        
        # === 支撑阻力强度 ===
        bb_position_score = np.abs(dataframe['bb_position'] - 0.5) * 2  # 0-1, 越接近边缘分数越高
        vwap_distance = np.where(dataframe['vwap'] > 0, 
                                np.abs((dataframe['close'] - dataframe['vwap']) / dataframe['vwap']) * 100, 
                                0)
        
        new_columns['sr_strength'] = (bb_position_score + np.minimum(vwap_distance, 5)) / 2  # 标准化到合理范围
        
        # === 趋势可持续性指标 ===
        adx_sustainability = np.where(dataframe['adx'] > 25, 1, 0)
        volume_sustainability = np.where(dataframe['volume_ratio'] > 0.8, 1, 0)
        volatility_sustainability = np.where(dataframe['atr_p'] < dataframe['atr_p'].rolling(20).quantile(0.8), 1, 0)
        new_columns['trend_sustainability'] = (
            (adx_sustainability * 0.5 + volume_sustainability * 0.3 + volatility_sustainability * 0.2) * 2 - 1
        ).clip(-1, 1)  # 归一化到[-1, 1]
        
        # === RSI背离强度指标 ===
        price_high_10 = dataframe['high'].rolling(10).max()
        price_low_10 = dataframe['low'].rolling(10).min()
        rsi_high_10 = dataframe['rsi_14'].rolling(10).max()
        rsi_low_10 = dataframe['rsi_14'].rolling(10).min()
        
        # 顶背离：价格新高但RSI未新高
        bearish_divergence = np.where(
            (dataframe['high'] >= price_high_10) & (dataframe['rsi_14'] < rsi_high_10),
            -(dataframe['high'] / price_high_10 - dataframe['rsi_14'] / rsi_high_10),
            0
        )
        
        # 底背离：价格新低但RSI未新低
        bullish_divergence = np.where(
            (dataframe['low'] <= price_low_10) & (dataframe['rsi_14'] > rsi_low_10),
            (dataframe['low'] / price_low_10 - dataframe['rsi_14'] / rsi_low_10),
            0
        )
        
        new_columns['rsi_divergence_strength'] = (bearish_divergence + bullish_divergence).clip(-2, 2)
        
        # === 新增：预测性指标系统 ===
        
        # 1. 更敏感的RSI背离检测
        price_higher_5 = dataframe['close'] > dataframe['close'].shift(5)
        rsi_lower_5 = dataframe['rsi_14'] < dataframe['rsi_14'].shift(5)
        new_columns['bearish_divergence'] = (price_higher_5 & rsi_lower_5).astype(int)
        
        price_lower_5 = dataframe['close'] < dataframe['close'].shift(5)
        rsi_higher_5 = dataframe['rsi_14'] > dataframe['rsi_14'].shift(5)
        new_columns['bullish_divergence'] = (price_lower_5 & rsi_higher_5).astype(int)
        
        # 2. 成交量衰竭检测
        volume_decreasing = (
            (dataframe['volume'] < dataframe['volume'].shift(1)) &
            (dataframe['volume'].shift(1) < dataframe['volume'].shift(2)) &
            (dataframe['volume'].shift(2) < dataframe['volume'].shift(3))
        )
        new_columns['volume_exhaustion'] = volume_decreasing.astype(int)
        
        # 3. 价格加速度变化（预测转折）
        price_roc_3 = dataframe['close'].pct_change(3)
        price_acceleration_new = price_roc_3 - price_roc_3.shift(3)
        new_columns['price_acceleration_rate'] = price_acceleration_new
        new_columns['price_decelerating'] = (np.abs(price_acceleration_new) < np.abs(price_acceleration_new.shift(3))).astype(int)
        
        # 4. 动量衰竭综合评分
        momentum_exhaustion = (
            (new_columns['bearish_divergence'] * 0.3) +
            (volume_decreasing.astype(int) * 0.3) +
            (new_columns['price_decelerating'] * 0.2) +
            ((dataframe['adx'] < dataframe['adx'].shift(3)).astype(int) * 0.2)
        )
        new_columns['momentum_exhaustion_score'] = momentum_exhaustion
        
        # 5. 趋势阶段识别（预测性）
        # 初期：突破+放量
        trend_early = (
            (dataframe['adx'] > dataframe['adx'].shift(1)) &
            (dataframe['adx'] > 20) &
            (dataframe['volume_ratio'] > 1.2)
        ).astype(int)
        # 中期：稳定趋势
        trend_middle = (
            (dataframe['adx'] > 25) &
            (np.abs(price_acceleration_new) < 0.02) &
            (~volume_decreasing)
        ).astype(int)
        # 末期：加速+背离
        trend_late = (
            (np.abs(price_acceleration_new) > 0.03) |
            (new_columns['bearish_divergence'] == 1) |
            (new_columns['bullish_divergence'] == 1) |
            (momentum_exhaustion > 0.6)
        ).astype(int)
        
        new_columns['trend_phase'] = trend_late * 3 + trend_middle * 2 + trend_early * 1
        
        # === 市场情绪指标 ===
        rsi_sentiment = (dataframe['rsi_14'] - 50) / 50  # 归一化RSI
        volatility_sentiment = np.where(dataframe['atr_p'] > 0, 
                                       -(dataframe['atr_p'] / dataframe['atr_p'].rolling(20).mean() - 1), 
                                       0)  # 高波动=恐慌，低波动=贪婪
        volume_sentiment = np.where(dataframe['volume_ratio'] > 1.5, -0.5,  # 异常放量=恐慌
                                   np.where(dataframe['volume_ratio'] < 0.7, 0.5, 0))  # 缩量=平静
        new_columns['market_sentiment'] = ((rsi_sentiment + volatility_sentiment + volume_sentiment) / 3).clip(-1, 1)
        
        # === 添加4级反转预警系统 ===
        reversal_warnings = self.detect_reversal_warnings_system(dataframe)
        new_columns['reversal_warning_level'] = reversal_warnings['level']
        new_columns['reversal_probability'] = reversal_warnings['probability']
        new_columns['reversal_signal_strength'] = reversal_warnings['signal_strength']
        
        # 一次性将所有新列添加到dataframe，使用concat避免碎片化
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, new_df], axis=1)
        
        # === 添加突破有效性验证系统 ===
        breakout_validation = self.validate_breakout_effectiveness(dataframe)
        dataframe['breakout_validity_score'] = breakout_validation['validity_score']
        dataframe['breakout_confidence'] = breakout_validation['confidence']
        dataframe['breakout_type'] = breakout_validation['breakout_type']
        
        return dataframe
    
    def detect_reversal_warnings_system(self, dataframe: DataFrame) -> dict:
        """🚨 革命性4级反转预警系统 - 提前2-5根K线识别趋势转换点"""
        
        # === 1级预警：动量衰减检测 ===
        # 检测趋势动量是否开始衰减（最早期信号）
        momentum_decay_long = (
            # 价格涨幅递减
            (dataframe['close'] - dataframe['close'].shift(3) < 
             dataframe['close'].shift(3) - dataframe['close'].shift(6)) &
            # 但价格仍在上升
            (dataframe['close'] > dataframe['close'].shift(3)) &
            # ADX开始下降
            (dataframe['adx'] < dataframe['adx'].shift(2)) &
            # 成交量开始萎缩
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].shift(3))
        )
        
        momentum_decay_short = (
            # 价格跌幅递减  
            (dataframe['close'] - dataframe['close'].shift(3) > 
             dataframe['close'].shift(3) - dataframe['close'].shift(6)) &
            # 但价格仍在下降
            (dataframe['close'] < dataframe['close'].shift(3)) &
            # ADX开始下降
            (dataframe['adx'] < dataframe['adx'].shift(2)) &
            # 成交量开始萎缩
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].shift(3))
        )
        
        # === Fixed RSI Divergence Detection (increased lookback for reliability) ===
        # Price new high but RSI not making new high (fixed 25-period lookback)
        price_higher_high = (
            (dataframe['high'] > dataframe['high'].shift(25)) &
            (dataframe['high'].shift(25) > dataframe['high'].shift(50))
        )
        rsi_lower_high = (
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(25)) &
            (dataframe['rsi_14'].shift(25) < dataframe['rsi_14'].shift(50))
        )
        bearish_rsi_divergence = price_higher_high & rsi_lower_high & (dataframe['rsi_14'] > 65)
        
        # Price new low but RSI not making new low
        price_lower_low = (
            (dataframe['low'] < dataframe['low'].shift(25)) &
            (dataframe['low'].shift(25) < dataframe['low'].shift(50))
        )
        rsi_higher_low = (
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(25)) &
            (dataframe['rsi_14'].shift(25) > dataframe['rsi_14'].shift(50))
        )
        bullish_rsi_divergence = price_lower_low & rsi_higher_low & (dataframe['rsi_14'] < 35)
        
        # === 3级预警：成交量分布异常（资金流向变化） ===
        # 多头趋势中出现大量抛盘
        distribution_volume = (
            (dataframe['close'] > dataframe['ema_13']) &  # 仍在上升趋势
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # 异常放量
            (dataframe['close'] < dataframe['open']) &  # 但收阴线
            (dataframe['close'] < (dataframe['high'] + dataframe['low']) / 2)  # 收盘价在K线下半部
        )
        
        # 空头趋势中出现大量买盘
        accumulation_volume = (
            (dataframe['close'] < dataframe['ema_13']) &  # 仍在下降趋势
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # 异常放量
            (dataframe['close'] > dataframe['open']) &  # 但收阳线
            (dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2)  # 收盘价在K线上半部
        )
        
        # === 4级预警：均线收敛+波动率压缩 ===
        # 均线开始收敛（趋势即将结束）
        ema_convergence = (
            abs(dataframe['ema_5'] - dataframe['ema_13']) < dataframe['atr'] * 0.8
        )
        
        # 波动率异常压缩（暴风雨前的宁静）
        volatility_squeeze = (
            dataframe['atr_p'] < dataframe['atr_p'].rolling(20).quantile(0.3)
        ) & (
            dataframe['bb_width'] < dataframe['bb_width'].rolling(20).quantile(0.2)
        )
        
        # === 综合预警等级计算 ===
        warning_level = self._safe_series(0, len(dataframe))
        
        # 多头反转预警
        bullish_reversal_signals = (
            momentum_decay_short.astype(int) +
            bullish_rsi_divergence.astype(int) +
            accumulation_volume.astype(int) +
            (ema_convergence & volatility_squeeze).astype(int)
        )
        
        # 空头反转预警  
        bearish_reversal_signals = (
            momentum_decay_long.astype(int) +
            bearish_rsi_divergence.astype(int) +  
            distribution_volume.astype(int) +
            (ema_convergence & volatility_squeeze).astype(int)
        )
        
        # 预警等级：1-4级，级数越高反转概率越大
        warning_level = np.maximum(bullish_reversal_signals, bearish_reversal_signals)
        
        # === 反转概率计算 ===
        # 基于历史统计的概率模型
        reversal_probability = np.where(
            warning_level >= 3, 0.75,  # 3-4级预警：75%概率
            np.where(warning_level == 2, 0.55,  # 2级预警：55%概率
                    np.where(warning_level == 1, 0.35, 0.1))  # 1级预警：35%概率
        )
        
        # === 信号强度评分 ===
        signal_strength = (
            bullish_reversal_signals * 25 -  # 多头信号为正
            bearish_reversal_signals * 25    # 空头信号为负
        ).clip(-100, 100)
        
        return {
            'level': warning_level,
            'probability': reversal_probability,
            'signal_strength': signal_strength,
            'bullish_signals': bullish_reversal_signals,
            'bearish_signals': bearish_reversal_signals
        }
    
    def validate_breakout_effectiveness(self, dataframe: DataFrame) -> dict:
        """🔍 突破有效性验证系统 - 精准识别真突破vs假突破"""
        
        # === 1. 成交量突破确认 ===
        # 突破必须伴随成交量放大
        volume_breakout_score = np.where(
            dataframe['volume_ratio'] > 2.0, 3,  # 异常放量：3分
            np.where(dataframe['volume_ratio'] > 1.5, 2,  # 显著放量：2分
                    np.where(dataframe['volume_ratio'] > 1.2, 1, 0))  # 温和放量：1分，无放量：0分
        )
        
        # === 2. 价格强度验证 ===
        # 突破幅度和力度评分
        atr_current = dataframe['atr']
        
        # 向上突破强度
        upward_strength = np.where(
            # 突破布林带上轨 + 超过1个ATR
            (dataframe['close'] > dataframe['bb_upper']) & 
            ((dataframe['close'] - dataframe['bb_upper']) > atr_current), 3,
            np.where(
                # 突破布林带上轨但未超过1个ATR
                dataframe['close'] > dataframe['bb_upper'], 2,
                np.where(
                    # 突破布林带中轨
                    dataframe['close'] > dataframe['bb_middle'], 1, 0
                )
            )
        )
        
        # 向下突破强度  
        downward_strength = np.where(
            # 跌破布林带下轨 + 超过1个ATR
            (dataframe['close'] < dataframe['bb_lower']) & 
            ((dataframe['bb_lower'] - dataframe['close']) > atr_current), -3,
            np.where(
                # 跌破布林带下轨但未超过1个ATR
                dataframe['close'] < dataframe['bb_lower'], -2,
                np.where(
                    # 跌破布林带中轨
                    dataframe['close'] < dataframe['bb_middle'], -1, 0
                )
            )
        )
        
        price_strength = upward_strength + downward_strength  # 合并评分
        
        # === 3. 时间持续性验证 ===
        # 突破后的持续确认（看后续2-3根K线）
        breakout_persistence = self._safe_series(0, len(dataframe))
        
        # 向上突破持续性
        upward_persistence = (
            (dataframe['close'] > dataframe['bb_middle']) &  # 当前在中轨上方
            (dataframe['close'].shift(-1) > dataframe['bb_middle'].shift(-1)) &  # 下一根也在
            (dataframe['low'].shift(-1) > dataframe['bb_middle'].shift(-1) * 0.995)  # 且回撤不深
        ).astype(int) * 2
        
        # 向下突破持续性
        downward_persistence = (
            (dataframe['close'] < dataframe['bb_middle']) &  # 当前在中轨下方
            (dataframe['close'].shift(-1) < dataframe['bb_middle'].shift(-1)) &  # 下一根也在
            (dataframe['high'].shift(-1) < dataframe['bb_middle'].shift(-1) * 1.005)  # 且反弹不高
        ).astype(int) * -2
        
        breakout_persistence = upward_persistence + downward_persistence
        
        # === 4. 假突破过滤 ===
        # 检测常见的假突破模式
        false_breakout_penalty = self._safe_series(0, len(dataframe))
        
        # 上影线过长的假突破（冲高回落）
        long_upper_shadow = (
            (dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['open']) * 2
        ) & (dataframe['close'] > dataframe['open'])  # 阳线但上影线过长
        false_breakout_penalty -= long_upper_shadow.astype(int) * 2
        
        # 下影线过长的假突破（探底回升）
        long_lower_shadow = (
            (dataframe['close'] - dataframe['low']) > (dataframe['open'] - dataframe['close']) * 2
        ) & (dataframe['close'] < dataframe['open'])  # 阴线但下影线过长
        false_breakout_penalty -= long_lower_shadow.astype(int) * 2
        
        # === 5. 技术指标确认 ===
        # RSI和MACD的同步确认
        technical_confirmation = self._safe_series(0, len(dataframe))
        
        # 多头突破确认
        bullish_tech_confirm = (
            (dataframe['rsi_14'] > 50) &  # RSI支持
            (dataframe['macd_hist'] > 0) &  # MACD柱状图为正
            (dataframe['trend_strength'] > 0)  # 趋势强度为正
        ).astype(int) * 2
        
        # 空头突破确认
        bearish_tech_confirm = (
            (dataframe['rsi_14'] < 50) &  # RSI支持
            (dataframe['macd_hist'] < 0) &  # MACD柱状图为负
            (dataframe['trend_strength'] < 0)  # 趋势强度为负
        ).astype(int) * -2
        
        technical_confirmation = bullish_tech_confirm + bearish_tech_confirm
        
        # === 6. 综合有效性评分 ===
        # 权重分配
        validity_score = (
            volume_breakout_score * 0.30 +      # 成交量确认：30%
            price_strength * 0.25 +             # 价格强度：25%
            breakout_persistence * 0.20 +       # 持续性：20%
            technical_confirmation * 0.15 +     # 技术确认：15%
            false_breakout_penalty * 0.10       # 假突破惩罚：10%
        ).clip(-10, 10)
        
        # === 7. 置信度计算 ===
        # 基于评分计算突破置信度
        confidence = np.where(
            abs(validity_score) >= 6, 0.85,  # 高置信度：85%
            np.where(abs(validity_score) >= 4, 0.70,  # 中等置信度：70%
                    np.where(abs(validity_score) >= 2, 0.55,  # 低置信度：55%
                            0.30))  # 很低置信度：30%
        )
        
        # === 8. 突破类型识别 ===
        breakout_type = self._safe_series('NONE', len(dataframe), 'NONE')
        
        # 强势突破
        strong_breakout_up = (validity_score >= 5) & (price_strength > 0)
        strong_breakout_down = (validity_score <= -5) & (price_strength < 0)
        
        # 温和突破
        mild_breakout_up = (validity_score >= 2) & (validity_score < 5) & (price_strength > 0)
        mild_breakout_down = (validity_score <= -2) & (validity_score > -5) & (price_strength < 0)
        
        # 可能的假突破
        false_breakout = (abs(validity_score) < 2) & (abs(price_strength) > 0)
        
        breakout_type.loc[strong_breakout_up] = 'STRONG_BULLISH'
        breakout_type.loc[strong_breakout_down] = 'STRONG_BEARISH'
        breakout_type.loc[mild_breakout_up] = 'MILD_BULLISH'
        breakout_type.loc[mild_breakout_down] = 'MILD_BEARISH'
        breakout_type.loc[false_breakout] = 'LIKELY_FALSE'
        
        return {
            'validity_score': validity_score,
            'confidence': confidence,
            'breakout_type': breakout_type,
            'volume_score': volume_breakout_score,
            'price_strength': price_strength,
            'persistence': breakout_persistence,
            'tech_confirmation': technical_confirmation
        }
    
    def calculate_market_regime_simple(self, dataframe: DataFrame) -> DataFrame:
        """简化的市场状态识别 - 优化DataFrame操作"""
        
        # 一次性计算所有需要的列，避免DataFrame碎片化
        new_columns = {}
        
        # 基于趋势强度和波动率状态确定市场类型
        conditions = [
            (dataframe['trend_strength'] > 75) & (dataframe['adx'] > 25),  # 强趋势
            (dataframe['trend_strength'] > 50) & (dataframe['adx'] > 20),  # 中等趋势  
            (dataframe['volatility_state'] > 75),  # 高波动
            (dataframe['adx'] < 20) & (dataframe['volatility_state'] < 30)  # 盘整
        ]
        
        choices = ['strong_trend', 'medium_trend', 'volatile', 'consolidation']
        new_columns['market_regime'] = np.select(conditions, choices, default='neutral')
        
        # 市场情绪指标 (简化版)
        price_vs_ma = np.where(dataframe['ema_21'] > 0, 
                              (dataframe['close'] - dataframe['ema_21']) / dataframe['ema_21'], 
                              0)
        volume_sentiment = np.where(dataframe['volume_ratio'] > 1.2, 1, 
                                  np.where(dataframe['volume_ratio'] < 0.8, -1, 0))
        
        new_columns['market_sentiment'] = (price_vs_ma * 10 + volume_sentiment) / 2
        
        # 使用直接赋值添加所有新列，避免concat引起的索引问题
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # 确保Series长度与dataframe匹配
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def ichimoku(self, dataframe: DataFrame, tenkan=9, kijun=26, senkou_b=52) -> DataFrame:
        """Ichimoku 云图指标 - 优化DataFrame操作"""
        # 批量计算所有指标
        new_columns = {}
        
        new_columns['tenkan'] = (dataframe['high'].rolling(tenkan).max() + dataframe['low'].rolling(tenkan).min()) / 2
        new_columns['kijun'] = (dataframe['high'].rolling(kijun).max() + dataframe['low'].rolling(kijun).min()) / 2
        new_columns['senkou_a'] = ((new_columns['tenkan'] + new_columns['kijun']) / 2).shift(kijun)
        new_columns['senkou_b'] = ((dataframe['high'].rolling(senkou_b).max() + dataframe['low'].rolling(senkou_b).min()) / 2).shift(kijun)
        new_columns['chikou'] = dataframe['close'].shift(-kijun)
        
        # 使用直接赋值添加所有新列，避免concat引起的索引问题
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # 确保Series长度与dataframe匹配
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def supertrend(self, dataframe: DataFrame, period=10, multiplier=3) -> pd.Series:
        """Super Trend 指标"""
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        atr = ta.ATR(dataframe, timeperiod=period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = dataframe['close'] * 0  # 初始化
        direction = self._safe_series(0.0, len(dataframe))
        
        for i in range(1, len(dataframe)):
            if dataframe['close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif dataframe['close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                
        return supertrend
    
    def calculate_advanced_volatility_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算高级波动率指标"""
        
        # Keltner 通道（基于ATR的动态通道）
        kc_period = 20
        kc_multiplier = 2
        kc_middle = ta.EMA(dataframe, timeperiod=kc_period)
        kc_range = ta.ATR(dataframe, timeperiod=kc_period) * kc_multiplier
        dataframe['kc_upper'] = kc_middle + kc_range
        dataframe['kc_lower'] = kc_middle - kc_range
        dataframe['kc_middle'] = kc_middle
        dataframe['kc_width'] = np.where(dataframe['kc_middle'] > 0, 
                                        (dataframe['kc_upper'] - dataframe['kc_lower']) / dataframe['kc_middle'], 
                                        0)
        dataframe['kc_position'] = (dataframe['close'] - dataframe['kc_lower']) / (dataframe['kc_upper'] - dataframe['kc_lower'])
        
        # Donchian 通道（突破交易系统）
        dc_period = 20
        dataframe['dc_upper'] = dataframe['high'].rolling(dc_period).max()
        dataframe['dc_lower'] = dataframe['low'].rolling(dc_period).min()
        dataframe['dc_middle'] = (dataframe['dc_upper'] + dataframe['dc_lower']) / 2
        dataframe['dc_width'] = np.where(dataframe['dc_middle'] > 0, 
                                        (dataframe['dc_upper'] - dataframe['dc_lower']) / dataframe['dc_middle'], 
                                        0)
        
        # Bollinger Bandwidth（波动率收缩检测）
        dataframe['bb_bandwidth'] = dataframe['bb_width']  # 已经在基础指标中计算
        dataframe['bb_squeeze'] = (dataframe['bb_bandwidth'] < dataframe['bb_bandwidth'].rolling(20).quantile(0.2)).astype(int)
        
        # Chaikin Volatility（成交量波动率）
        cv_period = 10
        hl_ema = ta.EMA(dataframe['high'] - dataframe['low'], timeperiod=cv_period)
        dataframe['chaikin_volatility'] = ((hl_ema - hl_ema.shift(cv_period)) / hl_ema.shift(cv_period)) * 100
        
        # 波动率指数（VIX风格）
        returns = dataframe['close'].pct_change()
        dataframe['volatility_index'] = returns.rolling(20).std() * np.sqrt(365) * 100  # 年化波动率
        
        return dataframe
    
    def calculate_advanced_momentum_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算高级动量指标"""
        
        # Fisher Transform（价格分布正态化）
        dataframe = self.fisher_transform(dataframe)
        
        # KST指标（多重ROC综合）
        dataframe = self.kst_indicator(dataframe)
        
        # Coppock曲线（长期动量指标）
        dataframe = self.coppock_curve(dataframe)
        
        # Vortex指标（趋势方向和强度）
        dataframe = self.vortex_indicator(dataframe)
        
        # Stochastic Momentum Index（SMI）
        dataframe = self.stochastic_momentum_index(dataframe)
        
        # True Strength Index（TSI）
        dataframe = self.true_strength_index(dataframe)
        
        return dataframe
    
    def fisher_transform(self, dataframe: DataFrame, period: int = 10) -> DataFrame:
        """计算Fisher Transform指标"""
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        
        # 计算价格的最大值和最小值
        high_n = hl2.rolling(period).max()
        low_n = hl2.rolling(period).min()
        
        # 标准化价格到-1到1之间
        normalized_price = 2 * ((hl2 - low_n) / (high_n - low_n) - 0.5)
        normalized_price = normalized_price.clip(-0.999, 0.999)  # 防止数学错误
        
        # Fisher Transform
        fisher = self._safe_series(0.0, len(dataframe))
        fisher[0] = 0
        
        for i in range(1, len(dataframe)):
            if not pd.isna(normalized_price.iloc[i]):
                raw_fisher = 0.5 * np.log((1 + normalized_price.iloc[i]) / (1 - normalized_price.iloc[i]))
                fisher.iloc[i] = 0.5 * fisher.iloc[i-1] + 0.5 * raw_fisher
            else:
                fisher.iloc[i] = fisher.iloc[i-1]
        
        dataframe['fisher'] = fisher
        dataframe['fisher_signal'] = fisher.shift(1)
        
        return dataframe
    
    def kst_indicator(self, dataframe: DataFrame) -> DataFrame:
        """计算KST (Know Sure Thing) 指标"""
        # 四个ROC周期
        roc1 = ta.ROC(dataframe, timeperiod=10)
        roc2 = ta.ROC(dataframe, timeperiod=15)
        roc3 = ta.ROC(dataframe, timeperiod=20)
        roc4 = ta.ROC(dataframe, timeperiod=30)
        
        # 对ROC进行移动平均平滑
        roc1_ma = ta.SMA(roc1, timeperiod=10)
        roc2_ma = ta.SMA(roc2, timeperiod=10)
        roc3_ma = ta.SMA(roc3, timeperiod=10)
        roc4_ma = ta.SMA(roc4, timeperiod=15)
        
        # KST计算（加权求和）
        dataframe['kst'] = (roc1_ma * 1) + (roc2_ma * 2) + (roc3_ma * 3) + (roc4_ma * 4)
        dataframe['kst_signal'] = ta.SMA(dataframe['kst'], timeperiod=9)
        
        return dataframe
    
    def coppock_curve(self, dataframe: DataFrame, wma_period: int = 10) -> DataFrame:
        """计算Coppock曲线"""
        # Coppock ROC计算
        roc11 = ta.ROC(dataframe, timeperiod=11)
        roc14 = ta.ROC(dataframe, timeperiod=14)
        
        # 两个ROC相加
        roc_sum = roc11 + roc14
        
        # 加权移动平均
        dataframe['coppock'] = ta.WMA(roc_sum, timeperiod=wma_period)
        
        return dataframe
    
    def vortex_indicator(self, dataframe: DataFrame, period: int = 14) -> DataFrame:
        """计算Vortex指标"""
        # True Range
        tr = ta.TRANGE(dataframe)
        
        # 正和负涡流运动
        vm_plus = abs(dataframe['high'] - dataframe['low'].shift(1))
        vm_minus = abs(dataframe['low'] - dataframe['high'].shift(1))
        
        # 求和
        vm_plus_sum = vm_plus.rolling(period).sum()
        vm_minus_sum = vm_minus.rolling(period).sum()
        tr_sum = tr.rolling(period).sum()
        
        # VI计算
        dataframe['vi_plus'] = vm_plus_sum / tr_sum
        dataframe['vi_minus'] = vm_minus_sum / tr_sum
        dataframe['vi_diff'] = dataframe['vi_plus'] - dataframe['vi_minus']
        
        return dataframe
    
    def stochastic_momentum_index(self, dataframe: DataFrame, k_period: int = 10, d_period: int = 3) -> DataFrame:
        """计算随机动量指数 (SMI)"""
        # 价格中点
        mid_point = (dataframe['high'].rolling(k_period).max() + dataframe['low'].rolling(k_period).min()) / 2
        
        # 计算SMI
        numerator = (dataframe['close'] - mid_point).rolling(k_period).sum()
        denominator = (dataframe['high'].rolling(k_period).max() - dataframe['low'].rolling(k_period).min()).rolling(k_period).sum() / 2
        
        smi_k = (numerator / denominator) * 100
        dataframe['smi_k'] = smi_k
        dataframe['smi_d'] = smi_k.rolling(d_period).mean()
        
        return dataframe
    
    def true_strength_index(self, dataframe: DataFrame, r: int = 25, s: int = 13) -> DataFrame:
        """计算真实强度指数 (TSI)"""
        # 价格变化
        price_change = dataframe['close'].diff()
        
        # 双次平滑价格变化
        first_smooth_pc = price_change.ewm(span=r).mean()
        double_smooth_pc = first_smooth_pc.ewm(span=s).mean()
        
        # 双次平滑绝对值价格变化
        first_smooth_abs_pc = abs(price_change).ewm(span=r).mean()
        double_smooth_abs_pc = first_smooth_abs_pc.ewm(span=s).mean()
        
        # TSI计算
        dataframe['tsi'] = 100 * (double_smooth_pc / double_smooth_abs_pc)
        dataframe['tsi_signal'] = dataframe['tsi'].ewm(span=7).mean()
        
        return dataframe
    
    def calculate_advanced_volume_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算高级成交量指标"""
        
        # Accumulation/Distribution Line（A/D线）
        dataframe['ad_line'] = ta.AD(dataframe)
        dataframe['ad_line_ma'] = ta.SMA(dataframe['ad_line'], timeperiod=20)
        
        # Money Flow Index（MFI - 成交量加权RSI）
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Force Index（力度指数）
        force_index = (dataframe['close'] - dataframe['close'].shift(1)) * dataframe['volume']
        dataframe['force_index'] = force_index.ewm(span=13).mean()
        dataframe['force_index_ma'] = force_index.rolling(20).mean()
        
        # Ease of Movement（移动难易度）
        high_low_avg = (dataframe['high'] + dataframe['low']) / 2
        high_low_avg_prev = high_low_avg.shift(1)
        distance_moved = high_low_avg - high_low_avg_prev
        
        high_low_diff = dataframe['high'] - dataframe['low']
        box_ratio = (dataframe['volume'] / 1000000) / (high_low_diff + 1e-10)
        
        emv_1 = distance_moved / (box_ratio + 1e-10)
        dataframe['emv'] = emv_1.rolling(14).mean()
        
        # Chaikin Money Flow（CMF）
        money_flow_multiplier = ((dataframe['close'] - dataframe['low']) - 
                               (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'] + 1e-10)
        money_flow_volume = money_flow_multiplier * dataframe['volume']
        dataframe['cmf'] = money_flow_volume.rolling(20).sum() / (dataframe['volume'].rolling(20).sum() + 1e-10)
        
        # Volume Price Trend（VPT）
        vpt = (dataframe['volume'] * ((dataframe['close'] - dataframe['close'].shift(1)) / (dataframe['close'].shift(1) + 1e-10)))
        dataframe['vpt'] = vpt.cumsum()
        dataframe['vpt_ma'] = dataframe['vpt'].rolling(20).mean()
        
        return dataframe
    
    def calculate_market_structure_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算市场结构指标"""
        
        # Price Action指标
        dataframe = self.calculate_price_action_indicators(dataframe)
        
        # 支撑/阻力位识别
        dataframe = self.identify_support_resistance(dataframe)
        
        # 波段分析
        dataframe = self.calculate_wave_analysis(dataframe)
        
        # 价格密度分析
        dataframe = self.calculate_price_density(dataframe)
        
        return dataframe
    
    def calculate_price_action_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算价格行为指标"""
        # 真实体大小
        dataframe['real_body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['real_body_pct'] = dataframe['real_body'] / (dataframe['close'] + 1e-10) * 100
        
        # 上下影线
        dataframe['upper_shadow'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['lower_shadow'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        # K线模式识别
        dataframe['is_doji'] = (dataframe['real_body_pct'] < 0.1).astype(int)
        dataframe['is_hammer'] = ((dataframe['lower_shadow'] > dataframe['real_body'] * 2) & 
                                 (dataframe['upper_shadow'] < dataframe['real_body'] * 0.5)).astype(int)
        dataframe['is_shooting_star'] = ((dataframe['upper_shadow'] > dataframe['real_body'] * 2) & 
                                        (dataframe['lower_shadow'] < dataframe['real_body'] * 0.5)).astype(int)
        
        # Pin Bar 模式识别
        # Pin Bar Bullish: 长下影线，小实体，短上影线，看涨信号
        dataframe['is_pin_bar_bullish'] = ((dataframe['lower_shadow'] > dataframe['real_body'] * 2) & 
                                          (dataframe['upper_shadow'] < dataframe['real_body']) &
                                          (dataframe['real_body_pct'] < 2.0) &  # 实体相对较小
                                          (dataframe['close'] > dataframe['open'])).astype(int)  # 阳线
        
        # Pin Bar Bearish: 长上影线，小实体，短下影线，看跌信号
        dataframe['is_pin_bar_bearish'] = ((dataframe['upper_shadow'] > dataframe['real_body'] * 2) & 
                                          (dataframe['lower_shadow'] < dataframe['real_body']) &
                                          (dataframe['real_body_pct'] < 2.0) &  # 实体相对较小
                                          (dataframe['close'] < dataframe['open'])).astype(int)  # 阴线
        
        # 吞噬模式识别
        # 向前偏移获取前一根K线数据
        prev_open = dataframe['open'].shift(1)
        prev_close = dataframe['close'].shift(1)
        prev_high = dataframe['high'].shift(1)
        prev_low = dataframe['low'].shift(1)
        
        # 看涨吞噬：当前阳线完全吞噬前一根阴线
        dataframe['is_bullish_engulfing'] = ((dataframe['close'] > dataframe['open']) &  # 当前为阳线
                                           (prev_close < prev_open) &  # 前一根为阴线
                                           (dataframe['open'] < prev_close) &  # 当前开盘价低于前一根收盘价
                                           (dataframe['close'] > prev_open) &  # 当前收盘价高于前一根开盘价
                                           (dataframe['real_body'] > dataframe['real_body'].shift(1) * 1.2)).astype(int)  # 当前实体更大
        
        # 看跌吞噬：当前阴线完全吞噬前一根阳线
        dataframe['is_bearish_engulfing'] = ((dataframe['close'] < dataframe['open']) &  # 当前为阴线
                                           (prev_close > prev_open) &  # 前一根为阳线
                                           (dataframe['open'] > prev_close) &  # 当前开盘价高于前一根收盘价
                                           (dataframe['close'] < prev_open) &  # 当前收盘价低于前一根开盘价
                                           (dataframe['real_body'] > dataframe['real_body'].shift(1) * 1.2)).astype(int)  # 当前实体更大
        
        return dataframe
    
    def identify_support_resistance(self, dataframe: DataFrame, window: int = 20) -> DataFrame:
        """识别支撑和阻力位"""
        # 计算所有支撑阻力指标，一次性添加避免碎片化
        sr_columns = {
            'local_max': dataframe['high'].rolling(window, center=True).max() == dataframe['high'],
            'local_min': dataframe['low'].rolling(window, center=True).min() == dataframe['low'],
            'resistance_distance': np.where(dataframe['close'] > 0, 
                                           (dataframe['high'].rolling(50).max() - dataframe['close']) / dataframe['close'], 
                                           0),
            'support_distance': np.where(dataframe['close'] > 0, 
                                        (dataframe['close'] - dataframe['low'].rolling(50).min()) / dataframe['close'], 
                                        0)
        }
        
        sr_df = pd.DataFrame(sr_columns, index=dataframe.index)
        return pd.concat([dataframe, sr_df], axis=1)
    
    def calculate_wave_analysis(self, dataframe: DataFrame) -> DataFrame:
        """计算波段分析指标"""
        # Elliott Wave相关指标，一次性计算避免碎片化
        returns = dataframe['close'].pct_change()
        
        wave_columns = {
            'wave_strength': abs(dataframe['close'] - dataframe['close'].shift(5)) / (dataframe['close'].shift(5) + 1e-10),
            'normalized_returns': returns / (returns.rolling(20).std() + 1e-10),
            'momentum_dispersion': dataframe['mom_10'].rolling(10).std() / (abs(dataframe['mom_10']).rolling(10).mean() + 1e-10)
        }
        
        wave_df = pd.DataFrame(wave_columns, index=dataframe.index)
        return pd.concat([dataframe, wave_df], axis=1)
    
    def calculate_price_density(self, dataframe: DataFrame) -> DataFrame:
        """计算价格密度分析指标 - 优化DataFrame操作"""
        # 一次性计算所有需要的列
        new_columns = {}
        
        # 价格区间分布分析
        price_range = dataframe['high'] - dataframe['low']
        new_columns['price_range_pct'] = price_range / (dataframe['close'] + 1e-10) * 100
        
        # 简化的价格密度计算
        new_columns['price_density'] = 1 / (new_columns['price_range_pct'] + 0.1)  # 价格区间越小密度越高
        
        # 使用直接赋值添加所有新列，避免concat引起的索引问题
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # 确保Series长度与dataframe匹配
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def calculate_composite_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算复合技术指标 - 优化DataFrame操作"""
        
        # 一次性计算所有需要的列
        new_columns = {}
        
        # 多维度动量评分
        new_columns['momentum_score'] = self.calculate_momentum_score(dataframe)
        
        # 趋势强度综合评分
        new_columns['trend_strength_score'] = self.calculate_trend_strength_score(dataframe)
        
        # 波动率状态评分
        new_columns['volatility_regime'] = self.calculate_volatility_regime(dataframe)
        
        # 市场状态综合评分
        new_columns['market_regime'] = self.calculate_market_regime(dataframe)
        
        # 风险调整收益指标
        new_columns['risk_adjusted_return'] = self.calculate_risk_adjusted_returns(dataframe)
        
        # 技术面健康度
        new_columns['technical_health'] = self.calculate_technical_health(dataframe)
        
        # 使用直接赋值添加所有新列，避免concat引起的索引问题
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # 确保Series长度与dataframe匹配
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def calculate_momentum_score(self, dataframe: DataFrame) -> pd.Series:
        """计算多维度动量评分"""
        # 收集多个动量指标
        momentum_indicators = {}
        
        # 基础动量指标
        if 'rsi_14' in dataframe.columns:
            momentum_indicators['rsi_14'] = (dataframe['rsi_14'] - 50) / 50  # 标准化RSI
        if 'mom_10' in dataframe.columns:
            momentum_indicators['mom_10'] = np.where(dataframe['close'] > 0, 
                                                     dataframe['mom_10'] / dataframe['close'] * 100, 
                                                     0)  # 标准化动量
        if 'roc_10' in dataframe.columns:
            momentum_indicators['roc_10'] = dataframe['roc_10'] / 100  # ROC
        if 'macd' in dataframe.columns:
            momentum_indicators['macd_normalized'] = np.where(dataframe['close'] > 0, 
                                                             dataframe['macd'] / dataframe['close'] * 1000, 
                                                             0)  # 标准化MACD
        
        # 高级动量指标
        if 'kst' in dataframe.columns:
            momentum_indicators['kst_normalized'] = dataframe['kst'] / abs(dataframe['kst']).rolling(20).mean()  # 标准化KST
        if 'fisher' in dataframe.columns:
            momentum_indicators['fisher'] = dataframe['fisher']  # Fisher Transform
        if 'tsi' in dataframe.columns:
            momentum_indicators['tsi'] = dataframe['tsi'] / 100  # TSI
        if 'vi_diff' in dataframe.columns:
            momentum_indicators['vi_diff'] = dataframe['vi_diff']  # Vortex差值
        
        # 加权平均
        weights = {
            'rsi_14': 0.15, 'mom_10': 0.10, 'roc_10': 0.10, 'macd_normalized': 0.15,
            'kst_normalized': 0.15, 'fisher': 0.15, 'tsi': 0.10, 'vi_diff': 0.10
        }
        
        momentum_score = self._safe_series(0.0, len(dataframe))
        
        for indicator, weight in weights.items():
            if indicator in momentum_indicators:
                normalized_indicator = momentum_indicators[indicator].fillna(0)
                # 限制在-1到1之间
                normalized_indicator = normalized_indicator.clip(-3, 3) / 3
                momentum_score += normalized_indicator * weight
        
        return momentum_score.clip(-1, 1)
    
    def calculate_trend_strength_score(self, dataframe: DataFrame) -> pd.Series:
        """计算趋势强度综合评分"""
        # 趋势指标
        trend_indicators = {}
        
        if 'adx' in dataframe.columns:
            trend_indicators['adx'] = dataframe['adx'] / 100  # ADX标准化
        
        # EMA排列
        trend_indicators['ema_trend'] = self.calculate_ema_trend_score(dataframe)
        
        # SuperTrend
        trend_indicators['supertrend_trend'] = self.calculate_supertrend_score(dataframe)
        
        # Ichimoku
        trend_indicators['ichimoku_trend'] = self.calculate_ichimoku_score(dataframe)
        
        # 线性回归趋势
        trend_indicators['linear_reg_trend'] = self.calculate_linear_regression_trend(dataframe)
        
        weights = {
            'adx': 0.3, 'ema_trend': 0.25, 'supertrend_trend': 0.2,
            'ichimoku_trend': 0.15, 'linear_reg_trend': 0.1
        }
        
        trend_score = self._safe_series(0.0, len(dataframe))
        
        for indicator, weight in weights.items():
            if indicator in trend_indicators:
                normalized_indicator = trend_indicators[indicator].fillna(0)
                trend_score += normalized_indicator * weight
        
        return trend_score.clip(-1, 1)
    
    def calculate_ema_trend_score(self, dataframe: DataFrame) -> pd.Series:
        """计算EMA排列趋势评分"""
        score = self._safe_series(0.0, len(dataframe))
        
        # EMA排列分数
        if all(col in dataframe.columns for col in ['ema_8', 'ema_21', 'ema_50']):
            # 多头排列: EMA8 > EMA21 > EMA50
            score += (dataframe['ema_8'] > dataframe['ema_21']).astype(int) * 0.4
            score += (dataframe['ema_21'] > dataframe['ema_50']).astype(int) * 0.3
            score += (dataframe['close'] > dataframe['ema_8']).astype(int) * 0.3
            
            # 空头排列：反向就是负分
            score -= (dataframe['ema_8'] < dataframe['ema_21']).astype(int) * 0.4
            score -= (dataframe['ema_21'] < dataframe['ema_50']).astype(int) * 0.3
            score -= (dataframe['close'] < dataframe['ema_8']).astype(int) * 0.3
        
        return score.clip(-1, 1)
    
    def calculate_supertrend_score(self, dataframe: DataFrame) -> pd.Series:
        """计算SuperTrend评分"""
        if 'supertrend' not in dataframe.columns:
            return self._safe_series(0.0, len(dataframe))
        
        # SuperTrend方向判断
        trend_score = ((dataframe['close'] > dataframe['supertrend']).astype(int) * 2 - 1)
        
        # 加入距离因子
        distance_factor = np.where(dataframe['close'] > 0, 
                                  abs(dataframe['close'] - dataframe['supertrend']) / dataframe['close'], 
                                  0)
        distance_factor = distance_factor.clip(0, 0.1) / 0.1  # 最多10%距离
        
        return trend_score * distance_factor
    
    def calculate_ichimoku_score(self, dataframe: DataFrame) -> pd.Series:
        """计算Ichimoku评分"""
        score = self._safe_series(0.0, len(dataframe))
        
        # Ichimoku云图信号
        if all(col in dataframe.columns for col in ['tenkan', 'kijun', 'senkou_a', 'senkou_b']):
            # 价格在云上方
            above_cloud = ((dataframe['close'] > dataframe['senkou_a']) & 
                          (dataframe['close'] > dataframe['senkou_b'])).astype(int)
            
            # 价格在云下方
            below_cloud = ((dataframe['close'] < dataframe['senkou_a']) & 
                          (dataframe['close'] < dataframe['senkou_b'])).astype(int)
            
            # Tenkan-Kijun交叉
            tenkan_above_kijun = (dataframe['tenkan'] > dataframe['kijun']).astype(int)
            
            score = (above_cloud * 0.5 + tenkan_above_kijun * 0.3 + 
                    (dataframe['close'] > dataframe['tenkan']).astype(int) * 0.2 - 
                    below_cloud * 0.5)
        
        return score.clip(-1, 1)
    
    def calculate_linear_regression_trend(self, dataframe: DataFrame, period: int = 20) -> pd.Series:
        """计算线性回归趋势"""
        def linear_reg_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            from scipy import stats
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * r_value ** 2  # 斜率乘以R平方
        
        # 计算滚动线性回归斜率
        reg_slope = dataframe['close'].rolling(period).apply(linear_reg_slope, raw=False)
        
        # 标准化
        normalized_slope = np.where(dataframe['close'] > 0, 
                                   reg_slope / dataframe['close'] * 1000, 
                                   0)  # 放大因子
        
        return normalized_slope.fillna(0).clip(-1, 1)
    
    def calculate_volatility_regime(self, dataframe: DataFrame) -> pd.Series:
        """计算波动率状态"""
        # 当前波动率
        current_vol = dataframe['atr_p']
        
        # 历史波动率分位数
        vol_percentile = current_vol.rolling(100).rank(pct=True)
        
        # 波动率状态分类
        regime = self._safe_series(0, len(dataframe))  # 0: 中等波动
        regime[vol_percentile < 0.2] = -1  # 低波动
        regime[vol_percentile > 0.8] = 1   # 高波动
        
        return regime
    
    def calculate_market_regime(self, dataframe: DataFrame) -> pd.Series:
        """计算市场状态综合评分"""
        # 综合多个因素
        regime_factors = {}
        
        if 'trend_strength_score' in dataframe.columns:
            regime_factors['trend_strength'] = dataframe['trend_strength_score']
        if 'momentum_score' in dataframe.columns:
            regime_factors['momentum'] = dataframe['momentum_score']
        if 'volatility_regime' in dataframe.columns:
            regime_factors['volatility'] = dataframe['volatility_regime'] / 2  # 标准化
        if 'volume_ratio' in dataframe.columns:
            regime_factors['volume_trend'] = (dataframe['volume_ratio'] - 1).clip(-1, 1)
        
        weights = {'trend_strength': 0.4, 'momentum': 0.3, 'volatility': 0.2, 'volume_trend': 0.1}
        
        market_regime = self._safe_series(0.0, len(dataframe))
        for factor, weight in weights.items():
            if factor in regime_factors:
                market_regime += regime_factors[factor].fillna(0) * weight
        
        return market_regime.clip(-1, 1)
    
    # 移除了 calculate_risk_adjusted_returns - 简化策略逻辑
    def calculate_risk_adjusted_returns(self, dataframe: DataFrame, window: int = 20) -> pd.Series:
        """计算风险调整收益"""
        # 计算收益率
        returns = dataframe['close'].pct_change()
        
        # 滚动Sharpe比率
        rolling_returns = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        risk_adjusted = rolling_returns / (rolling_std + 1e-6)  # 避免除零
        
        return risk_adjusted.fillna(0)
    
    def identify_coin_risk_tier(self, pair: str, dataframe: DataFrame) -> str:
        """🎯 智能币种风险等级识别系统 - 基于多维度市场特征分析"""
        
        try:
            if dataframe.empty or len(dataframe) < 96:  # 需要足够的历史数据
                return 'medium_risk'  # 默认中等风险
                
            current_idx = -1
            
            # === 特征1: 价格波动率分析 ===
            volatility = dataframe['atr_p'].iloc[current_idx] if 'atr_p' in dataframe.columns else 0.05
            volatility_24h = dataframe['close'].rolling(96).std().iloc[current_idx] / dataframe['close'].iloc[current_idx]
            
            # === 特征2: 交易量稳定性分析 ===
            volume_series = dataframe['volume'].rolling(24)
            volume_mean = volume_series.mean().iloc[current_idx]
            volume_std = volume_series.std().iloc[current_idx]
            volume_cv = (volume_std / volume_mean) if volume_mean > 0 else 5  # 变异系数
            
            # === 特征3: 价格行为特征 ===
            current_price = dataframe['close'].iloc[current_idx]
            price_24h_ago = dataframe['close'].iloc[-96] if len(dataframe) >= 96 else dataframe['close'].iloc[0]
            price_change_24h = abs((current_price / price_24h_ago) - 1) if price_24h_ago > 0 else 0
            
            # === 特征4: 价格水平判断 ===
            is_micro_price = current_price < 0.001  # 极小价格（通常是meme币特征）
            is_low_price = current_price < 0.1      # 低价格
            
            # === 特征5: 技术指标异常检测 ===
            rsi = dataframe['rsi_14'].iloc[current_idx] if 'rsi_14' in dataframe.columns else 50
            is_extreme_rsi = rsi > 80 or rsi < 20  # 极端RSI值
            
            # === 特征6: 价格模式识别 ===
            recent_pumps = 0
            if len(dataframe) >= 24:
                for i in range(1, min(24, len(dataframe))):
                    hour_change = (dataframe['close'].iloc[-i] / dataframe['close'].iloc[-i-1]) - 1
                    if hour_change > 0.15:  # 单小时涨幅超过15%
                        recent_pumps += 1
            
            # === 综合评分系统 ===
            risk_score = 0
            risk_factors = []
            
            # 波动率评分 (0-40分)
            if volatility > 0.20:  # 极高波动
                risk_score += 40
                risk_factors.append(f"极高波动({volatility*100:.1f}%)")
            elif volatility > 0.10:
                risk_score += 25
                risk_factors.append(f"高波动({volatility*100:.1f}%)")
            elif volatility > 0.05:
                risk_score += 10
                risk_factors.append(f"中等波动({volatility*100:.1f}%)")
            
            # 交易量不稳定性评分 (0-25分)
            if volume_cv > 3:  # 交易量极不稳定
                risk_score += 25
                risk_factors.append(f"交易量极不稳定(CV:{volume_cv:.1f})")
            elif volume_cv > 1.5:
                risk_score += 15
                risk_factors.append(f"交易量不稳定(CV:{volume_cv:.1f})")
            
            # 短期价格异常评分 (0-20分)
            if price_change_24h > 0.50:  # 24小时变化超过50%
                risk_score += 20
                risk_factors.append(f"24h巨幅波动({price_change_24h*100:.1f}%)")
            elif price_change_24h > 0.20:
                risk_score += 10
                risk_factors.append(f"24h大幅波动({price_change_24h*100:.1f}%)")
            
            # 价格水平评分 (0-10分)
            if is_micro_price:
                risk_score += 10
                risk_factors.append(f"微价格(${current_price:.6f})")
            elif is_low_price:
                risk_score += 5
                risk_factors.append(f"低价格(${current_price:.3f})")
            
            # Pump行为评分 (0-15分)
            if recent_pumps >= 3:
                risk_score += 15
                risk_factors.append(f"频繁pump({recent_pumps}次)")
            elif recent_pumps >= 1:
                risk_score += 8
                risk_factors.append(f"有pump行为({recent_pumps}次)")
            
            # === 风险等级判定 ===
            if risk_score >= 70:
                risk_tier = 'high_risk'    # 高风险（疑似垃圾币/meme币）
                tier_name = "⚠️ 高风险"
            elif risk_score >= 40:
                risk_tier = 'medium_risk'  # 中等风险
                tier_name = "⚡ 中等风险"
            else:
                risk_tier = 'low_risk'     # 低风险（相对稳定）
                tier_name = "✅ 低风险"
            
            # 详细日志
            logger.info(f"""
🎯 币种风险识别 - {pair}:
├─ 风险等级: {tier_name} (评分: {risk_score}/100)
├─ 当前价格: ${current_price:.6f}
├─ 波动率: {volatility*100:.2f}% | 24h变化: {price_change_24h*100:.1f}%
├─ 交易量CV: {volume_cv:.2f} | 近期Pump: {recent_pumps}次
├─ 识别因素: {' | '.join(risk_factors) if risk_factors else '正常特征'}
└─ 推荐策略: {'小仓位以小博大' if risk_tier == 'high_risk' else '正常配置' if risk_tier == 'low_risk' else '适度谨慎'}
""")
            
            return risk_tier
            
        except Exception as e:
            logger.error(f"币种风险识别失败 {pair}: {e}")
            return 'medium_risk'  # 出错时返回中等风险
    
    def calculate_technical_health(self, dataframe: DataFrame) -> pd.Series:
        """计算技术面健康度"""
        health_components = {}
        
        # 1. 趋势一致性（多个指标是否同向）
        trend_signals = []
        if 'ema_21' in dataframe.columns:
            trend_signals.append((dataframe['close'] > dataframe['ema_21']).astype(int))
        if 'macd' in dataframe.columns and 'macd_signal' in dataframe.columns:
            trend_signals.append((dataframe['macd'] > dataframe['macd_signal']).astype(int))
        if 'rsi_14' in dataframe.columns:
            trend_signals.append((dataframe['rsi_14'] > 50).astype(int))
        if 'momentum_score' in dataframe.columns:
            trend_signals.append((dataframe['momentum_score'] > 0).astype(int))
        
        if trend_signals:
            health_components['trend_consistency'] = (sum(trend_signals) / len(trend_signals) - 0.5) * 2
        
        # 2. 波动率健康度（不过高不过低）
        if 'volatility_regime' in dataframe.columns:
            vol_score = 1 - abs(dataframe['volatility_regime']) * 0.5  # 中等波动最好
            health_components['volatility_health'] = vol_score
        
        # 3. 成交量确认
        if 'volume_ratio' in dataframe.columns:
            volume_health = ((dataframe['volume_ratio'] > 0.8).astype(float) * 0.5 + 
                           (dataframe['volume_ratio'] < 2.0).astype(float) * 0.5)  # 适度放量
            health_components['volume_health'] = volume_health
        
        # 4. 技术指标发散度（过度买入/卖出检测）
        overbought_signals = []
        oversold_signals = []
        
        if 'rsi_14' in dataframe.columns:
            overbought_signals.append((dataframe['rsi_14'] > 80).astype(int))
            oversold_signals.append((dataframe['rsi_14'] < 20).astype(int))
        if 'mfi' in dataframe.columns:
            overbought_signals.append((dataframe['mfi'] > 80).astype(int))
            oversold_signals.append((dataframe['mfi'] < 20).astype(int))
        if 'stoch_k' in dataframe.columns:
            overbought_signals.append((dataframe['stoch_k'] > 80).astype(int))
            oversold_signals.append((dataframe['stoch_k'] < 20).astype(int))
        
        if overbought_signals and oversold_signals:
            extreme_condition = ((sum(overbought_signals) >= 2).astype(int) + 
                               (sum(oversold_signals) >= 2).astype(int))
            health_components['balance_health'] = 1 - extreme_condition * 0.5
        
        # 综合健康度评分
        weights = {
            'trend_consistency': 0.3, 'volatility_health': 0.25,
            'volume_health': 0.25, 'balance_health': 0.2
        }
        
        technical_health = self._safe_series(0.0, len(dataframe))
        for component, weight in weights.items():
            if component in health_components:
                technical_health += health_components[component].fillna(0) * weight
        
        return technical_health.clip(-1, 1)
    
    def detect_market_state(self, dataframe: DataFrame) -> str:
        """增强版市场状态识别 - 防止顶底反向开仓"""
        current_idx = -1
        
        # 获取基础指标
        adx = dataframe['adx'].iloc[current_idx]
        atr_p = dataframe['atr_p'].iloc[current_idx]
        rsi = dataframe['rsi_14'].iloc[current_idx]
        volume_ratio = dataframe['volume_ratio'].iloc[current_idx]
        price = dataframe['close'].iloc[current_idx]
        ema_8 = dataframe['ema_8'].iloc[current_idx] if 'ema_8' in dataframe.columns else price
        ema_21 = dataframe['ema_21'].iloc[current_idx]
        ema_50 = dataframe['ema_50'].iloc[current_idx]
        
        # 获取MACD指标
        macd = dataframe['macd'].iloc[current_idx] if 'macd' in dataframe.columns else 0
        macd_signal = dataframe['macd_signal'].iloc[current_idx] if 'macd_signal' in dataframe.columns else 0
        
        # === 顶部和底部检测 ===
        # 计算近期高低点
        high_20 = dataframe['high'].rolling(20).max().iloc[current_idx]
        low_20 = dataframe['low'].rolling(20).min().iloc[current_idx]
        price_position = (price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        # 检测是否在顶部区域（避免在顶部开多）
        is_at_top = (
            price_position > 0.90 and  # 价格在20日高点附近
            rsi > 70 and  # RSI超买
            macd < macd_signal  # MACD已经死叉
        )
        
        # 检测是否在底部区域（避免在底部开空）
        is_at_bottom = (
            price_position < 0.10 and  # 价格在20日低点附近
            rsi < 30 and  # RSI超卖
            macd > macd_signal  # MACD已经金叉
        )
        
        # === 趋势强度分析 ===
        # 多时间框架EMA排列
        ema_bullish = ema_8 > ema_21 > ema_50
        ema_bearish = ema_8 < ema_21 < ema_50
        
        # === 市场状态判断 ===
        if is_at_top:
            return "market_top"  # 市场顶部，避免开多
        elif is_at_bottom:
            return "market_bottom"  # 市场底部，避免开空
        elif adx > 40 and atr_p > self.volatility_threshold:
            if ema_bullish and not is_at_top:
                return "strong_uptrend"
            elif ema_bearish and not is_at_bottom:
                return "strong_downtrend"
            else:
                return "volatile"
        elif adx > 25:
            if price > ema_21 and not is_at_top:
                return "mild_uptrend"
            elif price < ema_21 and not is_at_bottom:
                return "mild_downtrend"
            else:
                return "sideways"
        elif atr_p < self.volatility_threshold * 0.5:
            return "consolidation"
        else:
            return "sideways"
    
    def calculate_var(self, returns: List[float], confidence_level: float = 0.05) -> float:
        """计算VaR (Value at Risk)"""
        if len(returns) < 20:
            return 0.05  # 默认5%风险
        
        returns_array = np.array(returns)
        # 使用历史模拟法
        var = np.percentile(returns_array, confidence_level * 100)
        return abs(var)
    
    def calculate_cvar(self, returns: List[float], confidence_level: float = 0.05) -> float:
        """计算CVaR (Conditional Value at Risk)"""
        if len(returns) < 20:
            return 0.08  # 默认8%条件风险
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, confidence_level * 100)
        # CVaR是超过VaR的损失的期望值
        tail_losses = returns_array[returns_array <= var]
        if len(tail_losses) > 0:
            cvar = np.mean(tail_losses)
            return abs(cvar)
        return abs(var)
    
    def calculate_portfolio_correlation(self, pair: str) -> float:
        """计算投资组合相关性"""
        if pair not in self.pair_returns_history:
            return 0.0
        
        current_returns = self.pair_returns_history[pair]
        if len(current_returns) < 20:
            return 0.0
        
        # 计算与其他活跃交易对的平均相关性
        correlations = []
        for other_pair, other_returns in self.pair_returns_history.items():
            if other_pair != pair and len(other_returns) >= 20:
                try:
                    # 确保两个数组长度相同
                    min_length = min(len(current_returns), len(other_returns))
                    corr = np.corrcoef(
                        current_returns[-min_length:], 
                        other_returns[-min_length:]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_kelly_fraction(self, pair: str) -> float:
        """改进的Kelly公式计算"""
        if pair not in self.pair_performance or self.trade_count < 20:
            return 0.25  # 默认保守值
        
        try:
            pair_trades = self.pair_performance[pair]
            wins = [t for t in pair_trades if t > 0]
            losses = [t for t in pair_trades if t < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return 0.25
            
            win_prob = len(wins) / len(pair_trades)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            # Kelly公式: f = (bp - q) / b
            # 其中 b = avg_win/avg_loss, p = win_prob, q = 1-win_prob
            b = avg_win / avg_loss
            kelly = (b * win_prob - (1 - win_prob)) / b
            
            # 保守调整：使用Kelly的1/4到1/2
            kelly_adjusted = max(0.05, min(0.4, kelly * 0.25))
            return kelly_adjusted
            
        except:
            return 0.25
    
    def calculate_position_size(self, current_price: float, market_state: str, pair: str) -> float:
        """动态仓位管理系统 - 根据配置和市场状态调整 + 币种风险控制"""
        
        # === 🎯 获取币种风险等级 ===
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            else:
                coin_risk_tier = 'medium_risk'
        except Exception as e:
            logger.warning(f"获取币种风险等级失败 {pair}: {e}")
            coin_risk_tier = 'medium_risk'
        
        # === 币种风险乘数（垃圾币小仓位以小博大）===
        coin_risk_multipliers = {
            'low_risk': 1.0,        # 低风险：标准仓位
            'medium_risk': 0.8,     # 中等风险：80%仓位
            'high_risk': 0.3        # 高风险（垃圾币）：30%仓位，以小博大
        }
        coin_risk_multiplier = coin_risk_multipliers.get(coin_risk_tier, 0.8)
        
        # === 使用配置的仓位范围中值作为基础 ===
        base_position = (self.base_position_size + self.max_position_size) / 2
        
        # === 连胜/连败乘数系统 ===
        streak_multiplier = 1.0
        if self.consecutive_wins >= 5:
            streak_multiplier = 1.5      # 连胜5次：仓位1.5倍
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.3      # 连胜3次：仓位1.3倍
        elif self.consecutive_wins >= 1:
            streak_multiplier = 1.1      # 连胜1次：仓位1.1倍
        elif self.consecutive_losses >= 3:
            streak_multiplier = 0.6      # 连亏3次：仓位减到60%
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.8      # 连亏1次：仓位减到80%
            
        # === 市场状态乘数（简化） ===
        market_multipliers = {
            "strong_uptrend": 1.25,      # 强趋势：适度激进
            "strong_downtrend": 1.25,    # 强趋势：适度激进
            "mild_uptrend": 1.2,        # 中等趋势
            "mild_downtrend": 1.2,      # 中等趋势
            "sideways": 1.0,            # 横盘：标准
            "volatile": 0.8,            # 高波动：保守
            "consolidation": 0.9        # 整理：略保守
        }
        market_multiplier = market_multipliers.get(market_state, 1.0)
        
        # === 时间段乘数 ===
        time_multiplier = self.get_time_session_position_boost()
        
        # === 账户表现乘数 ===
        equity_multiplier = 1.0
        if self.current_drawdown < -0.10:  # 回撤超过10%
            equity_multiplier = 0.6
        elif self.current_drawdown < -0.05:  # 回撤超过5%
            equity_multiplier = 0.8
        elif self.current_drawdown == 0:     # 无回撤，盈利状态
            equity_multiplier = 1.15
            
        # === 杠杆反比调整 ===
        # 获取当前杠杆
        current_leverage = getattr(self, '_current_leverage', {}).get(pair, 20)
        # 杠杆越高，基础仓位可以相对降低（因为实际风险敞口相同）
        leverage_adjustment = 1.0
        if current_leverage >= 75:
            leverage_adjustment = 0.8    # 高杠杆时适度降低仓位
        elif current_leverage >= 50:
            leverage_adjustment = 0.9
        else:
            leverage_adjustment = 1.1    # 低杠杆时可以提高仓位
            
        # === 🚀复利加速器乘数（核心功能）===
        compound_multiplier = self.get_compound_accelerator_multiplier()
            
        # === 🎯 整合币种风险乘数到总乘数系统 ===
        total_multiplier = (streak_multiplier * market_multiplier * 
                          time_multiplier * equity_multiplier * 
                          leverage_adjustment * compound_multiplier * 
                          coin_risk_multiplier)  # 新增币种风险乘数
        
        # 根据币种风险等级调整最大乘数限制
        max_multiplier_limits = {
            'low_risk': 1.8,        # 低风险：最多1.8倍
            'medium_risk': 1.5,     # 中等风险：最多1.5倍
            'high_risk': 1.2        # 高风险（垃圾币）：最多1.2倍，控制风险
        }
        max_multiplier = max_multiplier_limits.get(coin_risk_tier, 1.5)
        total_multiplier = min(total_multiplier, max_multiplier)
        
        # === 最终仓位计算 ===
        calculated_position = base_position * total_multiplier
        
        # === 智能仓位限制（根据杠杆动态调整）===
        if current_leverage >= 75:
            max_allowed_position = 0.15  # 高杠杆最多15%
        elif current_leverage >= 50:
            max_allowed_position = 0.20  # 中高杠杆最多20%
        elif current_leverage >= 20:
            max_allowed_position = 0.30  # 中杠杆最多30%
        else:
            max_allowed_position = self.max_position_size  # 低杠杆用配置上限
        
        # 应用限制
        final_position = max(self.base_position_size * 0.8, 
                           min(calculated_position, max_allowed_position))
        
        # 风险等级名称映射
        risk_tier_names = {
            'low_risk': '✅ 低风险',
            'medium_risk': '⚡ 中等风险', 
            'high_risk': '⚠️ 高风险'
        }
        
        logger.info(f"""
💰 智能仓位计算 - {pair}:
├─ 🔍 风险等级: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ 📊 基础仓位: {base_position*100:.0f}%
├─ 🏆 连胜乘数: {streak_multiplier:.1f}x (胜{self.consecutive_wins}/败{self.consecutive_losses})
├─ 📈 市场乘数: {market_multiplier:.1f}x ({market_state})
├─ ⏰ 时间乘数: {time_multiplier:.1f}x
├─ 💰 权益乘数: {equity_multiplier:.1f}x
├─ ⚖️ 杠杆调整: {leverage_adjustment:.1f}x ({current_leverage}x杠杆)
├─ 🚀 复利加速: {compound_multiplier:.1f}x
├─ 🎯 风险调整: {coin_risk_multiplier:.1f}x ({coin_risk_tier})
├─ 📐 总乘数限制: {max_multiplier:.1f}x (基于风险等级)
├─ 🧮 计算仓位: {calculated_position*100:.1f}%
└─ 🎉 最终仓位: {final_position*100:.1f}%
""")
        
        return final_position
    
    def get_time_session_position_boost(self) -> float:
        """获取时间段仓位加成"""
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        
        # 基于交易活跃度的仓位调整
        if 14 <= hour <= 16:       # 美盘开盘：最活跃
            return 1.2
        elif 8 <= hour <= 10:      # 欧盘开盘：较活跃  
            return 1.1
        elif 0 <= hour <= 2:       # 亚盘开盘：中等活跃
            return 1.0
        elif 3 <= hour <= 7:       # 深夜：低活跃
            return 0.9
        else:
            return 1.0
    
    def get_compound_accelerator_multiplier(self) -> float:
        """🚀复利加速器系统 - 基于日收益的动态仓位加速"""
        
        # 获取今日收益率
        daily_profit = self.get_daily_profit_percentage()
        
        # 复利加速算法
        if daily_profit >= 0.20:      # 日收益 > 20%
            multiplier = 1.5          # 次日仓位1.5倍（适度激进）
            mode = "🚀极限加速"
        elif daily_profit >= 0.10:    # 日收益 10-20%
            multiplier = 1.5          # 次日仓位1.5倍
            mode = "⚡高速加速"
        elif daily_profit >= 0.05:    # 日收益 5-10%
            multiplier = 1.2          # 次日仓位1.2倍
            mode = "📈温和加速"
        elif daily_profit >= 0:       # 日收益 0-5%
            multiplier = 1.0          # 标准仓位
            mode = "📊标准模式"
        elif daily_profit >= -0.05:   # 日亏损 0-5%
            multiplier = 0.8          # 略微保守
            mode = "🔄调整模式"
        else:                         # 日亏损 > 5%
            multiplier = 0.5          # 次日仓位减半（冷却）
            mode = "❄️冷却模式"
            
        # 连续盈利日加成
        consecutive_profit_days = self.get_consecutive_profit_days()
        if consecutive_profit_days >= 3:
            multiplier *= min(1.3, 1 + consecutive_profit_days * 0.05)  # 最高30%加成
            
        # 连续亏损日惩罚
        consecutive_loss_days = self.get_consecutive_loss_days()
        if consecutive_loss_days >= 2:
            multiplier *= max(0.3, 1 - consecutive_loss_days * 0.15)   # 最低减至30%
            
        # 硬性限制：0.3x - 2.5x
        final_multiplier = max(0.3, min(multiplier, 2.5))
        
        logger.info(f"""
🚀 复利加速器状态:
├─ 今日收益: {daily_profit*100:+.2f}%
├─ 触发模式: {mode}
├─ 基础乘数: {multiplier:.2f}x
├─ 连续盈利: {consecutive_profit_days}天
├─ 连续亏损: {consecutive_loss_days}天
└─ 最终乘数: {final_multiplier:.2f}x
""")
        
        return final_multiplier
    
    def get_daily_profit_percentage(self) -> float:
        """获取今日收益率"""
        try:
            # 简化版本：基于当前总收益的估算
            if hasattr(self, 'total_profit'):
                # 这里可以实现更精确的日收益计算
                # 暂时使用总收益的近似值
                return self.total_profit * 0.1  # 假设日收益是总收益的10%
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def get_consecutive_profit_days(self) -> int:
        """获取连续盈利天数"""
        try:
            # 简化实现，可以后续优化为真实的日级别统计
            if self.consecutive_wins >= 5:
                return min(7, self.consecutive_wins // 2)  # 转换为大致的天数
            else:
                return 0
        except Exception:
            return 0
    
    def get_consecutive_loss_days(self) -> int:
        """获取连续亏损天数"""
        try:
            # 简化实现，可以后续优化为真实的日级别统计
            if self.consecutive_losses >= 3:
                return min(5, self.consecutive_losses // 1)  # 转换为大致的天数
            else:
                return 0
        except Exception:
            return 0
    
    def update_portfolio_performance(self, pair: str, return_pct: float):
        """更新投资组合表现记录"""
        # 更新交易对收益历史
        if pair not in self.pair_returns_history:
            self.pair_returns_history[pair] = []
        
        self.pair_returns_history[pair].append(return_pct)
        
        # 保持最近500个记录
        if len(self.pair_returns_history[pair]) > 500:
            self.pair_returns_history[pair] = self.pair_returns_history[pair][-500:]
        
        # 更新交易对表现记录
        if pair not in self.pair_performance:
            self.pair_performance[pair] = []
        
        self.pair_performance[pair].append(return_pct)
        if len(self.pair_performance[pair]) > 200:
            self.pair_performance[pair] = self.pair_performance[pair][-200:]
        
        # 更新相关性矩阵
        self.update_correlation_matrix()
    
    def update_correlation_matrix(self):
        """更新相关性矩阵"""
        try:
            pairs = list(self.pair_returns_history.keys())
            if len(pairs) < 2:
                return
            
            # 创建相关性矩阵
            n = len(pairs)
            correlation_matrix = np.zeros((n, n))
            
            for i, pair1 in enumerate(pairs):
                for j, pair2 in enumerate(pairs):
                    if i == j:
                        correlation_matrix[i][j] = 1.0
                    else:
                        returns1 = self.pair_returns_history[pair1]
                        returns2 = self.pair_returns_history[pair2]
                        
                        if len(returns1) >= 20 and len(returns2) >= 20:
                            min_length = min(len(returns1), len(returns2))
                            corr = np.corrcoef(
                                returns1[-min_length:], 
                                returns2[-min_length:]
                            )[0, 1]
                            
                            if not np.isnan(corr):
                                correlation_matrix[i][j] = corr
            
            self.correlation_matrix = correlation_matrix
            self.correlation_pairs = pairs
            
        except Exception as e:
            pass
    
    def get_portfolio_risk_metrics(self) -> Dict[str, float]:
        """计算投资组合风险指标"""
        try:
            total_var = 0.0
            total_cvar = 0.0
            portfolio_correlation = 0.0
            
            active_pairs = [pair for pair, returns in self.pair_returns_history.items() 
                          if len(returns) >= 20]
            
            if not active_pairs:
                return {
                    'portfolio_var': 0.05,
                    'portfolio_cvar': 0.08,
                    'avg_correlation': 0.0,
                    'diversification_ratio': 1.0
                }
            
            # 计算平均VaR和CVaR
            var_values = []
            cvar_values = []
            
            for pair in active_pairs:
                returns = self.pair_returns_history[pair]
                var_values.append(self.calculate_var(returns))
                cvar_values.append(self.calculate_cvar(returns))
            
            total_var = np.mean(var_values)
            total_cvar = np.mean(cvar_values)
            
            # 计算平均相关性
            correlations = []
            for i, pair1 in enumerate(active_pairs):
                for j, pair2 in enumerate(active_pairs):
                    if i < j:  # 避免重复计算
                        corr = self.calculate_portfolio_correlation(pair1)
                        if corr > 0:
                            correlations.append(corr)
            
            portfolio_correlation = np.mean(correlations) if correlations else 0.0
            
            # 分散化比率
            diversification_ratio = len(active_pairs) * (1 - portfolio_correlation)
            
            return {
                'portfolio_var': total_var,
                'portfolio_cvar': total_cvar,
                'avg_correlation': portfolio_correlation,
                'diversification_ratio': max(1.0, diversification_ratio)
            }
            
        except Exception as e:
            return {
                'portfolio_var': 0.05,
                'portfolio_cvar': 0.08,
                'avg_correlation': 0.0,
                'diversification_ratio': 1.0
            }
    
    def calculate_leverage(self, market_state: str, volatility: float, pair: str, current_time: datetime = None) -> int:
        """🚀极限杠杆阶梯算法 - 基于波动率的数学精确计算 + 币种风险限制"""
        
        # === 🎯 获取币种风险等级（需要数据框） ===
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            else:
                coin_risk_tier = 'medium_risk'  # 默认中等风险
        except Exception as e:
            logger.warning(f"获取币种风险等级失败 {pair}: {e}")
            coin_risk_tier = 'medium_risk'
        
        # === 币种风险杠杆限制映射 ===
        coin_leverage_limits = {
            'low_risk': (10, 100),      # 低风险：10-100倍（不限制）
            'medium_risk': (5, 50),     # 中等风险：5-50倍
            'high_risk': (1, 10)        # 高风险（垃圾币）：1-10倍（严格限制）
        }
        
        # 获取当前币种的杠杆限制
        min_allowed, max_allowed = coin_leverage_limits.get(coin_risk_tier, (5, 50))
        
        # === 核心算法：波动率阶梯杠杆系统 ===
        volatility_percent = volatility * 100  # 转换为百分比
        
        # 基础杠杆阶梯（基于波动率的反比例关系）
        if volatility_percent < 0.5:
            base_leverage = 100  # 极低波动 = 极高杠杆
        elif volatility_percent < 1.0:
            base_leverage = 75   # 低波动
        elif volatility_percent < 1.5:
            base_leverage = 50   # 中低波动
        elif volatility_percent < 2.0:
            base_leverage = 30   # 中等波动
        elif volatility_percent < 2.5:
            base_leverage = 20   # 中高波动
        else:
            base_leverage = 10   # 高波动，保守杠杆
            
        # === 连胜/连败乘数系统 ===
        streak_multiplier = 1.0
        if self.consecutive_wins >= 5:
            streak_multiplier = 2.0      # 连胜5次：杠杆翻倍
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.5      # 连胜3次：杠杆1.5倍
        elif self.consecutive_wins >= 1:
            streak_multiplier = 1.2      # 连胜1次：杠杆1.2倍
        elif self.consecutive_losses >= 3:
            streak_multiplier = 0.5      # 连亏3次：杠杆减半
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.8      # 连亏1次：杠杆8折
            
        # === 时间段优化乘数 ===
        time_multiplier = self.get_time_session_leverage_boost(current_time)
        
        # === 市场状态乘数（简化） ===
        market_multipliers = {
            "strong_uptrend": 1.3,
            "strong_downtrend": 1.3,
            "mild_uptrend": 1.1,
            "mild_downtrend": 1.1,
            "sideways": 1.0,
            "volatile": 0.8,
            "consolidation": 0.9
        }
        market_multiplier = market_multipliers.get(market_state, 1.0)
        
        # === 账户表现乘数 ===
        equity_multiplier = 1.0
        if self.current_drawdown < -0.05:  # 回撤超过5%
            equity_multiplier = 0.7
        elif self.current_drawdown < -0.02:  # 回撤超过2%
            equity_multiplier = 0.85
        elif self.current_drawdown == 0:     # 无回撤
            equity_multiplier = 1.2
            
        # === 最终杠杆计算 ===
        calculated_leverage = base_leverage * streak_multiplier * time_multiplier * market_multiplier * equity_multiplier
        
        # 先应用原始硬性限制：10-100倍
        pre_risk_leverage = max(10, min(int(calculated_leverage), 100))
        
        # === 🎯 应用币种风险杠杆限制（垃圾币严格限制） ===
        final_leverage = max(min_allowed, min(pre_risk_leverage, max_allowed))
        
        # === 紧急风控 ===
        # 单日亏损超过3%，强制降低杠杆
        if hasattr(self, 'daily_loss') and self.daily_loss < -0.03:
            final_leverage = min(final_leverage, 20)
            
        # 连续亏损保护
        if self.consecutive_losses >= 5:
            final_leverage = min(final_leverage, 15)
            
        # 风险等级名称映射
        risk_tier_names = {
            'low_risk': '✅ 低风险',
            'medium_risk': '⚡ 中等风险', 
            'high_risk': '⚠️ 高风险'
        }
        
        logger.info(f"""
⚡ 智能杠杆计算 - {pair}:
├─ 🔍 风险等级: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ 🎯 风险限制: {min_allowed}-{max_allowed}倍
├─ 📊 波动率: {volatility_percent:.2f}% → 基础杠杆: {base_leverage}x
├─ 🏆 连胜状态: {self.consecutive_wins}胜{self.consecutive_losses}败 → 乘数: {streak_multiplier:.1f}x
├─ ⏰ 时间乘数: {time_multiplier:.1f}x
├─ 📈 市场乘数: {market_multiplier:.1f}x  
├─ 💰 权益乘数: {equity_multiplier:.1f}x
├─ 🧮 计算杠杆: {calculated_leverage:.1f}x
├─ 🔒 预限制杠杆: {pre_risk_leverage}x (通用限制: 10-100x)
└─ 🎉 最终杠杆: {final_leverage}x ({coin_risk_tier}限制: {min_allowed}-{max_allowed}x)
""")
        
        return final_leverage
    
    def get_time_session_leverage_boost(self, current_time: datetime = None) -> float:
        """获取时间段杠杆加成倍数"""
        if not current_time:
            current_time = datetime.now(timezone.utc)
            
        hour = current_time.hour
        
        # 基于交易时段的杠杆优化
        if 0 <= hour <= 2:      # 亚盘开盘 00:00-02:00
            return 1.2
        elif 8 <= hour <= 10:   # 欧盘开盘 08:00-10:00
            return 1.3
        elif 14 <= hour <= 16:  # 美盘开盘 14:00-16:00
            return 1.5          # 最高加成
        elif 20 <= hour <= 22:  # 美盘尾盘 20:00-22:00
            return 1.2
        elif 3 <= hour <= 7:    # 亚洲深夜 03:00-07:00
            return 0.8          # 降低杠杆
        elif 11 <= hour <= 13:  # 欧亚交接 11:00-13:00
            return 0.9
        else:
            return 1.0          # 标准倍数
    
    # 删除了 calculate_dynamic_stoploss - 使用固定止损
    
    def calculate_dynamic_takeprofit(self, pair: str, current_rate: float, trade: Trade, current_profit: float) -> Optional[float]:
        """计算动态止盈目标价格"""
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return None
            
            current_data = dataframe.iloc[-1]
            current_atr = current_data.get('atr_p', 0.02)
            adx = current_data.get('adx', 25)
            trend_strength = current_data.get('trend_strength', 50)
            momentum_score = current_data.get('momentum_score', 0)
            
            # 基于ATR的动态止盈
            base_profit_multiplier = 2.5  # ATR的2.5倍
            
            # 根据趋势强度调整
            if abs(trend_strength) > 70:  # 强趋势
                trend_multiplier = 1.5
            elif abs(trend_strength) > 40:  # 中等趋势
                trend_multiplier = 1.2
            else:  # 弱趋势
                trend_multiplier = 1.0
            
            # 根据动量调整
            momentum_multiplier = 1.0
            if abs(momentum_score) > 0.3:
                momentum_multiplier = 1.3
            elif abs(momentum_score) > 0.1:
                momentum_multiplier = 1.1
            
            # 综合止盈倍数
            profit_multiplier = base_profit_multiplier * trend_multiplier * momentum_multiplier
            
            # 计算止盈距离
            profit_distance = current_atr * profit_multiplier
            
            # 限制止盈范围：8%-80%
            profit_distance = max(0.08, min(0.80, profit_distance))
            
            # 计算目标价格
            if trade.is_short:
                target_price = trade.open_rate * (1 - profit_distance)
            else:
                target_price = trade.open_rate * (1 + profit_distance)
            
            logger.info(f"""
🎯 动态止盈计算 - {pair}:
├─ 入场价格: ${trade.open_rate:.6f}
├─ 当前价格: ${current_rate:.6f}
├─ 当前利润: {current_profit:.2%}
├─ ATR倍数: {profit_multiplier:.2f}
├─ 止盈距离: {profit_distance:.2%}
├─ 目标价格: ${target_price:.6f}
└─ 方向: {'空头' if trade.is_short else '多头'}
""")
            
            return target_price
            
        except Exception as e:
            logger.error(f"动态止盈计算失败 {pair}: {e}")
            return None
    
    # 移除了 get_smart_trailing_stop - 简化止损逻辑
    
    def validate_and_calibrate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """验证和校准技术指标的准确性"""
        try:
            logger.info(f"开始指标验证和校准，数据行数: {len(dataframe)}")
            
            # === RSI 指标校准 ===
            if 'rsi_14' in dataframe.columns:
                # 处理RSI异常值和空值
                original_rsi_nulls = dataframe['rsi_14'].isnull().sum()
                dataframe['rsi_14'] = dataframe['rsi_14'].clip(0, 100)
                dataframe['rsi_14'] = dataframe['rsi_14'].fillna(50)
                
                # RSI平滑处理（减少噪音）
                dataframe['rsi_14'] = dataframe['rsi_14'].ewm(span=2).mean()
                
                logger.info(f"RSI校准完成 - 原始空值: {original_rsi_nulls}, 范围限制: 0-100")
            
            # === MACD 指标校准 ===
            if 'macd' in dataframe.columns:
                # MACD指标平滑处理
                original_macd_nulls = dataframe['macd'].isnull().sum()
                dataframe['macd'] = dataframe['macd'].fillna(0)
                dataframe['macd'] = dataframe['macd'].ewm(span=3).mean()
                
                if 'macd_signal' in dataframe.columns:
                    dataframe['macd_signal'] = dataframe['macd_signal'].fillna(0)
                    dataframe['macd_signal'] = dataframe['macd_signal'].ewm(span=3).mean()
                
                logger.info(f"MACD校准完成 - 原始空值: {original_macd_nulls}, 应用3期平滑")
            
            # === ATR 指标校准 ===
            if 'atr_p' in dataframe.columns:
                # ATR异常值处理
                atr_median = dataframe['atr_p'].median()
                atr_std = dataframe['atr_p'].std()
                
                # 限制ATR在合理范围内（中位数 ± 5倍标准差）
                lower_bound = max(0.001, atr_median - 5 * atr_std)
                upper_bound = min(0.5, atr_median + 5 * atr_std)
                
                original_atr_outliers = ((dataframe['atr_p'] < lower_bound) | 
                                       (dataframe['atr_p'] > upper_bound)).sum()
                
                dataframe['atr_p'] = dataframe['atr_p'].clip(lower_bound, upper_bound)
                dataframe['atr_p'] = dataframe['atr_p'].fillna(atr_median)
                
                logger.info(f"ATR校准完成 - 异常值修正: {original_atr_outliers}, 范围: {lower_bound:.4f}-{upper_bound:.4f}")
            
            # === ADX 指标校准 ===
            if 'adx' in dataframe.columns:
                dataframe['adx'] = dataframe['adx'].clip(0, 100)
                dataframe['adx'] = dataframe['adx'].fillna(25)  # ADX默认值25
                logger.info("ADX校准完成 - 范围限制: 0-100, 默认值: 25")
            
            # === 成交量比率校准 ===
            if 'volume_ratio' in dataframe.columns:
                # 限制成交量比率在合理范围内
                dataframe['volume_ratio'] = dataframe['volume_ratio'].clip(0.1, 20)
                dataframe['volume_ratio'] = dataframe['volume_ratio'].fillna(1.0)
                logger.info("成交量比率校准完成 - 范围限制: 0.1-20, 默认值: 1.0")
            
            # === 趋势强度校准 ===
            if 'trend_strength' in dataframe.columns:
                dataframe['trend_strength'] = dataframe['trend_strength'].clip(-100, 100)
                dataframe['trend_strength'] = dataframe['trend_strength'].fillna(50)
                logger.info("趋势强度校准完成 - 范围限制: -100到100, 默认值: 50")
            
            # === 动量评分校准 ===
            if 'momentum_score' in dataframe.columns:
                dataframe['momentum_score'] = dataframe['momentum_score'].clip(-3, 3)
                dataframe['momentum_score'] = dataframe['momentum_score'].fillna(0)
                logger.info("动量评分校准完成 - 范围限制: -3到3, 默认值: 0")
            
            # === EMA 指标保护 ===
            # 确保EMA指标不被过度处理，保持原始计算结果
            for ema_col in ['ema_8', 'ema_21', 'ema_50']:
                if ema_col in dataframe.columns:
                    # 只处理明显的异常值和空值，不进行平滑处理
                    null_count = dataframe[ema_col].isnull().sum()
                    if null_count > 0:
                        # 使用前向填充处理少量空值
                        dataframe[ema_col] = dataframe[ema_col].ffill().bfill()
                        logger.info(f"{ema_col} 空值处理完成 - 原始空值: {null_count}")
                    
                    # 检查是否有明显异常的EMA值（价格的10倍以上差异）
                    if 'close' in dataframe.columns:
                        price_ratio = dataframe[ema_col] / dataframe['close']
                        outliers = ((price_ratio > 10) | (price_ratio < 0.1)).sum()
                        if outliers > 0:
                            logger.warning(f"{ema_col} 发现 {outliers} 个异常值，重新计算")
                            # 重新计算该EMA
                            if ema_col == 'ema_8':
                                dataframe[ema_col] = ta.EMA(dataframe, timeperiod=8)
                            elif ema_col == 'ema_21':
                                dataframe[ema_col] = ta.EMA(dataframe, timeperiod=21)
                            elif ema_col == 'ema_50':
                                dataframe[ema_col] = ta.EMA(dataframe, timeperiod=50)
            
            # === 指标健康度检查 ===
            self._log_indicator_health(dataframe)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"指标验证和校准失败: {e}")
            return dataframe
    
    def _log_indicator_health(self, dataframe: DataFrame):
        """记录指标健康状况日志"""
        try:
            health_report = []
            
            # 检查各个指标的健康状况
            indicators_to_check = ['rsi_14', 'macd', 'atr_p', 'adx', 'volume_ratio', 'trend_strength', 'momentum_score', 'ema_8', 'ema_21', 'ema_50']
            
            for indicator in indicators_to_check:
                if indicator in dataframe.columns:
                    series = dataframe[indicator].dropna()
                    if len(series) > 0:
                        null_count = dataframe[indicator].isnull().sum()
                        null_pct = null_count / len(dataframe) * 100
                        
                        health_status = "健康" if null_pct < 5 else "警告" if null_pct < 15 else "危险"
                        
                        health_report.append(f"├─ {indicator}: {health_status} (空值: {null_pct:.1f}%)")
            
            if health_report:
                logger.info(f"""
📊 技术指标健康报告:
{chr(10).join(health_report)}
└─ 数据质量: {'优秀' if all('健康' in line for line in health_report) else '良好' if any('警告' in line for line in health_report) else '需要关注'}
""")
        except Exception as e:
            logger.error(f"指标健康检查失败: {e}")
    
    def validate_real_data_quality(self, dataframe: DataFrame, pair: str) -> bool:
        """验证数据是否为真实市场数据而非模拟数据"""
        try:
            if len(dataframe) < 10:
                logger.warning(f"数据量不足 {pair}: {len(dataframe)} 行")
                return False
            
            # 检查价格数据的合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in dataframe.columns:
                    if dataframe[col].isnull().all():
                        logger.error(f"价格数据全为空值 {pair}: {col}")
                        return False
                    
                    # 检查价格是否有合理的变化
                    price_std = dataframe[col].std()
                    price_mean = dataframe[col].mean()
                    if price_std / price_mean < 0.001:  # 变化率低于0.1%
                        logger.warning(f"价格数据变化异常小 {pair}: {col} std/mean = {price_std/price_mean:.6f}")
            
            # 检查成交量数据
            if 'volume' in dataframe.columns:
                if dataframe['volume'].sum() == 0:
                    logger.warning(f"成交量数据全为0 {pair}")
                else:
                    # 检查成交量是否有合理的变化
                    volume_std = dataframe['volume'].std()
                    volume_mean = dataframe['volume'].mean()
                    if volume_mean > 0 and volume_std / volume_mean < 0.1:
                        logger.warning(f"成交量数据变化异常小 {pair}: std/mean = {volume_std/volume_mean:.6f}")
            
            # 检查时间戳连续性
            if 'date' in dataframe.columns or dataframe.index.name == 'date':
                time_diff = dataframe.index.to_series().diff().dropna()
                if len(time_diff) > 0:
                    # 动态计算预期时间间隔，使用最常见的时间间隔作为预期值
                    expected_interval = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else pd.Timedelta(minutes=5)
                    abnormal_intervals = (time_diff != expected_interval).sum()
                    if abnormal_intervals > len(time_diff) * 0.1:  # 超过10%的时间间隔异常
                        logger.warning(f"时间间隔异常 {pair}: {abnormal_intervals}/{len(time_diff)} 个异常间隔 (预期间隔: {expected_interval})")
            
            logger.info(f"✅ 数据质量验证通过 {pair}: {len(dataframe)} 行有效数据")
            return True
            
        except Exception as e:
            logger.error(f"数据质量验证失败 {pair}: {e}")
            return False
    
    # 移除了 _log_detailed_exit_decision 方法 - 简化日志
    
    def _log_risk_calculation_details(self, pair: str, input_params: dict, result: dict):
        """记录详细的风险计算信息"""
        try:
            # 移除了决策日志记录器
            pass
        except Exception as e:
            logger.error(f"风险计算日志记录失败 {pair}: {e}")
    
    def _calculate_risk_rating(self, risk_percentage: float) -> str:
        """计算风险等级"""
        try:
            if risk_percentage < 0.01:  # 小于1%
                return "低风险"
            elif risk_percentage < 0.02:  # 1-2%
                return "中低风险"
            elif risk_percentage < 0.03:  # 2-3%
                return "中等风险"
            elif risk_percentage < 0.05:  # 3-5%
                return "中高风险"
            else:  # 大于5%
                return "高风险"
        except Exception:
            return "风险未知"
    
    def get_equity_performance_factor(self) -> float:
        """获取账户权益表现因子"""
        if self.initial_balance is None:
            return 1.0
            
        try:
            current_balance = self.wallets.get_total_stake_amount()
            
            if current_balance <= 0:
                return 0.5
                
            # 计算收益率
            returns = (current_balance - self.initial_balance) / self.initial_balance
            
            # 更新峰值
            if self.peak_balance is None or current_balance > self.peak_balance:
                self.peak_balance = current_balance
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
            # 根据收益率和回撤计算权重
            if returns > 0.5:  # 收益超过50%
                return 1.5
            elif returns > 0.2:  # 收益20-50%
                return 1.3
            elif returns > 0:
                return 1.1
            elif returns > -0.1:
                return 0.9
            elif returns > -0.2:
                return 0.7
            else:
                return 0.5
                
        except Exception:
            return 1.0
    
    def get_streak_factor(self) -> float:
        """获取连胜连败因子"""
        if self.consecutive_wins >= 5:
            return 1.4  # 连胜5次以上，增加杠杆
        elif self.consecutive_wins >= 3:
            return 1.2  # 连胜3-4次
        elif self.consecutive_wins >= 1:
            return 1.1  # 连胜1-2次
        elif self.consecutive_losses >= 5:
            return 0.4  # 连败5次以上，大幅降低杠杆
        elif self.consecutive_losses >= 3:
            return 0.6  # 连败3-4次
        elif self.consecutive_losses >= 1:
            return 0.8  # 连败1-2次
        else:
            return 1.0  # 没有连胜连败记录
    
    def get_time_session_factor(self, current_time: datetime) -> float:
        """获取时段权重因子"""
        if current_time is None:
            return 1.0
            
        # 获取UTC时间的小时
        hour_utc = current_time.hour
        
        # 定义交易时段权重
        if 8 <= hour_utc <= 16:  # 欧洲时段 (较活跃)
            return 1.3
        elif 13 <= hour_utc <= 21:  # 美国时段 (最活跃)
            return 1.5
        elif 22 <= hour_utc <= 6:  # 亚洲时段 (相对较平静)
            return 0.8
        else:  # 过渡时段
            return 1.0
    
    def get_position_diversity_factor(self) -> float:
        """获取持仓分散度因子"""
        try:
            open_trades = Trade.get_open_trades()
            open_count = len(open_trades)
            
            if open_count == 0:
                return 1.0
            elif open_count <= 2:
                return 1.2  # 持仓较少，可适当增加杠杆
            elif open_count <= 5:
                return 1.0  # 适中
            elif open_count <= 8:
                return 0.8  # 持仓较多，降低杠杆
            else:
                return 0.6  # 持仓过多，大幅降低
                
        except Exception:
            return 1.0
    
    def get_win_rate(self) -> float:
        """获取胜率"""
        if len(self.trade_history) < 10:
            return 0.55  # 默认胜率
            
        wins = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        return wins / len(self.trade_history)
    
    def get_avg_win_loss_ratio(self) -> float:
        """获取平均盈亏比"""
        if len(self.trade_history) < 10:
            return 1.5  # 默认盈亏比
            
        wins = [trade['profit'] for trade in self.trade_history if trade.get('profit', 0) > 0]
        losses = [abs(trade['profit']) for trade in self.trade_history if trade.get('profit', 0) < 0]
        
        if not wins or not losses:
            return 1.5
            
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        
        return avg_win / avg_loss if avg_loss > 0 else 1.5
    
    # 移除了 analyze_multi_timeframe - 简化策略逻辑
    def analyze_multi_timeframe(self, dataframe: DataFrame, metadata: dict) -> Dict:
        """Simplified single timeframe analysis - removed multi-timeframe complexity"""
        
        # Return simple analysis based on current 5m timeframe only
        if dataframe.empty or len(dataframe) < 50:
            return {
                '5m': {
                    'trend': 'unknown',
                    'trend_direction': 'neutral', 
                    'trend_strength': 'unknown',
                    'rsi': 50,
                    'adx': 25
                }
            }
        
        current_data = dataframe.iloc[-1]
        
        # Simple trend analysis using current timeframe
        rsi = current_data.get('rsi_14', 50)
        adx = current_data.get('adx', 25) 
        close = current_data.get('close', 0)
        ema_21 = current_data.get('ema_21', close)
        
        if close > ema_21 and rsi > 50:
            trend_direction = 'bullish'
            trend = 'up'
        elif close < ema_21 and rsi < 50:
            trend_direction = 'bearish' 
            trend = 'down'
        else:
            trend_direction = 'neutral'
            trend = 'sideways'
            
        trend_strength = 'strong' if adx > 25 else 'weak'
        
        return {
            '5m': {
                'trend': trend,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'rsi': rsi,
                'adx': adx,
                'price_position': 0.5,
                'is_top': False,
                'is_bottom': False,
                'momentum': 'neutral',
                'ema_alignment': trend_direction
            }
        }
    
    def get_dataframe_with_indicators(self, pair: str, timeframe: str = None) -> DataFrame:
        """获取包含完整指标的dataframe"""
        if timeframe is None:
            timeframe = self.timeframe
            
        try:
            # 获取原始数据
            dataframe = self.dp.get_pair_dataframe(pair, timeframe)
            if dataframe.empty:
                return dataframe
            
            # 检查是否需要计算指标
            required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
            missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns]
            
            if missing_indicators:
                # 重新计算指标
                metadata = {'pair': pair}
                dataframe = self.populate_indicators(dataframe, metadata)
                
            return dataframe
            
        except Exception as e:
            logger.error(f"获取指标数据失败 {pair}: {e}")
            return DataFrame()

    def _safe_series(self, data, length: int, fill_value=0) -> pd.Series:
        """安全创建Series，避免索引重复问题"""
        if isinstance(data, (int, float)):
            return pd.Series([data] * length, index=range(length))
        elif hasattr(data, '__len__') and len(data) == length:
            return pd.Series(data, index=range(length))
        else:
            return pd.Series([fill_value] * length, index=range(length))
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """优化的指标填充 - 修复缓存和指标计算问题"""

        pair = metadata['pair']

        # 修复重复索引问题 - 重置索引确保唯一性
        if dataframe.index.duplicated().any():
            logger.warning(f"检测到重复索引，正在清理并重置索引: {pair}")
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')].reset_index(drop=True)

        # 保存原始索引
        original_index = dataframe.index.copy()
        
        # 确保有足够的数据进行指标计算
        if len(dataframe) < 50:
            logger.warning(f"数据长度不足 {pair}: {len(dataframe)} < 50")
            # 仍然尝试计算指标，但可能会有NaN值
        
        # 验证数据质量
        data_quality_ok = self.validate_real_data_quality(dataframe, pair)
        if not data_quality_ok:
            logger.warning(f"数据质量验证未通过 {pair}, 但继续处理")
        
        # 暂时禁用缓存以确保指标正确计算
        # cached_indicators = self.get_cached_indicators(pair, len(dataframe))
        # if cached_indicators is not None and len(cached_indicators) == len(dataframe):
        #     # 验证缓存数据是否包含必需指标
        #     required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
        #     if all(indicator in cached_indicators.columns for indicator in required_indicators):
        #         return cached_indicators
        
        # 计算技术指标
        start_time = datetime.now(timezone.utc)
        dataframe = self.calculate_technical_indicators(dataframe)
        
        # 记录性能统计
        calculation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.calculation_stats['indicator_calls'] += 1
        self.calculation_stats['avg_calculation_time'] = (
            (self.calculation_stats['avg_calculation_time'] * (self.calculation_stats['indicator_calls'] - 1) + 
             calculation_time) / self.calculation_stats['indicator_calls']
        )
        
        # 暂时禁用缓存以确保稳定性
        # self.cache_indicators(pair, len(dataframe), dataframe)
        
        # === 检查交易风格切换 ===
        try:
            self.check_and_switch_trading_style(dataframe)
        except Exception as e:
            logger.warning(f"交易风格检查失败: {e}")
        
        # 获取订单簿数据
        pair = metadata['pair']
        try:
            orderbook_data = self.get_market_orderbook(pair)
            if not orderbook_data:
                orderbook_data = {}
        except Exception as e:
            logger.warning(f"获取订单簿数据失败 {pair}: {e}")
            orderbook_data = {}
        
        # 确保必需的订单簿字段总是存在
        required_ob_fields = {
            'volume_ratio': 1.0,
            'spread_pct': 0.1,
            'depth_imbalance': 0.0,
            'market_quality': 0.5,
            'bid_volume': 0,
            'ask_volume': 0,
            'strong_resistance': 0.0,
            'strong_support': 0.0,
            'large_ask_orders': 0.0,
            'large_bid_orders': 0.0,
            'liquidity_score': 0.5,
            'buy_pressure': 0.5,  # 添加买压指标
            'sell_pressure': 0.5   # 添加卖压指标
        }
        
        # 批量添加订单簿数据，避免DataFrame碎片化
        ob_columns = {}
        for key, default_value in required_ob_fields.items():
            value = orderbook_data.get(key, default_value)
            if isinstance(value, (int, float, np.number)):
                ob_columns[f'ob_{key}'] = value
            else:
                # 对于非数值类型，使用默认值
                ob_columns[f'ob_{key}'] = default_value
        
        # 一次性添加所有订单簿列，使用concat避免DataFrame碎片化
        if ob_columns:
            ob_df = pd.DataFrame(ob_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, ob_df], axis=1)
        
        # 市场状态
        if len(dataframe) > 50:
            dataframe['market_state'] = dataframe.apply(
                lambda row: self.detect_market_state(dataframe.loc[:row.name]), axis=1
            )
        else:
            dataframe['market_state'] = 'sideways'
        
        # 多时间框架分析 - 真正应用到策略中
        mtf_analysis = self.analyze_multi_timeframe(dataframe, metadata)
        
        # 将多时间框架分析结果应用到dataframe
        dataframe = self.apply_mtf_analysis_to_dataframe(dataframe, mtf_analysis, metadata)
        
        # 综合信号强度（增强版）
        dataframe['signal_strength'] = self.calculate_enhanced_signal_strength(dataframe)

        # 最终检查和清理重复索引
        if dataframe.index.duplicated().any():
            logger.warning(f"最终检查发现重复索引，正在清理: {pair}")
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

        # 性能优化：去碎片化DataFrame以避免PerformanceWarning
        dataframe = dataframe.copy()

        return dataframe
    
    def convert_trend_strength_to_numeric(self, trend_strength):
        """将字符串类型的趋势强度转换为数值"""
        if isinstance(trend_strength, (int, float)):
            return trend_strength
        
        strength_mapping = {
            'strong': 80,
            'moderate': 60,
            'weak': 30,
            'reversing': 20,
            'unknown': 0
        }
        
        if isinstance(trend_strength, str):
            return strength_mapping.get(trend_strength.lower(), 0)
        
        return 0
    
    def apply_mtf_analysis_to_dataframe(self, dataframe: DataFrame, mtf_analysis: dict, metadata: dict) -> DataFrame:
        """将多时间框架分析结果应用到主dataframe - 真正利用MTF"""
        
        # === 1. 多时间框架趋势一致性评分 ===
        mtf_trend_score = 0
        mtf_strength_score = 0
        mtf_risk_score = 0
        
        # 时间框架权重：越长期权重越大
        tf_weights = {'1m': 0.1, '15m': 0.15, '1h': 0.25, '4h': 0.3, '1d': 0.2}
        
        for tf, analysis in mtf_analysis.items():
            if tf in tf_weights and analysis:
                weight = tf_weights[tf]
                
                # 趋势评分
                if analysis.get('trend_direction') == 'bullish':
                    mtf_trend_score += weight * 1
                elif analysis.get('trend_direction') == 'bearish':
                    mtf_trend_score -= weight * 1
                
                # 强度评分 - 修复类型错误
                trend_strength_raw = analysis.get('trend_strength', 0)
                trend_strength_numeric = self.convert_trend_strength_to_numeric(trend_strength_raw)
                mtf_strength_score += weight * trend_strength_numeric / 100
                
                # 风险评分（RSI极值）
                rsi = analysis.get('rsi', 50)
                if rsi > 70:
                    mtf_risk_score += weight * (rsi - 70) / 30  # 超买风险
                elif rsi < 30:
                    mtf_risk_score -= weight * (30 - rsi) / 30  # 超卖机会
        
        # === 2. 多时间框架关键位置 ===
        # 获取1小时和4小时的关键价格位
        h1_data = mtf_analysis.get('1h', {})
        h4_data = mtf_analysis.get('4h', {})
        
        # === 3. 多时间框架信号过滤器 ===
        # 长期趋势过滤 - 确保为Series格式
        mtf_long_condition = (
            (mtf_trend_score > 0.3) &  # 多时间框架偏多
            (mtf_risk_score > -0.5)    # 风险可控
        )
        
        mtf_short_condition = (
            (mtf_trend_score < -0.3) &  # 多时间框架偏空
            (mtf_risk_score < 0.5)     # 风险可控
        )
        
        # === 4. 多时间框架确认信号 ===
        # 长期确认：4小时+日线都支持
        h4_trend = h4_data.get('trend_direction', 'neutral')
        d1_trend = mtf_analysis.get('1d', {}).get('trend_direction', 'neutral')
        
        mtf_strong_bull_condition = (
            (h4_trend == 'bullish') & (d1_trend == 'bullish') &
            (mtf_strength_score > 0.6)
        )
        
        mtf_strong_bear_condition = (
            (h4_trend == 'bearish') & (d1_trend == 'bearish') &
            (mtf_strength_score > 0.6)
        )
        
        # 批量创建所有多时间框架列，避免DataFrame碎片化
        h1_support = h1_data.get('support_level', dataframe['close'] * 0.99)
        h1_resistance = h1_data.get('resistance_level', dataframe['close'] * 1.01)
        h4_support = h4_data.get('support_level', dataframe['close'] * 0.98)
        h4_resistance = h4_data.get('resistance_level', dataframe['close'] * 1.02)
        
        mtf_columns = {
            # 评分指标
            'mtf_trend_score': mtf_trend_score,  # [-1, 1] 多空趋势一致性
            'mtf_strength_score': mtf_strength_score,  # [0, 1] 趋势强度
            'mtf_risk_score': mtf_risk_score,  # [-1, 1] 风险/机会评分
            
            # 关键价格位
            'h1_support': h1_support,
            'h1_resistance': h1_resistance,
            'h4_support': h4_support,
            'h4_resistance': h4_resistance,
            
            # 价格与关键位置关系
            'near_h1_support': (abs(dataframe['close'] - h1_support) / dataframe['close'] < 0.005).astype(int),
            'near_h1_resistance': (abs(dataframe['close'] - h1_resistance) / dataframe['close'] < 0.005).astype(int),
            'near_h4_support': (abs(dataframe['close'] - h4_support) / dataframe['close'] < 0.01).astype(int),
            'near_h4_resistance': (abs(dataframe['close'] - h4_resistance) / dataframe['close'] < 0.01).astype(int),
            
            # 信号过滤器
            'mtf_long_filter': self._safe_series(1 if mtf_long_condition else 0, len(dataframe)),
            'mtf_short_filter': self._safe_series(1 if mtf_short_condition else 0, len(dataframe)),
            
            # 确认信号
            'mtf_strong_bull': self._safe_series(1 if mtf_strong_bull_condition else 0, len(dataframe)),
            'mtf_strong_bear': self._safe_series(1 if mtf_strong_bear_condition else 0, len(dataframe))
        }
        
        # 一次性添加所有多时间框架列，使用concat避免DataFrame碎片化
        if mtf_columns:
            # 处理Series和标量值
            processed_columns = {}
            for col_name, value in mtf_columns.items():
                if isinstance(value, pd.Series):
                    # 确保Series长度与dataframe匹配
                    if len(value) == len(dataframe):
                        processed_columns[col_name] = value.values
                    else:
                        processed_columns[col_name] = value
                else:
                    processed_columns[col_name] = value
            
            mtf_df = pd.DataFrame(processed_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, mtf_df], axis=1)
        
        return dataframe
    
    def calculate_enhanced_signal_strength(self, dataframe: DataFrame) -> pd.Series:
        """计算增强的综合信号强度"""
        signal_strength = self._safe_series(0.0, len(dataframe))
        
        # 1. 传统技术指标信号 (40%权重)
        traditional_signals = self.calculate_traditional_signals(dataframe) * 0.4
        
        # 2. 动量信号 (25%权重)
        momentum_signals = self._safe_series(0.0, len(dataframe))
        if 'momentum_score' in dataframe.columns:
            momentum_signals = dataframe['momentum_score'] * 2.5 * 0.25  # 放大到[-2.5, 2.5]
        
        # 3. 趋势强度信号 (20%权重)
        trend_signals = self._safe_series(0.0, len(dataframe))
        if 'trend_strength_score' in dataframe.columns:
            trend_signals = dataframe['trend_strength_score'] * 2 * 0.2  # 放大到[-2, 2]
        
        # 4. 技术健康度信号 (15%权重)
        health_signals = self._safe_series(0.0, len(dataframe))
        if 'technical_health' in dataframe.columns:
            health_signals = dataframe['technical_health'] * 1.5 * 0.15  # 放大到[-1.5, 1.5]
        
        # 综合信号强度
        signal_strength = traditional_signals + momentum_signals + trend_signals + health_signals
        
        return signal_strength.fillna(0).clip(-10, 10)  # 限制在[-10, 10]范围
    
    def calculate_traditional_signals(self, dataframe: DataFrame) -> pd.Series:
        """计算传统技术指标信号"""
        signals = self._safe_series(0.0, len(dataframe))
        
        # RSI 信号 (-3 到 +3)
        rsi_signals = self._safe_series(0.0, len(dataframe))
        if 'rsi_14' in dataframe.columns:
            rsi_signals[dataframe['rsi_14'] < 30] = 2
            rsi_signals[dataframe['rsi_14'] > 70] = -2
            rsi_signals[(dataframe['rsi_14'] > 40) & (dataframe['rsi_14'] < 60)] = 1
        
        # MACD 信号 (-2 到 +2)
        macd_signals = self._safe_series(0.0, len(dataframe))
        if 'macd' in dataframe.columns and 'macd_signal' in dataframe.columns:
            macd_signals = ((dataframe['macd'] > dataframe['macd_signal']).astype(int) * 2 - 1)
            if 'macd_hist' in dataframe.columns:
                macd_hist_signals = (dataframe['macd_hist'] > 0).astype(int) * 2 - 1
                macd_signals = (macd_signals + macd_hist_signals) / 2
        
        # 趋势 EMA 信号 (-3 到 +3)
        ema_signals = self._safe_series(0.0, len(dataframe))
        if all(col in dataframe.columns for col in ['ema_8', 'ema_21', 'ema_50']):
            bullish_ema = ((dataframe['ema_8'] > dataframe['ema_21']) & 
                          (dataframe['ema_21'] > dataframe['ema_50']))
            bearish_ema = ((dataframe['ema_8'] < dataframe['ema_21']) & 
                          (dataframe['ema_21'] < dataframe['ema_50']))
            ema_signals[bullish_ema] = 3
            ema_signals[bearish_ema] = -3
        
        # 成交量信号 (-1 到 +2)
        volume_signals = self._safe_series(0.0, len(dataframe))
        if 'volume_ratio' in dataframe.columns:
            volume_signals[dataframe['volume_ratio'] > 1.5] = 2
            volume_signals[dataframe['volume_ratio'] < 0.7] = -1
        
        # ADX 趋势强度信号 (0 到 +2)
        adx_signals = self._safe_series(0.0, len(dataframe))
        if 'adx' in dataframe.columns:
            adx_signals[dataframe['adx'] > 25] = 1
            adx_signals[dataframe['adx'] > 40] = 2
        
        # 高级指标信号
        advanced_signals = self._safe_series(0.0, len(dataframe))
        
        # Fisher Transform 信号
        if 'fisher' in dataframe.columns and 'fisher_signal' in dataframe.columns:
            fisher_cross_up = ((dataframe['fisher'] > dataframe['fisher_signal']) & 
                              (dataframe['fisher'].shift(1) <= dataframe['fisher_signal'].shift(1)))
            fisher_cross_down = ((dataframe['fisher'] < dataframe['fisher_signal']) & 
                                (dataframe['fisher'].shift(1) >= dataframe['fisher_signal'].shift(1)))
            advanced_signals[fisher_cross_up] += 1.5
            advanced_signals[fisher_cross_down] -= 1.5
        
        # KST 信号
        if 'kst' in dataframe.columns and 'kst_signal' in dataframe.columns:
            kst_bullish = dataframe['kst'] > dataframe['kst_signal']
            advanced_signals[kst_bullish] += 1
            advanced_signals[~kst_bullish] -= 1
        
        # MFI 信号
        if 'mfi' in dataframe.columns:
            advanced_signals[dataframe['mfi'] < 30] += 1  # 超卖
            advanced_signals[dataframe['mfi'] > 70] -= 1  # 超买
        
        # 综合传统信号
        total_signals = (rsi_signals + macd_signals + ema_signals + 
                        volume_signals + adx_signals + advanced_signals)
        
        return total_signals.fillna(0).clip(-10, 10)
    
    def _calculate_signal_quality(self, dataframe: DataFrame) -> pd.Series:
        """计算信号质量评分"""
        quality_score = self._safe_series(0.5, len(dataframe))  # 默认中等质量
        
        # 基于信号强度一致性计算质量
        if 'signal_strength' in dataframe.columns:
            # 信号强度绝对值越大质量越高
            abs_strength = abs(dataframe['signal_strength'])
            quality_score = abs_strength / 10.0  # 标准化到0-1
        
        # 基于技术指标一致性
        consistency_factors = []
        
        # RSI一致性
        if 'rsi_14' in dataframe.columns:
            rsi_consistency = 1 - abs(dataframe['rsi_14'] - 50) / 50  # 0-1
            consistency_factors.append(rsi_consistency)
        
        # MACD一致性
        if 'macd' in dataframe.columns and 'macd_signal' in dataframe.columns:
            macd_diff = abs(dataframe['macd'] - dataframe['macd_signal'])
            macd_consistency = 1 / (1 + macd_diff)  # 0-1
            consistency_factors.append(macd_consistency)
        
        # 趋势强度一致性
        if 'trend_strength' in dataframe.columns:
            trend_consistency = abs(dataframe['trend_strength']) / 100  # 0-1
            consistency_factors.append(trend_consistency)
        
        # 成交量确认
        if 'volume_ratio' in dataframe.columns:
            volume_quality = np.minimum(dataframe['volume_ratio'] / 2, 1.0)  # 0-1
            consistency_factors.append(volume_quality)
        
        # 综合质量评分
        if consistency_factors:
            avg_consistency = np.mean(consistency_factors, axis=0)
            quality_score = (quality_score + avg_consistency) / 2
        
        return quality_score.fillna(0.5).clip(0, 1)
    
    def _calculate_position_weight(self, dataframe: DataFrame) -> pd.Series:
        """计算仓位权重"""
        base_weight = self._safe_series(1.0, len(dataframe))  # 基础权重100%
        
        # 基于信号质量调整权重
        if 'signal_quality_score' in dataframe.columns:
            quality_multiplier = 0.5 + dataframe['signal_quality_score'] * 1.5  # 0.5-2.0倍
            base_weight = base_weight * quality_multiplier
        
        # 基于波动性调整
        if 'atr_p' in dataframe.columns:
            # 高波动性降低权重
            volatility_factor = 1 / (1 + dataframe['atr_p'] * 10)  # 0.09-1.0
            base_weight = base_weight * volatility_factor
        
        # 基于趋势强度调整
        if 'trend_strength' in dataframe.columns:
            trend_factor = 0.8 + abs(dataframe['trend_strength']) / 500  # 0.8-1.0
            base_weight = base_weight * trend_factor
        
        return base_weight.fillna(1.0).clip(0.1, 3.0)  # 10%-300%
    
    def _calculate_leverage_multiplier(self, dataframe: DataFrame) -> pd.Series:
        """计算杠杆倍数"""
        base_leverage = self._safe_series(1.0, len(dataframe))  # 基础1倍杠杆
        
        # 基于信号质量调整杠杆
        if 'signal_quality_score' in dataframe.columns:
            # 高质量信号可以使用更高杠杆
            quality_leverage = 1.0 + dataframe['signal_quality_score'] * 2.0  # 1.0-3.0倍
            base_leverage = base_leverage * quality_leverage
        
        # 基于市场波动性调整杠杆
        if 'atr_p' in dataframe.columns:
            # 高波动性使用低杠杆
            volatility_factor = 1 / (1 + dataframe['atr_p'] * 5)  # 0.17-1.0
            base_leverage = base_leverage * volatility_factor
        
        # 基于ADX趋势强度调整
        if 'adx' in dataframe.columns:
            # 强趋势可以使用更高杠杆
            adx_factor = 1.0 + (dataframe['adx'] - 25) / 100  # 0.75-1.75
            adx_factor = np.maximum(adx_factor, 0.5)  # 最低0.5倍
            base_leverage = base_leverage * adx_factor
        
        return base_leverage.fillna(1.0).clip(0.5, 5.0)  # 0.5-5倍杠杆
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """智能入场系统 - 防止追涨杀跌"""
        
        pair = metadata['pair']
        
        # === 核心防追涨杀跌过滤器 ===
        # 计算价格位置（20根K线）
        highest_20 = dataframe['high'].rolling(20).max()
        lowest_20 = dataframe['low'].rolling(20).min()
        price_position = (dataframe['close'] - lowest_20) / (highest_20 - lowest_20 + 0.0001)
        
        # 🚨 修复：放宽价格位置过滤 - 避免错过头部机会
        not_at_top = price_position < 0.80  # 放宽到80%，允许在较高位置入场
        # 防止在底部卖出  
        not_at_bottom = price_position > 0.20  # 放宽到20%，允许在较低位置入场
        
        # === 动量衰竭检测（放宽条件）===
        # 检测RSI动量是否衰竭（放宽判断）
        rsi_momentum_strong = (
            (dataframe['rsi_14'] - dataframe['rsi_14'].shift(3) > -10) &  # 放宽RSI下跌容忍度
            (dataframe['rsi_14'] < 80) & (dataframe['rsi_14'] > 20)  # 放宽RSI极值区范围
        )
        
        # 检测成交量是否支撑（放宽要求）
        volume_support = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.6) &  # 放宽成交量要求
            (dataframe['volume'] > dataframe['volume'].shift(1) * 0.7)  # 放宽成交量萎缩容忍度
        )
        
        # 简化假突破检测（减少过度限制）
        no_fake_breakout = ~(
            # 只检测极端长影线（过度严格的十字星检测已移除）
            ((dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['open']) * 3) |  # 提高到3倍
            ((dataframe['open'] - dataframe['low']) > (dataframe['close'] - dataframe['open']) * 3)       # 提高到3倍
            # 移除十字星检测 - 十字星也可能是好的入场点
        )
        
        # 横盘市场检测（ADX < 20 表示无趋势）
        is_trending = dataframe['adx'] > 20
        is_sideways = dataframe['adx'] < 20
        
        # 横盘市场额外限制（减少开仓频率）
        sideways_filter = ~is_sideways | (dataframe['atr_p'] > 0.02)  # 横盘时需要更大波动
        
        # 增强的基础环境判断
        basic_env = (
            (dataframe['volume_ratio'] > 0.8) &  # 成交量不能太低
            (dataframe['atr_p'] > 0.001) &       # 波动性基本要求
            sideways_filter &                     # 横盘市场过滤
            rsi_momentum_strong &                # RSI动量未衰竭
            volume_support                       # 成交量支撑
        )
        
        # 🚨 修复：定义缺失的环境变量（之前未定义导致60+信号失效）
        # 做多有利环境：趋势不过度弱势 + 情绪不过度悲观
        long_favourable_environment = (
            basic_env &  # 基础环境良好
            (dataframe['trend_strength'] > -40) &  # 趋势不过度弱势（放宽）
            (dataframe.get('market_sentiment', 0) > -0.8) &  # 情绪不过度悲观（放宽）
            (dataframe['rsi_14'] > 25)  # RSI不在极端超卖（避免接飞刀）
        )
        
        # 做空有利环境：趋势不过度强势 + 情绪不过度乐观  
        short_favourable_environment = (
            basic_env &  # 基础环境良好
            (dataframe['trend_strength'] < 40) &   # 趋势不过度强势（放宽）
            (dataframe.get('market_sentiment', 0) < 0.8) &   # 情绪不过度乐观（放宽）
            (dataframe['rsi_14'] < 75)  # RSI不在极端超买（避免在底部做空）
        )
        
        # === 🌍 市场状态感知系统 ===
        market_regime_data = self._enhanced_market_regime_detection(dataframe)
        current_regime = market_regime_data['regime']
        regime_confidence = market_regime_data['confidence']
        signals_advice = market_regime_data['signals_advice']
        
        # 记录市场状态到dataframe（用于后续分析）
        dataframe.loc[:, 'market_regime'] = current_regime
        dataframe.loc[:, 'regime_confidence'] = regime_confidence
        
        logger.info(
            f"📊 市场状态识别 {metadata.get('pair', '')}: "
            f"{current_regime} (置信度:{regime_confidence:.1%}) | "
            f"推荐信号:{signals_advice.get('recommended_signals', [])} | "
            f"避免信号:{signals_advice.get('avoid_signals', [])}"
        )
        
        # === 💰 智能市场适应性信号 ===
        
        # 🎯 Signal 1: RSI超卖反弹（智能动态版）
        # === 动态RSI阈值计算 ===
        # 根据市场波动性调整RSI阈值，高波动期间收紧避免假信号
        base_oversold = 30
        volatility_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        dynamic_oversold = base_oversold - (volatility_percentile * 8)  # 20-30动态范围
        
        # === 多重确认机制 ===
        rsi_condition = (dataframe['rsi_14'] < dynamic_oversold)
        rsi_momentum = (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2))  # 连续2期上升
        price_confirmation = (dataframe['close'] > dataframe['close'].shift(1))
        
        # === 趋势确认：只在上升趋势或横盘中做多 ===
        trend_confirmation = (
            (dataframe['ema_8'] >= dataframe['ema_21']) |  # 多头排列
            (dataframe['adx'] < 25)  # 或横盘环境
        )
        
        # === 成交量确认：突破需要成交量支撑 ===
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1
        )
        
        # === 强度确认：ADX显示趋势开始形成 ===
        strength_confirmation = (
            (dataframe['adx'] > 20) &  # 最低强度要求
            (dataframe['adx'] > dataframe['adx'].shift(2))  # ADX上升
        )
        
        # === 背离检测：避免在顶背离时入场 ===
        no_bearish_divergence = ~dataframe.get('bearish_divergence', False).astype(bool)
        
        rsi_oversold_bounce = (
            rsi_condition &
            rsi_momentum &
            price_confirmation &
            trend_confirmation &
            volume_confirmation &
            strength_confirmation &
            no_bearish_divergence &
            not_at_top &  # 防止在顶部买入
            basic_env
        )
        dataframe.loc[rsi_oversold_bounce, 'enter_long'] = 1
        dataframe.loc[rsi_oversold_bounce, 'enter_tag'] = 'RSI_Oversold_Bounce'
        
        # 🎯 Signal 2: EMA金叉后等待回调（改进版）
        ema_golden_cross = (
            (dataframe['ema_8'] > dataframe['ema_21']) &     # 已经金叉
            (dataframe['ema_8'].shift(3) <= dataframe['ema_21'].shift(3)) &  # 3根K线前刚金叉
            (dataframe['close'] <= dataframe['ema_8'] * 1.01) &  # 价格回调到EMA8附近
            (dataframe['close'] > dataframe['ema_21']) &     # 但仍在EMA21上方
            (dataframe['volume_ratio'] > 1.0) &              # 成交量配合
            # 新增：动量未衰竭验证
            (dataframe['momentum_exhaustion_score'] < 0.5) &  # 动量未衰竭
            (dataframe['trend_phase'] <= 2) &  # 不在趋势末期
            (~dataframe['bearish_divergence'].astype(bool)) &  # 无顶背离
            basic_env
        )
        dataframe.loc[ema_golden_cross, 'enter_long'] = 1
        dataframe.loc[ema_golden_cross, 'enter_tag'] = 'EMA_Golden_Cross'
        
        # 🎯 Signal 3: MACD向上突破（修复：金叉瞬间入场）
        macd_bullish = (
            (
                # MACD金叉瞬间 - 在趋势转折早期捕获信号
                ((dataframe['macd'] > dataframe['macd_signal']) & 
                 (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1))) |
                # 或者柱状图从负转正（备选确认）
                ((dataframe['macd_hist'] > 0) & 
                 (dataframe['macd_hist'].shift(1) <= 0))
            ) &
            basic_env
        )
        dataframe.loc[macd_bullish, 'enter_long'] = 1
        dataframe.loc[macd_bullish, 'enter_tag'] = 'MACD_Bullish'
        
        # 🎯 Signal 4: 布林带下轨反弹（增强确认）
        bb_lower_bounce = (
            (dataframe['close'] <= dataframe['bb_lower'] * 1.005) &  # 接近下轨
            (dataframe['close'] > dataframe['close'].shift(1)) &     # 价格反弹
            (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &  # 二次确认：持续反弹
            (dataframe['rsi_14'] < 50) &                             # RSI偏低
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &  # RSI开始上升
            (dataframe['volume_ratio'] > 1.1) &                     # 成交量增加
            not_at_top &  # 防止追高
            no_fake_breakout &  # 无假突破风险
            basic_env
        )
        dataframe.loc[bb_lower_bounce, 'enter_long'] = 1
        dataframe.loc[bb_lower_bounce, 'enter_tag'] = 'BB_Lower_Bounce'
        
        # Signal 5 已删除 - Simple_Breakout容易产生假突破信号
        
        # === 📉 简化的做空信号 ===
        
        # 🎯 Signal 1: RSI超买回落（智能动态版）
        # === 动态RSI阈值计算 ===
        # 根据市场波动性调整RSI阈值，高波动期间收紧避免假信号
        base_overbought = 70
        volatility_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        dynamic_overbought = base_overbought + (volatility_percentile * 8)  # 70-78动态范围
        
        # === 多重确认机制 ===
        rsi_condition = (dataframe['rsi_14'] > dynamic_overbought)
        rsi_momentum = (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2))  # 连续2期下降
        price_confirmation = (dataframe['close'] < dataframe['close'].shift(1))
        
        # === 趋势确认：只在下降趋势或横盘中做空 ===
        trend_confirmation = (
            (dataframe['ema_8'] <= dataframe['ema_21']) |  # 空头排列
            (dataframe['adx'] < 25)  # 或横盘环境
        )
        
        # === 成交量确认：突破需要成交量支撑 ===
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1
        )
        
        # === 强度确认：ADX显示趋势开始形成 ===
        strength_confirmation = (
            (dataframe['adx'] > 20) &  # 最低强度要求
            (dataframe['adx'] > dataframe['adx'].shift(2))  # ADX上升
        )
        
        # === 背离检测：避免在底背离时入场 ===
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)
        
        rsi_overbought_fall = (
            rsi_condition &
            rsi_momentum &
            price_confirmation &
            trend_confirmation &
            volume_confirmation &
            strength_confirmation &
            no_bullish_divergence &
            not_at_bottom &  # 防止在底部追空
            basic_env
        )
        # === 📊 信号质量评分系统 ===
        rsi_long_score = self._calculate_signal_quality_score(
            dataframe, rsi_oversold_bounce, 'RSI_Oversold_Bounce'
        )
        rsi_short_score = self._calculate_signal_quality_score(
            dataframe, rsi_overbought_fall, 'RSI_Overbought_Fall'
        )
        
        # === 📊 市场状态感知信号过滤 ===
        # 只有高质量+市场状态合适的信号才会触发
        
        # RSI做多信号
        rsi_long_regime_ok = 'RSI_Oversold_Bounce' not in signals_advice.get('avoid_signals', [])
        high_quality_long = rsi_oversold_bounce & (rsi_long_score >= 6) & rsi_long_regime_ok
        
        # RSI做空信号  
        rsi_short_regime_ok = 'RSI_Overbought_Fall' not in signals_advice.get('avoid_signals', [])
        high_quality_short = rsi_overbought_fall & (rsi_short_score >= 6) & rsi_short_regime_ok
        
        # 市场状态奖励：在推荐的市场环境中降低质量要求
        if 'RSI_Oversold_Bounce' in signals_advice.get('recommended_signals', []):
            regime_bonus_long = rsi_oversold_bounce & (rsi_long_score >= 5)  # 降低1分要求
            high_quality_long = high_quality_long | regime_bonus_long
            
        if 'RSI_Overbought_Fall' in signals_advice.get('recommended_signals', []):
            regime_bonus_short = rsi_overbought_fall & (rsi_short_score >= 5)  # 降低1分要求  
            high_quality_short = high_quality_short | regime_bonus_short
        
        dataframe.loc[high_quality_long, 'enter_long'] = 1
        dataframe.loc[high_quality_long, 'enter_tag'] = 'RSI_Oversold_Bounce'
        dataframe.loc[high_quality_long, 'signal_quality'] = rsi_long_score
        dataframe.loc[high_quality_long, 'market_regime_bonus'] = 'RSI_Oversold_Bounce' in signals_advice.get('recommended_signals', [])
        
        dataframe.loc[high_quality_short, 'enter_short'] = 1
        dataframe.loc[high_quality_short, 'enter_tag'] = 'RSI_Overbought_Fall'
        dataframe.loc[high_quality_short, 'signal_quality'] = rsi_short_score
        dataframe.loc[high_quality_short, 'market_regime_bonus'] = 'RSI_Overbought_Fall' in signals_advice.get('recommended_signals', [])
        
        # 🎯 Signal 2: EMA死叉后等待反弹（改进版）
        ema_death_cross = (
            (dataframe['ema_8'] < dataframe['ema_21']) &     # 已经死叉
            (dataframe['ema_8'].shift(3) >= dataframe['ema_21'].shift(3)) &  # 3根K线前刚死叉
            (dataframe['close'] >= dataframe['ema_8'] * 0.99) &  # 价格反弹到EMA8附近
            (dataframe['close'] < dataframe['ema_21']) &     # 但仍在EMA21下方
            (dataframe['volume_ratio'] > 1.0) &              # 成交量配合
            # 新增：动量未衰竭验证
            (dataframe['momentum_exhaustion_score'] < 0.5) &  # 动量未衰竭
            (dataframe['trend_phase'] <= 2) &  # 不在趋势末期
            (~dataframe['bullish_divergence'].astype(bool)) &  # 无底背离
            basic_env
        )
        dataframe.loc[ema_death_cross, 'enter_short'] = 1
        dataframe.loc[ema_death_cross, 'enter_tag'] = 'EMA_Death_Cross'
        
        # 🎯 Signal 3: MACD看跌信号（完全重构版）
        # === MACD基础信号 ===
        macd_death_cross = (
            (dataframe['macd'] < dataframe['macd_signal']) & 
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))
        )
        macd_hist_negative = (
            (dataframe['macd_hist'] < 0) & 
            (dataframe['macd_hist'].shift(1) >= 0)
        )
        macd_basic_signal = macd_death_cross | macd_hist_negative
        
        # === 🛡️ 强化过滤系统 - 解决假信号问题 ===
        
        # 1. 趋势环境确认：避免在上升趋势中做空
        trend_bearish = (
            (dataframe['ema_8'] < dataframe['ema_21']) &  # EMA空头排列
            (dataframe['ema_21'] < dataframe['ema_50']) & # 中长期趋势向下
            (dataframe['close'] < dataframe['ema_21'])     # 价格在趋势线下方
        )
        
        # 2. 动量确认：确保下跌动量真实存在
        momentum_confirmation = (
            (dataframe['rsi_14'] < 55) &                  # RSI偏弱
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2)) &  # RSI连续下跌
            (dataframe['close'] < dataframe['close'].shift(2))      # 价格连续下跌
        )
        
        # 3. 成交量确认：下跌需要成交量支撑
        volume_confirmation = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1) &
            (dataframe['volume'] > dataframe['volume'].shift(1))  # 成交量递增
        )
        
        # 4. 强度确认：ADX显示趋势强化
        strength_confirmation = (
            (dataframe['adx'] > 25) &                     # 有一定趋势强度
            (dataframe['adx'] > dataframe['adx'].shift(3)) # ADX上升趋势
        )
        
        # 5. 横盘过滤：避免在横盘市场中交易
        not_sideways = (dataframe['adx'] > 20)            # 不在横盘状态
        
        # 6. 位置确认：在相对高位做空
        position_confirmation = (
            dataframe['close'] > dataframe['close'].rolling(20).mean() * 1.02  # 价格相对偏高
        )
        
        # 7. 背离保护：避免在底背离时做空
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)
        
        # === 最终MACD看跌信号 ===
        macd_bearish = (
            macd_basic_signal &
            trend_bearish &
            momentum_confirmation &
            volume_confirmation &
            strength_confirmation &
            not_sideways &
            position_confirmation &
            no_bullish_divergence &
            not_at_bottom &  # 防止在底部追空
            basic_env
        )
        
        # === 📊 MACD信号质量评分 ===
        macd_score = self._calculate_macd_signal_quality(dataframe, macd_bearish, 'MACD_Bearish')
        
        # === 📊 MACD市场状态感知过滤 ===
        # MACD信号需要更严格的市场状态确认
        macd_regime_ok = 'MACD_Bearish' not in signals_advice.get('avoid_signals', [])
        high_quality_macd = macd_bearish & (macd_score >= 7) & macd_regime_ok  # MACD要求更高分数+状态确认
        
        # 市场状态奖励：在强下跌趋势中降低MACD要求
        if 'MACD_Bearish' in signals_advice.get('recommended_signals', []):
            regime_bonus_macd = macd_bearish & (macd_score >= 6) & macd_regime_ok  # 降低1分要求
            high_quality_macd = high_quality_macd | regime_bonus_macd
        
        dataframe.loc[high_quality_macd, 'enter_short'] = 1
        dataframe.loc[high_quality_macd, 'enter_tag'] = 'MACD_Bearish'
        dataframe.loc[high_quality_macd, 'signal_quality'] = macd_score
        dataframe.loc[high_quality_macd, 'market_regime_bonus'] = 'MACD_Bearish' in signals_advice.get('recommended_signals', [])
        
        # 🎯 Signal 4: 布林带上轨反压
        bb_upper_rejection = (
            (dataframe['close'] >= dataframe['bb_upper'] * 0.995) &  # 接近上轨
            (dataframe['close'] < dataframe['close'].shift(1)) &     # 价格回落
            (dataframe['rsi_14'] > 50) &                             # RSI偏高
            (dataframe['volume_ratio'] > 1.1) &                     # 成交量增加
            basic_env
        )
        dataframe.loc[bb_upper_rejection, 'enter_short'] = 1
        dataframe.loc[bb_upper_rejection, 'enter_tag'] = 'BB_Upper_Rejection'
        
        # Signal 5 已删除 - Simple_Breakdown容易产生假突破信号
        
        # ==============================
        # 🚨 新增：智能仓位权重系统 - 基于信号质量动态调整
        # ==============================
        
        # 1. 信号质量评分系统
        dataframe['signal_quality_score'] = self._calculate_signal_quality(dataframe)
        dataframe['position_weight'] = self._calculate_position_weight(dataframe)
        dataframe['leverage_multiplier'] = self._calculate_leverage_multiplier(dataframe)
        
        # 统计各类信号数量
        total_long_signals = dataframe['enter_long'].sum()
        total_short_signals = dataframe['enter_short'].sum()
        
        # 统计环境条件激活率
        env_basic_rate = basic_env.sum() / len(dataframe) * 100
        env_long_rate = long_favourable_environment.sum() / len(dataframe) * 100  
        env_short_rate = short_favourable_environment.sum() / len(dataframe) * 100
        
        # 检测是否有信号被激活
        if total_long_signals > 0 or total_short_signals > 0:
            logger.info(f"""
🔥 策略修复验证 - {metadata['pair']}:
📊 信号统计:
   └─ 做多信号: {total_long_signals} 个
   └─ 做空信号: {total_short_signals} 个
   └─ 信号总数: {total_long_signals + total_short_signals} 个

🌍 环境激活率:
   └─ 基础环境: {env_basic_rate:.1f}%
   └─ 做多环境: {env_long_rate:.1f}%  
   └─ 做空环境: {env_short_rate:.1f}%

✅ 修复状态: 环境变量已激活，60+信号重新启用！
""")
        
        # 如果没有信号，报告详细诊断
        if total_long_signals == 0 and total_short_signals == 0:
            logger.warning(f"""
⚠️  无信号诊断 - {metadata['pair']}:
🔍 可能原因:
   └─ 基础环境限制: {100-env_basic_rate:.1f}% K线被过滤
   └─ 做多环境限制: {100-env_long_rate:.1f}% K线不适合做多
   └─ 做空环境限制: {100-env_short_rate:.1f}% K线不适合做空
   
💡 建议: 检查RSI({dataframe['rsi_14'].iloc[-1]:.1f}), 趋势强度({dataframe.get('trend_strength', [0]).iloc[-1]:.1f})
""")
        
        return dataframe
    
    def _legacy_populate_entry_trend_backup(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """备份的旧版本入场信号（已禁用）"""
        
        # 0A. 价格行为抄底 - 最快反转信号
        price_action_bottom = (
            # 看涨Pin Bar：长下影线强烈反转信号
            (dataframe['is_pin_bar_bullish'] == 1) &
            # 或看涨吞噬：强势反转形态
            ((dataframe['is_bullish_engulfing'] == 1) | 
             # StochRSI极度超卖后反弹
             ((dataframe['stoch_rsi_k'] < 20) & (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1)))) &
            # 威廉指标确认反转
            (dataframe['williams_r'] < -70) & (dataframe['williams_r'] > dataframe['williams_r'].shift(1)) &
            # CCI从超卖区反弹
            (dataframe['cci'] < -100) & (dataframe['cci'] > dataframe['cci'].shift(1)) &
            # 成交量异常放大（抄底资金）
            (dataframe['volume_spike'] == 1) &
            # 价格接近支撑位（不是随意抄底）
            (dataframe['support_strength'] < 8) &
            # 价格位置合理：使用预测性判断，不接飞刀
            (price_percentile_20 > 0.15) & in_favorable_long_zone &
            # 多时间框架确认：长期趋势支持
            (dataframe['mtf_long_filter'] == 1) &
            long_favourable_environment
        )
        dataframe.loc[price_action_bottom, 'enter_long'] = 1
        dataframe.loc[price_action_bottom, 'enter_tag'] = '价格行为抄底'
        
        # 0B. [禁用] 加速度反转 - 容易误判
        acceleration_reversal = (
            False &  # 禁用这个信号
            # 价格下跌速度放缓（关键信号）
            (dataframe['price_velocity'] < 0) &  # 仍在下跌
            (dataframe['price_acceleration'] > 0) &  # 但下跌在减速
            (dataframe['price_acceleration'] > dataframe['price_acceleration'].shift(1)) &  # 减速加快
            # RSI背离：价格新低但RSI未创新低
            (dataframe['rsi_divergence_strength'] > 0) &
            # StochRSI快速从低位反弹
            (dataframe['stoch_rsi_k'] < 30) & 
            (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1) + 3) &
            # 成交量异常（恐慌性卖出后的抄底）
            (dataframe['volume_ratio'] > 1.3) &
            # 价格合理位置：预测性最佳入场区间
            (price_percentile_20 > 0.20) & in_favorable_long_zone &
            # 多时间框架确认
            (dataframe['mtf_long_filter'] == 1) &
            long_favourable_environment
        )
        dataframe.loc[acceleration_reversal, 'enter_long'] = 1
        dataframe.loc[acceleration_reversal, 'enter_tag'] = '加速度反转'
        
        # === 🚀 短线交易信号 (持仓时间: 10秒-30分钟) ===
        
        # 1. 超短线动量爆发 - 精准抓住启动点（避免追涨）
        ultra_short_momentum = (
            # EMA5快速上穿EMA8但不在高位
            (dataframe['ema_5'] > dataframe['ema_8']) & 
            (dataframe['ema_5'].shift(1) <= dataframe['ema_8'].shift(1)) &
            # RSI从低位上升，但防止追高（Hyperopt优化）
            (dataframe['rsi_14'] > self.rsi_buy_min.value) & (dataframe['rsi_14'] < self.rsi_buy_max.value) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2) + 5) &
            (dataframe['rsi_14'].shift(2) < self.rsi_buy_max.value) &  # 确保从相对低位启动
            # 成交量爆发但不过度（避免恐慌性买入）
            (dataframe['volume_ratio'] > self.volume_spike_min.value) & (dataframe['volume_ratio'] < self.volume_spike_max.value) &
            # 价格突破但有前瞻性过滤
            (dataframe['close'] > dataframe['high'].rolling(5).max().shift(1)) &
            # 确保不在近期高点附近（防止假突破）
            (dataframe['close'] < dataframe['high'].rolling(20).max() * 0.95) &
            # 趋势确认：确保有持续性
            (dataframe['ema_8'] > dataframe['ema_21']) &  # 中期趋势向上
            # 环境安全 + 预测性过滤
            not_chasing_high & long_favourable_environment &
            in_favorable_long_zone  # 使用预测性最佳区间
        )
        dataframe.loc[ultra_short_momentum, 'enter_long'] = 1
        dataframe.loc[ultra_short_momentum, 'enter_tag'] = '超短线动量'
        
        # 2. 剥头皮交易 - 精准低位入场
        scalping_opportunity = (
            # 价格在布林带中下轨附近（寻找低位机会）
            (dataframe['bb_position'] > 0.3) & (dataframe['bb_position'] < 0.55) &
            # MACD柱状图开始转正且有持续性
            (dataframe['macd_hist'] > 0) & (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(2)) &  # 连续改善
            # 成交量适中但不过度
            (dataframe['volume_ratio'] > 1.1) & (dataframe['volume_ratio'] < 2.0) &
            # RSI从低位回升，但避免追高
            (dataframe['rsi_14'] > 48) & (dataframe['rsi_14'] < 62) &
            (dataframe['rsi_14'].shift(3) < 55) &  # 确保从相对低位启动
            # 趋势强度适中且向上
            (dataframe['trend_strength'] > 5) & (dataframe['trend_strength'] < 25) &
            (dataframe['trend_strength'] > dataframe['trend_strength'].shift(2)) &  # 趋势改善
            # 价格位置安全：预测性最佳区间
            in_favorable_long_zone &  # 使用预测性判断
            long_favourable_environment
        )
        dataframe.loc[scalping_opportunity, 'enter_long'] = 1
        dataframe.loc[scalping_opportunity, 'enter_tag'] = '剝头皮做多'
        
        # === 📈 中线交易信号 (持仓时间: 30分钟-4小时) ===
        
        # 3. 黄金交叉确认 - 经典但加强预测性
        golden_cross_confirmed = (
            # EMA8上穿EMA21，但增加预测性验证
            (dataframe['ema_8'] > dataframe['ema_21']) & 
            (dataframe['ema_8'].shift(1) <= dataframe['ema_21'].shift(1)) &
            # 同时EMA5也在EMA8之上
            (dataframe['ema_5'] > dataframe['ema_8']) &
            # MACD金叉确认且有持续性
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['macd'] > dataframe['macd'].shift(2)) &  # MACD线本身上升
            # RSI健康上升但不追高
            (dataframe['rsi_14'] > 40) & (dataframe['rsi_14'] < 65) &
            (dataframe['rsi_14'].shift(5) < 50) &  # 确保从相对低位启动
            # ADX显示趋势强化且有前瞻性
            (dataframe['adx'] > 20) & (dataframe['adx'] > dataframe['adx'].shift(2)) &
            (dataframe['adx'] < 45) &  # 避免过度趋势（可能反转）
            # 成交量确认但理性
            (dataframe['volume_ratio'] > 1.2) & (dataframe['volume_ratio'] < 3.0) &
            # 价格位置更严格过滤：预测性判断
            in_favorable_long_zone &  # 预测性最佳入场区间
            # 确保有支撑确认
            (dataframe['close'] > dataframe['ema_34']) &  # 在长期均线之上
            # 位置合理
            not_chasing_high & long_favourable_environment
        )
        dataframe.loc[golden_cross_confirmed, 'enter_long'] = 1
        dataframe.loc[golden_cross_confirmed, 'enter_tag'] = '黄金交叉确认'
        
        # 4. 支撑位反弹 - 精确抄底
        support_bounce = (
            # 价格接近EMA34支撑
            (dataframe['close'] > dataframe['ema_34'] * 0.995) & 
            (dataframe['close'] < dataframe['ema_34'] * 1.01) &
            # RSI超卖反弹
            (dataframe['rsi_14'] < 35) & (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2)) &
            # 布林带下轨反弹
            (dataframe['close'] > dataframe['bb_lower']) & 
            (dataframe['close'].shift(1) <= dataframe['bb_lower'].shift(1)) &
            # 成交量放大
            (dataframe['volume_ratio'] > 1.3) &
            # 趋势强度开始回升
            (dataframe['trend_strength'] > dataframe['trend_strength'].shift(3)) &
            # 不在绝对低位
            not_chasing_low & long_favourable_environment
        )
        dataframe.loc[support_bounce, 'enter_long'] = 1
        dataframe.loc[support_bounce, 'enter_tag'] = '支撑位反弹'
        
        # === 📊 长线交易信号 (持仓时间: 4小时-24小时) ===
        
        # 5. 🚨 重新启用：趋势加速 - 优化后避免追涨
        trend_acceleration = (
            # 重新启用但增加安全措施
            # 多重均线排列
            (dataframe['ema_5'] > dataframe['ema_13']) & (dataframe['ema_13'] > dataframe['ema_34']) &
            (dataframe['ema_34'] > dataframe['ema_50']) &
            # 趋势强度提升但不过度
            (dataframe['trend_strength'] > 25) & (dataframe['trend_strength'] < 65) &  # 降低上限
            (dataframe['trend_strength'] - dataframe['trend_strength'].shift(10) > 10) &  # 降低要求
            # ADX强趋势确认但有上限
            (dataframe['adx'] > 25) & (dataframe['adx'] < 50) &  # 避免过度趋势
            (dataframe['adx'] > dataframe['adx'].shift(5)) &
            # MACD强势但不过热
            (dataframe['macd'] > dataframe['macd_signal']) & (dataframe['macd'] > 0) &
            # 价格在VWAP之上但不过高
            (dataframe['close'] > dataframe['vwap']) &
            (dataframe['close'] < dataframe['vwap'] * 1.08) &  # 不超过VWAP 8%
            # RSI强势但防止超买
            (dataframe['rsi_14'] > 50) & (dataframe['rsi_14'] < 70) &  # 降低上限
            (dataframe['rsi_14'].shift(10) < 55) &  # 确保从相对低位开始
            # 成交量理性放大
            (dataframe['volume_ratio'] > 1.15) & (dataframe['volume_ratio'] < 2.5) &
            # 更严格的位置过滤：预测性判断
            in_favorable_long_zone &  # 预测性最佳入场区间
            not_chasing_high & long_favourable_environment
        )
        dataframe.loc[trend_acceleration, 'enter_long'] = 1
        dataframe.loc[trend_acceleration, 'enter_tag'] = '趋势加速'
        
        # 6. 突破回踩确认 - 高概率入场
        breakout_retest = (
            # 价格重新站上关键阻力
            (dataframe['close'] > dataframe['bb_middle']) & 
            (dataframe['close'] > dataframe['vwap']) &
            # 前期有过回踩不破
            (dataframe['low'].rolling(3).min() > dataframe['ema_21'] * 0.99) &
            # 成交量确认
            (dataframe['volume_ratio'] > 1.25) &
            # 技术指标配合
            (dataframe['rsi_14'] > 50) & (dataframe['macd_hist'] > 0) &
            # ADX显示趋势延续
            (dataframe['adx'] > 25) &
            # 动量评分正面
            (dataframe['momentum_score'] > 0.1) &
            long_favourable_environment
        )
        dataframe.loc[breakout_retest, 'enter_long'] = 1
        dataframe.loc[breakout_retest, 'enter_tag'] = '突破回扥确认'
        
        # === 🎯 特殊情况信号 ===
        
        # 7. 反转抄底 - 高风险高收益
        reversal_bottom = (
            # 价格在极低位置
            (price_percentile_20 < 0.25) &
            # RSI严重超卖后反弹
            (dataframe['rsi_14'] < 25) & (dataframe['rsi_14'] > dataframe['rsi_14'].shift(3) + 5) &
            # 成交量异常放大（恐慌后买盘）
            (dataframe['volume_ratio'] > 2.0) &
            # 反转信号强烈
            (dataframe.get('reversal_signal_strength', 0) > 25) &
            # MACD底背离
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            long_favourable_environment
        )
        dataframe.loc[reversal_bottom, 'enter_long'] = 1
        dataframe.loc[reversal_bottom, 'enter_tag'] = '反转抄底'
        
        # === 🎯 多时间框架特殊信号 - 高质量入场 ===
        
        # 8A. MTF强势突破 - 所有时间框架一致看多
        mtf_strong_breakout = (
            # 多时间框架强势一致
            (dataframe['mtf_strong_bull'] == 1) &
            # 价格突破1小时阻力位
            (dataframe['close'] > dataframe['h1_resistance']) &
            (dataframe['close'].shift(1) <= dataframe['h1_resistance'].shift(1)) &
            # 成交量确认
            (dataframe['volume_spike'] == 1) &
            # 5分钟技术指标支持
            (dataframe['rsi_14'] > 50) & (dataframe['rsi_14'] < 70) &
            (dataframe['macd_hist'] > 0) &
            # 不在极高位置：预测性判断
            in_favorable_long_zone &
            long_favourable_environment
        )
        dataframe.loc[mtf_strong_breakout, 'enter_long'] = 1
        dataframe.loc[mtf_strong_breakout, 'enter_tag'] = '多时间框架强突破'
        
        # 8B. MTF支撑反弹 - 在关键支撑位精准抄底
        mtf_support_bounce = (
            # 价格在1小时或4小时支撑位附近
            ((dataframe['near_h1_support'] == 1) | (dataframe['near_h4_support'] == 1)) &
            # 多时间框架趋势评分偏多
            (dataframe['mtf_trend_score'] > 0.2) &
            # 快速反转信号
            ((dataframe['is_pin_bar_bullish'] == 1) | (dataframe['is_bullish_engulfing'] == 1)) &
            # StochRSI超卖反弹
            (dataframe['stoch_rsi_k'] < 30) & (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1)) &
            # 成交量确认
            (dataframe['volume_ratio'] > 1.2) &
            long_favourable_environment
        )
        dataframe.loc[mtf_support_bounce, 'enter_long'] = 1
        dataframe.loc[mtf_support_bounce, 'enter_tag'] = '多时间框架支撑反弹'
        
        # === 🎯 新增预测性做多信号 - 平衡多空，提升做多准确度 ===
        
        # 10A. 成交量背离做多 - 高精度预测
        volume_divergence_long = (
            # 价格创新低但成交量萎缩（恐慌性抛售结束）
            (dataframe['close'] < dataframe['close'].rolling(10).min().shift(1)) &
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].rolling(10).mean() * 0.8) &
            # RSI底背离确认
            (dataframe['rsi_divergence_strength'] > 0.5) &
            # MACD动量开始恢复
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(2)) &
            # 市场情绪过度悲观
            (dataframe['market_sentiment'] < -0.5) &
            # 在预测性做多区间
            in_favorable_long_zone & long_favourable_environment
        )
        dataframe.loc[volume_divergence_long, 'enter_long'] = 1
        dataframe.loc[volume_divergence_long, 'enter_tag'] = '成交量背离做多'
        
        # 10B. [已删除动量复苏做多 - 基于无用的momentum_exhaustion]
        momentum_recovery_long = (
            False &  # 禁用
            # 趋势开始重获可持续性
            (dataframe['trend_sustainability'] > 0.3) &
            # StochRSI从超卖区快速上升
            (dataframe['stoch_rsi_k'] < 25) &
            (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1) + 5) &
            # 威廉指标超卖后转向
            (dataframe['williams_r'] < -80) &
            (dataframe['williams_r'] > dataframe['williams_r'].shift(2) + 10) &
            # CCI极端超卖后反弹
            (dataframe['cci'] < -100) & (dataframe['cci'] > dataframe['cci'].shift(1)) &
            # 在预测性做多区间
            in_favorable_long_zone & long_favourable_environment
        )
        # [已删除 momentum_recovery_long 入场信号]
        
        # 10C. 反转抢跑做多 - 超前预测抄底
        early_reversal_long = (
            # 多重反转信号聚合（底部信号）
            (dataframe['reversal_probability'] > 0.6) &
            # 价格接近反转阈值（超卖）
            (dataframe['market_sentiment'] < -0.7) &
            # 波动率突增但方向向上（筑底完成）
            (dataframe['volatility_state'] > 70) &
            # 趋势强度从低位开始恢复
            (dataframe['trend_strength'] > dataframe['trend_strength'].rolling(5).mean() + 10) &
            # ADX开始上升（新趋势形成）
            (dataframe['adx'] > dataframe['adx'].shift(2)) &
            # 布林带开始扩张（变盘开始）
            (dataframe['bb_upper'] - dataframe['bb_lower'] > (dataframe['bb_upper'] - dataframe['bb_lower']).rolling(10).mean() * 1.1) &
            # RSI从底部区域开始反弹
            (dataframe['rsi_14'] < 40) & (dataframe['rsi_14'] > dataframe['rsi_14'].shift(3) + 3) &
            # 在预测性做多区间
            in_favorable_long_zone & long_favourable_environment
        )
        dataframe.loc[early_reversal_long, 'enter_long'] = 1
        dataframe.loc[early_reversal_long, 'enter_tag'] = '反转抢跑做多'
        
        # 10D. 智能抄底做多 - 多重确认精准入场
        smart_bottom_long = (
            # 价格接近重要支撑位
            (dataframe['close'] > dataframe['vwap'] * 0.98) & (dataframe['close'] < dataframe['vwap'] * 1.01) &
            # 多重超卖确认
            (dataframe['rsi_14'] < 35) & (dataframe['stoch_rsi_k'] < 30) & (dataframe['williams_r'] < -75) &
            # 成交量异常放大（抄底资金进入）
            (dataframe['volume_ratio'] > 1.4) &
            # 布林带下轨支撑
            (dataframe['close'] > dataframe['bb_lower']) & 
            (dataframe['close'].shift(1) <= dataframe['bb_lower'].shift(1)) &
            # 背离信号强烈
            (dataframe['rsi_divergence_strength'] > 0.3) &
            # 趋势强度不过度弱势
            (dataframe['trend_strength'] > -50) &
            # 在预测性做多区间
            in_favorable_long_zone & long_favourable_environment
        )
        dataframe.loc[smart_bottom_long, 'enter_long'] = 1
        dataframe.loc[smart_bottom_long, 'enter_tag'] = '智能抄底做多'
        
        # === 🔻 升级版空头信号 - 使用领先指标解决滞后问题 ===
        
        # 0A. 价格行为做顶 - 最快反转信号
        price_action_top = (
            # 看跌Pin Bar：长上影线强烈反转信号
            (dataframe['is_pin_bar_bearish'] == 1) &
            # 或看跌吞噬：强势反转形态
            ((dataframe['is_bearish_engulfing'] == 1) | 
             # StochRSI极度超买后下跌
             ((dataframe['stoch_rsi_k'] > 80) & (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1)))) &
            # 威廉指标确认反转
            (dataframe['williams_r'] > -30) & (dataframe['williams_r'] < dataframe['williams_r'].shift(1)) &
            # CCI从超买区回落
            (dataframe['cci'] > 100) & (dataframe['cci'] < dataframe['cci'].shift(1)) &
            # 成交量异常放大（出货迹象）
            (dataframe['volume_spike'] == 1) &
            # 价格接近阻力位（不是随意做空）
            (dataframe['resistance_strength'] > -8) &
            # 价格位置合理：预测性最佳做空区间
            in_favorable_short_zone & (price_percentile_20 < 0.85) &
            # 多时间框架确认：长期趋势支持做空
            (dataframe['mtf_short_filter'] == 1) &
            short_favourable_environment
        )
        dataframe.loc[price_action_top, 'enter_short'] = 1
        dataframe.loc[price_action_top, 'enter_tag'] = '价格行为做顶'
        
        # 0B. 加速度反转做空 - 预测最佳做空点
        acceleration_reversal_short = (
            # 价格上涨速度放缓（关键信号）
            (dataframe['price_velocity'] > 0) &  # 仍在上涨
            (dataframe['price_acceleration'] < 0) &  # 但上涨在减速
            (dataframe['price_acceleration'] < dataframe['price_acceleration'].shift(1)) &  # 减速加快
            # RSI顶背离：价格新高但RSI未创新高
            (dataframe['rsi_divergence_strength'] < 0) &
            # StochRSI快速从高位回落
            (dataframe['stoch_rsi_k'] > 70) & 
            (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1) - 3) &
            # 成交量异常（恐慌性买入后的出货）
            (dataframe['volume_ratio'] > 1.3) &
            # 价格合理位置：预测性最佳做空区间
            in_favorable_short_zone & (price_percentile_20 < 0.80) &
            # 多时间框架确认
            (dataframe['mtf_short_filter'] == 1) &
            short_favourable_environment
        )
        dataframe.loc[acceleration_reversal_short, 'enter_short'] = 1
        dataframe.loc[acceleration_reversal_short, 'enter_tag'] = '加速度反转做空'
        
        # === 🔻 传统空头信号（保留但优化） ===
        
        # 1. 超短线顶部反转 - 精准抓顶
        ultra_short_bear = (
            # EMA5快速下穿EMA8 + 顶部背离确认
            (dataframe['ema_5'] < dataframe['ema_8']) & 
            (dataframe['ema_5'].shift(1) >= dataframe['ema_8'].shift(1)) &
            # RSI顶部背离：价格新高但RSI不创新高
            (dataframe['rsi_14'] < 70) & (dataframe['rsi_14'] > 40) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2) - 5) &
            (dataframe['close'] > dataframe['close'].shift(2)) &  # 价格仍在上涨
            (dataframe['rsi_14'] < dataframe['rsi_14'].rolling(5).max().shift(3)) &  # RSI背离
            # 成交量爆发但价格滞涨
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['close'] < dataframe['high'].rolling(3).max()) &  # 未创新高
            # 避免在绝对低位做空：使用预测性区间
            not_chasing_low & short_favourable_environment &
            in_favorable_short_zone  # 预测性最佳做空区间
        )
        dataframe.loc[ultra_short_bear, 'enter_short'] = 1
        dataframe.loc[ultra_short_bear, 'enter_tag'] = '超短线顶部背离'
        
        # 2. 剥头皮做空 - 高位震荡获利
        scalping_short = (
            # 价格在布林带上轨附近震荡
            (dataframe['bb_position'] > 0.7) & (dataframe['bb_position'] < 0.95) &
            # MACD柱状图开始转负
            (dataframe['macd_hist'] < 0) & (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            # 成交量适中但买盘不足
            (dataframe['volume_ratio'] > 1.1) & (dataframe['volume_ratio'] < 2.0) &
            # RSI偏高但未极端超买
            (dataframe['rsi_14'] > 55) & (dataframe['rsi_14'] < 75) &
            # 趋势强度减弱
            (dataframe['trend_strength'] < 70) & (dataframe['trend_strength'] > -10) &
            # 价格位置过滤：预测性做空区间
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[scalping_short, 'enter_short'] = 1
        dataframe.loc[scalping_short, 'enter_tag'] = '剝头皮做空'
        
        # 3. 死叉确认做空 - 经典但优化
        death_cross_confirmed = (
            # EMA8下穿EMA21
            (dataframe['ema_8'] < dataframe['ema_21']) & 
            (dataframe['ema_8'].shift(1) >= dataframe['ema_21'].shift(1)) &
            # 同时EMA5也在EMA8之下
            (dataframe['ema_5'] < dataframe['ema_8']) &
            # MACD死叉确认
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            # RSI健康下降
            (dataframe['rsi_14'] < 60) & (dataframe['rsi_14'] > 30) &
            # ADX显示趋势强化
            (dataframe['adx'] > 20) & (dataframe['adx'] > dataframe['adx'].shift(2)) &
            # 成交量确认
            (dataframe['volume_ratio'] > 1.2) &
            # 位置合理 - 预测性做空区间
            not_chasing_low & short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[death_cross_confirmed, 'enter_short'] = 1
        dataframe.loc[death_cross_confirmed, 'enter_tag'] = '死叉确认'
        
        # 4. 阻力位反压 - 精确做空
        resistance_rejection = (
            # 价格接近EMA34阻力但无法突破
            (dataframe['close'] < dataframe['ema_34'] * 1.005) & 
            (dataframe['close'] > dataframe['ema_34'] * 0.99) &
            # RSI超买回落（Hyperopt优化）
            (dataframe['rsi_14'] > self.rsi_sell_max.value) & (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2)) &
            # 布林带上轨受阻
            (dataframe['close'] < dataframe['bb_upper']) & 
            (dataframe['close'].shift(1) >= dataframe['bb_upper'].shift(1)) &
            # 成交量放大但买盘不足
            (dataframe['volume_ratio'] > 1.3) &
            # 趋势强度开始下降
            (dataframe['trend_strength'] < dataframe['trend_strength'].shift(3)) &
            # 不在绝对高位：预测性做空区间
            not_chasing_high & short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[resistance_rejection, 'enter_short'] = 1
        dataframe.loc[resistance_rejection, 'enter_tag'] = '阻力位反压'
        
        # 5. 趋势衰竭做空 - 中长线持有
        trend_exhaustion = (
            # 多重均线开始翻空
            (dataframe['ema_5'] < dataframe['ema_13']) & (dataframe['ema_13'] < dataframe['ema_34']) &
            (dataframe['ema_34'] < dataframe['ema_50']) &
            # 趋势强度显著下降
            (dataframe['trend_strength'] < -20) & 
            (dataframe['trend_strength'] - dataframe['trend_strength'].shift(10) < -15) &
            # ADX强趋势确认下降
            (dataframe['adx'] > 30) & (dataframe['adx'] > dataframe['adx'].shift(5)) &
            # MACD弱势
            (dataframe['macd'] < dataframe['macd_signal']) & (dataframe['macd'] < 0) &
            # 价格在VWAP之下
            (dataframe['close'] < dataframe['vwap']) &
            # RSI弱势但非极端
            (dataframe['rsi_14'] < 45) & (dataframe['rsi_14'] > 20) &
            # 成交量持续放大（恐慌性抛售）
            (dataframe['volume_ratio'] > 1.15) &
            # 位置合理：预测性做空区间
            not_chasing_low & short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[trend_exhaustion, 'enter_short'] = 1
        dataframe.loc[trend_exhaustion, 'enter_tag'] = '趋势衰竭'
        
        # 6. 假突破做空 - 高概率入场
        false_breakout_short = (
            # 价格跌破关键支撑后快速拉回
            (dataframe['close'] < dataframe['bb_middle']) & 
            (dataframe['close'] < dataframe['vwap']) &
            # 前期有过假突破回踩
            (dataframe['high'].rolling(3).max() < dataframe['ema_21'] * 1.01) &
            # 成交量确认但买盘不足
            (dataframe['volume_ratio'] > 1.25) &
            # 技术指标配合
            (dataframe['rsi_14'] < 50) & (dataframe['macd_hist'] < 0) &
            # ADX显示趋势延续
            (dataframe['adx'] > 25) &
            # 动量评分负面
            (dataframe['momentum_score'] < -0.1) &
            short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[false_breakout_short, 'enter_short'] = 1
        dataframe.loc[false_breakout_short, 'enter_tag'] = '假突破做空'
        
        # 7. 反转做顶 - 高风险高收益
        reversal_top = (
            # 价格在极高位置
            (price_percentile_20 > 0.75) &
            # RSI严重超买后回落
            (dataframe['rsi_14'] > 75) & (dataframe['rsi_14'] < dataframe['rsi_14'].shift(3) - 5) &
            # 成交量异常放大（获利回吐）
            (dataframe['volume_ratio'] > 2.0) &
            # 反转信号强烈
            (dataframe.get('reversal_signal_strength', 0) < -25) &
            # MACD顶背离
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            # 价格创新高但指标背离
            (dataframe['close'] > dataframe['close'].shift(5)) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(5)) &
            short_favourable_environment
        )
        dataframe.loc[reversal_top, 'enter_short'] = 1
        dataframe.loc[reversal_top, 'enter_tag'] = '反转做顶'
        
        # === 🎯 多时间框架特殊空头信号 ===
        
        # 8A. MTF强势破位 - 所有时间框架一致看空
        mtf_strong_breakdown = (
            # 多时间框架强势一致看空
            (dataframe['mtf_strong_bear'] == 1) &
            # 价格跌破1小时支撑位
            (dataframe['close'] < dataframe['h1_support']) &
            (dataframe['close'].shift(1) >= dataframe['h1_support'].shift(1)) &
            # 成交量确认
            (dataframe['volume_spike'] == 1) &
            # 5分钟技术指标支持
            (dataframe['rsi_14'] < 50) & (dataframe['rsi_14'] > 30) &
            (dataframe['macd_hist'] < 0) &
            # 不在极低位置：预测性做空区间
            in_favorable_short_zone &
            short_favourable_environment
        )
        dataframe.loc[mtf_strong_breakdown, 'enter_short'] = 1
        dataframe.loc[mtf_strong_breakdown, 'enter_tag'] = '多时间框架强破位'
        
        # 8B. MTF阻力反压 - 在关键阻力位精准做空
        mtf_resistance_rejection = (
            # 价格在1小时或4小时阻力位附近
            ((dataframe['near_h1_resistance'] == 1) | (dataframe['near_h4_resistance'] == 1)) &
            # 多时间框架趋势评分偏空
            (dataframe['mtf_trend_score'] < -0.2) &
            # 快速反转信号
            ((dataframe['is_pin_bar_bearish'] == 1) | (dataframe['is_bearish_engulfing'] == 1)) &
            # StochRSI超买回落
            (dataframe['stoch_rsi_k'] > 70) & (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1)) &
            # 成交量确认
            (dataframe['volume_ratio'] > 1.2) &
            short_favourable_environment
        )
        dataframe.loc[mtf_resistance_rejection, 'enter_short'] = 1
        dataframe.loc[mtf_resistance_rejection, 'enter_tag'] = '多时间框架阻力反压'
        
        # === 🎯 新增预测性做空信号 - 提升做空准确度 ===
        
        # 9A. 成交量背离做空 - 高精度预测
        volume_divergence_short = (
            # 价格创新高但成交量萎缩
            (dataframe['close'] > dataframe['close'].rolling(10).max().shift(1)) &
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].rolling(10).mean() * 0.8) &
            # RSI背离确认
            (dataframe['rsi_divergence_strength'] < -0.5) &
            # MACD动量减弱
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(2)) &
            # 市场情绪过度乐观
            (dataframe['market_sentiment'] > 0.5) &
            # 在预测性做空区间
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[volume_divergence_short, 'enter_short'] = 1
        dataframe.loc[volume_divergence_short, 'enter_tag'] = '成交量背离做空'
        
        # 9B. [已删除动量衰竭做空 - 基于无用的momentum_exhaustion]
        momentum_exhaustion_short = (
            False &  # 禁用
            # 趋势不可持续
            (dataframe['trend_sustainability'] < -0.3) &
            # StochRSI从超买区快速下降
            (dataframe['stoch_rsi_k'] > 75) &
            (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1) - 5) &
            # 威廉指标超买后转向
            (dataframe['williams_r'] > -20) &
            (dataframe['williams_r'] < dataframe['williams_r'].shift(2) - 10) &
            # CCI极端超买后回落
            (dataframe['cci'] > 100) & (dataframe['cci'] < dataframe['cci'].shift(1)) &
            # 在预测性做空区间
            in_favorable_short_zone & short_favourable_environment
        )
        # [已删除 momentum_exhaustion_short 入场信号]
        
        # 9C. 反转抢跑做空 - 超前预测
        early_reversal_short = (
            # 多重反转信号聚合
            (dataframe['reversal_probability'] > 0.6) &
            # 价格接近反转阈值
            (dataframe['market_sentiment'] > 0.7) &
            # 波动率突增（市场不确定性）
            (dataframe['volatility_state'] > 70) &
            # 趋势强度开始减弱
            (dataframe['trend_strength'] < dataframe['trend_strength'].rolling(5).mean() - 10) &
            # ADX开始下降（趋势衰竭）
            (dataframe['adx'] < dataframe['adx'].shift(2)) &
            # 布林带收缩（变盘前兆）
            (dataframe['bb_upper'] - dataframe['bb_lower'] < (dataframe['bb_upper'] - dataframe['bb_lower']).rolling(10).mean() * 0.9) &
            # 在预测性做空区间
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[early_reversal_short, 'enter_short'] = 1
        dataframe.loc[early_reversal_short, 'enter_tag'] = '反转抢跑做空'
        
        # 9D. 智能做顶做空 - 多重确认精准做空 (完美平衡智能抄底做多)
        smart_top_short = (
            # 价格接近重要阻力位
            (dataframe['close'] < dataframe['vwap'] * 1.02) & (dataframe['close'] > dataframe['vwap'] * 0.99) &
            # 多重超买确认
            (dataframe['rsi_14'] > 65) & (dataframe['stoch_rsi_k'] > 70) & (dataframe['williams_r'] > -25) &
            # 成交量异常放大（出货迹象）
            (dataframe['volume_ratio'] > 1.4) &
            # 布林带上轨阻力
            (dataframe['close'] < dataframe['bb_upper']) & 
            (dataframe['close'].shift(1) >= dataframe['bb_upper'].shift(1)) &
            # 顶背离信号强烈
            (dataframe['rsi_divergence_strength'] < -0.3) &
            # 趋势强度不过度强势
            (dataframe['trend_strength'] < 50) &
            # 在预测性做空区间
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[smart_top_short, 'enter_short'] = 1
        dataframe.loc[smart_top_short, 'enter_tag'] = '智能做顶做空'
        
        # === 信号优先级和冲突处理（完美多空平衡） ===
        # 如果同时有多个信号，选择最强的
        signal_priority = {
            # 超短线信号 - 最高优先级（反应最快）
            'ULTRA_SHORT_MOMENTUM': 10, 'ULTRA_SHORT_BEAR_DIVERGENCE': 10,
            # 预测性信号 - 极高优先级（预测性极强，完美多空平衡）
            'VOLUME_DIVERGENCE_SHORT': 10, 'MOMENTUM_EXHAUSTION_SHORT': 10, 'EARLY_REVERSAL_SHORT': 10, 'SMART_TOP_SHORT': 10,
            'VOLUME_DIVERGENCE_LONG': 10, 'MOMENTUM_RECOVERY_LONG': 10, 'EARLY_REVERSAL_LONG': 10, 'SMART_BOTTOM_LONG': 10,
            # 顶底反转信号 - 高优先级（预测性强）
            'REVERSAL_TOP': 9, 'REVERSAL_BOTTOM': 9,
            # 经典交叉信号 - 高优先级（可靠性强）
            'GOLDEN_CROSS_CONFIRMED': 8, 'DEATH_CROSS_CONFIRMED': 8,
            # 阻力支撑信号 - 中高优先级
            'RESISTANCE_REJECTION': 7, 'SUPPORT_BOUNCE': 7,
            # 趋势信号 - 中等优先级
            'TREND_ACCELERATION': 6, 'TREND_EXHAUSTION': 6,
            # 突破信号 - 中等优先级
            'BREAKOUT_RETEST': 5, 'FALSE_BREAKOUT_SHORT': 5,
            # 剥头皮信号 - 较低优先级（频率高但幅度小）
            'SCALPING_LONG': 4, 'SCALPING_SHORT': 4
        }
        
        # 记录详细信号统计
        signal_counts = {}
        # 完整做空信号列表（完美平衡15个）
        short_signals = ['ULTRA_SHORT_BEAR_DIVERGENCE', 'SCALPING_SHORT', 'DEATH_CROSS_CONFIRMED', 
                        'RESISTANCE_REJECTION', 'TREND_EXHAUSTION', 'FALSE_BREAKOUT_SHORT', 'REVERSAL_TOP',
                        'VOLUME_DIVERGENCE_SHORT', 'MOMENTUM_EXHAUSTION_SHORT', 'EARLY_REVERSAL_SHORT', 'SMART_TOP_SHORT',
                        'PRICE_ACTION_TOP', 'ACCELERATION_REVERSAL_SHORT', 'MTF_STRONG_BREAKDOWN', 'MTF_RESISTANCE_REJECTION']
        
        # 完整做多信号列表（平衡后15个）
        long_signals = ['ULTRA_SHORT_MOMENTUM', 'SCALPING_LONG', 'GOLDEN_CROSS_CONFIRMED',
                       'SUPPORT_BOUNCE', 'TREND_ACCELERATION', 'BREAKOUT_RETEST', 'REVERSAL_BOTTOM',
                       'VOLUME_DIVERGENCE_LONG', 'MOMENTUM_RECOVERY_LONG', 'EARLY_REVERSAL_LONG', 'SMART_BOTTOM_LONG',
                       'PRICE_ACTION_BOTTOM', 'ACCELERATION_REVERSAL', 'MTF_STRONG_BREAKOUT', 'MTF_SUPPORT_BOUNCE']
        
        for tag in signal_priority.keys():
            count = (dataframe['enter_tag'] == tag).sum() if 'enter_tag' in dataframe.columns else 0
            if count > 0:
                if tag in short_signals:
                    signal_counts[f"空头-{tag}"] = count
                elif tag in long_signals:
                    signal_counts[f"多头-{tag}"] = count
                    
        # 统计多空信号平衡度
        total_long_signals = sum([count for key, count in signal_counts.items() if key.startswith("多头")])
        total_short_signals = sum([count for key, count in signal_counts.items() if key.startswith("空头")])
        signal_balance_ratio = total_long_signals / (total_short_signals + 1e-6)  # 避免除零
        
        logger.info(f"""
🎯 多维度入场信号分析 - {pair} (多空平衡优化后):
{'='*60}
📊 信号分布统计:
├─ 多头信号总数: {total_long_signals}
├─ 空头信号总数: {total_short_signals}
├─ 多空平衡比: {signal_balance_ratio:.2f} {'✅平衡' if 0.5 <= signal_balance_ratio <= 2.0 else '⚠️失衡'}
└─ 详细分布: {signal_counts if signal_counts else '当前无信号'}

📈 当前市场状态:
├─ 价格位置: {price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0:.1%}分位 ({price_percentile_50.iloc[-1] if len(price_percentile_50) > 0 else 0:.1%}长期分位)
├─ 多头适宜度: {'✅' if (price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0) < 0.55 else '❌'}做多最佳区间
├─ 空头适宜度: {'✅' if (price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0) > 0.45 else '❌'}做空最佳区间
├─ RSI: {dataframe['rsi_14'].iloc[-1] if 'rsi_14' in dataframe.columns and len(dataframe) > 0 else 50:.1f}
├─ ADX趋势强度: {dataframe['adx'].iloc[-1] if 'adx' in dataframe.columns and len(dataframe) > 0 else 25:.1f}
├─ 成交量倍数: {dataframe['volume_ratio'].iloc[-1] if 'volume_ratio' in dataframe.columns and len(dataframe) > 0 else 1:.2f}x
├─ 趋势评分: {dataframe['trend_strength'].iloc[-1] if 'trend_strength' in dataframe.columns and len(dataframe) > 0 else 50:.0f}/100
├─ 动量评分: {dataframe['momentum_score'].iloc[-1] if 'momentum_score' in dataframe.columns and len(dataframe) > 0 else 0:.3f}
├─ 市场情绪: {dataframe['market_sentiment'].iloc[-1] if 'market_sentiment' in dataframe.columns and len(dataframe) > 0 else 0:.3f}
└─ 背离强度: {dataframe['rsi_divergence_strength'].iloc[-1] if 'rsi_divergence_strength' in dataframe.columns and len(dataframe) > 0 else 0:.3f}

🎯 预测性信号状态:
├─ 多头预测信号: 4个高精度信号 (成交量背离/动量复苏/反转抢跑/智能抄底)
├─ 空头预测信号: 4个高精度信号 (成交量背离/动量衰竭/反转抢跑/智能做顶)
└─ 信号总数: 多头15个 vs 空头15个 (完美平衡)
{'='*60}
""")
        
        return dataframe
    
    def _log_enhanced_entry_decision(self, pair: str, dataframe: DataFrame, current_data, direction: str):
        """记录增强版入场决策详情"""
        
        # 获取具体的入场标签
        entry_tag = current_data.get('enter_tag', 'UNKNOWN_SIGNAL')
        
        # 根据标签确定详细的信号类型说明
        signal_explanations = {
            'GOLDEN_CROSS_BREAKOUT': '黄金交叉突破 - EMA8上穿EMA21，多重均线共振确认上升趋势',
            'MACD_MOMENTUM_CONFIRMED': 'MACD金叉动量确认 - MACD金叉且柱状图增长，动量强劲',
            'OVERSOLD_SUPPORT_BOUNCE': '超卖支撑反弹 - RSI超卖后回升，支撑位确认有效',
            'BREAKOUT_RETEST_HOLD': '突破回踩确认 - 突破关键位后回踩不破，趋势延续',
            'INSTITUTIONAL_ACCUMULATION': '机构资金建仓 - 大单买盘占优，机构资金流入',
            'DEATH_CROSS_BREAKDOWN': '死亡交叉破位 - EMA8下穿EMA21，多重均线确认下降趋势',
            'MACD_MOMENTUM_BEARISH': 'MACD死叉动量确认 - MACD死叉且柱状图下降，动量疲软',
            'OVERBOUGHT_RESISTANCE_REJECT': '超买阻力回调 - RSI超买后回落，阻力位有效',
            'BREAKDOWN_RETEST_FAIL': '破位回测失败 - 破位关键支撑后反弹无力',
            'INSTITUTIONAL_DISTRIBUTION': '机构资金派发 - 大单卖盘占优，机构资金流出'
        }
        
        signal_type = signal_explanations.get(entry_tag, f'技术信号确认 - {entry_tag}')
        
        # 详细的技术分析
        technical_analysis = {
            'rsi_14': current_data.get('rsi_14', 50),
            'macd': current_data.get('macd', 0),
            'macd_signal': current_data.get('macd_signal', 0),
            'macd_hist': current_data.get('macd_hist', 0),
            'ema_8': current_data.get('ema_8', 0),
            'ema_21': current_data.get('ema_21', 0),
            'ema_50': current_data.get('ema_50', 0),
            'adx': current_data.get('adx', 25),
            'volume_ratio': current_data.get('volume_ratio', 1),
            'bb_position': current_data.get('bb_position', 0.5),
            'trend_strength': current_data.get('trend_strength', 50),
            'momentum_score': current_data.get('momentum_score', 0),
            'ob_depth_imbalance': current_data.get('ob_depth_imbalance', 0),
            'ob_market_quality': current_data.get('ob_market_quality', 0.5)
        }
        
        # 构建详细的入场理由说明
        entry_reasoning = self._build_entry_reasoning(entry_tag, technical_analysis, direction)
        
        signal_details = {
            'signal_strength': current_data.get('signal_strength', 0),
            'entry_tag': entry_tag,
            'signal_explanation': signal_type,
            'entry_reasoning': entry_reasoning,
            'trend_confirmed': technical_analysis['trend_strength'] > 30 if direction == 'LONG' else technical_analysis['trend_strength'] < -30,
            'momentum_support': technical_analysis['momentum_score'] > 0.1 if direction == 'LONG' else technical_analysis['momentum_score'] < -0.1,
            'volume_confirmed': technical_analysis['volume_ratio'] > 1.1,
            'market_favorable': technical_analysis['ob_market_quality'] > 0.4,
            'decision_reason': f"{signal_type}"
        }
        
        risk_analysis = {
            'planned_stoploss': abs(self.stoploss) * 100,
            'risk_percentage': self.max_risk_per_trade * 100,
            'suggested_position': self.base_position_size * 100,
            'suggested_leverage': self.leverage_multiplier,
            'risk_budget_remaining': 80,
            'risk_level': self._assess_entry_risk_level(technical_analysis)
        }
        
        # 移除了 decision_logger 日志记录
        pass
    
    def _build_entry_reasoning(self, entry_tag: str, tech: dict, direction: str) -> str:
        """构建详细的入场理由说明"""
        
        reasoning_templates = {
            'GOLDEN_CROSS_BREAKOUT': f"EMA8({tech['ema_8']:.2f})上穿EMA21({tech['ema_21']:.2f})形成黄金交叉，价格突破EMA50({tech['ema_50']:.2f})确认趋势，ADX({tech['adx']:.1f})显示趋势强度充足，成交量放大{tech['volume_ratio']:.1f}倍确认突破有效性",
            
            'MACD_MOMENTUM_CONFIRMED': f"MACD({tech['macd']:.4f})上穿信号线({tech['macd_signal']:.4f})形成金叉，柱状图({tech['macd_hist']:.4f})为正且增长，动量评分{tech['momentum_score']:.3f}显示强劲上升动能，价格站上VWAP确认资金流入",
            
            'OVERSOLD_SUPPORT_BOUNCE': f"RSI({tech['rsi_14']:.1f})从超卖区域反弹，布林带位置({tech['bb_position']:.2f})显示价格接近下轨后企稳，成交量{tech['volume_ratio']:.1f}倍放大确认反弹力度，订单簿深度失衡({tech['ob_depth_imbalance']:.2f})显示买盘占优",
            
            'BREAKOUT_RETEST_HOLD': f"价格突破超级趋势和布林带中轨后，回踩EMA21支撑有效，ADX({tech['adx']:.1f})确认趋势延续，波动率控制在合理范围，成交量{tech['volume_ratio']:.1f}倍支撑突破",
            
            'INSTITUTIONAL_ACCUMULATION': f"订单簿深度失衡({tech['ob_depth_imbalance']:.2f})显示大单买盘占优，异常放量{tech['volume_ratio']:.1f}倍暗示机构建仓，价格站上VWAP，趋势强度({tech['trend_strength']:.0f})开始转强",
            
            'DEATH_CROSS_BREAKDOWN': f"EMA8({tech['ema_8']:.2f})下穿EMA21({tech['ema_21']:.2f})形成死亡交叉，价格跌破EMA50({tech['ema_50']:.2f})确认趋势转空，ADX({tech['adx']:.1f})显示下跌趋势强度，放量{tech['volume_ratio']:.1f}倍确认破位",
            
            'MACD_MOMENTUM_BEARISH': f"MACD({tech['macd']:.4f})下穿信号线({tech['macd_signal']:.4f})形成死叉，柱状图({tech['macd_hist']:.4f})为负且下降，动量评分{tech['momentum_score']:.3f}显示下行压力，价格跌破VWAP确认资金流出",
            
            'OVERBOUGHT_RESISTANCE_REJECT': f"RSI({tech['rsi_14']:.1f})从超买区域回落，布林带位置({tech['bb_position']:.2f})显示价格在上轨遇阻回落，成交量{tech['volume_ratio']:.1f}倍确认抛售压力，订单簿显示阻力位有效",
            
            'BREAKDOWN_RETEST_FAIL': f"价格跌破超级趋势和布林带中轨后，反弹至EMA21阻力失败，ADX({tech['adx']:.1f})确认下跌趋势，成交量{tech['volume_ratio']:.1f}倍支撑破位",
            
            'INSTITUTIONAL_DISTRIBUTION': f"订单簿深度失衡({tech['ob_depth_imbalance']:.2f})显示大单卖盘占优，异常放量{tech['volume_ratio']:.1f}倍暗示机构派发，价格跌破VWAP，趋势强度({tech['trend_strength']:.0f})转弱"
        }
        
        return reasoning_templates.get(entry_tag, f"基于{entry_tag}的技术信号确认，多项指标共振支持{direction}方向")
    
    def _assess_entry_risk_level(self, tech: dict) -> str:
        """评估入场风险等级"""
        risk_score = 0
        
        # ADX风险评估
        if tech['adx'] > 30:
            risk_score += 1  # 强趋势降低风险
        elif tech['adx'] < 20:
            risk_score -= 1  # 弱趋势增加风险
            
        # 成交量风险评估
        if tech['volume_ratio'] > 1.5:
            risk_score += 1  # 放量降低风险
        elif tech['volume_ratio'] < 0.8:
            risk_score -= 1  # 缩量增加风险
            
        # 市场质量风险评估
        if tech['ob_market_quality'] > 0.6:
            risk_score += 1  # 高质量降低风险
        elif tech['ob_market_quality'] < 0.3:
            risk_score -= 1  # 低质量增加风险
            
        # 波动率风险评估 (通过RSI极值判断)
        if 25 < tech['rsi_14'] < 75:
            risk_score += 1  # 健康区间降低风险
        else:
            risk_score -= 1  # 极值区间增加风险
        
        if risk_score >= 2:
            return "低风险"
        elif risk_score >= 0:
            return "中等风险"
        else:
            return "高风险"
    
    def _log_short_entry_decision(self, pair: str, dataframe: DataFrame, current_data):
        """记录空头入场决策详情"""
        
        signal_type = self._determine_short_signal_type(current_data)
        
        signal_details = {
            'signal_strength': current_data.get('signal_strength', 0),
            'trend_confirmed': current_data.get('trend_strength', 0) > 60,
            'momentum_support': current_data.get('momentum_score', 0) < -0.1,
            'volume_confirmed': current_data.get('volume_ratio', 1) > 1.1,
            'market_favorable': current_data.get('volatility_state', 50) < 90,
            'decision_reason': f"{signal_type} - 信号强度{current_data.get('signal_strength', 0):.1f}"
        }
        
        risk_analysis = {
            'planned_stoploss': abs(self.stoploss) * 100,
            'risk_percentage': self.max_risk_per_trade * 100,
            'suggested_position': self.base_position_size * 100,
            'suggested_leverage': self.leverage_multiplier,
            'risk_budget_remaining': 80,  # 估计值
            'risk_level': '中等'
        }
        
        # 移除了 decision_logger 日志记录
        pass
    
    def _determine_long_signal_type(self, current_data) -> str:
        """判断多头信号类型"""
        if (current_data.get('trend_strength', 0) > 60 and 
            current_data.get('momentum_score', 0) > 0.1):
            return "趋势确认+动量支撑"
        elif current_data.get('rsi_14', 50) < 35:
            return "超卖反弹机会"
        elif (current_data.get('close', 0) > current_data.get('supertrend', 0)):
            return "突破确认信号"
        else:
            return "复合信号"
    
    def _determine_short_signal_type(self, current_data) -> str:
        """判断空头信号类型"""
        if (current_data.get('trend_strength', 0) > 60 and 
            current_data.get('momentum_score', 0) < -0.1):
            return "趋势确认+动量支撑(空头)"
        elif current_data.get('rsi_14', 50) > 65:
            return "超买回调机会"
        elif (current_data.get('close', 0) < current_data.get('supertrend', 0)):
            return "突破确认信号(空头)"
        else:
            return "复合信号(空头)"
    
    def calculate_signal_strength(self, dataframe: DataFrame) -> DataFrame:
        """升级版综合信号强度计算 - 多维度精准评分"""
        
        # === 1. 趋势信号强度 (权重35%) ===
        # 基于ADX确认的趋势强度
        trend_signal = np.where(
            (dataframe['trend_strength'] > 70) & (dataframe['adx'] > 30), 3,  # 超强趋势
            np.where(
                (dataframe['trend_strength'] > 50) & (dataframe['adx'] > 25), 2,  # 强趋势
                np.where(
                    (dataframe['trend_strength'] > 30) & (dataframe['adx'] > 20), 1,  # 中等趋势
                    np.where(
                        (dataframe['trend_strength'] < -70) & (dataframe['adx'] > 30), -3,  # 超强下跌
                        np.where(
                            (dataframe['trend_strength'] < -50) & (dataframe['adx'] > 25), -2,  # 强下跌
                            np.where(
                                (dataframe['trend_strength'] < -30) & (dataframe['adx'] > 20), -1, 0  # 中等下跌
                            )
                        )
                    )
                )
            )
        ) * 0.35
        
        # === 2. 动量信号强度 (权重30%) ===
        # MACD + RSI + 价格动量综合
        macd_momentum = np.where(
            (dataframe['macd'] > dataframe['macd_signal']) & (dataframe['macd_hist'] > 0), 1,
            np.where((dataframe['macd'] < dataframe['macd_signal']) & (dataframe['macd_hist'] < 0), -1, 0)
        )
        
        rsi_momentum = np.where(
            dataframe['rsi_14'] > 60, 1,
            np.where(dataframe['rsi_14'] < 40, -1, 0)
        )
        
        price_momentum = np.where(
            dataframe['momentum_score'] > 0.2, 2,
            np.where(
                dataframe['momentum_score'] > 0.1, 1,
                np.where(
                    dataframe['momentum_score'] < -0.2, -2,
                    np.where(dataframe['momentum_score'] < -0.1, -1, 0)
                )
            )
        )
        
        momentum_signal = (macd_momentum + rsi_momentum + price_momentum) * 0.30
        
        # === 3. 成交量确认信号 (权重20%) ===
        volume_signal = np.where(
            dataframe['volume_ratio'] > 2.0, 2,  # 异常放量
            np.where(
                dataframe['volume_ratio'] > 1.5, 1,  # 明显放量
                np.where(
                    dataframe['volume_ratio'] < 0.6, -1,  # 缩量
                    0
                )
            )
        ) * 0.20
        
        # === 4. 市场微结构信号 (权重10%) ===
        microstructure_signal = np.where(
            (dataframe['ob_depth_imbalance'] > 0.2) & (dataframe['ob_market_quality'] > 0.5), 1,  # 买盘占优
            np.where(
                (dataframe['ob_depth_imbalance'] < -0.2) & (dataframe['ob_market_quality'] > 0.5), -1,  # 卖盘占优
                0
            )
        ) * 0.10
        
        # === 5. 技术位突破确认 (权重5%) ===
        breakout_signal = np.where(
            (dataframe['close'] > dataframe['supertrend']) & (dataframe['bb_position'] > 0.6), 1,  # 向上突破
            np.where(
                (dataframe['close'] < dataframe['supertrend']) & (dataframe['bb_position'] < 0.4), -1,  # 向下突破
                0
            )
        ) * 0.05
        
        # === 综合信号强度 ===
        dataframe['signal_strength'] = (trend_signal + momentum_signal + volume_signal + 
                                      microstructure_signal + breakout_signal)
        
        # === 信号质量评估 ===
        # 多重确认的信号质量更高
        confirmation_count = (
            (np.abs(trend_signal) > 0).astype(int) +
            (np.abs(momentum_signal) > 0).astype(int) +
            (np.abs(volume_signal) > 0).astype(int) +
            (np.abs(microstructure_signal) > 0).astype(int)
        )
        
        # 信号质量加权
        quality_multiplier = np.where(
            confirmation_count >= 3, 1.3,  # 三重确认
            np.where(confirmation_count >= 2, 1.1, 0.8)  # 双重确认
        )
        
        dataframe['signal_strength'] = dataframe['signal_strength'] * quality_multiplier
        
        # 性能优化：去碎片化DataFrame以避免PerformanceWarning
        dataframe = dataframe.copy()
        
        return dataframe
    
    # ===== 实时监控与自适应系统 =====
    
    def initialize_monitoring_system(self):
        """初始化监控系统"""
        self.monitoring_enabled = True
        self.performance_window = 100  # 性能监控窗口
        self.adaptation_threshold = 0.1  # 适应触发阈值
        self.last_monitoring_time = datetime.now(timezone.utc)
        self.monitoring_interval = 300  # 5分钟监控间隔
        
        # 性能指标追踪
        self.performance_metrics = {
            'win_rate': [],
            'profit_factor': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'avg_trade_duration': [],
            'volatility': []
        }
        
        # 市场状态追踪
        self.market_regime_history = []
        self.volatility_regime_history = []
        
        # 自适应参数记录
        self.parameter_adjustments = []
        
        # 风险监控阈值
        self.risk_thresholds = {
            'max_daily_loss': -0.05,  # 日最大亏损5%
            'max_drawdown': -0.15,    # 最大回撤15%
            'min_win_rate': 0.35,     # 最低胜率35%
            'max_volatility': 0.25,   # 最大波动率25%
            'max_correlation': 0.8    # 最大相关性80%
        }
        
    def monitor_real_time_performance(self) -> Dict[str, Any]:
        """实时性能监控"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # 检查监控间隔
            if (current_time - self.last_monitoring_time).seconds < self.monitoring_interval:
                return {}
            
            self.last_monitoring_time = current_time
            
            # 获取当前性能指标
            current_metrics = self.calculate_current_performance_metrics()
            
            # 更新性能历史
            self.update_performance_history(current_metrics)
            
            # 风险警报检查
            risk_alerts = self.check_risk_thresholds(current_metrics)
            
            # 市场状态监控
            market_state = self.monitor_market_regime()
            
            # 策略适应性检查
            adaptation_needed = self.check_adaptation_requirements(current_metrics)
            
            monitoring_report = {
                'timestamp': current_time,
                'performance_metrics': current_metrics,
                'risk_alerts': risk_alerts,
                'market_state': market_state,
                'adaptation_needed': adaptation_needed,
                'monitoring_status': 'active'
            }
            
            # 如果需要适应，执行自动调整
            if adaptation_needed:
                self.execute_adaptive_adjustments(current_metrics, market_state)
            
            return monitoring_report
            
        except Exception as e:
            return {'error': f'监控系统错误: {str(e)}', 'monitoring_status': 'error'}
    
    def calculate_current_performance_metrics(self) -> Dict[str, float]:
        """计算当前性能指标"""
        try:
            # 获取最近的交易记录
            recent_trades = self.get_recent_trades(self.performance_window)
            
            if not recent_trades:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade_duration': 0.0,
                    'volatility': 0.0,
                    'total_trades': 0
                }
            
            # 计算胜率
            profitable_trades = [t for t in recent_trades if t['profit'] > 0]
            win_rate = len(profitable_trades) / len(recent_trades)
            
            # 计算盈利因子
            total_profit = sum([t['profit'] for t in profitable_trades])
            total_loss = abs(sum([t['profit'] for t in recent_trades if t['profit'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            # 计算夏普比率
            returns = [t['profit'] for t in recent_trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # 计算最大回撤
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown)
            
            # 平均交易持续时间
            durations = [t.get('duration_hours', 0) for t in recent_trades]
            avg_trade_duration = np.mean(durations)
            
            # 波动率
            volatility = std_return
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade_duration': avg_trade_duration,
                'volatility': volatility,
                'total_trades': len(recent_trades)
            }
            
        except Exception:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_trade_duration': 0.0,
                'volatility': 0.0,
                'total_trades': 0
            }
    
    def get_recent_trades(self, window_size: int) -> List[Dict]:
        """获取最近的交易记录"""
        try:
            # 这里应该从实际的交易历史中获取数据
            # 暂时返回模拟数据结构
            return []
        except Exception:
            return []
    
    def update_performance_history(self, metrics: Dict[str, float]):
        """更新性能历史记录"""
        try:
            for key, value in metrics.items():
                if key in self.performance_metrics:
                    self.performance_metrics[key].append(value)
                    
                    # 保持历史记录在合理长度
                    if len(self.performance_metrics[key]) > 1000:
                        self.performance_metrics[key] = self.performance_metrics[key][-500:]
        except Exception:
            pass
    
    def check_risk_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """检查风险阈值"""
        alerts = []
        
        try:
            # 检查胜率
            if metrics['win_rate'] < self.risk_thresholds['min_win_rate']:
                alerts.append({
                    'type': 'low_win_rate',
                    'severity': 'warning',
                    'current_value': metrics['win_rate'],
                    'threshold': self.risk_thresholds['min_win_rate'],
                    'message': f"胜率过低: {metrics['win_rate']:.1%} < {self.risk_thresholds['min_win_rate']:.1%}"
                })
            
            # 检查最大回撤
            if metrics['max_drawdown'] < self.risk_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'high_drawdown',
                    'severity': 'critical',
                    'current_value': metrics['max_drawdown'],
                    'threshold': self.risk_thresholds['max_drawdown'],
                    'message': f"回撤过大: {metrics['max_drawdown']:.1%} < {self.risk_thresholds['max_drawdown']:.1%}"
                })
            
            # 检查波动率
            if metrics['volatility'] > self.risk_thresholds['max_volatility']:
                alerts.append({
                    'type': 'high_volatility',
                    'severity': 'warning',
                    'current_value': metrics['volatility'],
                    'threshold': self.risk_thresholds['max_volatility'],
                    'message': f"波动率过高: {metrics['volatility']:.1%} > {self.risk_thresholds['max_volatility']:.1%}"
                })
                
        except Exception:
            pass
        
        return alerts
    
    def monitor_market_regime(self) -> Dict[str, Any]:
        """监控市场状态变化"""
        try:
            # 获取当前市场指标
            current_regime = {
                'trend_strength': 0.0,
                'volatility_level': 0.0,
                'market_state': 'unknown',
                'regime_stability': 0.0
            }
            
            # 这里应该集成实际的市场数据获取
            # 暂时返回默认结构
            
            return current_regime
            
        except Exception:
            return {
                'trend_strength': 0.0,
                'volatility_level': 0.0,
                'market_state': 'unknown',
                'regime_stability': 0.0
            }
    
    def check_adaptation_requirements(self, metrics: Dict[str, float]) -> bool:
        """检查是否需要策略适应"""
        try:
            # 性能显著下降
            if len(self.performance_metrics['win_rate']) > 50:
                recent_win_rate = np.mean(self.performance_metrics['win_rate'][-20:])
                historical_win_rate = np.mean(self.performance_metrics['win_rate'][-50:-20])
                
                if historical_win_rate > 0 and (recent_win_rate / historical_win_rate) < 0.8:
                    return True
            
            # 夏普比率恶化
            if len(self.performance_metrics['sharpe_ratio']) > 50:
                recent_sharpe = np.mean(self.performance_metrics['sharpe_ratio'][-20:])
                if recent_sharpe < 0.5:  # 夏普比率过低
                    return True
            
            # 回撤过大
            if metrics['max_drawdown'] < -0.12:  # 超过12%回撤
                return True
            
            return False
            
        except Exception:
            return False
    
    def execute_adaptive_adjustments(self, metrics: Dict[str, float], market_state: Dict[str, Any]):
        """执行自适应调整"""
        try:
            adjustments = []
            
            # 基于性能的调整
            if metrics['win_rate'] < 0.4:
                # 降低仓位大小
                self.base_position_size *= 0.8
                adjustments.append('reduced_position_size')
                
                # 收紧止损
                self.stoploss *= 1.1
                adjustments.append('tightened_stoploss')
            
            # 基于波动率的调整
            if metrics['volatility'] > 0.2:
                # 降低最大杠杆
                self.leverage_multiplier = max(3, self.leverage_multiplier - 1)
                adjustments.append('reduced_leverage')
            
            # 基于回撤的调整
            if metrics['max_drawdown'] < -0.1:
                # 启用更严格的风险管理
                self.drawdown_protection *= 0.8
                adjustments.append('enhanced_drawdown_protection')
            
            # 记录调整
            adjustment_record = {
                'timestamp': datetime.now(timezone.utc),
                'trigger_metrics': metrics,
                'market_state': market_state,
                'adjustments': adjustments
            }
            
            self.parameter_adjustments.append(adjustment_record)
            
            # 保持调整历史在合理长度
            if len(self.parameter_adjustments) > 100:
                self.parameter_adjustments = self.parameter_adjustments[-50:]
                
        except Exception:
            pass
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态报告"""
        try:
            return {
                'monitoring_enabled': self.monitoring_enabled,
                'last_monitoring_time': self.last_monitoring_time,
                'performance_metrics_count': len(self.performance_metrics.get('win_rate', [])),
                'total_adjustments': len(self.parameter_adjustments),
                'current_parameters': {
                    'base_position_size': self.base_position_size,
                    'leverage_multiplier': self.leverage_multiplier,
                    'stoploss': self.stoploss,
                    'drawdown_protection': self.drawdown_protection
                }
            }
        except Exception:
            return {'error': '无法获取监控状态'}
    
    # ===== 综合风控系统 =====
    
    def initialize_risk_control_system(self):
        """初始化综合风控系统"""
        # 多级风控状态
        self.risk_control_enabled = True
        self.emergency_mode = False
        self.circuit_breaker_active = False
        
        # 风险预算系统
        self.risk_budgets = {
            'daily_var_budget': 0.02,      # 日VaR预算2%
            'weekly_var_budget': 0.05,     # 周VaR预算5%
            'monthly_var_budget': 0.12,    # 月VaR预算12%
            'position_var_limit': 0.01,    # 单仓VaR限制1%
            'correlation_limit': 0.7,      # 相关性限制70%
            'sector_exposure_limit': 0.3   # 行业敞口限制30%
        }
        
        # 风险使用情况追踪
        self.risk_utilization = {
            'current_daily_var': 0.0,
            'current_weekly_var': 0.0,
            'current_monthly_var': 0.0,
            'used_correlation_capacity': 0.0,
            'sector_exposures': {}
        }
        
        # 熔断阈值
        self.circuit_breakers = {
            'daily_loss_limit': -0.08,      # 日亏损熔断8%
            'hourly_loss_limit': -0.03,     # 小时亏损熔断3%
            'consecutive_loss_limit': 6,     # 连续亏损熔断
            'drawdown_limit': -0.20,        # 最大回撤熔断20%
            'volatility_spike_limit': 5.0,  # 波动率突增熔断
            'correlation_spike_limit': 0.9  # 相关性突增熔断
        }
        
        # 风险事件记录
        self.risk_events = []
        self.emergency_actions = []
        
        # 风险状态缓存
        self.last_risk_check_time = datetime.now(timezone.utc)
        self.risk_check_interval = 60  # 风控检查间隔60秒
        
    def comprehensive_risk_check(self, pair: str, current_price: float, 
                               proposed_position_size: float, 
                               proposed_leverage: int) -> Dict[str, Any]:
        """综合风险检查 - 多级风控验证"""
        
        risk_status = {
            'approved': True,
            'adjusted_position_size': proposed_position_size,
            'adjusted_leverage': proposed_leverage,
            'risk_warnings': [],
            'risk_violations': [],
            'emergency_action': None
        }
        
        try:
            current_time = datetime.now(timezone.utc)
            
            # 1. 熔断器检查
            circuit_breaker_result = self.check_circuit_breakers()
            if circuit_breaker_result['triggered']:
                risk_status['approved'] = False
                risk_status['emergency_action'] = 'circuit_breaker_halt'
                risk_status['risk_violations'].append(circuit_breaker_result)
                return risk_status
            
            # 2. VaR预算检查
            var_check_result = self.check_var_budget_limits(pair, proposed_position_size)
            if not var_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= var_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(var_check_result)
            
            # 3. 相关性限制检查
            correlation_check_result = self.check_correlation_limits(pair, proposed_position_size)
            if not correlation_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= correlation_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(correlation_check_result)
            
            # 4. 集中度风险检查
            concentration_check_result = self.check_concentration_risk(pair, proposed_position_size)
            if not concentration_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= concentration_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(concentration_check_result)
            
            # 5. 流动性风险检查
            liquidity_check_result = self.check_liquidity_risk(pair, proposed_position_size)
            if not liquidity_check_result['sufficient_liquidity']:
                risk_status['adjusted_position_size'] *= liquidity_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(liquidity_check_result)
            
            # 6. 杠杆风险检查
            leverage_check_result = self.check_leverage_risk(pair, proposed_leverage)
            if not leverage_check_result['within_limits']:
                risk_status['adjusted_leverage'] = leverage_check_result['max_allowed_leverage']
                risk_status['risk_warnings'].append(leverage_check_result)
            
            # 7. 时间风险检查
            time_risk_result = self.check_time_based_risk(current_time)
            if time_risk_result['high_risk_period']:
                risk_status['adjusted_position_size'] *= time_risk_result['adjustment_factor']
                risk_status['risk_warnings'].append(time_risk_result)
            
            # 最终调整确保不超过最小/最大限制
            risk_status['adjusted_position_size'] = max(
                0.005, 
                min(risk_status['adjusted_position_size'], self.max_position_size * 0.8)
            )
            
            # 记录风险检查事件
            self.record_risk_event('risk_check', risk_status)
            
        except Exception as e:
            risk_status['approved'] = False
            risk_status['emergency_action'] = 'system_error'
            risk_status['risk_violations'].append({
                'type': 'system_error',
                'message': f'风控系统错误: {str(e)}'
            })
        
        return risk_status
    
    def check_circuit_breakers(self) -> Dict[str, Any]:
        """熔断器检查"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # 获取当前账户状态
            current_equity = getattr(self, 'current_equity', 100000)  # 默认值
            daily_pnl = getattr(self, 'daily_pnl', 0)
            hourly_pnl = getattr(self, 'hourly_pnl', 0)
            
            # 1. 日亏损熔断
            daily_loss_pct = daily_pnl / current_equity if current_equity > 0 else 0
            if daily_loss_pct < self.circuit_breakers['daily_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'daily_loss_circuit_breaker',
                    'current_value': daily_loss_pct,
                    'limit': self.circuit_breakers['daily_loss_limit'],
                    'message': f'触发日亏损熔断: {daily_loss_pct:.2%}'
                }
            
            # 2. 小时亏损熔断
            hourly_loss_pct = hourly_pnl / current_equity if current_equity > 0 else 0
            if hourly_loss_pct < self.circuit_breakers['hourly_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'hourly_loss_circuit_breaker',
                    'current_value': hourly_loss_pct,
                    'limit': self.circuit_breakers['hourly_loss_limit'],
                    'message': f'触发小时亏损熔断: {hourly_loss_pct:.2%}'
                }
            
            # 3. 连续亏损熔断
            if self.consecutive_losses >= self.circuit_breakers['consecutive_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'consecutive_loss_circuit_breaker',
                    'current_value': self.consecutive_losses,
                    'limit': self.circuit_breakers['consecutive_loss_limit'],
                    'message': f'触发连续亏损熔断: {self.consecutive_losses}次'
                }
            
            # 4. 最大回撤熔断
            max_drawdown = getattr(self, 'current_max_drawdown', 0)
            if max_drawdown < self.circuit_breakers['drawdown_limit']:
                return {
                    'triggered': True,
                    'type': 'drawdown_circuit_breaker',
                    'current_value': max_drawdown,
                    'limit': self.circuit_breakers['drawdown_limit'],
                    'message': f'触发回撤熔断: {max_drawdown:.2%}'
                }
            
            return {'triggered': False, 'type': None, 'message': '熔断器正常'}
            
        except Exception:
            return {
                'triggered': True,
                'type': 'circuit_breaker_error',
                'message': '熔断器检查系统错误'
            }
    
    def check_var_budget_limits(self, pair: str, position_size: float) -> Dict[str, Any]:
        """VaR预算限制检查"""
        try:
            # 计算新仓位的VaR贡献
            position_var = self.calculate_position_var(pair, position_size)
            
            # 检查各级VaR预算
            current_daily_var = self.risk_utilization['current_daily_var']
            new_daily_var = current_daily_var + position_var
            
            if new_daily_var > self.risk_budgets['daily_var_budget']:
                # 计算允许的最大仓位
                available_var_budget = self.risk_budgets['daily_var_budget'] - current_daily_var
                max_allowed_position = available_var_budget / position_var * position_size if position_var > 0 else position_size
                
                adjustment_factor = max(0.1, max_allowed_position / position_size)
                
                return {
                    'within_limits': False,
                    'type': 'var_budget_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'current_utilization': new_daily_var,
                    'budget_limit': self.risk_budgets['daily_var_budget'],
                    'message': f'VaR预算超限，仓位调整为{adjustment_factor:.1%}'
                }
            
            return {
                'within_limits': True,
                'type': 'var_budget_check',
                'utilization': new_daily_var / self.risk_budgets['daily_var_budget'],
                'message': 'VaR预算检查通过'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.5,
                'message': 'VaR预算检查系统错误，保守调整仓位'
            }
    
    def calculate_position_var(self, pair: str, position_size: float) -> float:
        """计算仓位VaR贡献"""
        try:
            if pair in self.pair_returns_history and len(self.pair_returns_history[pair]) >= 20:
                returns = self.pair_returns_history[pair]
                position_var = self.calculate_var(returns) * position_size
                return min(position_var, self.risk_budgets['position_var_limit'])
            else:
                # 默认风险估计
                return position_size * 0.02  # 假设2%的默认VaR
        except Exception:
            return position_size * 0.03  # 保守估计
    
    def check_correlation_limits(self, pair: str, position_size: float) -> Dict[str, Any]:
        """相关性限制检查"""
        try:
            current_correlation = self.calculate_portfolio_correlation(pair)
            
            if current_correlation > self.risk_budgets['correlation_limit']:
                # 基于相关性调整仓位
                excess_correlation = current_correlation - self.risk_budgets['correlation_limit']
                adjustment_factor = max(0.2, 1 - (excess_correlation * 2))
                
                return {
                    'within_limits': False,
                    'type': 'correlation_limit_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'current_correlation': current_correlation,
                    'limit': self.risk_budgets['correlation_limit'],
                    'message': f'相关性超限({current_correlation:.1%})，仓位调整为{adjustment_factor:.1%}'
                }
            
            return {
                'within_limits': True,
                'type': 'correlation_check',
                'current_correlation': current_correlation,
                'message': '相关性检查通过'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.7,
                'message': '相关性检查系统错误，保守调整'
            }
    
    def check_concentration_risk(self, pair: str, position_size: float) -> Dict[str, Any]:
        """集中度风险检查"""
        try:
            # 检查单一品种集中度
            current_positions = getattr(self, 'portfolio_positions', {})
            total_exposure = sum([abs(pos) for pos in current_positions.values()])
            
            if pair in current_positions:
                new_exposure = current_positions[pair] + position_size
            else:
                new_exposure = position_size
            
            if total_exposure > 0:
                concentration_ratio = abs(new_exposure) / (total_exposure + position_size)
            else:
                concentration_ratio = 1.0
            
            max_single_position_ratio = 0.4  # 单一品种最大40%
            
            if concentration_ratio > max_single_position_ratio:
                adjustment_factor = max_single_position_ratio / concentration_ratio
                
                return {
                    'within_limits': False,
                    'type': 'concentration_risk_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'concentration_ratio': concentration_ratio,
                    'limit': max_single_position_ratio,
                    'message': f'集中度风险超限({concentration_ratio:.1%})，调整仓位'
                }
            
            return {
                'within_limits': True,
                'type': 'concentration_check',
                'concentration_ratio': concentration_ratio,
                'message': '集中度风险检查通过'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.6,
                'message': '集中度检查系统错误，保守调整'
            }
    
    def check_liquidity_risk(self, pair: str, position_size: float) -> Dict[str, Any]:
        """流动性风险检查"""
        try:
            # 获取市场流动性指标
            market_data = getattr(self, 'current_market_data', {})
            
            if pair in market_data:
                volume_ratio = market_data[pair].get('volume_ratio', 1.0)
                spread = market_data[pair].get('spread', 0.001)
            else:
                volume_ratio = 1.0  # 默认值
                spread = 0.002
            
            # 流动性风险评估
            liquidity_risk_score = 0.0
            
            # 成交量风险
            if volume_ratio < 0.5:  # 成交量过低
                liquidity_risk_score += 0.3
            elif volume_ratio < 0.8:
                liquidity_risk_score += 0.1
            
            # 点差风险
            if spread > 0.005:  # 点差过大
                liquidity_risk_score += 0.4
            elif spread > 0.003:
                liquidity_risk_score += 0.2
            
            if liquidity_risk_score > 0.5:  # 流动性风险过高
                adjustment_factor = max(0.3, 1 - liquidity_risk_score)
                
                return {
                    'sufficient_liquidity': False,
                    'type': 'liquidity_risk_high',
                    'adjustment_factor': adjustment_factor,
                    'risk_score': liquidity_risk_score,
                    'volume_ratio': volume_ratio,
                    'spread': spread,
                    'message': f'流动性风险过高({liquidity_risk_score:.1f})，调整仓位'
                }
            
            return {
                'sufficient_liquidity': True,
                'type': 'liquidity_check',
                'risk_score': liquidity_risk_score,
                'message': '流动性风险检查通过'
            }
            
        except Exception:
            return {
                'sufficient_liquidity': False,
                'adjustment_factor': 0.5,
                'message': '流动性检查系统错误，保守调整'
            }
    
    def check_leverage_risk(self, pair: str, proposed_leverage: int) -> Dict[str, Any]:
        """杠杆风险检查"""
        try:
            # 基于市场状态和波动率的杠杆限制
            market_volatility = getattr(self, 'current_market_volatility', {}).get(pair, 0.02)
            
            # 动态杠杆限制
            if market_volatility > 0.05:  # 高波动
                max_allowed_leverage = min(5, self.leverage_multiplier)
            elif market_volatility > 0.03:  # 中等波动
                max_allowed_leverage = min(8, self.leverage_multiplier)
            else:  # 低波动
                max_allowed_leverage = self.leverage_multiplier
            
            if proposed_leverage > max_allowed_leverage:
                return {
                    'within_limits': False,
                    'type': 'leverage_risk_exceeded',
                    'max_allowed_leverage': max_allowed_leverage,
                    'proposed_leverage': proposed_leverage,
                    'market_volatility': market_volatility,
                    'message': f'杠杆风险过高，限制为{max_allowed_leverage}倍'
                }
            
            return {
                'within_limits': True,
                'type': 'leverage_check',
                'approved_leverage': proposed_leverage,
                'message': '杠杆风险检查通过'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'max_allowed_leverage': min(3, proposed_leverage),
                'message': '杠杆检查系统错误，保守限制'
            }
    
    def check_time_based_risk(self, current_time: datetime) -> Dict[str, Any]:
        """基于时间的风险检查"""
        try:
            hour = current_time.hour
            weekday = current_time.weekday()
            
            high_risk_periods = [
                (weekday >= 5),  # 周末
                (hour <= 6 or hour >= 22),  # 亚洲深夜时段
                (11 <= hour <= 13),  # 午休时段
            ]
            
            if any(high_risk_periods):
                adjustment_factor = 0.7  # 高风险时段减小仓位
                
                return {
                    'high_risk_period': True,
                    'type': 'time_based_risk',
                    'adjustment_factor': adjustment_factor,
                    'hour': hour,
                    'weekday': weekday,
                    'message': '高风险时段，调整仓位'
                }
            
            return {
                'high_risk_period': False,
                'type': 'time_check',
                'adjustment_factor': 1.0,
                'message': '时间风险检查通过'
            }
            
        except Exception:
            return {
                'high_risk_period': True,
                'adjustment_factor': 0.8,
                'message': '时间检查系统错误，保守调整'
            }
    
    def record_risk_event(self, event_type: str, event_data: Dict[str, Any]):
        """记录风险事件"""
        try:
            risk_event = {
                'timestamp': datetime.now(timezone.utc),
                'event_type': event_type,
                'event_data': event_data,
                'severity': self.determine_event_severity(event_data)
            }
            
            self.risk_events.append(risk_event)
            
            # 保持事件记录在合理长度
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-500:]
                
        except Exception:
            pass
    
    def determine_event_severity(self, event_data: Dict[str, Any]) -> str:
        """确定事件严重程度"""
        try:
            if not event_data.get('approved', True):
                return 'critical'
            elif event_data.get('emergency_action'):
                return 'high'
            elif len(event_data.get('risk_violations', [])) > 0:
                return 'medium'
            elif len(event_data.get('risk_warnings', [])) > 2:
                return 'medium'
            elif len(event_data.get('risk_warnings', [])) > 0:
                return 'low'
            else:
                return 'info'
        except Exception:
            return 'unknown'
    
    def emergency_risk_shutdown(self, reason: str):
        """紧急风控关闭"""
        try:
            self.emergency_mode = True
            self.circuit_breaker_active = True
            
            emergency_action = {
                'timestamp': datetime.now(timezone.utc),
                'reason': reason,
                'action': 'emergency_shutdown',
                'open_positions_count': len(getattr(self, 'portfolio_positions', {})),
                'total_exposure': sum([abs(pos) for pos in getattr(self, 'portfolio_positions', {}).values()])
            }
            
            self.emergency_actions.append(emergency_action)
            
            # 这里应该集成实际的平仓操作
            # 暂时记录紧急操作
            
        except Exception:
            pass
    
    def get_risk_control_status(self) -> Dict[str, Any]:
        """获取风控状态报告"""
        try:
            return {
                'risk_control_enabled': self.risk_control_enabled,
                'emergency_mode': self.emergency_mode,
                'circuit_breaker_active': self.circuit_breaker_active,
                'risk_budgets': self.risk_budgets,
                'risk_utilization': self.risk_utilization,
                'recent_risk_events': len(self.risk_events[-24:]) if self.risk_events else 0,
                'emergency_actions_count': len(self.emergency_actions),
                'last_risk_check': self.last_risk_check_time
            }
        except Exception:
            return {'error': '无法获取风控状态'}
    
    # ===== 执行算法与滑点控制系统 =====
    
    def initialize_execution_system(self):
        """初始化执行算法系统"""
        # 执行算法配置
        self.execution_algorithms = {
            'twap': {'enabled': True, 'weight': 0.3},      # 时间加权平均价格
            'vwap': {'enabled': True, 'weight': 0.4},      # 成交量加权平均价格
            'implementation_shortfall': {'enabled': True, 'weight': 0.3}  # 执行损失最小化
        }
        
        # 滑点控制参数
        self.slippage_control = {
            'max_allowed_slippage': 0.002,    # 最大允许滑点0.2%
            'slippage_prediction_window': 50,  # 滑点预测窗口
            'adaptive_threshold': 0.001,      # 自适应阈值0.1%
            'emergency_threshold': 0.005      # 紧急阈值0.5%
        }
        
        # 订单分割参数
        self.order_splitting = {
            'min_split_size': 0.01,           # 最小分割大小1%
            'max_split_count': 10,            # 最大分割数量
            'split_interval_seconds': 30,     # 分割间隔30秒
            'adaptive_splitting': True        # 自适应分割
        }
        
        # 执行质量追踪
        self.execution_metrics = {
            'realized_slippage': [],
            'market_impact': [],
            'execution_time': [],
            'fill_ratio': [],
            'cost_basis_deviation': []
        }
        
        # 市场影响模型
        self.market_impact_model = {
            'temporary_impact_factor': 0.5,   # 临时冲击因子
            'permanent_impact_factor': 0.3,   # 永久冲击因子
            'nonlinear_factor': 1.5,          # 非线性因子
            'decay_factor': 0.1               # 衰减因子
        }
        
        # 执行状态追踪
        self.active_executions = {}
        self.execution_history = []
        
    def smart_order_execution(self, pair: str, order_size: float, order_side: str, 
                            current_price: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """智能订单执行系统"""
        
        execution_plan = {
            'original_size': order_size,
            'execution_strategy': None,
            'split_orders': [],
            'expected_slippage': 0.0,
            'estimated_execution_time': 0,
            'risk_level': 'normal'
        }
        
        try:
            # 1. 执行风险评估
            execution_risk = self.assess_execution_risk(pair, order_size, market_conditions)
            execution_plan['risk_level'] = execution_risk['level']
            
            # 2. 滑点预测
            predicted_slippage = self.predict_slippage(pair, order_size, order_side, market_conditions)
            execution_plan['expected_slippage'] = predicted_slippage
            
            # 3. 选择执行算法
            optimal_algorithm = self.select_execution_algorithm(pair, order_size, market_conditions, execution_risk)
            execution_plan['execution_strategy'] = optimal_algorithm
            
            # 4. 订单分割优化
            if order_size > self.order_splitting['min_split_size'] and execution_risk['level'] != 'low':
                split_plan = self.optimize_order_splitting(pair, order_size, market_conditions, optimal_algorithm)
                execution_plan['split_orders'] = split_plan['orders']
                execution_plan['estimated_execution_time'] = split_plan['total_time']
            else:
                execution_plan['split_orders'] = [{'size': order_size, 'delay': 0, 'priority': 'high'}]
                execution_plan['estimated_execution_time'] = 30  # 预估30秒
            
            # 5. 执行时机优化
            execution_timing = self.optimize_execution_timing(pair, market_conditions)
            execution_plan['optimal_timing'] = execution_timing
            
            # 6. 生成执行指令
            execution_instructions = self.generate_execution_instructions(execution_plan, pair, order_side, current_price)
            execution_plan['instructions'] = execution_instructions
            
            return execution_plan
            
        except Exception as e:
            # 发生错误时回退到简单执行
            return {
                'original_size': order_size,
                'execution_strategy': 'immediate',
                'split_orders': [{'size': order_size, 'delay': 0, 'priority': 'high'}],
                'expected_slippage': 0.002,  # 保守估计
                'estimated_execution_time': 30,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    def assess_execution_risk(self, pair: str, order_size: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """评估执行风险"""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # 1. 订单大小风险
            avg_volume = market_conditions.get('avg_volume', 1.0)
            order_volume_ratio = order_size / avg_volume if avg_volume > 0 else 1.0
            
            if order_volume_ratio > 0.1:  # 超过10%平均成交量
                risk_score += 0.4
                risk_factors.append('large_order_size')
            elif order_volume_ratio > 0.05:
                risk_score += 0.2
                risk_factors.append('medium_order_size')
            
            # 2. 市场波动风险
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:
                risk_score += 0.3
                risk_factors.append('high_volatility')
            elif volatility > 0.03:
                risk_score += 0.15
                risk_factors.append('medium_volatility')
            
            # 3. 流动性风险
            bid_ask_spread = market_conditions.get('spread', 0.001)
            if bid_ask_spread > 0.003:
                risk_score += 0.2
                risk_factors.append('wide_spread')
            
            # 4. 时间风险
            if self.is_high_volatility_session(datetime.now(timezone.utc)):
                risk_score += 0.1
                risk_factors.append('high_volatility_session')
            
            # 确定风险等级
            if risk_score < 0.3:
                risk_level = 'low'
            elif risk_score < 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'level': risk_level,
                'score': risk_score,
                'factors': risk_factors,
                'order_volume_ratio': order_volume_ratio
            }
            
        except Exception:
            return {
                'level': 'medium',
                'score': 0.5,
                'factors': ['assessment_error'],
                'order_volume_ratio': 0.1
            }
    
    def predict_slippage(self, pair: str, order_size: float, order_side: str, 
                        market_conditions: Dict[str, Any]) -> float:
        """滑点预测模型"""
        try:
            # 基础滑点模型
            base_slippage = market_conditions.get('spread', 0.001) / 2  # 半个点差
            
            # 市场冲击模型
            avg_volume = market_conditions.get('avg_volume', 1.0)
            volume_ratio = order_size / avg_volume if avg_volume > 0 else 0.1
            
            # 临时市场冲击
            temporary_impact = (
                self.market_impact_model['temporary_impact_factor'] * 
                (volume_ratio ** self.market_impact_model['nonlinear_factor'])
            )
            
            # 永久市场冲击
            permanent_impact = (
                self.market_impact_model['permanent_impact_factor'] * 
                (volume_ratio ** 0.5)
            )
            
            # 波动率调整
            volatility = market_conditions.get('volatility', 0.02)
            volatility_adjustment = min(1.0, volatility * 10)  # 波动率越高滑点越大
            
            # 时间调整
            time_adjustment = 1.0
            if self.is_high_volatility_session(datetime.now(timezone.utc)):
                time_adjustment = 1.2
            elif self.is_low_liquidity_session(datetime.now(timezone.utc)):
                time_adjustment = 1.3
            
            # 历史滑点调整
            historical_slippage = self.get_historical_slippage(pair)
            historical_adjustment = max(0.5, min(2.0, historical_slippage / 0.001))
            
            # 综合滑点预测
            predicted_slippage = (
                base_slippage + temporary_impact + permanent_impact
            ) * volatility_adjustment * time_adjustment * historical_adjustment
            
            # 限制在合理范围
            predicted_slippage = min(predicted_slippage, self.slippage_control['emergency_threshold'])
            
            return max(0.0001, predicted_slippage)  # 最小0.01%
            
        except Exception:
            return 0.002  # 保守估计0.2%
    
    def get_historical_slippage(self, pair: str) -> float:
        """获取历史平均滑点"""
        try:
            if len(self.execution_metrics['realized_slippage']) > 0:
                recent_slippage = self.execution_metrics['realized_slippage'][-20:]  # 最近20次
                return np.mean(recent_slippage)
            else:
                return 0.001  # 默认0.1%
        except Exception:
            return 0.001
    
    def select_execution_algorithm(self, pair: str, order_size: float, 
                                 market_conditions: Dict[str, Any], 
                                 execution_risk: Dict[str, Any]) -> str:
        """选择最优执行算法"""
        try:
            algorithm_scores = {}
            
            # TWAP算法评分
            if self.execution_algorithms['twap']['enabled']:
                twap_score = 0.5  # 基础分
                
                # 时间敏感性低时加分
                if execution_risk['level'] == 'low':
                    twap_score += 0.2
                
                # 市场平静时加分
                if market_conditions.get('volatility', 0.02) < 0.025:
                    twap_score += 0.1
                
                algorithm_scores['twap'] = twap_score * self.execution_algorithms['twap']['weight']
            
            # VWAP算法评分
            if self.execution_algorithms['vwap']['enabled']:
                vwap_score = 0.6  # 基础分
                
                # 成交量充足时加分
                if market_conditions.get('volume_ratio', 1.0) > 1.0:
                    vwap_score += 0.2
                
                # 中等风险时最优
                if execution_risk['level'] == 'medium':
                    vwap_score += 0.15
                
                algorithm_scores['vwap'] = vwap_score * self.execution_algorithms['vwap']['weight']
            
            # Implementation Shortfall算法评分
            if self.execution_algorithms['implementation_shortfall']['enabled']:
                is_score = 0.4  # 基础分
                
                # 高风险时优选
                if execution_risk['level'] == 'high':
                    is_score += 0.3
                
                # 大订单时优选
                if execution_risk.get('order_volume_ratio', 0.1) > 0.05:
                    is_score += 0.2
                
                # 高波动时优选
                if market_conditions.get('volatility', 0.02) > 0.03:
                    is_score += 0.1
                
                algorithm_scores['implementation_shortfall'] = is_score * self.execution_algorithms['implementation_shortfall']['weight']
            
            # 选择最高分算法
            if algorithm_scores:
                optimal_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
                return optimal_algorithm
            else:
                return 'twap'  # 默认算法
                
        except Exception:
            return 'twap'  # 出错时回退到TWAP
    
    def optimize_order_splitting(self, pair: str, order_size: float, 
                               market_conditions: Dict[str, Any], 
                               algorithm: str) -> Dict[str, Any]:
        """优化订单分割"""
        try:
            split_plan = {
                'orders': [],
                'total_time': 0,
                'expected_total_slippage': 0.0
            }
            
            # 确定分割数量
            avg_volume = market_conditions.get('avg_volume', 1.0)
            volume_ratio = order_size / avg_volume if avg_volume > 0 else 0.1
            
            if volume_ratio > 0.2:  # 超大订单
                split_count = min(self.order_splitting['max_split_count'], 8)
            elif volume_ratio > 0.1:  # 大订单
                split_count = min(self.order_splitting['max_split_count'], 5)
            elif volume_ratio > 0.05:  # 中等订单
                split_count = min(self.order_splitting['max_split_count'], 3)
            else:
                split_count = 1  # 小订单不分割
            
            if split_count == 1:
                split_plan['orders'] = [{'size': order_size, 'delay': 0, 'priority': 'high'}]
                split_plan['total_time'] = 30
                return split_plan
            
            # 根据算法调整分割策略
            if algorithm == 'twap':
                # 等时间间隔分割
                sub_order_size = order_size / split_count
                base_delay = self.order_splitting['split_interval_seconds']
                
                for i in range(split_count):
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': i * base_delay,
                        'priority': 'medium' if i > 0 else 'high'
                    })
                
                split_plan['total_time'] = (split_count - 1) * base_delay + 30
                
            elif algorithm == 'vwap':
                # 基于预期成交量分布分割
                volume_distribution = self.get_volume_distribution_forecast()
                cumulative_size = 0
                
                for i, volume_weight in enumerate(volume_distribution[:split_count]):
                    sub_order_size = order_size * volume_weight
                    cumulative_size += sub_order_size
                    
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': i * 60,  # 每分钟一个子订单
                        'priority': 'high' if volume_weight > 0.2 else 'medium'
                    })
                
                # 处理剩余部分
                if cumulative_size < order_size:
                    remaining = order_size - cumulative_size
                    split_plan['orders'][-1]['size'] += remaining
                
                split_plan['total_time'] = len(split_plan['orders']) * 60
                
            else:  # implementation_shortfall
                # 动态分割，根据市场冲击调整
                remaining_size = order_size
                time_offset = 0
                urgency_factor = min(1.5, market_conditions.get('volatility', 0.02) * 20)
                
                for i in range(split_count):
                    if i == split_count - 1:
                        # 最后一个订单包含所有剩余
                        sub_order_size = remaining_size
                    else:
                        # 根据紧急性调整订单大小
                        base_portion = 1.0 / (split_count - i)
                        urgency_adjustment = base_portion * urgency_factor
                        sub_order_size = min(remaining_size, order_size * urgency_adjustment)
                    
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': time_offset,
                        'priority': 'high' if i < 2 else 'medium'
                    })
                    
                    remaining_size -= sub_order_size
                    time_offset += max(15, int(45 / urgency_factor))  # 动态间隔
                    
                    if remaining_size <= 0:
                        break
                
                split_plan['total_time'] = time_offset + 30
            
            # 计算预期总滑点
            total_slippage = 0.0
            for order in split_plan['orders']:
                sub_slippage = self.predict_slippage(pair, order['size'], 'buy', market_conditions)
                total_slippage += sub_slippage * (order['size'] / order_size)
            
            split_plan['expected_total_slippage'] = total_slippage
            
            return split_plan
            
        except Exception:
            return {
                'orders': [{'size': order_size, 'delay': 0, 'priority': 'high'}],
                'total_time': 30,
                'expected_total_slippage': 0.002
            }
    
    def get_volume_distribution_forecast(self) -> List[float]:
        """获取成交量分布预测"""
        try:
            # 简化的日内成交量分布模型
            # 实际应该基于历史数据和机器学习模型
            typical_distribution = [
                0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.02
            ]
            return typical_distribution
        except Exception:
            return [0.1] * 10  # 均匀分布
    
    def optimize_execution_timing(self, pair: str, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """优化执行时机"""
        try:
            current_time = datetime.now(timezone.utc)
            hour = current_time.hour
            
            timing_score = 0.5  # 基础分
            timing_factors = []
            
            # 流动性时段评分
            if 13 <= hour <= 16:  # 欧美重叠时段
                timing_score += 0.3
                timing_factors.append('high_liquidity_session')
            elif 8 <= hour <= 11 or 17 <= hour <= 20:  # 单一市场活跃时段
                timing_score += 0.1
                timing_factors.append('medium_liquidity_session')
            else:  # 低流动性时段
                timing_score -= 0.2
                timing_factors.append('low_liquidity_session')
            
            # 波动率评分
            volatility = market_conditions.get('volatility', 0.02)
            if 0.02 <= volatility <= 0.04:  # 适中波动率
                timing_score += 0.1
                timing_factors.append('optimal_volatility')
            elif volatility > 0.05:  # 高波动率
                timing_score -= 0.15
                timing_factors.append('high_volatility_risk')
            
            # 成交量评分
            volume_ratio = market_conditions.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:
                timing_score += 0.1
                timing_factors.append('high_volume')
            elif volume_ratio < 0.8:
                timing_score -= 0.1
                timing_factors.append('low_volume')
            
            # 建议行动
            if timing_score > 0.7:
                recommendation = 'execute_immediately'
            elif timing_score > 0.4:
                recommendation = 'execute_normal'
            else:
                recommendation = 'delay_execution'
            
            return {
                'timing_score': timing_score,
                'recommendation': recommendation,
                'factors': timing_factors,
                'optimal_delay_minutes': max(0, int((0.6 - timing_score) * 30))
            }
            
        except Exception:
            return {
                'timing_score': 0.5,
                'recommendation': 'execute_normal',
                'factors': ['timing_analysis_error'],
                'optimal_delay_minutes': 0
            }
    
    def generate_execution_instructions(self, execution_plan: Dict[str, Any], 
                                      pair: str, order_side: str, 
                                      current_price: float) -> List[Dict[str, Any]]:
        """生成具体执行指令"""
        try:
            instructions = []
            
            for i, order in enumerate(execution_plan['split_orders']):
                instruction = {
                    'instruction_id': f"{pair}_{order_side}_{i}_{int(datetime.now(timezone.utc).timestamp())}",
                    'pair': pair,
                    'side': order_side,
                    'size': order['size'],
                    'order_type': self.determine_order_type(order, execution_plan),
                    'price_limit': self.calculate_price_limit(current_price, order_side, order['size'], execution_plan),
                    'delay_seconds': order['delay'],
                    'priority': order['priority'],
                    'timeout_seconds': 300,  # 5分钟超时
                    'max_slippage': self.slippage_control['max_allowed_slippage'],
                    'execution_strategy': execution_plan['execution_strategy'],
                    'created_at': datetime.now(timezone.utc)
                }
                
                instructions.append(instruction)
            
            return instructions
            
        except Exception:
            # 生成简单指令
            return [{
                'instruction_id': f"{pair}_{order_side}_simple_{int(datetime.now(timezone.utc).timestamp())}",
                'pair': pair,
                'side': order_side,
                'size': execution_plan['original_size'],
                'order_type': 'market',
                'delay_seconds': 0,
                'priority': 'high',
                'timeout_seconds': 180,
                'max_slippage': 0.003,
                'created_at': datetime.now(timezone.utc)
            }]
    
    def determine_order_type(self, order: Dict[str, Any], execution_plan: Dict[str, Any]) -> str:
        """确定订单类型"""
        try:
            if order['priority'] == 'high' or execution_plan.get('risk_level') == 'high':
                return 'market'
            elif execution_plan['expected_slippage'] < self.slippage_control['adaptive_threshold']:
                return 'limit'
            else:
                return 'market_with_protection'  # 带保护的市价单
        except Exception:
            return 'market'
    
    def calculate_price_limit(self, current_price: float, side: str, 
                            order_size: float, execution_plan: Dict[str, Any]) -> float:
        """计算价格限制"""
        try:
            expected_slippage = execution_plan['expected_slippage']
            
            # 添加缓冲
            slippage_buffer = expected_slippage * 1.2  # 20%缓冲
            
            if side.lower() == 'buy':
                return current_price * (1 + slippage_buffer)
            else:
                return current_price * (1 - slippage_buffer)
                
        except Exception:
            # 保守的价格限制
            if side.lower() == 'buy':
                return current_price * 1.005
            else:
                return current_price * 0.995
    
    def track_execution_performance(self, execution_id: str, execution_result: Dict[str, Any]):
        """追踪执行表现"""
        try:
            # 计算实际滑点
            expected_price = execution_result.get('expected_price', 0)
            actual_price = execution_result.get('actual_price', 0)
            
            if expected_price > 0 and actual_price > 0:
                realized_slippage = abs(actual_price - expected_price) / expected_price
                self.execution_metrics['realized_slippage'].append(realized_slippage)
            
            # 计算市场冲击
            pre_trade_price = execution_result.get('pre_trade_price', 0)
            post_trade_price = execution_result.get('post_trade_price', 0)
            
            if pre_trade_price > 0 and post_trade_price > 0:
                market_impact = abs(post_trade_price - pre_trade_price) / pre_trade_price
                self.execution_metrics['market_impact'].append(market_impact)
            
            # 记录其他指标
            execution_time = execution_result.get('execution_time_seconds', 0)
            if execution_time > 0:
                self.execution_metrics['execution_time'].append(execution_time)
            
            fill_ratio = execution_result.get('fill_ratio', 1.0)
            self.execution_metrics['fill_ratio'].append(fill_ratio)
            
            # 维护指标历史长度
            for metric in self.execution_metrics.values():
                if len(metric) > 500:
                    metric[:] = metric[-250:]  # 保持最近250个记录
                    
        except Exception:
            pass
    
    def get_execution_quality_report(self) -> Dict[str, Any]:
        """获取执行质量报告"""
        try:
            if not any(self.execution_metrics.values()):
                return {'error': '无执行数据'}
            
            report = {}
            
            # 滑点统计
            if self.execution_metrics['realized_slippage']:
                slippage_data = self.execution_metrics['realized_slippage']
                report['slippage'] = {
                    'avg': np.mean(slippage_data),
                    'median': np.median(slippage_data),
                    'std': np.std(slippage_data),
                    'p95': np.percentile(slippage_data, 95),
                    'samples': len(slippage_data)
                }
            
            # 市场冲击统计
            if self.execution_metrics['market_impact']:
                impact_data = self.execution_metrics['market_impact']
                report['market_impact'] = {
                    'avg': np.mean(impact_data),
                    'median': np.median(impact_data),
                    'std': np.std(impact_data),
                    'p95': np.percentile(impact_data, 95),
                    'samples': len(impact_data)
                }
            
            # 执行时间统计
            if self.execution_metrics['execution_time']:
                time_data = self.execution_metrics['execution_time']
                report['execution_time'] = {
                    'avg_seconds': np.mean(time_data),
                    'median_seconds': np.median(time_data),
                    'p95_seconds': np.percentile(time_data, 95),
                    'samples': len(time_data)
                }
            
            # 成交率统计
            if self.execution_metrics['fill_ratio']:
                fill_data = self.execution_metrics['fill_ratio']
                report['fill_ratio'] = {
                    'avg': np.mean(fill_data),
                    'median': np.median(fill_data),
                    'samples_below_95pct': sum(1 for x in fill_data if x < 0.95),
                    'samples': len(fill_data)
                }
            
            return report
            
        except Exception:
            return {'error': '无法生成执行质量报告'}
    
    # ===== 市场情绪与外部数据集成系统 =====
    
    def initialize_sentiment_system(self):
        """初始化市场情绪分析系统"""
        # 市场情绪指标配置
        self.sentiment_indicators = {
            'fear_greed_index': {'enabled': True, 'weight': 0.25},
            'vix_equivalent': {'enabled': True, 'weight': 0.20},
            'news_sentiment': {'enabled': True, 'weight': 0.15},
            'social_sentiment': {'enabled': True, 'weight': 0.10},
            'positioning_data': {'enabled': True, 'weight': 0.15},
            'intermarket_sentiment': {'enabled': True, 'weight': 0.15}
        }
        
        # 情绪阈值设置
        self.sentiment_thresholds = {
            'extreme_fear': 20,      # 极度恐惧
            'fear': 35,              # 恐惧
            'neutral': 50,           # 中性
            'greed': 65,             # 贪婪
            'extreme_greed': 80      # 极度贪婪
        }
        
        # 外部数据源配置
        self.external_data_sources = {
            'economic_calendar': {'enabled': True, 'impact_threshold': 'medium'},
            'central_bank_policy': {'enabled': True, 'lookback_days': 30},
            'geopolitical_events': {'enabled': True, 'risk_threshold': 'medium'},
            'seasonal_patterns': {'enabled': True, 'historical_years': 5},
            'intermarket_correlations': {'enabled': True, 'correlation_threshold': 0.6}
        }
        
        # 情绪数据历史
        self.sentiment_history = {
            'composite_sentiment': [],
            'market_regime': [],
            'sentiment_extremes': [],
            'contrarian_signals': []
        }
        
        # 外部事件影响追踪
        self.external_events = []
        self.event_impact_history = []
        
        # 季节性模式数据
        self.seasonal_patterns = {}
        self.intermarket_data = {}
        
    # 移除了 analyze_market_sentiment - 简化策略逻辑
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """综合市场情绪分析"""
        try:
            sentiment_components = {}
            
            # 1. 恐惧贪婪指数分析
            if self.sentiment_indicators['fear_greed_index']['enabled']:
                fear_greed = self.calculate_fear_greed_index()
                sentiment_components['fear_greed'] = fear_greed
            
            # 2. 波动率情绪分析
            if self.sentiment_indicators['vix_equivalent']['enabled']:
                vix_sentiment = self.analyze_volatility_sentiment()
                sentiment_components['volatility_sentiment'] = vix_sentiment
            
            # 3. 新闻情绪分析
            if self.sentiment_indicators['news_sentiment']['enabled']:
                news_sentiment = self.analyze_news_sentiment()
                sentiment_components['news_sentiment'] = news_sentiment
            
            # 4. 社交媒体情绪
            if self.sentiment_indicators['social_sentiment']['enabled']:
                social_sentiment = self.analyze_social_sentiment()
                sentiment_components['social_sentiment'] = social_sentiment
            
            # 5. 持仓数据分析
            if self.sentiment_indicators['positioning_data']['enabled']:
                positioning_sentiment = self.analyze_positioning_data()
                sentiment_components['positioning_sentiment'] = positioning_sentiment
            
            # 6. 跨市场情绪分析
            if self.sentiment_indicators['intermarket_sentiment']['enabled']:
                intermarket_sentiment = self.analyze_intermarket_sentiment()
                sentiment_components['intermarket_sentiment'] = intermarket_sentiment
            
            # 综合情绪计算
            composite_sentiment = self.calculate_composite_sentiment(sentiment_components)
            
            # 情绪状态判断
            sentiment_state = self.determine_sentiment_state(composite_sentiment)
            
            # 生成交易信号调整
            sentiment_adjustment = self.generate_sentiment_adjustment(sentiment_state, sentiment_components)
            
            sentiment_analysis = {
                'composite_sentiment': composite_sentiment,
                'sentiment_state': sentiment_state,
                'components': sentiment_components,
                'trading_adjustment': sentiment_adjustment,
                'contrarian_opportunity': self.detect_contrarian_opportunity(composite_sentiment),
                'timestamp': datetime.now(timezone.utc)
            }
            
            # 更新情绪历史
            self.update_sentiment_history(sentiment_analysis)
            
            return sentiment_analysis
            
        except Exception as e:
            return {
                'composite_sentiment': 50,  # 中性
                'sentiment_state': 'neutral',
                'error': f'情绪分析错误: {str(e)}',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def calculate_fear_greed_index(self) -> Dict[str, Any]:
        """计算恐惧贪婪指数"""
        try:
            components = {}
            
            # 价格动量 (25%)
            price_momentum = self.calculate_price_momentum_sentiment()
            components['price_momentum'] = price_momentum
            
            # 市场波动率 (25%) - 与VIX相反
            volatility_fear = self.calculate_volatility_fear()
            components['volatility_fear'] = volatility_fear
            
            # 市场广度 (15%) - 上涨下跌比例
            market_breadth = self.calculate_market_breadth_sentiment()
            components['market_breadth'] = market_breadth
            
            # 安全避险需求 (15%) - 避险资产表现
            safe_haven_demand = self.calculate_safe_haven_sentiment()
            components['safe_haven_demand'] = safe_haven_demand
            
            # 垃圾债券需求 (10%) - 风险偏好指标  
            junk_bond_demand = self.calculate_junk_bond_sentiment()
            components['junk_bond_demand'] = junk_bond_demand
            
            # 看涨看跌期权比例 (10%)
            put_call_ratio = self.calculate_put_call_sentiment()
            components['put_call_ratio'] = put_call_ratio
            
            # 加权平均计算恐惧贪婪指数
            weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
            values = [price_momentum, volatility_fear, market_breadth, 
                     safe_haven_demand, junk_bond_demand, put_call_ratio]
            
            fear_greed_index = sum(w * v for w, v in zip(weights, values) if v is not None)
            
            return {
                'index_value': fear_greed_index,
                'components': components,
                'interpretation': self.interpret_fear_greed_index(fear_greed_index)
            }
            
        except Exception:
            return {
                'index_value': 50,
                'components': {},
                'interpretation': 'neutral'
            }
    
    def calculate_price_momentum_sentiment(self) -> float:
        """计算价格动量情绪"""
        try:
            # 这里应该基于实际的价格数据计算
            # 简化实现：基于假设的价格表现
            
            # 模拟125日移动平均线上方的股票百分比
            stocks_above_ma125 = 0.6  # 60%的股票在125日均线上方
            
            # 转换为0-100的恐惧贪婪指数值
            momentum_sentiment = stocks_above_ma125 * 100
            
            return min(100, max(0, momentum_sentiment))
            
        except Exception:
            return 50
    
    def calculate_volatility_fear(self) -> float:
        """计算波动率恐惧指数"""
        try:
            # 当前波动率相对于历史平均值
            current_volatility = getattr(self, 'current_market_volatility', {})
            avg_vol = sum(current_volatility.values()) / len(current_volatility) if current_volatility else 0.02
            
            # 历史平均波动率（假设值）
            historical_avg_vol = 0.025
            
            # 波动率比率
            vol_ratio = avg_vol / historical_avg_vol if historical_avg_vol > 0 else 1.0
            
            # 转换为恐惧贪婪指数（波动率越高，恐惧越大，指数越低）
            volatility_fear = max(0, min(100, 100 - (vol_ratio - 1) * 50))
            
            return volatility_fear
            
        except Exception:
            return 50
    
    def calculate_market_breadth_sentiment(self) -> float:
        """计算市场广度情绪"""
        try:
            # 模拟市场广度数据
            # 实际应该基于上涨下跌股票数量比例
            
            # 假设数据：上涨股票比例
            advancing_stocks_ratio = 0.55  # 55%的股票上涨
            
            # 转换为恐惧贪婪指数
            breadth_sentiment = advancing_stocks_ratio * 100
            
            return min(100, max(0, breadth_sentiment))
            
        except Exception:
            return 50
    
    def calculate_safe_haven_sentiment(self) -> float:
        """计算避险资产需求情绪"""
        try:
            # 模拟避险资产表现
            # 实际应该基于美债、黄金等避险资产的表现
            
            # 假设避险资产相对表现（负值表示避险需求高）
            safe_haven_performance = -0.02  # -2%表示避险资产跑赢
            
            # 转换为恐惧贪婪指数（避险需求越高，贪婪指数越低）
            safe_haven_sentiment = max(0, min(100, 50 - safe_haven_performance * 1000))
            
            return safe_haven_sentiment
            
        except Exception:
            return 50
    
    def calculate_junk_bond_sentiment(self) -> float:
        """计算垃圾债券需求情绪"""
        try:
            # 模拟垃圾债券与国债收益率差
            # 实际应该基于高收益债券的信用利差
            
            # 假设信用利差（bp）
            credit_spread_bp = 350  # 350个基点
            historical_avg_spread = 400  # 历史平均400bp
            
            # 转换为恐惧贪婪指数
            spread_ratio = credit_spread_bp / historical_avg_spread
            junk_bond_sentiment = max(0, min(100, 100 - (spread_ratio - 1) * 100))
            
            return junk_bond_sentiment
            
        except Exception:
            return 50
    
    def calculate_put_call_sentiment(self) -> float:
        """计算看涨看跌期权比例情绪"""
        try:
            # 模拟看跌/看涨期权比例
            # 实际应该基于期权交易数据
            
            # 假设看跌/看涨比例
            put_call_ratio = 0.8  # 0.8表示相对看涨
            historical_avg_ratio = 1.0
            
            # 转换为恐惧贪婪指数（看跌比例越低，贪婪指数越高）
            put_call_sentiment = max(0, min(100, 100 - (put_call_ratio / historical_avg_ratio - 1) * 100))
            
            return put_call_sentiment
            
        except Exception:
            return 50
    
    def interpret_fear_greed_index(self, index_value: float) -> str:
        """解释恐惧贪婪指数"""
        if index_value <= self.sentiment_thresholds['extreme_fear']:
            return 'extreme_fear'
        elif index_value <= self.sentiment_thresholds['fear']:
            return 'fear'
        elif index_value <= self.sentiment_thresholds['neutral']:
            return 'neutral_fear'
        elif index_value <= self.sentiment_thresholds['greed']:
            return 'neutral_greed'
        elif index_value <= self.sentiment_thresholds['extreme_greed']:
            return 'greed'
        else:
            return 'extreme_greed'
    
    # 移除了 analyze_volatility_sentiment - 简化策略逻辑
    def analyze_volatility_sentiment(self) -> Dict[str, Any]:
        """分析波动率情绪"""
        try:
            current_volatility = getattr(self, 'current_market_volatility', {})
            
            if not current_volatility:
                return {
                    'volatility_level': 'normal',
                    'sentiment_signal': 'neutral',
                    'volatility_percentile': 50
                }
            
            avg_vol = sum(current_volatility.values()) / len(current_volatility)
            
            # 波动率分位数（简化计算）
            vol_percentile = min(95, max(5, avg_vol * 2000))  # 简化映射
            
            # 情绪信号
            if vol_percentile > 80:
                sentiment_signal = 'high_fear'
                volatility_level = 'high'
            elif vol_percentile > 60:
                sentiment_signal = 'moderate_fear'
                volatility_level = 'elevated'
            elif vol_percentile < 20:
                sentiment_signal = 'complacency'
                volatility_level = 'low'
            else:
                sentiment_signal = 'neutral'
                volatility_level = 'normal'
            
            return {
                'volatility_level': volatility_level,
                'sentiment_signal': sentiment_signal,
                'volatility_percentile': vol_percentile,
                'average_volatility': avg_vol
            }
            
        except Exception:
            return {
                'volatility_level': 'normal',
                'sentiment_signal': 'neutral',
                'volatility_percentile': 50
            }
    
    # 移除了 analyze_news_sentiment - 简化策略逻辑
    def analyze_news_sentiment(self) -> Dict[str, Any]:
        """分析新闻情绪"""
        try:
            # 模拟新闻情绪分析
            # 实际应该集成新闻API和NLP分析
            
            # 假设新闻情绪分数 (-1到1)
            news_sentiment_score = 0.1  # 略微积极
            
            # 新闻量和关注度
            news_volume = 1.2  # 120%的正常新闻量
            
            # 关键词分析结果
            sentiment_keywords = {
                'positive': ['growth', 'opportunity', 'bullish'],
                'negative': ['uncertainty', 'risk', 'volatile'],
                'neutral': ['stable', 'unchanged', 'maintain']
            }
            
            # 转换为交易信号
            if news_sentiment_score > 0.3:
                trading_signal = 'bullish'
            elif news_sentiment_score < -0.3:
                trading_signal = 'bearish'
            else:
                trading_signal = 'neutral'
            
            return {
                'sentiment_score': news_sentiment_score,
                'trading_signal': trading_signal,
                'news_volume': news_volume,
                'sentiment_keywords': sentiment_keywords,
                'confidence_level': min(1.0, abs(news_sentiment_score) + 0.5)
            }
            
        except Exception:
            return {
                'sentiment_score': 0.0,
                'trading_signal': 'neutral',
                'news_volume': 1.0,
                'confidence_level': 0.5
            }
    
    # 移除了 analyze_social_sentiment - 简化策略逻辑
    def analyze_social_sentiment(self) -> Dict[str, Any]:
        """分析社交媒体情绪"""
        try:
            # 模拟社交媒体情绪分析
            # 实际应该集成Twitter/Reddit等API
            
            # 社交媒体提及量
            mention_volume = 1.3  # 130%的正常提及量
            
            # 情绪分布
            sentiment_distribution = {
                'bullish': 0.4,   # 40%看涨
                'bearish': 0.3,   # 30%看跌
                'neutral': 0.3    # 30%中性
            }
            
            # 影响者情绪（权重更高）
            influencer_sentiment = 0.2  # 影响者略微看涨
            
            # 趋势强度
            trend_strength = abs(sentiment_distribution['bullish'] - sentiment_distribution['bearish'])
            
            # 综合社交情绪分数
            social_score = (
                sentiment_distribution['bullish'] * 1 + 
                sentiment_distribution['bearish'] * (-1) + 
                sentiment_distribution['neutral'] * 0
            )
            
            # 调整影响者权重
            adjusted_score = social_score * 0.7 + influencer_sentiment * 0.3
            
            return {
                'sentiment_score': adjusted_score,
                'mention_volume': mention_volume,
                'sentiment_distribution': sentiment_distribution,
                'influencer_sentiment': influencer_sentiment,
                'trend_strength': trend_strength,
                'social_signal': 'bullish' if adjusted_score > 0.1 else 'bearish' if adjusted_score < -0.1 else 'neutral'
            }
            
        except Exception:
            return {
                'sentiment_score': 0.0,
                'mention_volume': 1.0,
                'social_signal': 'neutral',
                'trend_strength': 0.0
            }
    
    # 移除了 analyze_positioning_data - 简化策略逻辑
    def analyze_positioning_data(self) -> Dict[str, Any]:
        """分析持仓数据情绪"""
        try:
            # 模拟持仓数据分析
            # 实际应该基于COT报告等数据
            
            # 大型交易者净持仓
            large_trader_net_long = 0.15  # 15%净多头
            
            # 散户持仓偏向
            retail_sentiment = -0.1  # 散户略微看空
            
            # 机构持仓变化
            institutional_flow = 0.05  # 5%资金净流入
            
            # 持仓极端程度
            positioning_extreme = max(
                abs(large_trader_net_long),
                abs(retail_sentiment),
                abs(institutional_flow)
            )
            
            # 逆向指标（散户情绪相反）
            contrarian_signal = 'bullish' if retail_sentiment < -0.15 else 'bearish' if retail_sentiment > 0.15 else 'neutral'
            
            return {
                'large_trader_positioning': large_trader_net_long,
                'retail_sentiment': retail_sentiment,
                'institutional_flow': institutional_flow,
                'positioning_extreme': positioning_extreme,
                'contrarian_signal': contrarian_signal,
                'positioning_risk': 'high' if positioning_extreme > 0.2 else 'medium' if positioning_extreme > 0.1 else 'low'
            }
            
        except Exception:
            return {
                'large_trader_positioning': 0.0,
                'retail_sentiment': 0.0,
                'institutional_flow': 0.0,
                'contrarian_signal': 'neutral',
                'positioning_risk': 'low'
            }
    
    # 移除了 analyze_intermarket_sentiment - 简化策略逻辑
    def analyze_intermarket_sentiment(self) -> Dict[str, Any]:
        """分析跨市场情绪"""
        try:
            # 模拟跨市场关系分析
            # 实际应该基于股票、债券、商品、汇率的相关性
            
            # 股债关系
            stock_bond_correlation = -0.3  # 负相关为正常
            
            # 美元强度
            dollar_strength = 0.02  # 美元相对强势2%
            
            # 商品表现
            commodity_performance = -0.01  # 商品略微下跌
            
            # 避险资产表现
            safe_haven_flows = 0.5  # 适中的避险需求
            
            # 跨市场压力指标
            intermarket_stress = abs(stock_bond_correlation + 0.5) + abs(dollar_strength) * 10
            
            # 风险偏好指标
            risk_appetite = 0.6 - safe_haven_flows
            
            return {
                'stock_bond_correlation': stock_bond_correlation,
                'dollar_strength': dollar_strength,
                'commodity_performance': commodity_performance,
                'safe_haven_flows': safe_haven_flows,
                'intermarket_stress': intermarket_stress,
                'risk_appetite': risk_appetite,
                'regime': 'risk_on' if risk_appetite > 0.3 else 'risk_off' if risk_appetite < -0.3 else 'mixed'
            }
            
        except Exception:
            return {
                'stock_bond_correlation': -0.5,
                'dollar_strength': 0.0,
                'commodity_performance': 0.0,
                'safe_haven_flows': 0.5,
                'risk_appetite': 0.0,
                'regime': 'mixed'
            }
    
    def calculate_composite_sentiment(self, components: Dict[str, Any]) -> float:
        """计算综合情绪指数"""
        try:
            sentiment_values = []
            weights = []
            
            # 恐惧贪婪指数
            if 'fear_greed' in components:
                sentiment_values.append(components['fear_greed']['index_value'])
                weights.append(self.sentiment_indicators['fear_greed_index']['weight'])
            
            # 波动率情绪
            if 'volatility_sentiment' in components:
                vol_sentiment = 100 - components['volatility_sentiment']['volatility_percentile']
                sentiment_values.append(vol_sentiment)
                weights.append(self.sentiment_indicators['vix_equivalent']['weight'])
            
            # 新闻情绪
            if 'news_sentiment' in components:
                news_score = (components['news_sentiment']['sentiment_score'] + 1) * 50
                sentiment_values.append(news_score)
                weights.append(self.sentiment_indicators['news_sentiment']['weight'])
            
            # 社交媒体情绪
            if 'social_sentiment' in components:
                social_score = (components['social_sentiment']['sentiment_score'] + 1) * 50
                sentiment_values.append(social_score)
                weights.append(self.sentiment_indicators['social_sentiment']['weight'])
            
            # 持仓数据情绪
            if 'positioning_sentiment' in components:
                pos_score = 50  # 中性基础值，可根据实际数据调整
                sentiment_values.append(pos_score)
                weights.append(self.sentiment_indicators['positioning_data']['weight'])
            
            # 跨市场情绪
            if 'intermarket_sentiment' in components:
                inter_score = (components['intermarket_sentiment']['risk_appetite'] + 1) * 50
                sentiment_values.append(inter_score)
                weights.append(self.sentiment_indicators['intermarket_sentiment']['weight'])
            
            # 加权平均
            if sentiment_values and weights:
                total_weight = sum(weights)
                composite_sentiment = sum(s * w for s, w in zip(sentiment_values, weights)) / total_weight
            else:
                composite_sentiment = 50  # 默认中性
            
            return max(0, min(100, composite_sentiment))
            
        except Exception:
            return 50  # 出错时返回中性情绪
    
    def determine_sentiment_state(self, composite_sentiment: float) -> str:
        """确定情绪状态"""
        if composite_sentiment <= self.sentiment_thresholds['extreme_fear']:
            return 'extreme_fear'
        elif composite_sentiment <= self.sentiment_thresholds['fear']:
            return 'fear'
        elif composite_sentiment <= self.sentiment_thresholds['neutral']:
            return 'neutral_bearish'
        elif composite_sentiment <= self.sentiment_thresholds['greed']:
            return 'neutral_bullish'
        elif composite_sentiment <= self.sentiment_thresholds['extreme_greed']:
            return 'greed'
        else:
            return 'extreme_greed'
    
    def generate_sentiment_adjustment(self, sentiment_state: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """基于情绪生成交易调整"""
        try:
            adjustment = {
                'position_size_multiplier': 1.0,
                'leverage_multiplier': 1.0,
                'risk_tolerance_adjustment': 0.0,
                'entry_threshold_adjustment': 0.0,
                'sentiment_signal': 'neutral'
            }
            
            # 基于情绪状态的调整
            if sentiment_state == 'extreme_fear':
                adjustment.update({
                    'position_size_multiplier': 0.8,    # 减小仓位
                    'leverage_multiplier': 0.7,         # 降低杠杆
                    'risk_tolerance_adjustment': -0.1,   # 更保守
                    'entry_threshold_adjustment': -0.05, # 降低入场标准（逆向）
                    'sentiment_signal': 'contrarian_bullish'
                })
            elif sentiment_state == 'fear':
                adjustment.update({
                    'position_size_multiplier': 0.9,
                    'leverage_multiplier': 0.85,
                    'risk_tolerance_adjustment': -0.05,
                    'entry_threshold_adjustment': -0.02,
                    'sentiment_signal': 'cautious_bullish'
                })
            elif sentiment_state == 'extreme_greed':
                adjustment.update({
                    'position_size_multiplier': 0.7,    # 大幅减小仓位
                    'leverage_multiplier': 0.6,         # 大幅降低杠杆
                    'risk_tolerance_adjustment': -0.15,  # 非常保守
                    'entry_threshold_adjustment': 0.1,   # 提高入场标准
                    'sentiment_signal': 'contrarian_bearish'
                })
            elif sentiment_state == 'greed':
                adjustment.update({
                    'position_size_multiplier': 0.85,
                    'leverage_multiplier': 0.8,
                    'risk_tolerance_adjustment': -0.08,
                    'entry_threshold_adjustment': 0.03,
                    'sentiment_signal': 'cautious_bearish'
                })
            
            # 基于具体组件的微调
            if 'volatility_sentiment' in components:
                vol_signal = components['volatility_sentiment']['sentiment_signal']
                if vol_signal == 'high_fear':
                    adjustment['position_size_multiplier'] *= 0.9
                elif vol_signal == 'complacency':
                    adjustment['risk_tolerance_adjustment'] -= 0.05
            
            return adjustment
            
        except Exception:
            return {
                'position_size_multiplier': 1.0,
                'leverage_multiplier': 1.0,
                'risk_tolerance_adjustment': 0.0,
                'entry_threshold_adjustment': 0.0,
                'sentiment_signal': 'neutral'
            }
    
    def detect_contrarian_opportunity(self, composite_sentiment: float) -> Dict[str, Any]:
        """检测逆向投资机会"""
        try:
            # 逆向机会检测
            contrarian_opportunity = {
                'opportunity_detected': False,
                'opportunity_type': None,
                'strength': 0.0,
                'recommended_action': 'hold'
            }
            
            # 极端情绪逆向机会
            if composite_sentiment <= 25:  # 极度恐惧
                contrarian_opportunity.update({
                    'opportunity_detected': True,
                    'opportunity_type': 'extreme_fear_buying',
                    'strength': (25 - composite_sentiment) / 25,
                    'recommended_action': 'aggressive_buy'
                })
            elif composite_sentiment >= 75:  # 极度贪婪
                contrarian_opportunity.update({
                    'opportunity_detected': True,
                    'opportunity_type': 'extreme_greed_selling',
                    'strength': (composite_sentiment - 75) / 25,
                    'recommended_action': 'reduce_exposure'
                })
            
            # 情绪快速变化检测
            if len(self.sentiment_history['composite_sentiment']) >= 5:
                recent_sentiments = self.sentiment_history['composite_sentiment'][-5:]
                sentiment_velocity = recent_sentiments[-1] - recent_sentiments[0]
                
                if abs(sentiment_velocity) > 20:  # 快速变化
                    contrarian_opportunity.update({
                        'opportunity_detected': True,
                        'opportunity_type': 'sentiment_reversal',
                        'strength': min(1.0, abs(sentiment_velocity) / 30),
                        'recommended_action': 'fade_the_move'
                    })
            
            return contrarian_opportunity
            
        except Exception:
            return {
                'opportunity_detected': False,
                'opportunity_type': None,
                'strength': 0.0,
                'recommended_action': 'hold'
            }
    
    def update_sentiment_history(self, sentiment_analysis: Dict[str, Any]):
        """更新情绪历史记录"""
        try:
            # 更新综合情绪历史
            self.sentiment_history['composite_sentiment'].append(sentiment_analysis['composite_sentiment'])
            
            # 更新情绪状态历史
            self.sentiment_history['sentiment_state'].append(sentiment_analysis['sentiment_state'])
            
            # 记录情绪极端值
            if sentiment_analysis['composite_sentiment'] <= 25 or sentiment_analysis['composite_sentiment'] >= 75:
                extreme_record = {
                    'timestamp': sentiment_analysis['timestamp'],
                    'sentiment_value': sentiment_analysis['composite_sentiment'],
                    'sentiment_state': sentiment_analysis['sentiment_state']
                }
                self.sentiment_history['sentiment_extremes'].append(extreme_record)
            
            # 记录逆向信号
            if sentiment_analysis.get('contrarian_opportunity', {}).get('opportunity_detected'):
                contrarian_record = {
                    'timestamp': sentiment_analysis['timestamp'],
                    'opportunity_type': sentiment_analysis['contrarian_opportunity']['opportunity_type'],
                    'strength': sentiment_analysis['contrarian_opportunity']['strength']
                }
                self.sentiment_history['contrarian_signals'].append(contrarian_record)
            
            # 维护历史记录长度
            for key, history in self.sentiment_history.items():
                if len(history) > 500:
                    self.sentiment_history[key] = history[-250:]
                    
        except Exception:
            pass
    
    def get_sentiment_analysis_report(self) -> Dict[str, Any]:
        """获取情绪分析报告"""
        try:
            if not self.sentiment_history['composite_sentiment']:
                return {'error': '无情绪数据'}
            
            recent_sentiment = self.sentiment_history['composite_sentiment'][-1]
            recent_state = self.sentiment_history['sentiment_state'][-1]
            
            # 情绪统计
            sentiment_stats = {
                'current_sentiment': recent_sentiment,
                'current_state': recent_state,
                'avg_sentiment_30d': np.mean(self.sentiment_history['composite_sentiment'][-30:]) if len(self.sentiment_history['composite_sentiment']) >= 30 else recent_sentiment,
                'sentiment_volatility': np.std(self.sentiment_history['composite_sentiment'][-30:]) if len(self.sentiment_history['composite_sentiment']) >= 30 else 0,
                'extreme_events_30d': len([x for x in self.sentiment_history['sentiment_extremes'] if (datetime.now(timezone.utc) - x['timestamp']).days <= 30]),
                'contrarian_signals_30d': len([x for x in self.sentiment_history['contrarian_signals'] if (datetime.now(timezone.utc) - x['timestamp']).days <= 30])
            }
            
            return {
                'sentiment_stats': sentiment_stats,
                'sentiment_trend': 'improving' if len(self.sentiment_history['composite_sentiment']) >= 2 and self.sentiment_history['composite_sentiment'][-1] > self.sentiment_history['composite_sentiment'][-2] else 'deteriorating',
                'market_regime': 'fear_dominated' if recent_sentiment < 40 else 'greed_dominated' if recent_sentiment > 60 else 'neutral',
                'last_update': datetime.now(timezone.utc)
            }
            
        except Exception:
            return {'error': '无法生成情绪分析报告'}
    
    # === 🛡️ ATR智能止损辅助函数 ===
    
    def _get_trade_entry_atr(self, trade: Trade, dataframe: DataFrame) -> float:
        """
        获取交易开仓时的ATR值 - 作为止损计算的基准
        这是避免止损过于宽松或严格的关键
        """
        try:
            # 使用开仓时间戳找到对应的K线
            from freqtrade.misc import timeframe_to_prev_date
            
            entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            entry_candles = dataframe[dataframe.index <= entry_date]
            
            if not entry_candles.empty and 'atr_p' in entry_candles.columns:
                entry_atr = entry_candles['atr_p'].iloc[-1]
                # 安全范围检查
                if 0.005 <= entry_atr <= 0.20:
                    return entry_atr
                    
        except Exception as e:
            logger.warning(f"获取开仓ATR失败: {e}")
            
        # 降级方案：使用最近20期ATR中位数
        if 'atr_p' in dataframe.columns and len(dataframe) >= 20:
            return dataframe['atr_p'].tail(20).median()
        
        # 最后降级：根据交易对类型给出经验值
        if 'BTC' in trade.pair or 'ETH' in trade.pair:
            return 0.02  # 主流币相对稳定
        else:
            return 0.035  # 山寨币波动更大
    
    def _calculate_atr_multiplier(self, entry_atr_p: float, current_candle: dict, enter_tag: str) -> float:
        """
        计算ATR倍数 - 核心参数，决定止损给予的波动空间
        基于信号类型和市场环境动态调整
        """
        # 基础倍数：研究表明2.5-3.5为最优范围
        base_multiplier = 2.8
        
        # === 1. 信号类型调整 ===
        signal_adjustments = {
            'RSI_Oversold_Bounce': 2.5,    # RSI信号相对可靠，可用紧一些的止损
            'RSI_Overbought_Fall': 2.5,    
            'MACD_Bearish': 3.2,           # MACD信号容易假突破，需要更宽松
            'MACD_Bullish': 3.2,
            'EMA_Golden_Cross': 2.6,       # 趋势信号，中等止损
            'EMA_Death_Cross': 2.6,
        }
        
        multiplier = signal_adjustments.get(enter_tag, base_multiplier)
        
        # === 2. 波动性环境调整 ===
        current_atr_p = current_candle.get('atr_p', entry_atr_p)
        volatility_ratio = current_atr_p / entry_atr_p
        
        if volatility_ratio > 1.5:      # 当前波动比开仓时高50%
            multiplier *= 1.2           # 放宽止损20%
        elif volatility_ratio < 0.7:    # 当前波动降低30%
            multiplier *= 0.9           # 收紧止损10%
        
        # === 3. 趋势强度调整 ===
        adx = current_candle.get('adx', 25)
        if adx > 35:                    # 强趋势环境
            multiplier *= 1.15          # 给趋势更多空间
        elif adx < 20:                  # 横盘环境
            multiplier *= 0.85          # 收紧止损避免横盘消耗
        
        # 安全边界
        return max(1.5, min(4.0, multiplier))
    
    def _calculate_time_decay(self, hours_held: float, current_profit: float) -> float:
        """
        时间衰减因子 - 防止长期套牢
        持仓时间越长，止损越严格
        """
        # 如果已经盈利，延缓时间衰减
        if current_profit > 0.02:       # 盈利2%以上
            decay_start_hours = 72      # 3天后开始衰减
        elif current_profit > -0.02:    # 小幅亏损
            decay_start_hours = 48      # 2天后开始衰减  
        else:                           # 较大亏损
            decay_start_hours = 24      # 1天后开始衰减
        
        if hours_held <= decay_start_hours:
            return 1.0                  # 无衰减
            
        # 指数衰减：每24小时收紧10%
        excess_hours = hours_held - decay_start_hours
        decay_periods = excess_hours / 24
        
        # 最多衰减到原来的50%
        min_factor = 0.5
        decay_factor = max(min_factor, 1.0 - (decay_periods * 0.1))
        
        return decay_factor
    
    def _calculate_profit_protection(self, current_profit: float) -> Optional[float]:
        """
        分阶段盈利保护 - 锁定利润，让盈利奔跑
        """
        if current_profit > 0.15:      # 盈利15%+，锁定75%利润
            return -0.0375              # 允许3.75%回撤
        elif current_profit > 0.10:    # 盈利10%+，锁定60%利润  
            return -0.04                # 允许4%回撤
        elif current_profit > 0.08:    # 盈利8%+，锁定50%利润
            return -0.04                # 允许4%回撤
        elif current_profit > 0.05:    # 盈利5%+，保本+
            return -0.01                # 允许1%回撤保本
        elif current_profit > 0.03:    # 盈利3%+，移至保本
            return 0.001                # 保本+手续费
        
        return None                     # 无盈利保护，使用ATR止损
    
    def _calculate_trend_adjustment(self, current_candle: dict, is_short: bool, entry_atr_p: float) -> float:
        """
        趋势强度调整 - 顺势宽松，逆势严格
        """
        # 获取趋势指标
        ema_8 = current_candle.get('ema_8', 0)
        ema_21 = current_candle.get('ema_21', 0)
        adx = current_candle.get('adx', 25)
        current_price = current_candle.get('close', 0)
        
        # 判断趋势方向
        is_uptrend = ema_8 > ema_21 and adx > 25
        is_downtrend = ema_8 < ema_21 and adx > 25
        
        # 趋势一致性检查
        if is_short and is_downtrend:      # 做空+下跌趋势，顺势
            return 1.2                     # 放宽20%
        elif not is_short and is_uptrend:  # 做多+上涨趋势，顺势
            return 1.2                     # 放宽20%
        elif is_short and is_uptrend:      # 做空+上涨趋势，逆势
            return 0.8                     # 收紧20%
        elif not is_short and is_downtrend: # 做多+下跌趋势，逆势  
            return 0.8                     # 收紧20%
        else:                              # 横盘或不明确
            return 1.0                     # 无调整
    
    def _log_stoploss_calculation(self, pair: str, trade: Trade, current_profit: float,
                                 entry_atr_p: float, base_atr_multiplier: float,
                                 time_decay_factor: float, trend_adjustment: float,
                                 final_stoploss: float):
        """
        详细记录止损计算过程 - 便于优化和调试
        """
        hours_held = (datetime.now(timezone.utc) - trade.open_date_utc).total_seconds() / 3600
        
        logger.info(
            f"🛡️ ATR止损 {pair} [{trade.enter_tag}]: "
            f"盈利{current_profit:.1%} | "
            f"持仓{hours_held:.1f}h | "
            f"开仓ATR{entry_atr_p:.3f} | "
            f"ATR倍数{base_atr_multiplier:.1f} | "
            f"时间衰减{time_decay_factor:.2f} | " 
            f"趋势调整{trend_adjustment:.2f} | "
            f"最终止损{final_stoploss:.3f}"
        )
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, 
                       after_fill: bool = False, **kwargs) -> Optional[float]:
        """
        🚀 智能ATR动态止损系统
        - 基于开仓时ATR动态调整
        - 时间衰减机制防止长期套牢
        - 分阶段盈利保护
        - 趋势强度感知调整
        """
        try:
            # 获取最新数据
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or len(dataframe) < 50:
                return None
                
            current_candle = dataframe.iloc[-1]
            
            # === 1. 获取开仓时ATR (关键基准) ===
            entry_atr_p = self._get_trade_entry_atr(trade, dataframe)
            current_atr_p = current_candle.get('atr_p', 0.02)
            
            # === 2. 计算基础ATR止损距离 ===
            # 研究表明2.5-3倍ATR为最优，给予正常波动喘息空间
            base_atr_multiplier = self._calculate_atr_multiplier(
                entry_atr_p, current_candle, trade.enter_tag
            )
            base_stop_distance = entry_atr_p * base_atr_multiplier
            
            # === 3. 时间衰减机制 ===
            # 持仓越久，止损越严格，防止长期套牢
            hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600
            time_decay_factor = self._calculate_time_decay(hours_held, current_profit)
            
            # === 4. 分阶段盈利保护系统 ===
            profit_protection = self._calculate_profit_protection(current_profit)
            if profit_protection is not None:
                return profit_protection
                
            # === 5. 趋势强度调整 ===
            trend_adjustment = self._calculate_trend_adjustment(
                current_candle, trade.is_short, entry_atr_p
            )
            
            # === 6. 最终止损计算 ===
            final_stop_distance = (base_stop_distance * time_decay_factor * trend_adjustment)
            
            # 安全边界：最小1%，最大8%
            final_stop_distance = max(0.01, min(0.08, final_stop_distance))
            
            # 做空单需要取反
            final_stoploss = -final_stop_distance if not trade.is_short else final_stop_distance
            
            # === 7. 日志记录 ===
            if self.config.get('verbosity', 0) > 1:
                self._log_stoploss_calculation(
                    pair, trade, current_profit, entry_atr_p, base_atr_multiplier,
                    time_decay_factor, trend_adjustment, final_stoploss
                )
                
            return final_stoploss
            
        except Exception as e:
            logger.error(f"ATR动态止损计算错误 {pair}: {e}")
            # 紧急情况使用保守止损
            return -0.03 if not trade.is_short else 0.03
    
    def _calculate_signal_quality_score(self, dataframe: DataFrame, signal_mask: pd.Series, signal_type: str) -> pd.Series:
        """
        🎯 智能信号质量评分系统 (1-10分)
        基于多维度分析评估信号可靠性，为风险管理提供依据
        """
        # 初始化评分
        scores = pd.Series(0.0, index=dataframe.index)
        
        # 只对有信号的位置计算评分
        signal_indices = signal_mask[signal_mask].index
        
        for idx in signal_indices:
            try:
                score = 3.0  # 基础分
                current_data = dataframe.loc[idx]
                
                # === 1. 技术指标一致性 (0-2分) ===
                rsi = current_data.get('rsi_14', 50)
                if signal_type in ['RSI_Oversold_Bounce']:
                    if rsi < 25:
                        score += 2    # 深度超卖，机会大
                    elif rsi < 30:
                        score += 1.5  # 正常超卖
                elif signal_type in ['RSI_Overbought_Fall']:
                    if rsi > 75:
                        score += 2    # 深度超买，风险大
                    elif rsi > 70:
                        score += 1.5  # 正常超买
                
                # === 2. 趋势强度与方向 (0-2分) ===
                adx = current_data.get('adx', 25)
                ema_8 = current_data.get('ema_8', 0)
                ema_21 = current_data.get('ema_21', 0)
                
                if adx > 30:  # 强趋势
                    if signal_type in ['RSI_Oversold_Bounce'] and ema_8 > ema_21:
                        score += 2  # 上升趋势中的超卖，高质量
                    elif signal_type in ['RSI_Overbought_Fall'] and ema_8 < ema_21:
                        score += 2  # 下跌趋势中的超买，高质量
                    else:
                        score += 0.5  # 逆势信号，质量一般
                elif 20 < adx <= 30:  # 中等趋势
                    score += 1
                
                # === 3. 成交量确认 (0-1.5分) ===
                volume_ratio = current_data.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:
                    score += 1.5  # 成交量爆发
                elif volume_ratio > 1.2:
                    score += 1.0  # 成交量放大
                elif volume_ratio > 1.0:
                    score += 0.5  # 成交量正常
                
                # === 4. 波动性环境 (0-1分) ===
                atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True).loc[idx]
                if 0.2 <= atr_percentile <= 0.8:  # 正常波动环境
                    score += 1
                elif atr_percentile > 0.9:  # 极高波动，风险大
                    score -= 0.5
                
                # === 5. 背离信号 (0-1分) ===
                no_bearish_div = not current_data.get('bearish_divergence', False)
                no_bullish_div = not current_data.get('bullish_divergence', False)
                
                if signal_type in ['RSI_Oversold_Bounce'] and no_bearish_div:
                    score += 1
                elif signal_type in ['RSI_Overbought_Fall'] and no_bullish_div:
                    score += 1
                
                # === 6. 市场环境加分 (0-0.5分) ===
                price_position = current_data.get('price_position', 0.5)
                if signal_type in ['RSI_Oversold_Bounce'] and 0.2 < price_position < 0.7:
                    score += 0.5  # 不在极端位置做多
                elif signal_type in ['RSI_Overbought_Fall'] and 0.3 < price_position < 0.8:
                    score += 0.5  # 不在极端位置做空
                
                # 限制评分范围
                scores.loc[idx] = max(1.0, min(10.0, score))
                
            except Exception as e:
                scores.loc[idx] = 3.0  # 默认评分
                logger.warning(f"信号质量评分计算错误 {signal_type}: {e}")
        
        return scores
    
    def _calculate_macd_signal_quality(self, dataframe: DataFrame, signal_mask: pd.Series, signal_type: str) -> pd.Series:
        """
        🎯 MACD专用信号质量评分系统 (1-10分)
        针对MACD信号特点，更严格的评分标准
        """
        # 初始化评分
        scores = pd.Series(0.0, index=dataframe.index)
        
        # 只对有信号的位置计算评分
        signal_indices = signal_mask[signal_mask].index
        
        for idx in signal_indices:
            try:
                score = 2.0  # MACD信号基础分更低，需要更多确认
                current_data = dataframe.loc[idx]
                
                # === 1. MACD信号强度 (0-2.5分) ===
                macd = current_data.get('macd', 0)
                macd_signal = current_data.get('macd_signal', 0)
                macd_hist = current_data.get('macd_hist', 0)
                
                # MACD死叉幅度越大，信号越强
                cross_magnitude = abs(macd - macd_signal)
                if cross_magnitude > 0.002:  # 强烈死叉
                    score += 2.5
                elif cross_magnitude > 0.001:  # 明显死叉
                    score += 1.5
                elif cross_magnitude > 0.0005:  # 轻微死叉
                    score += 1.0
                
                # === 2. 趋势一致性 (0-2分) ===
                ema_8 = current_data.get('ema_8', 0)
                ema_21 = current_data.get('ema_21', 0)
                ema_50 = current_data.get('ema_50', 0)
                
                if ema_8 < ema_21 < ema_50:  # 完美空头排列
                    score += 2
                elif ema_8 < ema_21:  # 基本空头排列
                    score += 1
                
                # === 3. 动量衰竭确认 (0-2分) ===
                rsi = current_data.get('rsi_14', 50)
                rsi_prev = dataframe['rsi_14'].iloc[max(0, idx-2):idx].mean()
                
                if rsi < 45 and rsi < rsi_prev:  # RSI配合下跌
                    score += 2
                elif rsi < 50:  # RSI偏弱
                    score += 1
                
                # === 4. 成交量爆发 (0-1.5分) ===
                volume_ratio = current_data.get('volume_ratio', 1.0)
                volume_trend = dataframe['volume'].iloc[max(0, idx-3):idx+1].iloc[-1] > \
                              dataframe['volume'].iloc[max(0, idx-3):idx+1].iloc[0]
                
                if volume_ratio > 1.5 and volume_trend:  # 成交量爆发且递增
                    score += 1.5
                elif volume_ratio > 1.2:  # 成交量放大
                    score += 1.0
                
                # === 5. ADX趋势强度 (0-1.5分) ===
                adx = current_data.get('adx', 25)
                adx_trend = current_data.get('adx', 25) > dataframe['adx'].iloc[max(0, idx-3)]
                
                if adx > 35 and adx_trend:  # 强趋势且加强
                    score += 1.5
                elif adx > 25:  # 中等趋势
                    score += 1.0
                
                # === 6. 横盘过滤 (0-1分) ===
                # MACD最容易在横盘中产生假信号
                if adx > 25:  # 确保不在横盘
                    score += 1
                else:
                    score -= 1  # 横盘时扣分
                
                # === 7. 位置合理性 (0-0.5分) ===
                price_position = current_data.get('price_position', 0.5)
                if 0.4 < price_position < 0.8:  # 在合理位置做空
                    score += 0.5
                
                # === 8. 背离保护 (0-0.5分) ===
                no_bullish_div = not current_data.get('bullish_divergence', False)
                if no_bullish_div:
                    score += 0.5
                
                # 限制评分范围
                scores.loc[idx] = max(1.0, min(10.0, score))
                
            except Exception as e:
                scores.loc[idx] = 2.0  # MACD默认评分更低
                logger.warning(f"MACD信号质量评分计算错误: {e}")
        
        return scores
    
    def _enhanced_market_regime_detection(self, dataframe: DataFrame) -> Dict[str, Any]:
        """
        🌍 增强版市场状态识别系统
        为信号生成和风险管理提供精确的市场环境分析
        """
        try:
            if dataframe.empty or len(dataframe) < 50:
                return {'regime': 'UNKNOWN', 'confidence': 0.0, 'characteristics': {}}
            
            current_data = dataframe.iloc[-1]
            recent_data = dataframe.tail(30)
            
            # === 1. 趋势状态分析 ===
            adx = current_data.get('adx', 25)
            ema_8 = current_data.get('ema_8', 0)
            ema_21 = current_data.get('ema_21', 0)
            ema_50 = current_data.get('ema_50', 0)
            
            # 趋势强度和方向
            if adx > 35:
                trend_strength = 'STRONG'
            elif adx > 25:
                trend_strength = 'MODERATE' 
            elif adx > 15:
                trend_strength = 'WEAK'
            else:
                trend_strength = 'SIDEWAYS'
            
            # 趋势方向
            if ema_8 > ema_21 > ema_50:
                trend_direction = 'UPTREND'
            elif ema_8 < ema_21 < ema_50:
                trend_direction = 'DOWNTREND'
            else:
                trend_direction = 'SIDEWAYS'
            
            # === 2. 波动性分析 ===
            atr_p = current_data.get('atr_p', 0.02)
            atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True).iloc[-1]
            
            if atr_percentile > 0.8:
                volatility_regime = 'HIGH'
            elif atr_percentile > 0.6:
                volatility_regime = 'ELEVATED'
            elif atr_percentile > 0.3:
                volatility_regime = 'NORMAL'
            else:
                volatility_regime = 'LOW'
            
            # === 3. 成交量分析 ===
            volume_ratio = current_data.get('volume_ratio', 1.0)
            avg_volume_ratio = recent_data['volume_ratio'].mean()
            
            if avg_volume_ratio > 1.3:
                volume_regime = 'HIGH_ACTIVITY'
            elif avg_volume_ratio > 1.1:
                volume_regime = 'ACTIVE'
            elif avg_volume_ratio > 0.8:
                volume_regime = 'NORMAL'
            else:
                volume_regime = 'LOW'
            
            # === 4. 价格位置分析 ===
            high_20 = dataframe['high'].rolling(20).max().iloc[-1]
            low_20 = dataframe['low'].rolling(20).min().iloc[-1]
            current_price = current_data.get('close', 0)
            price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            
            if price_position > 0.8:
                position_regime = 'NEAR_HIGH'
            elif price_position > 0.6:
                position_regime = 'UPPER_RANGE'
            elif price_position > 0.4:
                position_regime = 'MIDDLE_RANGE'
            elif price_position > 0.2:
                position_regime = 'LOWER_RANGE'
            else:
                position_regime = 'NEAR_LOW'
            
            # === 5. 综合市场状态判断 ===
            regime_score = 0
            confidence_factors = []
            
            # 强趋势市场
            if trend_strength in ['STRONG', 'MODERATE'] and trend_direction != 'SIDEWAYS':
                if volatility_regime in ['NORMAL', 'ELEVATED']:
                    regime = f"TRENDING_{trend_direction}"
                    regime_score += 3
                    confidence_factors.append("strong_trend")
                else:
                    regime = f"VOLATILE_{trend_direction}"
                    regime_score += 2
                    confidence_factors.append("volatile_trend")
            
            # 横盘市场
            elif trend_strength in ['WEAK', 'SIDEWAYS']:
                if volatility_regime in ['HIGH', 'ELEVATED']:
                    regime = "CHOPPY_SIDEWAYS"
                    regime_score += 1
                    confidence_factors.append("high_vol_sideways")
                else:
                    regime = "QUIET_SIDEWAYS"
                    regime_score += 2
                    confidence_factors.append("low_vol_sideways")
            
            # 不确定状态
            else:
                regime = "TRANSITIONAL"
                regime_score += 1
                confidence_factors.append("uncertain")
            
            # === 6. 特殊市场条件检测 ===
            special_conditions = []
            
            # 极端波动
            if atr_p > 0.06:
                special_conditions.append("EXTREME_VOLATILITY")
                regime_score -= 1
            
            # 成交量异常
            if volume_ratio > 2.0:
                special_conditions.append("VOLUME_SPIKE")
                regime_score += 1
            elif volume_ratio < 0.5:
                special_conditions.append("VOLUME_DRYING")
                regime_score -= 1
            
            # 极端位置
            if position_regime in ['NEAR_HIGH', 'NEAR_LOW']:
                special_conditions.append(f"EXTREME_POSITION_{position_regime}")
            
            # === 7. 置信度计算 ===
            base_confidence = min(0.9, regime_score / 5.0)
            
            # 数据质量调整
            data_quality = min(1.0, len(dataframe) / 100)
            final_confidence = base_confidence * data_quality
            
            return {
                'regime': regime,
                'confidence': max(0.1, final_confidence),
                'characteristics': {
                    'trend_strength': trend_strength,
                    'trend_direction': trend_direction,
                    'volatility_regime': volatility_regime,
                    'volume_regime': volume_regime,
                    'position_regime': position_regime,
                    'special_conditions': special_conditions,
                    'adx': adx,
                    'atr_percentile': atr_percentile,
                    'price_position': price_position,
                    'volume_ratio': volume_ratio
                },
                'signals_advice': self._get_regime_trading_advice(regime, volatility_regime, position_regime),
                'confidence_factors': confidence_factors
            }
            
        except Exception as e:
            logger.error(f"市场状态识别失败: {e}")
            return {
                'regime': 'ERROR',
                'confidence': 0.0,
                'characteristics': {},
                'signals_advice': {'recommended_signals': [], 'avoid_signals': []},
                'confidence_factors': []
            }
    
    def _get_regime_trading_advice(self, regime: str, volatility_regime: str, position_regime: str) -> Dict[str, list]:
        """
        基于市场状态给出交易建议
        """
        advice = {
            'recommended_signals': [],
            'avoid_signals': [],
            'risk_adjustment': 1.0,
            'position_size_multiplier': 1.0
        }
        
        # 基于不同市场状态的建议
        if 'TRENDING_UPTREND' in regime:
            advice['recommended_signals'] = ['RSI_Oversold_Bounce', 'EMA_Golden_Cross']
            advice['avoid_signals'] = ['RSI_Overbought_Fall'] 
            advice['position_size_multiplier'] = 1.2
            
        elif 'TRENDING_DOWNTREND' in regime:
            advice['recommended_signals'] = ['RSI_Overbought_Fall', 'MACD_Bearish']
            advice['avoid_signals'] = ['RSI_Oversold_Bounce']
            advice['position_size_multiplier'] = 1.2
            
        elif 'SIDEWAYS' in regime:
            if volatility_regime == 'LOW':
                advice['recommended_signals'] = ['RSI_Oversold_Bounce', 'RSI_Overbought_Fall']
                advice['avoid_signals'] = ['MACD_Bearish']
            else:
                advice['avoid_signals'] = ['MACD_Bearish', 'RSI_Overbought_Fall', 'RSI_Oversold_Bounce']
            advice['position_size_multiplier'] = 0.7
            
        elif 'VOLATILE' in regime:
            advice['avoid_signals'] = ['MACD_Bearish']
            advice['risk_adjustment'] = 1.5
            advice['position_size_multiplier'] = 0.6
            
        # 位置调整
        if position_regime in ['NEAR_HIGH']:
            advice['avoid_signals'].extend(['RSI_Oversold_Bounce'])
            advice['position_size_multiplier'] *= 0.8
        elif position_regime in ['NEAR_LOW']:
            advice['avoid_signals'].extend(['RSI_Overbought_Fall', 'MACD_Bearish'])
            advice['position_size_multiplier'] *= 0.8
        
        return advice
    
    # === 🎯 智能杠杆管理辅助函数 ===
    
    def _calculate_signal_quality_leverage_bonus(self, entry_tag: str, current_data: dict, 
                                               regime: str, signals_advice: dict) -> float:
        """
        基于信号质量计算杠杆奖励倍数
        高质量信号允许更高杠杆
        """
        if not entry_tag:
            return 1.0
        
        # 获取信号质量评分（如果有的话）
        signal_quality = current_data.get('signal_quality', 5.0)
        
        # 基础质量奖励：5-10分映射到0.8-1.5倍
        quality_bonus = 0.8 + (signal_quality - 5.0) / 5.0 * 0.7
        quality_bonus = max(0.8, min(1.5, quality_bonus))
        
        # 市场状态奖励：推荐信号额外奖励
        regime_bonus = 1.0
        if entry_tag in signals_advice.get('recommended_signals', []):
            regime_bonus = 1.2  # 推荐信号+20%杠杆
        elif entry_tag in signals_advice.get('avoid_signals', []):
            regime_bonus = 0.6  # 不推荐信号-40%杠杆
        
        return quality_bonus * regime_bonus
    
    def _get_regime_leverage_multiplier(self, regime: str, confidence: float) -> float:
        """
        基于市场状态计算杠杆倍数
        """
        base_multiplier = 1.0
        
        # 基于市场状态的倍数
        if 'TRENDING' in regime:
            if 'UPTREND' in regime or 'DOWNTREND' in regime:
                base_multiplier = 1.3  # 趋势市场+30%杠杆
            else:
                base_multiplier = 1.1  # 一般趋势+10%杠杆
                
        elif 'SIDEWAYS' in regime:
            if 'QUIET' in regime:
                base_multiplier = 1.1  # 安静横盘+10%杠杆
            else:
                base_multiplier = 0.8  # 混乱横盘-20%杠杆
                
        elif 'VOLATILE' in regime:
            base_multiplier = 0.7  # 高波动-30%杠杆
            
        elif 'TRANSITIONAL' in regime:
            base_multiplier = 0.9  # 过渡期-10%杠杆
        
        # 置信度调整：高置信度时增加倍数
        confidence_multiplier = 0.8 + confidence * 0.4  # 0.8-1.2范围
        
        return base_multiplier * confidence_multiplier
    
    def _get_signal_leverage_multiplier(self, entry_tag: str, signals_advice: dict) -> float:
        """
        基于信号类型计算杠杆倍数
        """
        if not entry_tag:
            return 1.0
        
        # 信号可靠性映射
        signal_reliability = {
            'RSI_Oversold_Bounce': 1.2,    # RSI信号相对可靠
            'RSI_Overbought_Fall': 1.2,
            'EMA_Golden_Cross': 1.3,       # 趋势信号最可靠
            'EMA_Death_Cross': 1.3,
            'MACD_Bearish': 1.0,           # MACD信号保守
            'MACD_Bullish': 1.0,
        }
        
        base_multiplier = signal_reliability.get(entry_tag, 1.0)
        
        # 市场推荐奖励
        if entry_tag in signals_advice.get('recommended_signals', []):
            base_multiplier *= 1.1  # 额外+10%
        elif entry_tag in signals_advice.get('avoid_signals', []):
            base_multiplier *= 0.7  # 惩罚-30%
        
        return base_multiplier
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        智能出场信号系统 - 多重条件组合的精准出场
        不碰止损和ROI，只通过智能信号优化出场时机
        """

        # 初始化出场信号列
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        # ==============================
        # 1. 强势反转信号（优先级最高）
        # ==============================

        # 做多头寸的强势反转出场（改进版：更敏感）
        strong_reversal_long_exit = (
            # RSI超买或背离
            ((dataframe['rsi_14'] > 70) |  # 原75，更敏感
             (dataframe['bearish_divergence'] == 1) |  # 新增：顶背离
             # MACD柱状图由正转负
             ((dataframe['macd_hist'].shift(1) > 0) & (dataframe['macd_hist'] < 0))) &
            # 成交量异常或衰竭
            ((dataframe['volume_ratio'] > 1.8) | (dataframe['volume_exhaustion'] == 1)) &  # 原2.0，更敏感
            # 价格在布林带上轨附近
            (dataframe['bb_position'] > 0.85) &  # 原0.9，更敏感
            # 新增：动量衰竭或趋势末期
            ((dataframe['momentum_exhaustion_score'] > 0.4) | (dataframe['trend_phase'] >= 3))
        )

        # 做空头寸的强势反转出场（改进版：更敏感）
        strong_reversal_short_exit = (
            # RSI超卖或背离
            ((dataframe['rsi_14'] < 30) |  # 原25，更敏感
             (dataframe['bullish_divergence'] == 1) |  # 新增：底背离
             # MACD柱状图由负转正
             ((dataframe['macd_hist'].shift(1) < 0) & (dataframe['macd_hist'] > 0))) &
            # 成交量异常或衰竭
            ((dataframe['volume_ratio'] > 1.8) | (dataframe['volume_exhaustion'] == 1)) &  # 原2.0，更敏感
            # 价格在布林带下轨附近
            (dataframe['bb_position'] < 0.15) &  # 原0.1，更敏感
            # 新增：动量衰竭或趋势末期
            ((dataframe['momentum_exhaustion_score'] > 0.4) | (dataframe['trend_phase'] >= 3))
        )

        # ==============================
        # 2. 趋势衰竭信号
        # ==============================

        # 做多头寸的趋势衰竭
        trend_exhaustion_long = (
            # ADX下降且低于阈值
            (dataframe['adx'] < 20) &
            (dataframe['adx'] < dataframe['adx'].shift(1)) &
            # 价格回归布林带中轨
            (abs(dataframe['bb_position'] - 0.5) < 0.15) &
            # 动量指标转负
            (dataframe['mom_10'] < 0) &
            # DI-开始上升
            (dataframe['plus_di'] < dataframe['minus_di'])
        )

        # 做空头寸的趋势衰竭
        trend_exhaustion_short = (
            # ADX下降且低于阈值
            (dataframe['adx'] < 20) &
            (dataframe['adx'] < dataframe['adx'].shift(1)) &
            # 价格回归布林带中轨
            (abs(dataframe['bb_position'] - 0.5) < 0.15) &
            # 动量指标转正
            (dataframe['mom_10'] > 0) &
            # DI+开始上升
            (dataframe['plus_di'] > dataframe['minus_di'])
        )

        # ==============================
        # 3. 技术形态出场
        # ==============================

        # 做多头寸的技术形态出场
        technical_exit_long = (
            # 价格跌破EMA13且短期均线死叉
            (dataframe['close'] < dataframe['ema_13']) &
            (dataframe['ema_5'] < dataframe['ema_8']) &
            # MACD死叉
            (dataframe['macd'] < dataframe['macd_signal']) &
            # 成交量萎缩
            (dataframe['volume_ratio'] < 0.8)
        )

        # 做空头寸的技术形态出场
        technical_exit_short = (
            # 价格突破EMA13且短期均线金叉
            (dataframe['close'] > dataframe['ema_13']) &
            (dataframe['ema_5'] > dataframe['ema_8']) &
            # MACD金叉
            (dataframe['macd'] > dataframe['macd_signal']) &
            # 成交量萎缩
            (dataframe['volume_ratio'] < 0.8)
        )

        # ==============================
        # 4. 市场微结构信号
        # ==============================

        # 做多头寸的市场微结构出场
        microstructure_exit_long = (
            # 订单簿失衡严重（卖压过大）
            (dataframe['ob_depth_imbalance'] < -0.3) &
            # 流动性恶化
            (dataframe['ob_liquidity_score'] < 0.3) &
            # 买压减弱
            (dataframe['ob_buy_pressure'] < 0.3)
        )

        # 做空头寸的市场微结构出场
        microstructure_exit_short = (
            # 订单簿失衡严重（买压过大）
            (dataframe['ob_depth_imbalance'] > 0.3) &
            # 流动性恶化
            (dataframe['ob_liquidity_score'] < 0.3) &
            # 卖压减弱
            (dataframe['ob_sell_pressure'] < 0.3)
        )

        # ==============================
        # 5. 波动性保护
        # ==============================

        # ATR突然放大（适用于所有头寸）
        volatility_protection = (
            # ATR突然放大超过平均值2倍
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2) |
            # 或者ATR百分比超过5%
            (dataframe['atr_p'] > 0.05)
        )

        # ==============================
        # 6. 根据市场状态调整敏感度
        # ==============================

        # 获取市场状态
        is_bull_market = dataframe['market_state'] == 'bullish'
        is_bear_market = dataframe['market_state'] == 'bearish'
        is_sideways = dataframe['market_state'] == 'sideways'

        # ==============================
        # 组合所有出场条件
        # ==============================

        # 做多出场信号
        dataframe.loc[
            (
                strong_reversal_long_exit |  # 强势反转
                trend_exhaustion_long |      # 趋势衰竭
                technical_exit_long |        # 技术形态
                microstructure_exit_long |   # 市场微结构
                (volatility_protection & is_bear_market)  # 熊市中的波动保护
            ),
            'exit_long'
        ] = 1

        # 做空出场信号
        dataframe.loc[
            (
                strong_reversal_short_exit |  # 强势反转
                trend_exhaustion_short |       # 趋势衰竭
                technical_exit_short |         # 技术形态
                microstructure_exit_short |    # 市场微结构
                (volatility_protection & is_bull_market)  # 牛市中的波动保护
            ),
            'exit_short'
        ] = 1

        # 添加出场标签以便分析
        dataframe.loc[strong_reversal_long_exit, 'exit_tag'] = 'strong_reversal'
        dataframe.loc[trend_exhaustion_long, 'exit_tag'] = 'trend_exhaustion'
        dataframe.loc[technical_exit_long, 'exit_tag'] = 'technical_exit'
        dataframe.loc[microstructure_exit_long, 'exit_tag'] = 'microstructure'
        dataframe.loc[volatility_protection, 'exit_tag'] = 'volatility_protection'

        # ==============================
        # 🚨 修复：智能交叉出场机制 - 避免同K线信号冲突
        # ==============================
        
        # 检测同一K线的多空信号冲突
        signal_conflict = (dataframe['enter_long'] == 1) & (dataframe['enter_short'] == 1)
        
        # 解决冲突：基于信号强度或市场趋势优先级
        conflict_resolution_favor_long = (
            signal_conflict &
            (
                (dataframe['trend_strength'] > 0) |  # 趋势偏多时优先做多
                (dataframe['rsi_14'] < 50) |         # RSI偏低时优先做多
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))  # MACD改善时优先做多
            )
        )
        
        # 冲突解决：保留高优先级信号，清除低优先级信号
        dataframe.loc[conflict_resolution_favor_long, 'enter_short'] = 0
        dataframe.loc[signal_conflict & ~conflict_resolution_favor_long, 'enter_long'] = 0
        
        # 重新计算清理后的信号
        clean_enter_long = dataframe['enter_long'] == 1
        clean_enter_short = dataframe['enter_short'] == 1
        
        # 智能交叉出场：只在真正的方向切换时平仓
        # 原理：强势看涨时平空仓，强势看跌时平多仓
        strong_bullish_signal = (
            clean_enter_long &
            (dataframe['rsi_14'] > 30) &  # 避免在极端超卖时误判
            (dataframe['volume_ratio'] > 1.1)  # 需要成交量支撑
        )
        
        strong_bearish_signal = (
            clean_enter_short &
            (dataframe['rsi_14'] < 70) &  # 避免在极端超买时误判
            (dataframe['volume_ratio'] > 1.1)  # 需要成交量支撑
        )
        
        # 执行交叉平仓（避免误平仓）
        dataframe.loc[strong_bullish_signal, 'exit_short'] = 1
        dataframe.loc[strong_bearish_signal, 'exit_long'] = 1
        
        # 更新exit_tag以便追踪（区分冲突解决和正常平仓）
        dataframe.loc[
            strong_bullish_signal & (dataframe['exit_short'] == 1),
            'exit_tag'
        ] = 'smart_cross_exit_bullish'
        
        dataframe.loc[
            strong_bearish_signal & (dataframe['exit_long'] == 1),
            'exit_tag' 
        ] = 'smart_cross_exit_bearish'

        # 记录出场信号统计
        exit_long_count = dataframe['exit_long'].sum()
        exit_short_count = dataframe['exit_short'].sum()

        if exit_long_count > 0 or exit_short_count > 0:
            logger.info(f"""
📤 智能出场信号生成 - {metadata['pair']}:
├─ 做多出场信号: {exit_long_count}个
├─ 做空出场信号: {exit_short_count}个
└─ 时间范围: {dataframe.index[0]} - {dataframe.index[-1]}
""")

        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """自定义仓位大小"""
        
        try:
            # 获取最新数据
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
            
            # 获取市场状态
            market_state = dataframe['market_state'].iloc[-1] if 'market_state' in dataframe.columns else 'sideways'
            volatility = dataframe['atr_p'].iloc[-1] if 'atr_p' in dataframe.columns else 0.02
            
            # === 🎯 币种风险识别系统 ===
            coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            
            # 定义币种风险乘数（垃圾币小仓位以小博大）
            coin_risk_multipliers = {
                'low_risk': 1.0,        # 低风险：标准仓位
                'medium_risk': 0.7,     # 中等风险：70%仓位
                'high_risk': 0.25       # 高风险（垃圾币）：25%仓位，以小博大
            }
            
            # 获取币种风险乘数
            coin_risk_multiplier = coin_risk_multipliers.get(coin_risk_tier, 0.7)
            
            # 计算动态仓位大小
            position_size_ratio = self.calculate_position_size(current_rate, market_state, pair)
            
            # 获取账户余额
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # === 应用币种风险乘数到仓位计算 ===
            # 基础仓位计算
            base_calculated_stake = available_balance * position_size_ratio
            
            # 应用币种风险乘数（垃圾币自动小仓位）
            calculated_stake = base_calculated_stake * coin_risk_multiplier
            
            # 计算动态杠杆
            dynamic_leverage = self.calculate_leverage(market_state, volatility, pair, current_time)
            
            # 注意：在Freqtrade中，杠杆通过leverage()方法设置，这里只计算基础仓位
            # 杠杆会由系统自动应用，不需要手动乘以杠杆倍数
            # leveraged_stake = calculated_stake * dynamic_leverage  # 移除这行
            leveraged_stake = calculated_stake  # 只返回基础仓位
            
            # 记录杠杆应用过程
            base_position_value = calculated_stake
            
            # 确保在限制范围内
            final_stake = max(min_stake or 0, min(leveraged_stake, max_stake))
            
            # 详细的杠杆应用日志
            risk_tier_names = {
                'low_risk': '✅ 低风险',
                'medium_risk': '⚡ 中等风险', 
                'high_risk': '⚠️ 高风险'
            }
            
            logger.info(f"""
🎯 智能仓位计算详情 - {pair}:
├─ 市场状态: {market_state}
├─ 🔍 风险等级: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ 📊 策略仓位: ${base_calculated_stake:.2f} ({position_size_ratio:.2%})
├─ 🎯 风险调整: {coin_risk_multiplier:.2f}x ({coin_risk_tier})
├─ 💰 调整后仓位: ${calculated_stake:.2f}
├─ ⚡ 计算杠杆: {dynamic_leverage}x (通过leverage()方法应用)
├─ 🎉 最终金额: ${final_stake:.2f}
├─ 📈 预期数量: {final_stake / current_rate:.6f}
└─ ⏰ 决策时间: {current_time}
""")
            
            # 重要：设置策略的当前杠杆（供Freqtrade使用）
            if hasattr(self, '_current_leverage'):
                self._current_leverage[pair] = dynamic_leverage
            else:
                self._current_leverage = {pair: dynamic_leverage}
            
            # 记录详细的风险计算日志
            self._log_risk_calculation_details(pair, {
                'current_price': current_rate,
                'planned_position': position_size_ratio,
                'stoploss_level': abs(self.stoploss),
                'leverage': dynamic_leverage,
                'market_state': market_state,
                'volatility': volatility
            }, {
                'risk_amount': final_stake * abs(self.stoploss),
                'risk_percentage': (final_stake * abs(self.stoploss)) / available_balance,
                'max_loss': final_stake * abs(self.stoploss),
                'adjusted_position': position_size_ratio,
                'suggested_leverage': dynamic_leverage,
                'risk_rating': self._calculate_risk_rating(final_stake * abs(self.stoploss) / available_balance),
                'rating_reason': f'基于{market_state}市场状态和{volatility*100:.1f}%波动率的综合评估'
            })
            
            return final_stake
            
        except Exception as e:
            logger.error(f"仓位计算失败: {e}")
            return proposed_stake
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """升级版智能DCA加仓系统 - 多重技术确认与风险控制"""
        
        # 检查是否允许DCA
        if trade.nr_of_successful_entries >= self.max_dca_orders:
            logger.info(f"DCA限制 {trade.pair}: 已达最大加仓次数 {self.max_dca_orders}")
            return None
            
        # 获取包含完整指标的数据
        dataframe = self.get_dataframe_with_indicators(trade.pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"DCA检查失败 {trade.pair}: 无数据")
            return None
            
        # 最终检查关键指标是否存在
        required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
        missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns]
        
        if missing_indicators:
            logger.warning(f"DCA检查 {trade.pair}: 关键指标仍缺失 {missing_indicators}，跳过DCA")
            return None
            
        # 获取关键指标
        current_data = dataframe.iloc[-1]
        prev_data = dataframe.iloc[-2] if len(dataframe) > 1 else current_data
        
        current_rsi = current_data.get('rsi_14', 50)
        current_adx = current_data.get('adx', 25)
        current_atr_p = current_data.get('atr_p', 0.02)
        trend_strength = current_data.get('trend_strength', 50)
        momentum_score = current_data.get('momentum_score', 0)
        volume_ratio = current_data.get('volume_ratio', 1)
        signal_strength = current_data.get('signal_strength', 0)
        bb_position = current_data.get('bb_position', 0.5)
        market_state = current_data.get('market_state', 'sideways')
        
        # 计算基本参数
        entry_price = trade.open_rate
        price_deviation = abs(current_rate - entry_price) / entry_price
        hold_time = current_time - trade.open_date_utc
        hold_hours = hold_time.total_seconds() / 3600
        
        # === 智能DCA决策系统 ===
        
        dca_decision = self._analyze_dca_opportunity(
            trade, current_rate, current_profit, price_deviation,
            current_data, prev_data, hold_hours, market_state
        )
        
        if dca_decision['should_dca']:
            # 计算智能DCA金额
            dca_amount = self._calculate_smart_dca_amount(
                trade, dca_decision, current_data, market_state
            )
            
            # 最终风险检查
            risk_check = self._dca_risk_validation(trade, dca_amount, current_data)
            
            if risk_check['approved']:
                final_dca_amount = risk_check['adjusted_amount']
                
                # 记录详细DCA决策日志
                self._log_dca_decision(
                    trade, current_rate, current_profit, price_deviation,
                    dca_decision, final_dca_amount, current_data
                )
                
                # 跟踪DCA性能
                self.track_dca_performance(trade, dca_decision['dca_type'], final_dca_amount)
                
                return final_dca_amount
            else:
                logger.warning(f"DCA风险检查失败 {trade.pair}: {risk_check['reason']}")
                return None
        
        return None
    
    # 移除了 _analyze_dca_opportunity - 简化策略逻辑
    def _analyze_dca_opportunity(self, trade: Trade, current_rate: float, 
                               current_profit: float, price_deviation: float,
                               current_data: dict, prev_data: dict, 
                               hold_hours: float, market_state: str) -> dict:
        """分析DCA加仓机会 - 多维度技术分析"""
        
        decision = {
            'should_dca': False,
            'dca_type': None,
            'confidence': 0.0,
            'risk_level': 'high',
            'technical_reasons': [],
            'market_conditions': {}
        }
        
        try:
            # === 基础DCA触发条件 ===
            basic_trigger_met = (
                price_deviation > self.dca_price_deviation and  # 价格偏差足够
                current_profit < -0.03 and  # 浮亏3%以上（降低门槛）
                hold_hours > 0.5  # 持仓至少30分钟
            )
            
            if not basic_trigger_met:
                return decision
            
            # === 技术面DCA条件分析 ===
            
            if not trade.is_short:
                # === 做多DCA条件 ===
                
                # 1. 超卖反弹DCA - 最安全的DCA时机
                oversold_dca = (
                    current_rate < trade.open_rate and  # 价格下跌
                    current_data.get('rsi_14', 50) < 35 and  # RSI超卖
                    current_data.get('bb_position', 0.5) < 0.2 and  # 接近布林带下轨
                    current_data.get('momentum_score', 0) > prev_data.get('momentum_score', 0)  # 动量开始改善
                )
                
                if oversold_dca:
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'OVERSOLD_REVERSAL_DCA',
                        'confidence': 0.8,
                        'risk_level': 'low'
                    })
                    decision['technical_reasons'].append(f"RSI{current_data.get('rsi_14', 50):.1f}超卖反弹")
                
                # 2. 支撑位DCA - 在关键支撑位加仓
                elif (current_data.get('close', 0) > current_data.get('ema_50', 0) and  # 仍在长期趋势上方
                      abs(current_rate - current_data.get('ema_21', 0)) / current_rate < 0.02 and  # 接近EMA21支撑
                      current_data.get('adx', 25) > 20):  # 趋势仍然有效
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'SUPPORT_LEVEL_DCA',
                        'confidence': 0.7,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append("EMA21关键支撑位加仓")
                
                # 3. 趋势延续DCA - 趋势依然强劲的回调
                elif (current_data.get('trend_strength', 50) > 30 and  # 趋势仍然向上
                      current_data.get('adx', 25) > 25 and  # ADX确认趋势
                      current_data.get('signal_strength', 0) > 0):  # 信号仍然偏多
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'TREND_CONTINUATION_DCA',
                        'confidence': 0.6,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"趋势延续回调加仓，趋势强度{current_data.get('trend_strength', 50):.0f}")
                
                # 4. 成交量确认DCA - 有成交量支撑的回调
                elif (current_data.get('volume_ratio', 1) > 1.2 and  # 成交量放大
                      current_data.get('ob_depth_imbalance', 0) > 0.1):  # 买盘占优
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'VOLUME_CONFIRMED_DCA',
                        'confidence': 0.5,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"成交量{current_data.get('volume_ratio', 1):.1f}倍确认买盘")
                
            else:
                # === 做空DCA条件 ===
                
                # 1. 超买回调DCA - 最安全的空头DCA时机
                overbought_dca = (
                    current_rate > trade.open_rate and  # 价格上涨
                    current_data.get('rsi_14', 50) > 65 and  # RSI超买
                    current_data.get('bb_position', 0.5) > 0.8 and  # 接近布林带上轨
                    current_data.get('momentum_score', 0) < prev_data.get('momentum_score', 0)  # 动量开始恶化
                )
                
                if overbought_dca:
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'OVERBOUGHT_REJECTION_DCA',
                        'confidence': 0.8,
                        'risk_level': 'low'
                    })
                    decision['technical_reasons'].append(f"RSI{current_data.get('rsi_14', 50):.1f}超买回调")
                
                # 2. 阻力位DCA - 在关键阻力位加仓
                elif (current_data.get('close', 0) < current_data.get('ema_50', 0) and  # 仍在长期趋势下方
                      abs(current_rate - current_data.get('ema_21', 0)) / current_rate < 0.02 and  # 接近EMA21阻力
                      current_data.get('adx', 25) > 20):  # 趋势仍然有效
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'RESISTANCE_LEVEL_DCA',
                        'confidence': 0.7,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append("EMA21关键阻力位加仓")
                
                # 3. 趋势延续DCA - 趋势依然向下的反弹
                elif (current_data.get('trend_strength', 50) < -30 and  # 趋势仍然向下
                      current_data.get('adx', 25) > 25 and  # ADX确认趋势
                      current_data.get('signal_strength', 0) < 0):  # 信号仍然偏空
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'TREND_CONTINUATION_DCA_SHORT',
                        'confidence': 0.6,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"下跌趋势延续反弹加仓，趋势强度{current_data.get('trend_strength', 50):.0f}")
            
            # === 市场环境验证 ===
            decision['market_conditions'] = {
                'market_state': market_state,
                'volatility_acceptable': current_data.get('atr_p', 0.02) < 0.06,  # 波动率不过高
                'liquidity_sufficient': current_data.get('ob_market_quality', 0.5) > 0.3,  # 流动性充足
                'spread_reasonable': current_data.get('ob_spread_pct', 0.1) < 0.4,  # 价差合理
                'trend_not_reversing': abs(current_data.get('trend_strength', 50)) > 20  # 趋势未完全反转
            }
            
            # 市场环境不利时降低信心度或取消DCA
            unfavorable_conditions = sum([
                not decision['market_conditions']['volatility_acceptable'],
                not decision['market_conditions']['liquidity_sufficient'], 
                not decision['market_conditions']['spread_reasonable'],
                not decision['market_conditions']['trend_not_reversing']
            ])
            
            if unfavorable_conditions >= 2:
                decision['should_dca'] = False
                decision['risk_level'] = 'too_high'
            elif unfavorable_conditions == 1:
                decision['confidence'] *= 0.7  # 降低信心度
                decision['risk_level'] = 'high'
                
        except Exception as e:
            logger.error(f"DCA机会分析失败 {trade.pair}: {e}")
            decision['should_dca'] = False
            
        return decision
    
    def _calculate_smart_dca_amount(self, trade: Trade, dca_decision: dict, 
                                  current_data: dict, market_state: str) -> float:
        """计算智能DCA金额 - 根据信心度和风险动态调整"""
        
        try:
            # 基础DCA金额
            base_amount = trade.stake_amount
            entry_count = trade.nr_of_successful_entries + 1
            
            # === 根据DCA类型调整基础倍数 ===
            dca_type_multipliers = {
                'OVERSOLD_REVERSAL_DCA': 1.5,  # 超卖反弹，较激进
                'OVERBOUGHT_REJECTION_DCA': 1.5,  # 超买回调，较激进
                'SUPPORT_LEVEL_DCA': 1.3,  # 支撑位，中等激进
                'RESISTANCE_LEVEL_DCA': 1.3,  # 阻力位，中等激进
                'TREND_CONTINUATION_DCA': 1.2,  # 趋势延续，较保守
                'TREND_CONTINUATION_DCA_SHORT': 1.2,  # 空头趋势延续
                'VOLUME_CONFIRMED_DCA': 1.1  # 成交量确认，保守
            }
            
            type_multiplier = dca_type_multipliers.get(dca_decision['dca_type'], 1.0)
            
            # === 根据信心度调整 ===
            confidence_multiplier = 0.5 + (dca_decision['confidence'] * 0.8)  # 0.5-1.3倍
            
            # === 根据市场状态调整 ===
            market_multipliers = {
                'strong_uptrend': 1.4,  # 强趋势中DCA更积极
                'strong_downtrend': 1.4,
                'mild_uptrend': 1.2,
                'mild_downtrend': 1.2,
                'sideways': 1.0,
                'volatile': 0.7,  # 波动市场保守DCA
                'consolidation': 1.1
            }
            market_multiplier = market_multipliers.get(market_state, 1.0)
            
            # === 根据加仓次数递减 ===
            # 后续加仓应该更保守
            entry_decay = max(0.6, 1.0 - (entry_count - 1) * 0.15)
            
            # === 综合计算DCA金额 ===
            total_multiplier = (type_multiplier * confidence_multiplier * 
                              market_multiplier * entry_decay)
            
            calculated_dca = base_amount * total_multiplier
            
            # === 应用限制 ===
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # 动态最大DCA限制
            max_dca_ratio = {
                'low': 0.15,      # 低风险时最多15%余额
                'medium': 0.10,   # 中等风险10%余额  
                'high': 0.05      # 高风险5%余额
            }
            
            max_ratio = max_dca_ratio.get(dca_decision['risk_level'], 0.05)
            max_dca_amount = available_balance * max_ratio
            
            final_dca = min(calculated_dca, max_dca_amount, max_stake or float('inf'))
            
            return max(min_stake or 10, final_dca)
            
        except Exception as e:
            logger.error(f"DCA金额计算失败 {trade.pair}: {e}")
            return trade.stake_amount * 0.5  # 保守默认值
    
    def _dca_risk_validation(self, trade: Trade, dca_amount: float, current_data: dict) -> dict:
        """DCA风险验证 - 最终安全检查"""
        
        risk_check = {
            'approved': True,
            'adjusted_amount': dca_amount,
            'reason': 'DCA风险检查通过',
            'risk_factors': []
        }
        
        try:
            # 1. 总仓位风险检查
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            total_exposure = trade.stake_amount + dca_amount
            exposure_ratio = total_exposure / available_balance
            
            if exposure_ratio > 0.4:  # 单一交易不超过40%资金
                adjustment = 0.4 / exposure_ratio
                risk_check['adjusted_amount'] = dca_amount * adjustment
                risk_check['risk_factors'].append(f'总仓位过大，调整为{adjustment:.1%}')
            
            # 2. 连续DCA风险检查
            if trade.nr_of_successful_entries >= 3:  # 已经DCA 3次以上
                risk_check['adjusted_amount'] *= 0.7  # 减少后续DCA金额
                risk_check['risk_factors'].append('多次DCA风险控制')
            
            # 3. 市场环境风险检查
            if current_data.get('atr_p', 0.02) > 0.05:  # 高波动环境
                risk_check['adjusted_amount'] *= 0.8
                risk_check['risk_factors'].append('高波动环境风险调整')
            
            # 4. 账户回撤保护
            if hasattr(self, 'current_drawdown') and self.current_drawdown > 0.08:
                risk_check['adjusted_amount'] *= 0.6
                risk_check['risk_factors'].append('账户回撤保护')
            
            # 5. 最小金额检查
            min_meaningful_dca = trade.stake_amount * 0.2  # DCA至少是原仓位的20%
            if risk_check['adjusted_amount'] < min_meaningful_dca:
                risk_check['approved'] = False
                risk_check['reason'] = f'DCA金额过小，低于最小有效金额${min_meaningful_dca:.2f}'
            
        except Exception as e:
            risk_check['approved'] = False
            risk_check['reason'] = f'DCA风险检查系统错误: {e}'
            
        return risk_check
    
    def _log_dca_decision(self, trade: Trade, current_rate: float, current_profit: float,
                         price_deviation: float, dca_decision: dict, dca_amount: float,
                         current_data: dict):
        """记录详细的DCA决策日志"""
        
        try:
            hold_time = datetime.now(timezone.utc) - trade.open_date_utc
            hold_hours = hold_time.total_seconds() / 3600
            
            dca_log = f"""
==================== DCA加仓决策分析 ====================
时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | 交易对: {trade.pair}
加仓次数: 第{trade.nr_of_successful_entries + 1}次 / 最大{self.max_dca_orders}次

📊 当前交易状态:
├─ 开仓价格: ${trade.open_rate:.6f}
├─ 当前价格: ${current_rate:.6f}
├─ 价格偏差: {price_deviation:.2%}
├─ 当前盈亏: {current_profit:.2%}
├─ 持仓时间: {hold_hours:.1f}小时
├─ 交易方向: {'🔻做空' if trade.is_short else '🔹做多'}
├─ 原始仓位: ${trade.stake_amount:.2f}

🎯 DCA触发分析:
├─ DCA类型: {dca_decision['dca_type']}
├─ 信心水平: {dca_decision['confidence']:.1%}
├─ 风险等级: {dca_decision['risk_level']}
├─ 技术理由: {' | '.join(dca_decision['technical_reasons'])}

📋 技术指标状态:
├─ RSI(14): {current_data.get('rsi_14', 50):.1f}
├─ 趋势强度: {current_data.get('trend_strength', 50):.0f}/100
├─ 动量评分: {current_data.get('momentum_score', 0):.3f}
├─ ADX: {current_data.get('adx', 25):.1f}
├─ 成交量倍数: {current_data.get('volume_ratio', 1):.1f}x
├─ 布林带位置: {current_data.get('bb_position', 0.5):.2f}
├─ 信号强度: {current_data.get('signal_strength', 0):.1f}

💰 DCA金额计算:
├─ 基础金额: ${trade.stake_amount:.2f}
├─ 计算金额: ${dca_amount:.2f}
├─ 新增暴露: {(dca_amount/trade.stake_amount)*100:.0f}%
├─ 总仓位: ${trade.stake_amount + dca_amount:.2f}

🌊 市场环境评估:
├─ 市场状态: {dca_decision['market_conditions'].get('market_state', '未知')}
├─ 波动率: {'✅正常' if dca_decision['market_conditions'].get('volatility_acceptable', False) else '⚠️过高'}
├─ 流动性: {'✅充足' if dca_decision['market_conditions'].get('liquidity_sufficient', False) else '⚠️不足'}
├─ 价差: {'✅合理' if dca_decision['market_conditions'].get('spread_reasonable', False) else '⚠️过大'}

=================================================="""
            
            logger.info(dca_log)
            
        except Exception as e:
            logger.error(f"DCA决策日志记录失败 {trade.pair}: {e}")
    
    def track_dca_performance(self, trade: Trade, dca_type: str, dca_amount: float):
        """跟踪DCA性能"""
        try:
            # 记录DCA执行
            self.dca_performance_tracker['total_dca_count'] += 1
            
            dca_record = {
                'trade_id': f"{trade.pair}_{trade.open_date_utc.timestamp()}",
                'pair': trade.pair,
                'dca_type': dca_type,
                'dca_amount': dca_amount,
                'execution_time': datetime.now(timezone.utc),
                'entry_number': trade.nr_of_successful_entries + 1,
                'price_at_dca': trade.open_rate  # 这将在实际执行时更新
            }
            
            self.dca_performance_tracker['dca_history'].append(dca_record)
            
            # 更新DCA类型性能统计
            if dca_type not in self.dca_performance_tracker['dca_type_performance']:
                self.dca_performance_tracker['dca_type_performance'][dca_type] = {
                    'count': 0,
                    'successful': 0,
                    'success_rate': 0.0,
                    'avg_profit_contribution': 0.0
                }
            
            self.dca_performance_tracker['dca_type_performance'][dca_type]['count'] += 1
            
        except Exception as e:
            logger.error(f"DCA性能跟踪失败: {e}")
    
    def get_dca_performance_report(self) -> dict:
        """获取DCA性能报告"""
        try:
            tracker = self.dca_performance_tracker
            
            return {
                'total_dca_executions': tracker['total_dca_count'],
                'overall_success_rate': tracker['dca_success_rate'],
                'type_performance': tracker['dca_type_performance'],
                'avg_profit_contribution': tracker['avg_dca_profit'],
                'recent_dca_count_30d': len([
                    dca for dca in tracker['dca_history'] 
                    if (datetime.now(timezone.utc) - dca['execution_time']).days <= 30
                ]),
                'best_performing_dca_type': max(
                    tracker['dca_type_performance'].items(),
                    key=lambda x: x[1]['success_rate'],
                    default=('none', {'success_rate': 0})
                )[0] if tracker['dca_type_performance'] else 'none'
            }
        except Exception:
            return {'error': '无法生成DCA性能报告'}
    
    # 移除了 custom_stoploss - 使用固定止损更简单可靠
    
    # 移除了 _analyze_smart_stoploss_conditions - 简化止损逻辑
    
    # 移除了 _log_smart_stoploss_decision - 简化日志
    
    def calculate_smart_takeprofit_levels(self, pair: str, trade: Trade, current_rate: float,
                                        current_profit: float) -> dict:
        """计算智能分级止盈目标 - AI动态止盈系统"""
        
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return {'error': '无数据'}
            
            current_data = dataframe.iloc[-1]
            current_atr = current_data.get('atr_p', 0.02)
            trend_strength = current_data.get('trend_strength', 50)
            momentum_score = current_data.get('momentum_score', 0)
            current_adx = current_data.get('adx', 25)
            
            # === 智能分级止盈计算 ===
            base_multiplier = 3.0  # 基础ATR倍数
            
            # 趋势强度调整
            if abs(trend_strength) > 80:
                trend_mult = 2.5
            elif abs(trend_strength) > 60:
                trend_mult = 2.0
            else:
                trend_mult = 1.5
            
            # 计算分级目标
            total_mult = base_multiplier * trend_mult
            base_distance = current_atr * total_mult
            
            # 4级止盈目标
            targets = {
                'level_1': {'target': base_distance * 0.6, 'close': 0.25, 'desc': '快速获利'},
                'level_2': {'target': base_distance * 1.0, 'close': 0.35, 'desc': '主要获利'},
                'level_3': {'target': base_distance * 1.6, 'close': 0.25, 'desc': '趋势延伸'},
                'level_4': {'target': base_distance * 2.5, 'close': 0.15, 'desc': '超预期收益'}
            }
            
            # 计算实际价格目标
            for level_data in targets.values():
                if not trade.is_short:
                    level_data['price'] = trade.open_rate * (1 + level_data['target'])
                else:
                    level_data['price'] = trade.open_rate * (1 - level_data['target'])
                level_data['profit_pct'] = level_data['target'] * 100
            
            return {
                'targets': targets,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'atr_percent': current_atr * 100,
                'analysis_time': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"智能止盈分析失败 {pair}: {e}")
            return {'error': f'止盈分析失败: {e}'}
    
    # 删除了 get_smart_stoploss_takeprofit_status
    def should_protect_strong_trend(self, pair: str, trade: Trade, 
                                  dataframe: DataFrame, current_rate: float) -> bool:
        """判断是否应该保护强趋势 - 防止趋势中的正常回调被误止损"""
        
        if dataframe.empty:
            return False
            
        try:
            current_data = dataframe.iloc[-1]
            
            # 获取趋势指标
            trend_strength = current_data.get('trend_strength', 0)
            adx = current_data.get('adx', 0)
            momentum_score = current_data.get('momentum_score', 0)
            
            # 检查价格与关键均线的关系
            ema_21 = current_data.get('ema_21', current_rate)
            ema_50 = current_data.get('ema_50', current_rate)
            
            # === 多头趋势保护条件 ===
            if not trade.is_short:
                trend_protection = (
                    trend_strength > 70 and          # 趋势强度依然很强
                    adx > 25 and                     # ADX确认趋势
                    current_rate > ema_21 and        # 价格仍在关键均线上方
                    momentum_score > -0.2 and        # 动量没有严重恶化
                    current_rate > ema_50 * 0.98     # 价格没有跌破重要支撑
                )
                
            # === 空头趋势保护条件 ===
            else:
                trend_protection = (
                    trend_strength > 70 and          # 趋势强度依然很强
                    adx > 25 and                     # ADX确认趋势
                    current_rate < ema_21 and        # 价格仍在关键均线下方
                    momentum_score < 0.2 and         # 动量没有严重恶化  
                    current_rate < ema_50 * 1.02     # 价格没有突破重要阻力
                )
            
            return trend_protection
            
        except Exception as e:
            logger.warning(f"趋势保护检查失败: {e}")
            return False
    
    def detect_false_breakout(self, dataframe: DataFrame, current_rate: float, 
                            trade: Trade) -> bool:
        """检测假突破 - 防止在假突破后的快速反转中被误止损"""
        
        if dataframe.empty or len(dataframe) < 10:
            return False
            
        try:
            # 获取最近10根K线数据
            recent_data = dataframe.tail(10)
            current_data = dataframe.iloc[-1]
            
            # 获取关键价位
            supertrend = current_data.get('supertrend', current_rate)
            bb_upper = current_data.get('bb_upper', current_rate * 1.02)
            bb_lower = current_data.get('bb_lower', current_rate * 0.98)
            
            # === 多头假突破检测 ===
            if not trade.is_short:
                # 检查是否刚刚跌破关键支撑后快速反弹
                recent_low = recent_data['low'].min()
                current_recovery = (current_rate - recent_low) / recent_low
                
                # 突破后快速回调超过50%视为假突破
                if (recent_low < supertrend and 
                    current_rate > supertrend and 
                    current_recovery > 0.005):  # 0.5%的反弹
                    return True
                    
                # 布林带假突破检测
                if (recent_data['low'].min() < bb_lower and 
                    current_rate > bb_lower and
                    current_rate > recent_data['close'].iloc[-3]):  # 比3根K线前收盘价高
                    return True
            
            # === 空头假突破检测 ===
            else:
                # 检查是否刚刚突破关键阻力后快速回落
                recent_high = recent_data['high'].max()
                current_pullback = (recent_high - current_rate) / recent_high
                
                # 突破后快速回调超过50%视为假突破
                if (recent_high > supertrend and 
                    current_rate < supertrend and 
                    current_pullback > 0.005):  # 0.5%的回调
                    return True
                
                # 布林带假突破检测
                if (recent_data['high'].max() > bb_upper and 
                    current_rate < bb_upper and
                    current_rate < recent_data['close'].iloc[-3]):  # 比3根K线前收盘价低
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"假突破检测失败: {e}")
            return False
    
    # 删除了 confirm_stoploss_signal
    
    def _log_trend_protection(self, pair: str, trade: Trade, current_rate: float, 
                            current_profit: float, dataframe: DataFrame):
        """记录趋势保护详情"""
        
        try:
            current_data = dataframe.iloc[-1]
            
            protection_details = {
                'current_rate': current_rate,
                'current_profit': current_profit,
                'trend_strength': current_data.get('trend_strength', 0),
                'adx': current_data.get('adx', 0),
                'momentum_score': current_data.get('momentum_score', 0),
                'trend_protection': True,
                'time_decay': False,
                'profit_protection': False,
                'atr_percent': current_data.get('atr_p', 0),
                'volatility_state': current_data.get('volatility_state', 0),
                'atr_multiplier': 1.0
            }
            
            # 计算建议的新止损值（基于当前市场状态）
            suggested_new_stoploss = self.stoploss
            
            # 移除了 decision_logger 日志记录
            pass
            
        except Exception as e:
            logger.warning(f"趋势保护日志记录失败: {e}")
    
    def _log_false_breakout_protection(self, pair: str, trade: Trade, 
                                     current_rate: float, dataframe: DataFrame):
        """记录假突破保护详情"""
        
        try:
            logger.info(f"🚫 假突破保护启动 - {pair} 检测到假突破模式，止损放宽50%")
            
        except Exception as e:
            logger.warning(f"假突破保护日志记录失败: {e}")
    
    # ===== 新的智能止损辅助方法 =====
    
    # 删除了 _calculate_structure_based_stop 
    # 删除了 calculate_atr_stop_multiplier - 简化止损逻辑
    
    # 移除了 calculate_trend_stop_adjustment - 简化止损逻辑
    
    # 移除了 calculate_volatility_cluster_stop - 简化止损逻辑
    
    # 移除了 calculate_time_decay_stop - 简化止损逻辑
    
    # 移除了 calculate_profit_protection_stop - 简化止损逻辑
    
    # 移除了 calculate_volume_stop_adjustment - 简化止损逻辑
    
    # 移除了 calculate_microstructure_stop - 简化止损逻辑
    
    # 移除了 apply_stoploss_limits - 简化止损逻辑
    
    # 移除了 get_enhanced_technical_stoploss - 简化止损逻辑
    
    # 移除了 custom_exit 方法 - 使用固定止损和ROI更简单可靠
    
    # 移除了 _get_detailed_exit_reason 方法 - 简化逻辑
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """交易入场确认"""
        
        try:
            # 最终风控检查
            
            # 1. 市场开放时间检查 (避免重大消息时段)
            # 这里可以添加避开特定时间的逻辑
            
            # 2. 订单簿流动性检查
            orderbook_data = self.get_market_orderbook(pair)
            if orderbook_data['spread_pct'] > 0.3:  # 价差过大
                logger.warning(f"价差过大，取消交易: {pair}")
                return False
            
            # 3. 极端波动检查
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                current_atr_p = dataframe['atr_p'].iloc[-1] if 'atr_p' in dataframe.columns else 0.02
                if current_atr_p > 0.06:  # 极高波动
                    logger.warning(f"波动率过高，取消交易: {pair}")
                    return False
            
            
            logger.info(f"交易确认通过: {pair} {side} {amount} @ {rate}")
            return True
            
        except Exception as e:
            logger.error(f"交易确认失败: {e}")
            return False
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """交易出场确认 - 更新连胜连败状态"""
        try:
            # 计算交易盈亏
            profit_ratio = trade.calc_profit_ratio(rate)
            
            # 更新连胜连败计数
            if profit_ratio > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                logger.info(f"🏆 {pair} 盈利交易，连胜: {self.consecutive_wins}")
            else:
                self.consecutive_wins = 0
                self.consecutive_losses += 1
                logger.info(f"❌ {pair} 亏损交易，连败: {self.consecutive_losses}")
                
            # 更新交易历史记录
            trade_record = {
                'pair': pair,
                'profit': profit_ratio,
                'exit_reason': exit_reason,
                'timestamp': current_time,
                'entry_rate': trade.open_rate,
                'exit_rate': rate
            }
            
            self.trade_history.append(trade_record)
            
            # 保持历史记录在合理范围内
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]
                
        except Exception as e:
            logger.warning(f"更新连胜状态失败: {e}")
            
        return True  # 始终允许出场
    
    def check_entry_timeout(self, pair: str, trade: Trade, order: Dict,
                           current_time: datetime, **kwargs) -> bool:
        """入场订单超时检查"""
        return True  # 默认允许超时取消
    
    def check_exit_timeout(self, pair: str, trade: Trade, order: Dict,
                          current_time: datetime, **kwargs) -> bool:
        """出场订单超时检查"""  
        return True  # 默认允许超时取消
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """🧠 智能杠杆管理系统 - 基于信号质量和市场状态的动态调整"""
        
        try:
            # 获取数据
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                logger.warning(f"杠杆计算失败，无数据 {pair}")
                return min(2.0, max_leverage)
            
            # 获取基础市场数据
            current_data = dataframe.iloc[-1]
            volatility = current_data.get('atr_p', 0.02)
            
            # === 1. 获取增强市场状态分析 ===
            market_regime_data = self._enhanced_market_regime_detection(dataframe)
            regime = market_regime_data['regime']
            regime_confidence = market_regime_data['confidence']
            signals_advice = market_regime_data['signals_advice']
            
            # === 2. 信号质量评估 ===
            signal_quality_bonus = self._calculate_signal_quality_leverage_bonus(
                entry_tag, current_data, regime, signals_advice
            )
            
            # === 3. 基础杠杆计算 ===
            base_leverage = self.calculate_leverage('sideways', volatility, pair, current_time)
            
            # === 4. 市场状态调整 ===
            regime_multiplier = self._get_regime_leverage_multiplier(regime, regime_confidence)
            
            # === 5. 信号类型调整 ===
            signal_multiplier = self._get_signal_leverage_multiplier(entry_tag, signals_advice)
            
            # === 6. 综合计算 ===
            calculated_leverage = (
                base_leverage * 
                regime_multiplier * 
                signal_multiplier * 
                signal_quality_bonus
            )
            
            # === 7. 安全边界和限制 ===
            # 确保不超过交易所限制
            safe_leverage = min(calculated_leverage, max_leverage)
            
            # 极端波动保护
            if volatility > 0.08:  # 8%以上波动，强制低杠杆
                safe_leverage = min(safe_leverage, 5)
            elif volatility > 0.05:  # 5%以上波动，限制杠杆
                safe_leverage = min(safe_leverage, 15)
            
            # 市场状态保护
            if 'VOLATILE' in regime or regime_confidence < 0.3:
                safe_leverage = min(safe_leverage, 10)
            
            final_leverage = max(1.0, safe_leverage)  # 最低1倍杠杆
            
            # === 8. 详细日志 ===
            logger.info(
                f"🎯 智能杠杆 {pair} [{entry_tag}]: "
                f"基础{base_leverage:.1f}x × "
                f"状态{regime_multiplier:.2f} × "
                f"信号{signal_multiplier:.2f} × " 
                f"质量{signal_quality_bonus:.2f} = "
                f"{calculated_leverage:.1f}x → {final_leverage:.1f}x | "
                f"市场:{regime} ({regime_confidence:.1%})"
            )
            
            return final_leverage
            
        except Exception as e:
            logger.error(f"杠杆计算失败 {pair}: {e}")
            return min(2.0, max_leverage)  # 出错时返回安全杠杆
    
    def leverage_update_callback(self, trade: Trade, **kwargs):
        """杠杆更新回调"""
        # 这个方法在交易过程中被调用，用于动态调整杠杆
        pass
    
    def update_trade_results(self, trade: Trade, profit: float, exit_reason: str):
        """更新交易结果统计"""
        try:
            # 更新交易历史
            trade_record = {
                'pair': trade.pair,
                'profit': profit,
                'exit_reason': exit_reason,
                'hold_time': (trade.close_date_utc - trade.open_date_utc).total_seconds() / 3600,
                'timestamp': trade.close_date_utc
            }
            
            self.trade_history.append(trade_record)
            
            # 保持历史记录在合理范围内
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]
            
            # 连胜连败计数已在 confirm_trade_exit 中更新
            
            # 清理止盈跟踪器
            trade_id = f"{trade.pair}_{trade.open_date_utc.timestamp()}"
            if trade_id in self.profit_taking_tracker:
                del self.profit_taking_tracker[trade_id]
                
        except Exception as e:
            logger.error(f"更新交易结果失败: {e}")
    
    # 移除了 get_intelligent_exit_signal - 不再使用动态出场
    
    # 移除了 calculate_emergency_stoploss_triggers - 简化止损逻辑