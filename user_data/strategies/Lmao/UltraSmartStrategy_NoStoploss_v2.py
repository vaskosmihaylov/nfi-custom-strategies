# === 核心库导入 ===
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import Counter
from typing import Optional, Dict, List, Tuple, Any
import logging

# === Freqtrade 导入 ===
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas_ta as pta  # 🔧 新增：pandas_ta用于多时间框架指标计算
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

# Structured logging helper to keep strategy messages compact and consistent.
class StrategyLogHelper:
    """Provide configurable logs for strategy events with lightweight formatting."""

    def __init__(self, base_logger: logging.Logger, prefix: str = "UltraSmart", verbosity: int = 0):
        self._logger = base_logger
        self._prefix = prefix
        self._verbosity = verbosity

    def set_verbosity(self, level: int) -> None:
        self._verbosity = level

    def debug(self, event: str, *, pair: str | None = None, **fields: Any) -> None:
        self._log(logging.DEBUG, event, pair, fields)

    def info(self, event: str, *, pair: str | None = None, importance: str = "verbose", **fields: Any) -> None:
        level = self._determine_level(logging.INFO, importance)
        self._log(level, event, pair, fields)

    def warning(self, event: str, *, pair: str | None = None, **fields: Any) -> None:
        self._log(logging.WARNING, event, pair, fields)

    def error(self, event: str, *, pair: str | None = None, **fields: Any) -> None:
        self._log(logging.ERROR, event, pair, fields)

    def critical(self, event: str, *, pair: str | None = None, **fields: Any) -> None:
        self._log(logging.CRITICAL, event, pair, fields)

    def _determine_level(self, base_level: int, importance: str) -> int:
        importance = (importance or "verbose").lower()
        if importance in ("critical", "always"):
            return base_level
        if importance in ("summary", "high"):
            return logging.INFO  # 强制输出所有日志
        if importance in ("verbose", "normal"):
            return logging.INFO  # 强制输出所有日志
        if importance == "debug":
            return logging.DEBUG
        return logging.INFO  # 强制输出所有日志

    def _log(self, level: int, event: str, pair: str | None, fields: Dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return

        parts: list[str] = [f"[{self._prefix}:{event.upper()}]"]
        if pair:
            parts.append(f"pair={pair}")

        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={self._format_value(value)}")

        self._logger.log(level, " | ".join(parts))

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            formatted = f"{value:.6f}"
            formatted = formatted.rstrip("0").rstrip(".")
            return formatted or "0"
        if isinstance(value, (list, tuple, set)):
            return "[" + ",".join(str(item) for item in value) + "]"
        if isinstance(value, dict):
            return "{" + ",".join(f"{k}:{v}" for k, v in value.items()) + "}"
        return str(value)

# 移除了 StrategyDecisionLogger 类 - 简化日志系统

class TradingStyleManager:
    """交易风格管理器 - 根据市场状态自动切换稳定/横盘/激进模式"""
    
    def __init__(self, log_callback=None, verbosity: int = 0):
        self.current_style = "stable"  # 默认稳定模式
        self.style_switch_cooldown = 0
        self.min_switch_interval = 0.5  # 最少30分钟才能切换一次 (提升响应速度)
        self._log_callback = log_callback
        self._verbosity = verbosity
        
        # === 稳定模式配置 ===
        self.STABLE_CONFIG = {
            'name': '稳定模式',
            'leverage_range': (5, 12),  # 大幅提升杠杆：5-12倍
            'position_range': (0.15, 0.35),  # 增大仓位：15-35%
            'entry_threshold': 6.5,  # 适度放宽入场要求
            'risk_per_trade': 0.025,  # 提升风险到2.5%
            'max_trades': 4,         # 增加并发交易从3到4
            'description': '平衡稳健，稳定收益与适度风险结合'
        }
        
        # === 横盘模式配置 ===  
        self.SIDEWAYS_CONFIG = {
            'name': '横盘模式',
            'leverage_range': (8, 15),  # 大幅提升杠杆：8-15倍
            'position_range': (0.20, 0.40),  # 增大仓位：20-40%
            'entry_threshold': 5.0,  # 适度放宽入场要求
            'risk_per_trade': 0.03, # 提升风险到3%
            'max_trades': 5,         # 增加并发交易从4到5
            'description': '积极震荡交易，快速进出，中高风险收益'
        }
        
        # === 激进模式配置 ===
        self.AGGRESSIVE_CONFIG = {
            'name': '激进模式',
            'leverage_range': (12, 25),  # 激进杠杆：12-25倍
            'position_range': (0.25, 0.50),  # 大仓位：25-50%
            'entry_threshold': 3.5,  # 更灵活的入场要求
  
            'risk_per_trade': 0.04,  # 提升风险到4%
            'max_trades': 8,         # 增加并发交易从6到8
            'description': '积极进取，追求高收益，高风险高回报'
        }
        
        self.style_configs = {
            'stable': self.STABLE_CONFIG,
            'sideways': self.SIDEWAYS_CONFIG,
            'aggressive': self.AGGRESSIVE_CONFIG
        }
        
    def _log_message(self, message: str, *, importance: str = "summary") -> None:
        if callable(self._log_callback):
            try:
                self._log_callback(message, importance=importance)
                return
            except Exception:
                logger.debug("TradingStyleManager 日志回调失败，退回默认logger")
        log_level = logging.INFO if importance in ("critical", "summary", "high") else logging.DEBUG
        logger.log(log_level, message)

    def get_current_config(self) -> dict:
        """获取当前风格配置"""
        return self.style_configs[self.current_style]
    
    def detect_reversal_warning(self, dataframe: DataFrame) -> Dict[str, Any]:
        """
        🚨 早期预警系统 - 检测趋势转向前兆

        返回预警信号和严重程度（0-100）
        """
        if dataframe.empty or len(dataframe) < 20:
            return {'warning_level': 0, 'signals': []}

        try:
            current = dataframe.iloc[-1]
            recent = dataframe.tail(10)

            warning_signals = []
            warning_score = 0

            # === 1. 动量连续衰减检测 ===
            if 'macd' in dataframe.columns and len(dataframe) >= 5:
                macd_values = dataframe['macd'].tail(5).values
                # 检查连续3根K线动量衰减
                if len(macd_values) >= 3:
                    bullish_momentum_decay = all(
                        macd_values[i] < macd_values[i-1]
                        for i in range(-3, 0)
                    ) and macd_values[-1] > 0

                    bearish_momentum_decay = all(
                        macd_values[i] > macd_values[i-1]
                        for i in range(-3, 0)
                    ) and macd_values[-1] < 0

                    if bullish_momentum_decay:
                        warning_signals.append('bullish_momentum_decay')
                        warning_score += 25
                    elif bearish_momentum_decay:
                        warning_signals.append('bearish_momentum_decay')
                        warning_score += 25

            # === 2. RSI背离检测（简化版）===
            if 'rsi_14' in dataframe.columns and len(dataframe) >= 10:
                price_trend = dataframe['close'].tail(10)
                rsi_trend = dataframe['rsi_14'].tail(10)

                # 价格创新高但RSI未创新高（顶背离）
                if price_trend.iloc[-1] > price_trend.iloc[-5]:
                    if rsi_trend.iloc[-1] < rsi_trend.iloc[-5]:
                        warning_signals.append('bearish_divergence')
                        warning_score += 30

                # 价格创新低但RSI未创新低（底背离）
                elif price_trend.iloc[-1] < price_trend.iloc[-5]:
                    if rsi_trend.iloc[-1] > rsi_trend.iloc[-5]:
                        warning_signals.append('bullish_divergence')
                        warning_score += 30

            # === 3. 关键支撑/阻力突破 ===
            if 'ema_21' in dataframe.columns and 'ema_50' in dataframe.columns:
                ema_21 = current.get('ema_21', 0)
                ema_50 = current.get('ema_50', 0)
                close = current.get('close', 0)
                prev_close = dataframe['close'].iloc[-2] if len(dataframe) > 1 else close

                # 跌破EMA21（上升趋势警报）
                if prev_close > ema_21 and close < ema_21:
                    warning_signals.append('break_ema21_down')
                    warning_score += 20

                # 突破EMA21（下降趋势警报）
                elif prev_close < ema_21 and close > ema_21:
                    warning_signals.append('break_ema21_up')
                    warning_score += 20

            # === 4. 反向大成交量 ===
            if 'volume_ratio' in dataframe.columns:
                volume_ratio = current.get('volume_ratio', 1.0)
                close_change = (current['close'] - dataframe['close'].iloc[-2]) / dataframe['close'].iloc[-2]

                # 大成交量下跌（多头警报）
                if volume_ratio > 1.5 and close_change < -0.01:
                    warning_signals.append('high_volume_selloff')
                    warning_score += 25

                # 大成交量上涨（空头警报）
                elif volume_ratio > 1.5 and close_change > 0.01:
                    warning_signals.append('high_volume_rally')
                    warning_score += 25

            # === 5. ADX趋势减弱 ===
            if 'adx' in dataframe.columns and len(dataframe) >= 3:
                adx_values = dataframe['adx'].tail(3).values
                # ADX连续下降且之前在高位
                if adx_values[-3] > 30 and adx_values[-1] < adx_values[-2] < adx_values[-3]:
                    warning_signals.append('adx_weakening')
                    warning_score += 20

            return {
                'warning_level': min(warning_score, 100),
                'signals': warning_signals,
                'has_warning': warning_score > 30  # 30分以上算有预警
            }

        except Exception as e:
            logger.warning(f"预警系统检测失败: {e}")
            return {'warning_level': 0, 'signals': []}

    def classify_market_regime(self, dataframe: DataFrame) -> str:
        """
        识别当前市场状态以决定适合的交易风格

        🚀 优化：多层级分析系统
        - 快速层（10根K线/50分钟）：权重60% - 捕捉急速转向
        - 中期层（30根K线/150分钟）：权重30% - 趋势确认
        - 长期层（50根K线/250分钟）：权重10% - 强趋势过滤
        """

        if dataframe.empty or len(dataframe) < 50:
            return "stable"  # 数据不足时使用稳定模式

        try:
            current_data = dataframe.iloc[-1]

            # === 获取早期预警信号 ===
            warning = self.detect_reversal_warning(dataframe)
            has_warning = warning.get('has_warning', False)
            warning_level = warning.get('warning_level', 0)

            # === 多层级市场特征分析 ===

            # 快速层（10根K线 - 捕捉快速变化）
            fast_data = dataframe.tail(10)
            fast_trend = current_data.get('trend_strength', 50)
            fast_adx = current_data.get('adx', 20)
            fast_volatility = current_data.get('volatility_state', 50)
            fast_atr = fast_data['atr_p'].mean() if 'atr_p' in fast_data.columns else 0.02
            fast_price_range = (fast_data['high'].max() - fast_data['low'].min()) / fast_data['close'].mean() if fast_data['close'].mean() > 0 else 0

            # 中期层（30根K线 - 趋势确认）
            mid_data = dataframe.tail(30)
            mid_trend = mid_data['trend_strength'].mean() if 'trend_strength' in mid_data.columns else 50
            mid_adx = mid_data['adx'].mean() if 'adx' in mid_data.columns else 20
            mid_volatility = mid_data['volatility_state'].mean() if 'volatility_state' in mid_data.columns else 50
            mid_atr = mid_data['atr_p'].mean() if 'atr_p' in mid_data.columns else 0.02
            mid_price_range = (mid_data['high'].max() - mid_data['low'].min()) / mid_data['close'].mean() if mid_data['close'].mean() > 0 else 0

            # 长期层（50根K线 - 强趋势过滤）
            long_data = dataframe.tail(50)
            long_trend = long_data['trend_strength'].mean() if 'trend_strength' in long_data.columns else 50
            long_adx = long_data['adx'].mean() if 'adx' in long_data.columns else 20
            long_volatility = long_data['volatility_state'].mean() if 'volatility_state' in long_data.columns else 50
            long_atr = long_data['atr_p'].mean() if 'atr_p' in long_data.columns else 0.02

            # === 加权综合评分 ===
            # 快速层权重60%（激进型配置）
            # 中期层权重30%
            # 长期层权重10%

            weighted_trend = fast_trend * 0.6 + mid_trend * 0.3 + long_trend * 0.1
            weighted_adx = fast_adx * 0.6 + mid_adx * 0.3 + long_adx * 0.1
            weighted_volatility = fast_volatility * 0.6 + mid_volatility * 0.3 + long_volatility * 0.1
            weighted_atr = fast_atr * 0.6 + mid_atr * 0.3 + long_atr * 0.1
            weighted_price_range = fast_price_range * 0.6 + mid_price_range * 0.3

            # === 早期预警调整 ===
            # 如果有强烈预警，降低激进模式的门槛，更快切换到稳定模式
            if warning_level > 60:
                # 高度预警：强制降级
                weighted_trend *= 0.7
                weighted_adx *= 0.7
            elif warning_level > 30:
                # 中度预警：适度降级
                weighted_trend *= 0.85
                weighted_adx *= 0.85

            # === 市场状态判断逻辑（调整后阈值）===

            # 激进模式条件：强趋势 + 高波动 + 明确方向 + 无严重预警
            if (weighted_trend > 75 and weighted_adx > 30 and
                weighted_volatility > 60 and weighted_atr > 0.025 and
                warning_level < 50):  # 预警不能太高
                return "aggressive"

            # 横盘模式条件：弱趋势 + 低波动 + 区间震荡
            elif (weighted_trend < 45 and weighted_adx < 18 and
                  weighted_volatility < 35 and weighted_price_range < 0.12):
                return "sideways"

            # 稳定模式：其他情况或不确定状态（包括有预警时）
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
        
        self._log_message(
            f"🔄 交易风格切换: {old_style} → {new_style} | 原因: {reason}",
            importance="critical"
        )
        
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

class UltraSmartStrategy_NoStoploss_v2(IStrategy):
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.log_verbosity = int(config.get('verbosity', 0)) if config else 0
        self.event_log = StrategyLogHelper(logger, verbosity=self.log_verbosity)
        self._log_verbosity = self.log_verbosity
        self._last_signal_inactive_log: Dict[str, datetime] = {}


        # 🎯 设置出场利润偏移参数（基于HYPEROPT参数）
        try:
            if hasattr(self, 'exit_profit_offset_param') and hasattr(self.exit_profit_offset_param, 'value'):
                self.exit_profit_offset = self.exit_profit_offset_param
            else:
                # 默认出场利润偏移
                self.exit_profit_offset = 0.0
        except Exception:
            # 如果出现任何异常，使用默认值
            self.exit_profit_offset = 0.0

    INTERFACE_VERSION = 3
    
    # 策略核心参数
    timeframe = '15m'  # 15分钟 - 中长线交易，更好的信噪比和趋势稳定性
    can_short: bool = True
    
    # === 功能开关 ===
    # [已删除] enable_profit_protection - 功能已整合到 adjust_trade_position
    enforce_small_stake_for_non_bluechips = False   # 关闭强制小仓位限制 - 让其他币有足够资金盈利
    use_mtf_entry_filter = True      # 多时间框架过滤开关 - 已优化为只用1h，延迟小且保持准确度
    entry_confidence_threshold_long = 0.55    # 多头信心阈值
    entry_confidence_threshold_short = 0.55   # 空头信心阈值
    enable_signal_inactive_logging = True     # 是否记录信号缺失日志
    signal_inactive_log_interval = 600        # 同一交易对信号缺失日志最小间隔（秒）
    enable_dca_logging = True                 # 是否记录DCA和分批止盈日志

    # 滑点和手续费缓冲配置（用于计算实际盈亏）
    trailing_min_profit_buffer = 0.002  # 0.2% 基础收益缓冲
    trailing_slippage_per_leverage = 0.0002  # 每倍杠杆额外增加0.02%

    # 🔧 Meme币优化：ATR-based 动态止损 + 分级利润锁定
    # 基于PEPE实战分析（-26.76% vs 拿住+72.74%）改进：
    # 1. 用ATR替代固定比例，自动适应波动性
    # 2. 分级锁定利润，防止大幅回吐
    # 3. 趋势确认，避免徘徊时误判反转

    # ATR 倍数配置（基于最佳实践：2.5-3.5 ATR for trend following）
    # Meme 币/其他币配置（高波动，需要更大空间避免被扫止损）
    atr_multiplier_strong_trend_meme = 5.0      # 强趋势：大幅放宽避免被震出
    atr_multiplier_moderate_trend_meme = 4.5    # 中等趋势：充足空间
    atr_multiplier_choppy_meme = 4.0            # 震荡：适度空间
    atr_multiplier_trend_broken_meme = 3.5      # 趋势破坏：保守退出但不过紧

    # 主流币配置（仅BTC/ETH，增加止损空间）
    atr_multiplier_strong_trend_bluechip = 4.5      # 强趋势：大幅增加
    atr_multiplier_moderate_trend_bluechip = 4.0    # 中等趋势：大幅增加
    atr_multiplier_choppy_bluechip = 3.5            # 震荡：增加空间
    atr_multiplier_trend_broken_bluechip = 3.0      # 趋势破坏：增加空间

    # 分级利润锁定里程碑（永久锁定，不会再降低）- 放宽阈值，减少过早锁定
    profit_lock_milestones = [
        (0.40, 0.30),  # 达到40%利润 → 永久锁定30%（从 30%→25% 放宽）
        (0.25, 0.18),  # 达到25%利润 → 永久锁定18%（从 20%→15% 放宽）
        (0.15, 0.10),  # 达到15%利润 → 永久锁定10%（从 10%→7% 放宽）
        (0.08, 0.05),  # 达到8%利润 → 永久锁定5%（从 5%→3% 放宽）
    ]

    # 趋势确认时间（根据趋势强度动态调整）
    confirmation_time_trend_broken = 0     # 趋势破坏：立即退出
    confirmation_time_choppy = 5           # 震荡：5分钟确认 - 🎯 加快反应
    confirmation_time_moderate = 10        # 中等趋势：10分钟确认 - 🎯 加快反应
    confirmation_time_strong = 15          # 强趋势：15分钟确认 - 🎯 加快反应

    # 头部币名单 & 非头部币仓位限制
    # 支持现货和合约格式
    bluechip_pairs = {
        'BTC/USDT', 'ETH/USDT',  # 只保留真正的蓝筹币
        'BTC/USDT:USDT', 'ETH/USDT:USDT'
        # 移除 SOL/BNB - 它们波动性和其他币类似，不应占用大量资金
    }
    non_bluechip_stake_multiplier = 0.8  # 非蓝筹币资金倍率（大幅提升，从 0.25 → 0.8）

    # 🎯 信号止盈系数配置（基于网络研究 + 信号类型）
    # 用于动态计算止盈目标：base_target × quality_multiplier × signal_multiplier
    SIGNAL_PROFIT_MULTIPLIERS = {
        # === 反转/背离信号（高目标 1.3-1.4x）===
        'RSI_Bearish_Divergence_Short': 1.4,
        'Volume_Divergence_Long': 1.3,             # 背离信号
        'Volume_Divergence_Short': 1.3,
        'BB_Fake_Rejection_Breakout': 1.35,

        # === 超买超卖反转（中高目标 1.2-1.25x）===
        'MACD_Golden_Reversal_Short': 1.25,
        'MACD_Bearish_Reversal': 1.25,

        # === 支撑阻力反弹（中等目标 1.1-1.15x）===
        'RSI_Rebound_Short': 1.1,

        # === 趋势追随（保守 0.9-1.0x）===
        'Strong_Bullish_Follow': 0.9,
        'Strong_Bearish_Follow': 0.9,
        'MACD_Bearish': 1.0,

        # === 默认（未分类信号）===
        'default': 1.0
    }

    # 多时间框架数据预加载配置
    # 说明：本策略在 analyze_multi_timeframe() 中使用多个时间框架进行趋势分析
    # 必须在此声明，否则 DataProvider 不会预加载数据，导致运行时错误
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """
        声明所需的额外时间框架数据，供 DataProvider 预加载

        当前策略配置 (15m主框架 - 长线优化版):
        - 15m: 主时间框架 (策略 timeframe，无需声明)
        - 1h:  短期趋势确认 (需要预加载)

        Returns:
            List[Tuple[str, str]]: [(交易对, 时间框架), ...]
        """
        if not hasattr(self, 'dp') or self.dp is None:
            return []

        pairs = self.dp.current_whitelist()
        # 仅添加与主时间框架不同的时间框架，避免重复加载
        # ⚠️ 注意：这里的时间框架列表必须与 analyze_multi_timeframe() 中的配置一致
        informative_tfs = [tf for tf in ['1h'] if tf != self.timeframe]
        return [(pair, tf) for pair in pairs for tf in informative_tfs]
    
    # 增强指标计算: 支持所有高级技术分析功能
    startup_candle_count: int = 250  # 需要足够数据计算EMA_200 (200周期) + 缓冲
    
    # 智能交易模式: 精准入场后的优化配置
    position_adjustment_enable = True
    max_dca_orders = 5  # 精准入场后减少DCA依赖，提高资金效率
    strong_signal_cooldown_bars = 2  # 缩短冷却，适度提高入场频率
    
    # 🔧 DCA功能开关 - 用于加速回测和简化策略
    enable_dca = True  # 设为False可完全禁用DCA功能，加速回测

    # === 📊 信心阈值配置 ===
    confidence_threshold_dca = 0.4            # DCA最低信心要求（只有高信心交易才加仓）
    confidence_threshold_low = 0.55            # 低信心阈值（触发第3批全清仓）

    # === 🛡️ 利润保护参数 ===
    enable_profit_protection = True            # 回撤保护开关
    profit_drawdown_threshold = 0.5            # 回撤50%峰值利润时触发清仓
    low_confidence_full_exit = True            # 低信心交易第3批止盈时全清仓

    # === 🎯 放宽后的DCA触发参数 ===
    dca_min_drawdown = 0.01                    # 最小回撤1%（原1.5%）
    dca_max_drawdown = 0.20                    # 最大回撤20%（原15%）
    dca_price_tolerance_upper = 0.02           # 价格上容差2%（原0.8%）
    dca_price_tolerance_lower = 0.10           # 价格下容差10%（原5%）
    dca_min_signals_first = 0                  # 首次DCA信号要求0个（原1个）
    dca_min_signals_after = 1                  # 后续DCA信号要求1个（原2个）

    # === 🎯 智能跟踪止损系统配置 ===
    enable_trailing_stop = True                    # 启用智能跟踪止损
    trailing_only_in_profit = True                 # 仅在盈利时跟踪（符合NoStoploss哲学）

    # 基于信心的三级激活点（决定何时开始跟踪）
    trailing_activation_low_confidence = 0.03      # 低信心(≤0.55): 3%激活（优化：从1.5%提高）
    trailing_activation_mid_confidence = 0.06      # 中等信心(0.55-0.75): 6%激活（优化：从4%提高）
    trailing_activation_high_confidence = 0.10     # 高信心(>0.75): 10%激活（优化：从6%提高）

    # 基于信心的距离系数（乘以多因子计算的基础距离）
    trailing_distance_low_confidence = 1.0         # 低信心标准（优化：从0.7放宽，给更多回撤空间）
    trailing_distance_mid_confidence = 1.3         # 中等信心放宽30%（优化：从1.0提高）
    trailing_distance_high_confidence = 1.6        # 高信心放宽60%（优化：从1.3提高）

    # partial_exit完成后的收紧系数
    trailing_tighten_after_exits = 0.8             # 完成3批止盈后收紧20%（优化：从0.6放宽，减少过早退出）

    # 与profit_protection的配合模式
    trailing_mode = "cooperative"                   # cooperative=取更宽松的，aggressive=取更严格的

    # 🎯 价格位置过滤器参数 (HYPEROPT优化参数)
    price_percentile_long_max = 0.70    # 做多最大分位数 - 参考V3放宽到0.70
    price_percentile_long_best = 0.40   # 做多最佳区间
    price_percentile_short_min = 0.60   # 做空最小分位数
    price_percentile_short_best = 0.75  # 做空最佳区间
    
    # 🎯 RSI parameters - 固定值
    rsi_long_min = 25        # Long RSI lower bound - 参考V3
    rsi_long_max = 75        # Long RSI upper bound - 参考V3
    rsi_short_min = 25       # Short RSI lower bound - 参考V3
    rsi_short_max = 75       # Short RSI upper bound - 参考V3
    
    # 🎯 成交量和趋势参数 - 固定值（参考V3）
    volume_long_threshold = 0.8       # 做多成交量要求 - 参考V3放宽
    volume_short_threshold = 0.8      # 做空成交量要求 - 参考V3放宽
    volume_spike_threshold = 2.0      # 异常放量阈值
    adx_long_min = 20                 # 做多ADX要求 - 参考V3
    adx_short_min = 20                # 做空ADX要求 - 参考V3
    trend_strength_threshold = 30     # 强趋势阈值
    
    # 🎯 极端价格区过滤参数 - 固定值（参考V3统一标准）
    overextended_long_pos_cap_bluechip = 0.80  # 蓝筹做多高位阈值 - 参考V3放宽
    overextended_long_pos_cap_meme = 0.80      # Meme做多高位阈值 - 参考V3放宽
    overextended_long_rsi_cap = 70             # 做多放弃RSI阈值
    overextended_long_ema_mult = 1.10          # 做多价差阈值
    overextended_long_bb_cap = 0.85            # 做多布林位置阈值

    oversold_short_pos_floor_bluechip = 0.20   # 蓝筹做空低位阈值 - 参考V3
    oversold_short_pos_floor_meme = 0.20       # Meme做空低位阈值 - 参考V3
    oversold_short_rsi_floor = 30              # 做空放弃RSI阈值
    oversold_short_ema_mult = 0.90             # 做空价差阈值
    oversold_short_bb_floor = 0.15             # 做空布林位置阈值

    breakout_base_pos_cap_bluechip = 0.75      # 蓝筹突破前位置上限
    breakout_base_pos_cap_meme = 0.75          # Meme突破前位置上限
    breakout_volatility_multiplier = 1.20     # 突破波动倍数
    breakout_ema_distance_cap = 1.08          # 突破离EMA上限

    strong_bullish_pos_cap_bluechip = 0.75    # 蓝筹强趋势多头上限
    strong_bullish_pos_cap_meme = 0.75        # Meme强趋势多头上限
    strong_bearish_pos_floor = 0.25           # 强趋势空头位置下限
    reversal_pos_cap_bluechip = 0.75          # 蓝筹反指多头上限
    reversal_pos_cap_meme = 0.75              # Meme反指多头上限

    # 🎯 技术指标参数 - 固定值
    macd_fast = 12                        # MACD快线
    macd_slow = 26                        # MACD慢线
    macd_signal = 9                       # MACD信号线
    bb_period = 20                        # 布林带周期
    bb_std = 2.0                          # 布林带标准差
    
    # 简化风险管理 - 使用固定止损
    # 移除了复杂的动态止损，使用简单可靠的固定值
    
    # 🎯 ROI配置优化 - 完全关闭以避免错失趋势
    minimal_roi = {
        "0": 999  # 实际禁用ROI，让智能出场信号接管
    }
    
    # 完全关闭止损（
    stoploss = -0.99  # 禁用止损  # 默认7%止损，将被动态止损系统覆盖
    use_custom_stoploss = True  # 启用智能跟踪止损系统

    # 禁用Freqtrade原生跟踪止损（使用custom_stoploss实现更智能的跟踪）
    trailing_stop = False
    trailing_stop_positive = 0.0
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # 启用智能出场信号
    ignore_roi_if_entry_signal = False  # 不忽略ROI

    # 订单类型配置
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,  # 启用交易所止损保护资金安全
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

    # 针对非主流币的仓位限制与风险乘数配置
    NON_MAINSTREAM_POSITION_CAP = 0.12    # 非主流币单笔最大仓位占比（相对总资金）
    NON_MAINSTREAM_MIN_POSITION = 0.04    # 非主流币最低仓位占比，避免完全失效

    COIN_RISK_MULTIPLIERS = {
        'mainstream': 1.0,      # 从 1.2 → 1.0 (减少蓝筹优势)
        'low_risk': 0.8,        # 从 0.5 → 0.8 (增加其他币资金)
        'medium_risk': 0.8,     # 从 0.3 → 0.8 (大幅增加)
        'high_risk': 0.5        # 从 0.15 → 0.5 (适度增加)
    }

    DYNAMIC_COIN_RISK_MULTIPLIERS = {
        'mainstream': 1.0,      # 从 1.3 → 1.0 (减少蓝筹优势)
        'low_risk': 0.8,        # 从 0.5 → 0.8
        'medium_risk': 0.8,     # 从 0.3 → 0.8 (大幅增加)
        'high_risk': 0.5        # 从 0.15 → 0.5
    }
    
    # === 技术指标参数（固定经典值） ===
    @property
    def rsi_period(self):
        return 14  # RSI周期保持固定
        
    atr_period = 14
    adx_period = 14
    
    # === 简化的市场状态参数 ===
    volatility_threshold = 0.025     # 稍微提高波动率阈值
    trend_strength_min = 50          # 提高趋势强度要求
    # volume_spike_threshold moved to HYPEROPT parameters above
    
    # 🎯 风险管理参数 (HYPEROPT优化参数)
    dca_multiplier = 1.3                    # DCA倍数
    dca_price_deviation = 0.025             # DCA触发偏差
    min_meaningful_dca_ratio = 0.20         # 最小DCA占比
    max_risk_per_trade = 0.025              # 单笔最大风险
    kelly_lookback = 50                     # Kelly回看期
    drawdown_protection = 0.12              # 回撤保护阈值
    max_portfolio_heat = 0.30               # 最大组合风险度
    correlation_threshold = 0.70            # 相关性阈值
    rebalance_threshold = 0.10              # 再平衡阈值
    
    # 固定高级资金管理参数 (暂时不优化)
    var_confidence_level = 0.95    # VaR置信度
    cvar_confidence_level = 0.99   # CVaR置信度
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
        self.event_log.info("startup", importance="summary", version="v2", strategy="UltraSmartStrategy")
        
        # === 交易风格管理系统初始化 ===
        self.style_manager = TradingStyleManager(log_callback=self._log_message, verbosity=self._log_verbosity)
        self.event_log.info(
            "style_manager_init",
            importance="summary",
            current_style=self.style_manager.current_style,
            switch_cooldown=self.style_manager.style_switch_cooldown,
        )
        
        # 初始化风格切换记录
        self.last_style_check = datetime.now(timezone.utc)
        self.style_check_interval = 300  # 5分钟检查一次风格切换

        # === 智能跟踪止损状态字典 ===
        self._trailing_stop_state = {}  # {trade_key: {'peak_profit': float, 'exits_completed': bool, 'last_distance': float}}

    def _resolve_log_level(self, importance: str) -> int:
        importance = (importance or "verbose").lower()
        if importance in ("critical", "always"):
            return logging.INFO
        if importance in ("summary", "high"):
            return logging.INFO  # 强制输出所有日志
        if importance in ("verbose", "normal"):
            return logging.INFO  # 强制输出所有日志
        if importance == "debug":
            return logging.DEBUG
        return logging.INFO  # 强制输出所有日志

    def _log_message(self, message: str, *, importance: str = "verbose", extra: Optional[dict] = None) -> None:
        level = self._resolve_log_level(importance)
        if extra:
            logger.log(level, message, extra=extra)
        else:
            logger.log(level, message)

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
            'rsi_oversold': self.rsi_long_max,
            'rsi_overbought': self.rsi_short_min, 
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
    
    @property
    def protections(self):
        """保护机制配置 - 防追空巨亏"""
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            }
        ]
    
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
            
            self._log_message(switch_log, importance="summary")
            
            # 记录风格切换
            style_summary = self.style_manager.get_style_summary()
            self._log_message(f"🔄 风格切换完成: {style_summary}", importance="summary")
            
        except Exception as e:
            logger.error(f"风格切换日志记录失败: {e}")
    
    def get_current_trading_style_info(self) -> dict:
        """获取当前交易风格的详细信息"""
        return self.style_manager.get_style_summary()
        
    # Removed informative_pairs() method - no longer needed without informative timeframes
    
    def get_market_orderbook(self, pair: str) -> Dict:
        """获取订单簿数据"""
        try:
            # 在回测/HYPEROPT模式下，订单簿数据不可用
            if not hasattr(self, 'dp') or self.dp is None:
                return self._get_default_orderbook()
                
            if not hasattr(self.dp, 'orderbook') or self.dp.orderbook is None:
                return self._get_default_orderbook()
            
            # 检查是否在回测模式（exchange为None时不可用）
            if not hasattr(self.dp, '_exchange') or self.dp._exchange is None:
                return self._get_default_orderbook()
                
            # 额外检查，确保exchange对象存在且有fetch_l2_order_book方法
            if self.dp._exchange is None or not hasattr(self.dp._exchange, 'fetch_l2_order_book'):
                return self._get_default_orderbook()
                
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
        except Exception:
            # 在回测/HYPEROPT模式下，订单簿获取失败是正常的，不记录警告
            return self._get_default_orderbook()
            
    def _get_default_orderbook(self) -> Dict:
        """返回默认订单簿数据（用于回测/HYPEROPT模式）"""
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
        new_columns['ema_200'] = ta.EMA(dataframe, timeperiod=200)  # 超长期趋势过滤
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
        
        # === 🎯 动态RSI阈值系统 - 基于市场环境智能调整 ===
        # 注意：必须在所有基础指标计算完成后调用，因为需要依赖trend_strength等指标
        # 这个调用会在后续添加，确保所有依赖指标都已计算完成
        
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
                              'ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_50', 'ema_200', 'mom_10', 'roc_10']
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
                elif indicator in ['ema_5', 'ema_13', 'ema_34', 'ema_200']:
                    # 如果EMA指标缺失，重新计算
                    if indicator == 'ema_5':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=5)
                    elif indicator == 'ema_13':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=13)
                    elif indicator == 'ema_34':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=34)
                    elif indicator == 'ema_200':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=200)
            
            # 一次性添加所有默认值
            if default_values:
                defaults_df = pd.DataFrame(default_values, index=dataframe.index)
                dataframe = pd.concat([dataframe, defaults_df], axis=1)
        # 指标计算完成
        
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
        bearish_rsi_divergence = price_higher_high & rsi_lower_high & (dataframe['rsi_14'] > self.rsi_short_min)
        
        # Price new low but RSI not making new low
        price_lower_low = (
            (dataframe['low'] < dataframe['low'].shift(25)) &
            (dataframe['low'].shift(25) < dataframe['low'].shift(50))
        )
        rsi_higher_low = (
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(25)) &
            (dataframe['rsi_14'].shift(25) > dataframe['rsi_14'].shift(50))
        )
        bullish_rsi_divergence = price_lower_low & rsi_higher_low & (dataframe['rsi_14'] < self.rsi_long_max)
        
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
        """计算线性回归趋势 - 优化为向量化实现"""
        close = dataframe['close'].values
        n = len(close)
        reg_slope = np.zeros(n)

        # 向量化计算线性回归斜率
        for i in range(period - 1, n):
            y = close[i - period + 1:i + 1]
            if len(y) == period:
                x = np.arange(period)
                # 使用 NumPy 的向量化计算而非 scipy
                x_mean = x.mean()
                y_mean = y.mean()

                # 计算斜率和 R²
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator_x = np.sum((x - x_mean) ** 2)

                if denominator_x > 0:
                    slope = numerator / denominator_x

                    # 计算 R²
                    y_pred = slope * (x - x_mean) + y_mean
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y_mean) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    reg_slope[i] = slope * r_squared

        # 转换为 Series
        reg_slope_series = pd.Series(reg_slope, index=dataframe.index)

        # 标准化
        normalized_slope = np.where(dataframe['close'] > 0,
                                   reg_slope_series / dataframe['close'] * 1000,
                                   0)  # 放大因子

        return pd.Series(normalized_slope, index=dataframe.index).fillna(0).clip(-1, 1)
    
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
            # === 首先识别主流币 ===
            pair_upper = pair.upper()
            normalized_pair = pair_upper.split(':')[0]
            # 主流币白名单（这些币种可以用大仓位）
            if normalized_pair in self.bluechip_pairs:
                return 'mainstream'  # 主流币特殊标识
            
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
            
            # 币种风险识别
            if risk_tier == 'high_risk':
                self._log_message(
                    f"⚠️ {pair} 高风险币种: {risk_score}/100, 因素: {' | '.join(risk_factors)}",
                    importance="summary"
                )
            elif risk_tier == 'low_risk' and risk_score < 20:
                self._log_message(
                    f"✅ {pair} 低风险币种: {risk_score}/100",
                    importance="verbose"
                )
            
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
    
    def _detect_market_state_vectorized(self, dataframe: DataFrame, pair: str) -> pd.Series:
        """向量化版本的市场状态识别 - 性能优化，避免 O(n²) 复杂度"""
        # 计算所有需要的指标（向量化）
        high_20 = dataframe['high'].rolling(20).max()
        low_20 = dataframe['low'].rolling(20).min()
        price_position = (dataframe['close'] - low_20) / (high_20 - low_20).replace(0, 1)

        # 根据币种类型选择参数
        is_bluechip = pair in self.bluechip_pairs
        overextended_long_pos_cap = self.overextended_long_pos_cap_bluechip if is_bluechip else self.overextended_long_pos_cap_meme
        oversold_short_pos_floor = self.oversold_short_pos_floor_bluechip if is_bluechip else self.oversold_short_pos_floor_meme

        # 向量化条件判断
        is_at_top = (
            (price_position > overextended_long_pos_cap) &
            (dataframe['rsi_14'] > self.overextended_long_rsi_cap) &
            (dataframe['macd'] < dataframe['macd_signal'])
        )

        is_at_bottom = (
            (price_position < oversold_short_pos_floor) &
            (dataframe['rsi_14'] < self.oversold_short_rsi_floor) &
            (dataframe['macd'] > dataframe['macd_signal'])
        )

        # EMA 排列
        ema_8 = dataframe['ema_8'] if 'ema_8' in dataframe.columns else dataframe['close']
        ema_bullish = (ema_8 > dataframe['ema_21']) & (dataframe['ema_21'] > dataframe['ema_50'])
        ema_bearish = (ema_8 < dataframe['ema_21']) & (dataframe['ema_21'] < dataframe['ema_50'])

        # 初始化 market_state 为 'sideways'
        market_state = pd.Series('sideways', index=dataframe.index)

        # 按优先级应用条件（从低到高）
        market_state.loc[dataframe['atr_p'] < self.volatility_threshold * 0.5] = 'consolidation'

        # ADX > 25 的情况
        mask_adx_25 = dataframe['adx'] > 25
        market_state.loc[mask_adx_25 & (dataframe['close'] > dataframe['ema_21']) & ~is_at_top] = 'mild_uptrend'
        market_state.loc[mask_adx_25 & (dataframe['close'] < dataframe['ema_21']) & ~is_at_bottom] = 'mild_downtrend'

        # ADX > 40 的强趋势
        mask_adx_40 = (dataframe['adx'] > 40) & (dataframe['atr_p'] > self.volatility_threshold)
        market_state.loc[mask_adx_40 & ema_bullish & ~is_at_top] = 'strong_uptrend'
        market_state.loc[mask_adx_40 & ema_bearish & ~is_at_bottom] = 'strong_downtrend'
        market_state.loc[mask_adx_40 & ~ema_bullish & ~ema_bearish] = 'volatile'

        # 顶底检测（最高优先级）
        market_state.loc[is_at_bottom] = 'market_bottom'
        market_state.loc[is_at_top] = 'market_top'

        return market_state

    def detect_market_state(self, dataframe: DataFrame, pair: str) -> str:
        """增强版市场状态识别 - 防止顶底反向开仓（保留用于向后兼容）"""
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

        # 🎯 根据币种类型选择参数（蓝筹 vs Meme）
        is_bluechip = pair in self.bluechip_pairs
        overextended_long_pos_cap = self.overextended_long_pos_cap_bluechip if is_bluechip else self.overextended_long_pos_cap_meme
        oversold_short_pos_floor = self.oversold_short_pos_floor_bluechip if is_bluechip else self.oversold_short_pos_floor_meme

        # 检测是否在顶部区域（避免在顶部开多）
        is_at_top = (
            price_position > overextended_long_pos_cap and  # 价格在20日高点附近
            rsi > self.overextended_long_rsi_cap and  # RSI超买
            macd < macd_signal  # MACD已经死叉
        )

        # 检测是否在底部区域（避免在底部开空）
        is_at_bottom = (
            price_position < oversold_short_pos_floor and  # 价格在20日低点附近
            rsi < self.oversold_short_rsi_floor and  # RSI超卖
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
                except Exception:
                    # 跳过计算失败的相关性
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
            
        except Exception as e:
            logger.debug(f"Kelly分数计算失败: {e}")
            return 0.25
    
    def calculate_dynamic_position_size(self, dataframe: DataFrame, current_price: float, market_state: str, pair: str, signal_direction: str = 'long') -> Dict[str, float]:
        """🎯 增强动态仓位管理系统 - 整合所有优化功能"""
        
        # === 1. 基础仓位计算 ===
        base_position = (self.base_position_size + self.max_position_size) / 2
        
        # === 2. 信号质量仓位调整 (最重要的因素 - 40%权重) ===
        signal_quality_multiplier = 1.0
        
        # 获取信号质量评分
        quality_column = f'{signal_direction}_signal_quality_score'
        grade_column = f'{signal_direction}_signal_quality_grade'
        
        if quality_column in dataframe.columns and len(dataframe) > 0:
            current_quality = dataframe[quality_column].iloc[-1]
            current_grade = dataframe[grade_column].iloc[-1] if grade_column in dataframe.columns else 'C'
            
            # 基于信号质量等级的仓位倍数
            quality_multipliers = {
                'A+': 1.8,   # 极优质信号：1.8倍仓位
                'A': 1.6,    # 优质信号：1.6倍仓位  
                'A-': 1.4,   # 良好信号：1.4倍仓位
                'B+': 1.2,   # 较好信号：1.2倍仓位
                'B': 1.0,    # 标准信号：标准仓位
                'B-': 0.8,   # 一般信号：0.8倍仓位
                'C+': 0.6,   # 较差信号：0.6倍仓位
                'C': 0.4,    # 差信号：0.4倍仓位
                'C-': 0.3,   # 很差信号：0.3倍仓位
                'D': 0.2,    # 极差信号：0.2倍仓位
                'F': 0.1     # 垃圾信号：0.1倍仓位
            }
            signal_quality_multiplier = quality_multipliers.get(current_grade, 1.0)
            
        # === 3. MTF确认强度调整 (25%权重) ===
        mtf_multiplier = 1.0
        
        if 'mtf_confirmation_score' in dataframe.columns and len(dataframe) > 0:
            mtf_confirmation = dataframe['mtf_confirmation_score'].iloc[-1]
            
            # MTF确认越强，仓位越大
            if mtf_confirmation > 0.8:
                mtf_multiplier = 1.4      # 极强MTF确认
            elif mtf_confirmation > 0.6:
                mtf_multiplier = 1.2      # 强MTF确认  
            elif mtf_confirmation > 0.4:
                mtf_multiplier = 1.0      # 中等MTF确认
            elif mtf_confirmation > 0.2:
                mtf_multiplier = 0.8      # 弱MTF确认
            else:
                mtf_multiplier = 0.6      # 很弱/无MTF确认
        
        # === 4. 噪音环境调整 (15%权重) ===
        noise_multiplier = 1.0
        
        if 'noise_score' in dataframe.columns and len(dataframe) > 0:
            noise_level = dataframe['noise_score'].iloc[-1]
            
            # 噪音越低，仓位可以越大
            if noise_level < 0.2:
                noise_multiplier = 1.2    # 极低噪音环境
            elif noise_level < 0.4:  
                noise_multiplier = 1.0    # 低噪音环境
            elif noise_level < 0.6:
                noise_multiplier = 0.8    # 中等噪音环境
            else:
                noise_multiplier = 0.5    # 高噪音环境
        
        # === 5. 波动率调整 (10%权重) ===
        volatility_multiplier = 1.0
        
        if 'atr_p' in dataframe.columns and len(dataframe) > 0:
            atr = dataframe['atr_p'].iloc[-1]
            
            # 波动率调整：中等波动最佳，过高过低都降低仓位
            if 0.01 <= atr <= 0.03:      # 1%-3% ATR为最佳波动率
                volatility_multiplier = 1.1
            elif 0.005 <= atr < 0.01:     # 0.5%-1% ATR稍低
                volatility_multiplier = 0.9
            elif 0.03 < atr <= 0.05:      # 3%-5% ATR稍高
                volatility_multiplier = 0.8
            elif atr > 0.05:              # >5% ATR过高
                volatility_multiplier = 0.6
            else:                         # <0.5% ATR过低
                volatility_multiplier = 0.7
        
        # === 6. 成交量确认调整 (10%权重) ===
        volume_multiplier = 1.0
        
        volume_quality_col = f'{signal_direction}_volume_quality_score'
        if volume_quality_col in dataframe.columns and len(dataframe) > 0:
            volume_quality = dataframe[volume_quality_col].iloc[-1]
            
            if volume_quality > 80:
                volume_multiplier = 1.3   # 优秀成交量确认
            elif volume_quality > 70:
                volume_multiplier = 1.1   # 良好成交量确认
            elif volume_quality > 50:
                volume_multiplier = 1.0   # 一般成交量确认
            elif volume_quality > 30:
                volume_multiplier = 0.8   # 较差成交量确认
            else:
                volume_multiplier = 0.6   # 很差成交量确认
        
        # === 7. 账户状态调整 ===
        account_multiplier = 1.0
        
        # 连胜/连败调整
        if self.consecutive_wins >= 5:
            account_multiplier *= 1.4
        elif self.consecutive_wins >= 3:
            account_multiplier *= 1.2
        elif self.consecutive_wins >= 1:
            account_multiplier *= 1.1
        elif self.consecutive_losses >= 5:
            account_multiplier *= 0.4
        elif self.consecutive_losses >= 3:
            account_multiplier *= 0.6
        elif self.consecutive_losses >= 1:
            account_multiplier *= 0.8
            
        # 回撤调整
        if hasattr(self, 'current_drawdown'):
            if self.current_drawdown < -0.05:        # 回撤超过5%
                account_multiplier *= 0.5
            elif self.current_drawdown < -0.02:      # 回撤超过2%
                account_multiplier *= 0.7
            elif self.current_drawdown == 0:         # 无回撤
                account_multiplier *= 1.2
        
        # === 8. 币种风险调整 ===
        try:
            coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            coin_risk_multipliers = self.DYNAMIC_COIN_RISK_MULTIPLIERS
            coin_risk_multiplier = coin_risk_multipliers.get(
                coin_risk_tier, self.DYNAMIC_COIN_RISK_MULTIPLIERS.get('medium_risk', 0.3)
            )

            if self.enforce_small_stake_for_non_bluechips:
                if coin_risk_tier != 'mainstream':
                    coin_risk_multiplier = min(coin_risk_multiplier, self.non_bluechip_stake_multiplier)
            else:
                if coin_risk_tier != 'mainstream':
                    coin_risk_multiplier = max(coin_risk_multiplier, 1.0)
        except Exception as e:
            logger.debug(f"币种风险评估失败: {e}")
            coin_risk_multiplier = 1.0
            coin_risk_tier = 'medium_risk'
        
        # === 9. 综合仓位计算 ===
        # 分层乘数应用（避免过度放大）
        
        # 第一层：信号质量（最重要）
        adjusted_position = base_position * signal_quality_multiplier
        
        # 第二层：市场环境（MTF + 噪音 + 波动率的几何平均，避免极端值）
        market_environment_multiplier = (mtf_multiplier * noise_multiplier * volatility_multiplier) ** (1/3)
        adjusted_position *= market_environment_multiplier
        
        # 第三层：成交量确认
        adjusted_position *= volume_multiplier
        
        # 第四层：账户状态
        adjusted_position *= account_multiplier
        
        # 第五层：币种风险（最后应用，确保风控）
        final_position = adjusted_position * coin_risk_multiplier
        
        # === 10. 最终风险控制 ===
        # 硬性上下限
        min_position = self.base_position_size * 0.5   # 最小仓位
        max_position = self.max_position_size * 1.2    # 最大仓位（允许略微超过配置）

        # 非主流币整体缩减仓位，保持主流币不变
        if self.enforce_small_stake_for_non_bluechips and coin_risk_tier != 'mainstream':
            max_position = min(max_position, self.NON_MAINSTREAM_POSITION_CAP)
            min_position = min(
                min_position,
                max(self.NON_MAINSTREAM_MIN_POSITION, max_position * 0.5)
            )
            min_position = min(min_position, max_position)
        
        # 应用限制
        final_position = max(min_position, min(final_position, max_position))
        
        # === 11. 紧急风控检查 ===
        # 在极端情况下进一步降低仓位
        emergency_multiplier = 1.0
        
        # 高波动 + 高噪音 + 低质量信号的组合
        if (volatility_multiplier <= 0.6 and noise_multiplier <= 0.6 and signal_quality_multiplier <= 0.8):
            emergency_multiplier = 0.5
            logger.warning(f"紧急风控触发 - {pair}: 高风险环境，仓位减半")
        
        final_position *= emergency_multiplier
        
        # === 12. 返回详细信息 ===
        return {
            'final_position_size': final_position,
            'base_position': base_position,
            'signal_quality_multiplier': signal_quality_multiplier,
            'mtf_multiplier': mtf_multiplier,
            'noise_multiplier': noise_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'volume_multiplier': volume_multiplier,
            'account_multiplier': account_multiplier,
            'coin_risk_multiplier': coin_risk_multiplier,
            'coin_risk_tier': coin_risk_tier,
            'emergency_multiplier': emergency_multiplier,
            'market_environment_multiplier': market_environment_multiplier,
            'position_utilization': final_position / max_position,  # 仓位利用率
            'risk_level': self._assess_position_risk_level(final_position, max_position)
        }
    
    def _assess_position_risk_level(self, position_size: float, max_position: float) -> str:
        """评估仓位风险等级"""
        utilization = position_size / max_position
        
        if utilization > 0.8:
            return "高风险"
        elif utilization > 0.6:
            return "中高风险"
        elif utilization > 0.4:
            return "中等风险"
        elif utilization > 0.2:
            return "中低风险"
        else:
            return "低风险"
    
    def calculate_position_size(self, current_price: float, market_state: str, pair: str) -> float:
        """动态仓位管理系统 - 增强版整合所有优化功能"""
        
        try:
            # 获取最新的dataframe数据
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            
            if dataframe.empty or len(dataframe) == 0:
                logger.warning(f"无法获取{pair}数据，使用基础仓位")
                return self.base_position_size
            
            # === 🎯 使用增强动态仓位系统 ===
            # 这里假设主要是long方向，实际应根据信号方向动态调整
            enhanced_position_info = self.calculate_dynamic_position_size(
                dataframe=dataframe,
                current_price=current_price,
                market_state=market_state,
                pair=pair,
                signal_direction='long'  # 默认long，实际使用时应根据当前信号方向调整
            )
            
            final_position = enhanced_position_info['final_position_size']
            
            # 增强仓位计算完成
            if enhanced_position_info['risk_level'] == 'HIGH':
                logger.warning(f"⚠️ {pair} 高风险: {final_position*100:.1f}% 仓位 | 因子: {enhanced_position_info['signal_quality_multiplier']:.1f}x")
            elif final_position > 0.05:  # 仅大仓位时记录
                self._log_message(
                    f"💰 {pair} 仓位: {final_position*100:.1f}% | 质量: {enhanced_position_info['signal_quality_multiplier']:.1f}x",
                    importance="verbose"
                )
            
            return final_position
            
        except Exception as e:
            logger.error(f"增强仓位计算失败 {pair}: {e}")
            # 降级到原始系统
            return self._fallback_position_calculation(current_price, market_state, pair)
    
    def _fallback_position_calculation(self, current_price: float, market_state: str, pair: str) -> float:
        """备用仓位计算系统（当增强系统失败时使用）"""
        
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
        coin_risk_multiplier = self.COIN_RISK_MULTIPLIERS.get(
            coin_risk_tier, self.COIN_RISK_MULTIPLIERS.get('medium_risk', 0.3)
        )

        if self.enforce_small_stake_for_non_bluechips:
            if coin_risk_tier != 'mainstream':
                coin_risk_multiplier = min(coin_risk_multiplier, self.non_bluechip_stake_multiplier)
        else:
            if coin_risk_tier != 'mainstream':
                coin_risk_multiplier = max(coin_risk_multiplier, 1.0)

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
        
        min_allowed_position = self.base_position_size * 0.8
        if self.enforce_small_stake_for_non_bluechips and coin_risk_tier != 'mainstream':
            max_allowed_position = min(max_allowed_position, self.NON_MAINSTREAM_POSITION_CAP)
            min_allowed_position = min(
                min_allowed_position,
                max(self.NON_MAINSTREAM_MIN_POSITION, max_allowed_position * 0.5)
            )
            min_allowed_position = min(min_allowed_position, max_allowed_position)

        # 应用限制
        final_position = max(
            min_allowed_position,
            min(calculated_position, max_allowed_position)
        )
        
        risk_tier_labels = {
            'low_risk': 'low',
            'medium_risk': 'medium',
            'high_risk': 'high'
        }

        self.event_log.info(
            "position_calc",
            pair=pair,
            risk_tier=coin_risk_tier,
            risk_label=risk_tier_labels.get(coin_risk_tier, coin_risk_tier),
            base_pct=f"{base_position*100:.0f}%",
            streak=f"{self.consecutive_wins}w/{self.consecutive_losses}l",
            streak_multiplier=f"{streak_multiplier:.2f}x",
            market_state=market_state,
            market_multiplier=f"{market_multiplier:.2f}x",
            time_multiplier=f"{time_multiplier:.2f}x",
            equity_multiplier=f"{equity_multiplier:.2f}x",
            leverage_adjustment=f"{leverage_adjustment:.2f}x",
            current_leverage=f"{int(current_leverage)}x",
            compound_multiplier=f"{compound_multiplier:.2f}x",
            risk_multiplier=f"{coin_risk_multiplier:.2f}x",
            multiplier_cap=f"{max_multiplier:.2f}x",
            calculated_pct=f"{calculated_position*100:.2f}%",
            final_pct=f"{final_position*100:.2f}%",
        )
        
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
        
        # 复利加速器状态
        if final_multiplier > 1.2 or final_multiplier < 0.8:
            self._log_message(
                f"🚀 复利加速: {final_multiplier:.1f}x | 今日: {daily_profit*100:+.1f}% | {mode}",
                importance="summary"
            )
        
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
        
        # === 💪 15m优化：币种风险杠杆限制（中长线稳健配置）===
        # 15m中长线交易，更注重稳定性，适度降低杠杆上限
        coin_leverage_limits = {
            'low_risk': (5, 15),        # 低风险(BTC/ETH等)：5-15倍（原8-30倍太激进）
            'medium_risk': (3, 8),      # 中等风险：3-8倍（原3-12倍，进一步限制） 🎯
            'high_risk': (2, 5)         # 高风险（山寨币）：2-5倍（原2-8倍，进一步限制） 🎯
        }

        # 获取当前币种的杠杆限制
        min_allowed, max_allowed = coin_leverage_limits.get(coin_risk_tier, (2, 10))

        # === 💡 15m框架专用波动率阶梯系统 ===
        # 15m波动特性：比5m更平稳，阈值需要相应调整
        volatility_percent = volatility * 100  # 转换为百分比

        # 15m优化的基础杠杆阶梯（中长线稳健配置）
        if volatility_percent < 1.2:
            base_leverage = 12   # 极低波动 (15m: 1.2%以下)
        elif volatility_percent < 2.0:
            base_leverage = 10   # 低波动 (15m: 1.2%-2.0%)
        elif volatility_percent < 3.0:
            base_leverage = 8    # 中低波动 (15m: 2.0%-3.0%)
        elif volatility_percent < 4.0:
            base_leverage = 6    # 中等波动 (15m: 3.0%-4.0%)
        elif volatility_percent < 5.5:
            base_leverage = 4    # 中高波动 (15m: 4.0%-5.5%)
        else:
            base_leverage = 3    # 高波动 (15m: 5.5%+)，保守配置
        
        # === 🎯 信号质量调整系统 ===
        # 基于信号质量动态调整杠杆（新增功能）
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty and len(dataframe) > 0:
                # 获取最新的信号质量评分
                signal_quality = 0.5  # 默认值
                
                # 尝试获取EMA确认分数
                if 'ema_bullish_score' in dataframe.columns:
                    ema_score = dataframe['ema_bullish_score'].iloc[-1]
                    signal_quality = max(signal_quality, ema_score / 7.0)  # 标准化到0-1
                
                # 信号质量乘数
                if signal_quality > 0.8:
                    quality_multiplier = 1.2    # 高质量信号，适度增加杠杆
                elif signal_quality > 0.6:
                    quality_multiplier = 1.0    # 中等质量，保持不变
                elif signal_quality > 0.4:
                    quality_multiplier = 0.8    # 低质量，降低杠杆
                else:
                    quality_multiplier = 0.6    # 极低质量，大幅降低
                
                base_leverage = int(base_leverage * quality_multiplier)
        except Exception:
            pass  # 信号质量获取失败时忽略
            
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
        
        # 应用放宽后的硬性限制：1-35倍
        pre_risk_leverage = max(1, min(int(calculated_leverage), 35))
        
        # === 🎯 应用币种风险杠杆限制（垃圾币严格限制） ===
        final_leverage = max(min_allowed, min(pre_risk_leverage, max_allowed))
        
        # === 紧急风控 ===
        # 单日亏损超过3%，强制降低杠杆
        if hasattr(self, 'daily_loss') and self.daily_loss < -0.03:
            final_leverage = min(final_leverage, 8)    # 降低到8倍
            
        # 连续亏损保护
        if self.consecutive_losses >= 5:
            final_leverage = min(final_leverage, 5)    # 降低到5倍
            
        risk_tier_labels = {
            'low_risk': 'low',
            'medium_risk': 'medium',
            'high_risk': 'high'
        }

        self.event_log.info(
            "leverage_calc",
            pair=pair,
            risk_tier=coin_risk_tier,
            risk_label=risk_tier_labels.get(coin_risk_tier, coin_risk_tier),
            volatility_pct=f"{volatility_percent:.2f}%",
            base_leverage=f"{base_leverage:.2f}x",
            streak=f"{self.consecutive_wins}w/{self.consecutive_losses}l",
            streak_multiplier=f"{streak_multiplier:.2f}x",
            time_multiplier=f"{time_multiplier:.2f}x",
            market_multiplier=f"{market_multiplier:.2f}x",
            equity_multiplier=f"{equity_multiplier:.2f}x",
            pre_risk_leverage=f"{pre_risk_leverage}x",
            risk_limits=f"{min_allowed}-{max_allowed}x",
            calculated_leverage=f"{calculated_leverage:.2f}x",
            final_leverage=f"{final_leverage}x",
        )
        
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
    
    def _log_trade_entry_targets(self, pair: str, entry_price: float, leverage_params: dict):
        """
        📊 记录详细的交易入场目标价格
        清晰显示所有预计算的价格级别
        """
        try:
            lp = leverage_params  # 简化引用
            
            # 构建止盈价格字符串
            tp_lines = []
            for tp_key in ['tp1', 'tp2', 'tp3', 'tp4']:
                if tp_key in lp['take_profit']:
                    tp = lp['take_profit'][tp_key]
                    tp_lines.append(
                        f"  ├─ {tp_key.upper()}: ${tp['price']:.4f} "
                        f"(+{tp['profit_pct']:.1f}%) [{int(tp['close_ratio']*100)}%] "
                        f"- {tp['description']}"
                    )
            
            # 构建DCA价格字符串
            dca_lines = []
            for dca_level in lp['dca']['price_levels'][:3]:  # 只显示前3级
                dca_lines.append(
                    f"  ├─ DCA{dca_level['level']}: ${dca_level['price']:.4f} "
                    f"(-{dca_level['deviation_pct']:.1f}%) "
                    f"[{dca_level['amount_multiplier']:.1f}x]"
                )
            
            # 杠杆风控配置记录
            self._log_message(
                f"🎯 {pair} 杠杆{lp['leverage']}x | 止损${lp['stop_loss']['price']:.4f}(-{lp['stop_loss']['distance_pct']:.1f}%) | 风险{lp['risk_score']:.0f}/100",
                importance="summary"
            )
            
        except Exception as e:
            logger.error(f"记录交易目标失败 {pair}: {e}")

    def _verify_stoploss_calculation(self, pair: str, leverage: int, final_stoploss: float, 
                                    risk_factors: dict, components: dict):
        """
        🔍 验证止损计算结果的合理性
        """
        try:
            # 验证最终止损值是否在合理范围
            if final_stoploss < 0.01:
                logger.warning(
                    f"⚠️ 止损过小 [{pair}]: {final_stoploss:.1%} < 1%, "
                    f"杠杆={leverage}x, 币种={risk_factors['asset_type']}"
                )
            elif final_stoploss > 0.45:
                logger.warning(
                    f"⚠️ 止损过大[{pair}]: {final_stoploss:.1%} > 45%, "
                    f"杠杆={leverage}x, 币种={risk_factors['asset_type']}"
                )
            
            # 验证各组件是否合理
            if components['base_stoploss'] <= 0:
                logger.error(f"❌ 基础止损计算错误: {components['base_stoploss']}")
            
            if components['atr_component'] < 0:
                logger.error(f"❌ ATR组件计算错误: {components['atr_component']}")
            
            # 记录详细计算过程（DEBUG模式）
            logger.debug(
                f"📋 止损计算详情 {pair}:\n"
                f"  · 输入: 杠杆={leverage}x, 币种={risk_factors['asset_type']}\n"
                f"  · 风险因子: {risk_factors}\n"
                f"  · 计算组件: {components}\n"
                f"  · 结果: {final_stoploss:.1%}"
            )
            
        except Exception as e:
            logger.error(f"止损验证异常: {e}")

    def calculate_leverage_adjusted_params(self, leverage: int, atr_value: float, entry_price: float, is_short: bool = False) -> dict:
        """
        🎯 杠杆自适应参数计算系统
        根据不同杠杆等级自动调整所有风控参数和价格目标
        
        参数:
            leverage: 当前使用的杠杆倍数 (1-20)
            atr_value: ATR绝对值（价格单位）
            entry_price: 入场价格
            is_short: 是否做空
        
        返回:
            包含所有调整后参数和价格目标的字典
        """
        
        # === 1. 杠杆风险等级分类 ===
        if leverage <= 3:
            risk_level = "低风险"
            risk_emoji = "🟢"
            stop_multiplier = 3.0
            trail_activation = 0.05
            trail_distance = 0.03
            dca_trigger = 0.03
            dca_multiplier = 1.5
            max_dca = 5
        elif leverage <= 6:
            risk_level = "中低风险" 
            risk_emoji = "🔵"
            stop_multiplier = 2.0
            trail_activation = 0.04
            trail_distance = 0.025
            dca_trigger = 0.025
            dca_multiplier = 1.3
            max_dca = 4
        elif leverage <= 10:
            risk_level = "中等风险"
            risk_emoji = "🟡"
            stop_multiplier = 1.5
            trail_activation = 0.03
            trail_distance = 0.02
            dca_trigger = 0.02
            dca_multiplier = 1.2
            max_dca = 3
        elif leverage <= 15:
            risk_level = "高风险"
            risk_emoji = "🟠"
            stop_multiplier = 1.0
            trail_activation = 0.02
            trail_distance = 0.015
            dca_trigger = 0.015
            dca_multiplier = 1.0
            max_dca = 2
        else:  # 16-20x
            risk_level = "极高风险"
            risk_emoji = "🔴"
            stop_multiplier = 0.7
            trail_activation = 0.015
            trail_distance = 0.01
            dca_trigger = 0.01
            dca_multiplier = 0.7
            max_dca = 2
        
        # === 2. 计算止损价格 ===
        stop_distance = atr_value * stop_multiplier
        if not is_short:
            stop_loss_price = entry_price - stop_distance
            trailing_trigger_price = entry_price * (1 + trail_activation)
            liquidation_price = entry_price * (1 - 1.0 / leverage * 0.95)  # 考虑5%安全边际
        else:
            stop_loss_price = entry_price + stop_distance
            trailing_trigger_price = entry_price * (1 - trail_activation)
            liquidation_price = entry_price * (1 + 1.0 / leverage * 0.95)
        
        # === 3. 计算止盈价格（4级系统）===
        # 基础倍数根据杠杆调整
        tp_base_multipliers = {
            1: 1.5 if leverage <= 5 else 1.2,   # TP1倍数
            2: 2.5 if leverage <= 5 else 2.0,   # TP2倍数
            3: 4.0 if leverage <= 5 else 3.0,   # TP3倍数
            4: 6.0 if leverage <= 5 else 4.5    # TP4倍数
        }
        
        take_profit_prices = {}
        for i in range(1, 5):
            tp_distance = atr_value * tp_base_multipliers[i]
            if not is_short:
                tp_price = entry_price + tp_distance
                tp_pct = (tp_price / entry_price - 1) * 100
            else:
                tp_price = entry_price - tp_distance
                tp_pct = (1 - tp_price / entry_price) * 100
            
            # 分配平仓比例
            close_ratio = [0.25, 0.35, 0.25, 0.15][i-1]
            
            take_profit_prices[f'tp{i}'] = {
                'price': tp_price,
                'profit_pct': tp_pct,
                'close_ratio': close_ratio,
                'description': ['快速获利', '主要获利', '趋势延伸', '超额收益'][i-1]
            }
        
        # === 4. 计算DCA价格点 ===
        dca_prices = []
        dca_deviation = dca_trigger
        for i in range(max_dca):
            if not is_short:
                dca_price = entry_price * (1 - dca_deviation)
            else:
                dca_price = entry_price * (1 + dca_deviation)
            
            dca_prices.append({
                'level': i + 1,
                'price': dca_price,
                'deviation_pct': dca_deviation * 100,
                'amount_multiplier': dca_multiplier ** i  # 指数增长或固定
            })
            
            # 下一级DCA偏差递增
            dca_deviation *= 1.5  # 每级增加50%偏差
        
        # === 5. 计算风险指标 ===
        if not is_short:
            distance_to_stop = (entry_price - stop_loss_price) / entry_price * 100
            distance_to_liquidation = (entry_price - liquidation_price) / entry_price * 100
        else:
            distance_to_stop = (stop_loss_price - entry_price) / entry_price * 100
            distance_to_liquidation = (liquidation_price - entry_price) / entry_price * 100
        
        # 风险评分（0-100，越低越安全）
        risk_score = min(100, leverage * 5 + (100 - distance_to_liquidation * 2))
        
        return {
            'leverage': leverage,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'risk_score': risk_score,
            
            # 止损配置
            'stop_loss': {
                'price': stop_loss_price,
                'distance_pct': distance_to_stop,
                'atr_multiplier': stop_multiplier
            },
            
            # 跟踪止损配置
            'trailing_stop': {
                'activation_pct': trail_activation * 100,
                'activation_price': trailing_trigger_price,
                'distance_pct': trail_distance * 100
            },
            
            # 止盈目标
            'take_profit': take_profit_prices,
            
            # DCA配置
            'dca': {
                'trigger_pct': dca_trigger * 100,
                'multiplier': dca_multiplier,
                'max_orders': max_dca,
                'price_levels': dca_prices
            },
            
            # 爆仓警告
            'liquidation': {
                'price': liquidation_price,
                'distance_pct': distance_to_liquidation,
                'warning': distance_to_liquidation < 5  # 距离爆仓小于5%时警告
            },
            
            # 时间戳
            'calculated_at': datetime.now(timezone.utc)
        }
    
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
            
            self.event_log.info(
                "takeprofit_calc",
                pair=pair,
                trade_direction="short" if trade.is_short else "long",
                entry_price=f"{trade.open_rate:.6f}",
                current_price=f"{current_rate:.6f}",
                current_profit=f"{current_profit:.2%}",
                atr_multiplier=f"{profit_multiplier:.2f}",
                distance=f"{profit_distance:.2%}",
                target_price=f"{target_price:.6f}",
            )
            
            return target_price
            
        except Exception as e:
            logger.error(f"动态止盈计算失败 {pair}: {e}")
            return None
    
    # 移除了 get_smart_trailing_stop - 简化止损逻辑
    
    def validate_and_calibrate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """验证和校准技术指标的准确性"""
        try:
            self.event_log.info("indicator_validation_start", rows=len(dataframe))
            
            # === RSI 指标校准 ===
            if 'rsi_14' in dataframe.columns:
                # 处理RSI异常值和空值
                original_rsi_nulls = dataframe['rsi_14'].isnull().sum()
                dataframe['rsi_14'] = dataframe['rsi_14'].clip(0, 100)
                dataframe['rsi_14'] = dataframe['rsi_14'].fillna(50)
                
                # RSI平滑处理（减少噪音）
                dataframe['rsi_14'] = dataframe['rsi_14'].ewm(span=2).mean()
                
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="rsi_14",
                    nulls=original_rsi_nulls,
                    actions="clip[0,100]|fill=50|ewm_span=2",
                )
            
            # === MACD 指标校准 ===
            if 'macd' in dataframe.columns:
                # MACD指标平滑处理
                original_macd_nulls = dataframe['macd'].isnull().sum()
                dataframe['macd'] = dataframe['macd'].fillna(0)
                dataframe['macd'] = dataframe['macd'].ewm(span=3).mean()
                
                if 'macd_signal' in dataframe.columns:
                    dataframe['macd_signal'] = dataframe['macd_signal'].fillna(0)
                    dataframe['macd_signal'] = dataframe['macd_signal'].ewm(span=3).mean()
                
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="macd",
                    nulls=original_macd_nulls,
                    actions="fill=0|ewm_span=3",
                )
            
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
                
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="atr_p",
                    outliers=original_atr_outliers,
                    bounds=f"{lower_bound:.4f}-{upper_bound:.4f}",
                )
            
            # === ADX 指标校准 ===
            if 'adx' in dataframe.columns:
                dataframe['adx'] = dataframe['adx'].clip(0, 100)
                dataframe['adx'] = dataframe['adx'].fillna(25)  # ADX默认值25
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="adx",
                    actions="clip[0,100]|fill=25",
                )
            
            # === 成交量比率校准 ===
            if 'volume_ratio' in dataframe.columns:
                # 限制成交量比率在合理范围内
                dataframe['volume_ratio'] = dataframe['volume_ratio'].clip(0.1, 20)
                dataframe['volume_ratio'] = dataframe['volume_ratio'].fillna(1.0)
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="volume_ratio",
                    actions="clip[0.1,20]|fill=1.0",
                )
            
            # === 趋势强度校准 ===
            if 'trend_strength' in dataframe.columns:
                dataframe['trend_strength'] = dataframe['trend_strength'].clip(-100, 100)
                dataframe['trend_strength'] = dataframe['trend_strength'].fillna(50)
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="trend_strength",
                    actions="clip[-100,100]|fill=50",
                )
            
            # === 动量评分校准 ===
            if 'momentum_score' in dataframe.columns:
                dataframe['momentum_score'] = dataframe['momentum_score'].clip(-3, 3)
                dataframe['momentum_score'] = dataframe['momentum_score'].fillna(0)
                self.event_log.info(
                    "indicator_calibrated",
                    indicator="momentum_score",
                    actions="clip[-3,3]|fill=0",
                )
            
            # === EMA 指标保护 ===
            # 确保EMA指标不被过度处理，保持原始计算结果
            for ema_col in ['ema_8', 'ema_21', 'ema_50']:
                if ema_col in dataframe.columns:
                    # 只处理明显的异常值和空值，不进行平滑处理
                    null_count = dataframe[ema_col].isnull().sum()
                    if null_count > 0:
                        # 使用前向填充处理少量空值
                        dataframe[ema_col] = dataframe[ema_col].ffill().bfill()
                        self.event_log.info(
                            "ema_null_fill",
                            indicator=ema_col,
                            filled_nulls=null_count,
                        )
                    
                    # 检查是否有明显异常的EMA值（价格的10倍以上差异）
                    if 'close' in dataframe.columns:
                        price_ratio = dataframe[ema_col] / dataframe['close']
                        outliers = ((price_ratio > 10) | (price_ratio < 0.1)).sum()
                        if outliers > 0:
                            self.event_log.warning(
                                "ema_outlier_reset",
                                indicator=ema_col,
                                outliers=outliers,
                            )
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
            indicator_statuses: list[str] = []

            # 检查各个指标的健康状况
            indicators_to_check = ['rsi_14', 'macd', 'atr_p', 'adx', 'volume_ratio', 'trend_strength', 'momentum_score', 'ema_8', 'ema_21', 'ema_50']

            for indicator in indicators_to_check:
                if indicator in dataframe.columns:
                    series = dataframe[indicator].dropna()
                    if len(series) > 0:
                        null_count = dataframe[indicator].isnull().sum()
                        null_pct = null_count / len(dataframe) * 100

                        if null_pct < 5:
                            health_status = "healthy"
                        elif null_pct < 15:
                            health_status = "warning"
                        else:
                            health_status = "critical"

                        indicator_statuses.append(health_status)
                        self.event_log.info(
                            "indicator_health",
                            indicator=indicator,
                            status=health_status,
                            null_pct=f"{null_pct:.1f}%",
                        )

            if indicator_statuses:
                if all(status == "healthy" for status in indicator_statuses):
                    overall = "excellent"
                elif any(status == "critical" for status in indicator_statuses):
                    overall = "attention"
                else:
                    overall = "good"

                self.event_log.info(
                    "indicator_health_summary",
                    indicators=len(indicator_statuses),
                    overall=overall,
                )
        except Exception as e:
            logger.error(f"指标健康检查失败: {e}")
    
    def validate_real_data_quality(self, dataframe: DataFrame, pair: str) -> bool:
        """验证数据是否为真实市场数据而非模拟数据"""
        try:
            if len(dataframe) < 10:
                self.event_log.warning("data_quality_rows", pair=pair, rows=len(dataframe))
                return False
            
            # 检查价格数据的合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in dataframe.columns:
                    if dataframe[col].isnull().all():
                        self.event_log.error("data_quality_price_null", pair=pair, column=col)
                        return False
                    
                    # 检查价格是否有合理的变化
                    price_std = dataframe[col].std()
                    price_mean = dataframe[col].mean()
                    if price_std / price_mean < 0.001:  # 变化率低于0.1%
                        self.event_log.warning(
                            "data_quality_price_variation",
                            pair=pair,
                            column=col,
                            ratio=f"{price_std/price_mean:.6f}",
                        )
            
            # 检查成交量数据
            if 'volume' in dataframe.columns:
                if dataframe['volume'].sum() == 0:
                    self.event_log.warning("data_quality_volume_zero", pair=pair)
                else:
                    # 检查成交量是否有合理的变化
                    volume_std = dataframe['volume'].std()
                    volume_mean = dataframe['volume'].mean()
                    if volume_mean > 0 and volume_std / volume_mean < 0.1:
                        self.event_log.warning(
                            "data_quality_volume_variation",
                            pair=pair,
                            ratio=f"{volume_std/volume_mean:.6f}",
                        )
            
            # 检查时间戳连续性
            if 'date' in dataframe.columns or dataframe.index.name == 'date':
                time_diff = dataframe.index.to_series().diff().dropna()
                if len(time_diff) > 0:
                    # 动态计算预期时间间隔，使用最常见的时间间隔作为预期值
                    expected_interval = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else pd.Timedelta(minutes=5)
                    abnormal_intervals = (time_diff != expected_interval).sum()
                    if abnormal_intervals > len(time_diff) * 0.1:  # 超过10%的时间间隔异常
                        self.event_log.warning(
                            "data_quality_time_interval",
                            pair=pair,
                            abnormal=f"{abnormal_intervals}/{len(time_diff)}",
                            expected=str(expected_interval),
                        )

            self.event_log.info("data_quality_pass", pair=pair, rows=len(dataframe))
            return True
        
        except Exception as e:
            self.event_log.error("data_quality_failure", pair=pair, error=str(e))
            return False
    
    # 移除了 _log_detailed_exit_decision 方法 - 简化日志
    
    def _log_risk_calculation_details(self, pair: str, input_params: dict, result: dict):
        """记录详细的风险计算信息"""
        try:
            risk_pct = result.get('risk_percentage')
            risk_label = result.get('risk_rating')

            log_fields: Dict[str, Any] = {
                'pair': pair,
                'side': input_params.get('side', 'unknown'),
                'entry_tag': input_params.get('entry_tag'),
                'planned_position': f"{input_params.get('planned_position', 0):.2%}" if input_params.get('planned_position') is not None else None,
                'leverage': result.get('suggested_leverage'),
                'risk_amt': result.get('risk_amount'),
                'risk_pct': f"{risk_pct:.2%}" if isinstance(risk_pct, (float, int)) else None,
                'risk_rating': risk_label,
                'market_state': input_params.get('market_state'),
            }

            # 过滤掉空字段，保持日志精简
            log_fields = {k: v for k, v in log_fields.items() if v is not None}
            self.event_log.info("risk_calculation", importance="summary", **log_fields)
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
    
    def analyze_multi_timeframe(self, dataframe: DataFrame, metadata: dict) -> Dict:
        """🕐 增强MTF多时间框架确认机制 - 提升信号质量"""
        
        pair = metadata.get('pair', 'UNKNOWN')
        
        # === 时间框架配置（长线优化版：15m+1h双重确认）===
        # 保持快速响应，避免4h延迟
        timeframes = {
            '15m': {'weight': 0.60, 'required_candles': 100},  # 主周期
            '1h': {'weight': 0.40, 'required_candles': 50},    # 短期趋势确认
        }
        
        mtf_analysis = {}
        
        for tf, config in timeframes.items():
            try:
                # 获取指定时间框架数据
                # 修正：只有当 tf 等于策略主时间框架(self.timeframe)时，才使用传入的 dataframe。
                # 其他时间框架统一从 DataProvider 获取，避免错误地把15m当作5m使用。
                if tf == self.timeframe:
                    tf_dataframe = dataframe
                else:
                    tf_dataframe = self.dp.get_pair_dataframe(pair, tf)
                
                # 数据有效性检查
                if tf_dataframe.empty or len(tf_dataframe) < config['required_candles']:
                    logger.debug(f"MTF {tf}: 数据不足，使用默认分析")
                    mtf_analysis[tf] = self._get_default_tf_analysis()
                    continue
                
                # === 核心技术分析 ===
                current_data = tf_dataframe.iloc[-1]
                recent_data = tf_dataframe.tail(20)  # 最近20根K线
                
                # 基础指标获取
                close = current_data.get('close', 0)
                high = current_data.get('high', close)
                low = current_data.get('low', close)
                
                # 计算缺失指标（如果不存在）
                if 'rsi_14' not in tf_dataframe.columns:
                    tf_dataframe['rsi_14'] = ta.RSI(tf_dataframe['close'], timeperiod=14)
                if 'ema_21' not in tf_dataframe.columns:
                    tf_dataframe['ema_21'] = ta.EMA(tf_dataframe['close'], timeperiod=21)
                if 'ema_50' not in tf_dataframe.columns:
                    tf_dataframe['ema_50'] = ta.EMA(tf_dataframe['close'], timeperiod=50)
                if 'adx' not in tf_dataframe.columns:
                    tf_dataframe['adx'] = ta.ADX(tf_dataframe['high'], tf_dataframe['low'], tf_dataframe['close'], timeperiod=14)
                if 'macd' not in tf_dataframe.columns:
                    macd, macd_signal, macd_hist = ta.MACD(tf_dataframe['close'])
                    tf_dataframe['macd'] = macd
                    tf_dataframe['macd_signal'] = macd_signal
                
                # 重新获取当前数据（包含新计算的指标）
                current_data = tf_dataframe.iloc[-1]
                
                rsi = current_data.get('rsi_14', 50)
                ema_21 = current_data.get('ema_21', close)
                ema_50 = current_data.get('ema_50', close)
                adx = current_data.get('adx', 25)
                macd = current_data.get('macd', 0)
                macd_signal = current_data.get('macd_signal', 0)
                
                # === 1. 趋势方向分析 ===
                # EMA排列分析
                ema_bullish = close > ema_21 > ema_50
                ema_bearish = close < ema_21 < ema_50
                
                # MACD趋势确认
                macd_bullish = macd > macd_signal
                macd_bearish = macd < macd_signal
                
                # RSI趋势确认（放宽空头判定，提高对下行的敏感度）
                rsi_bullish = rsi > 55
                rsi_bearish = rsi < 48
                
                # 综合趋势评分 (-1 to 1)
                trend_factors = [
                    1 if ema_bullish else -1 if ema_bearish else 0,
                    1 if macd_bullish else -1 if macd_bearish else 0,
                    1 if rsi_bullish else -1 if rsi_bearish else 0
                ]
                trend_score = sum(trend_factors) / len(trend_factors)
                
                # 放宽空头阈值：更容易识别为看空
                if trend_score > 0.33:
                    trend_direction = 'bullish'
                    trend = 'up'
                elif trend_score < -0.20:
                    trend_direction = 'bearish'
                    trend = 'down'
                else:
                    trend_direction = 'neutral'
                    trend = 'sideways'
                
                # === 2. 趋势强度分析 ===
                # ADX强度评估
                if adx > 35:
                    adx_strength = 'very_strong'
                elif adx > 25:
                    adx_strength = 'strong'
                elif adx > 20:
                    adx_strength = 'moderate'
                else:
                    adx_strength = 'weak'
                
                # 价格位置分析（20期高低点）
                highest_20 = recent_data['high'].max()
                lowest_20 = recent_data['low'].min()
                price_position = (close - lowest_20) / (highest_20 - lowest_20 + 0.0001)
                
                # === 3. 动量分析 ===
                # 价格动量（5期ROC）
                price_momentum = ((close - tf_dataframe['close'].shift(5).iloc[-1]) / 
                                tf_dataframe['close'].shift(5).iloc[-1] * 100) if len(tf_dataframe) > 5 else 0
                
                # RSI动量
                rsi_momentum = rsi - tf_dataframe['rsi_14'].shift(3).iloc[-1] if len(tf_dataframe) > 3 else 0
                
                if price_momentum > 2 and rsi_momentum > 5:
                    momentum = 'strong_bullish'
                elif price_momentum > 0.5 and rsi_momentum > 0:
                    momentum = 'bullish'
                elif price_momentum < -2 and rsi_momentum < -5:
                    momentum = 'strong_bearish'
                elif price_momentum < -0.5 and rsi_momentum < 0:
                    momentum = 'bearish'
                else:
                    momentum = 'neutral'
                
                # === 4. 关键位置识别 ===
                is_top = price_position > 0.85 and rsi > 70
                is_bottom = price_position < 0.15 and rsi < 30
                
                # === 5. 确认信号强度评估 ===
                confirmation_factors = [
                    1 if ema_bullish or ema_bearish else 0,  # EMA排列确认
                    1 if abs(rsi - 50) > 10 else 0,          # RSI方向明确
                    1 if adx > 20 else 0,                    # 有趋势
                    1 if abs(price_momentum) > 1 else 0      # 动量明确
                ]
                confirmation_strength = sum(confirmation_factors) / len(confirmation_factors)
                
                # === 组装结果 ===
                mtf_analysis[tf] = {
                    'trend': trend,
                    'trend_direction': trend_direction,
                    'trend_strength': adx_strength,
                    'trend_score': trend_score,  # -1 to 1
                    'rsi': rsi,
                    'adx': adx,
                    'price_position': price_position,
                    'price_momentum': price_momentum,
                    'rsi_momentum': rsi_momentum,
                    'momentum': momentum,
                    'is_top': is_top,
                    'is_bottom': is_bottom,
                    'ema_alignment': trend_direction,
                    'confirmation_strength': confirmation_strength,  # 0 to 1
                    'macd_trend': 'bullish' if macd_bullish else 'bearish' if macd_bearish else 'neutral'
                }
                
                logger.debug(f"MTF {tf}: {trend_direction} (强度:{adx_strength}, 位置:{price_position:.2f})")
                
            except Exception as e:
                logger.warning(f"MTF {tf} 分析失败: {e}")
                mtf_analysis[tf] = self._get_default_tf_analysis()
        
        # === 多时间框架一致性检查 ===
        if len(mtf_analysis) >= 2:
            mtf_analysis['mtf_consensus'] = self._calculate_mtf_consensus(mtf_analysis, timeframes)
        
        return mtf_analysis
    
    def _get_default_tf_analysis(self) -> Dict:
        """返回默认的时间框架分析结果"""
        return {
            'trend': 'unknown',
            'trend_direction': 'neutral',
            'trend_strength': 'weak',
            'trend_score': 0,
            'rsi': 50,
            'adx': 20,
            'price_position': 0.5,
            'price_momentum': 0,
            'rsi_momentum': 0,
            'momentum': 'neutral',
            'is_top': False,
            'is_bottom': False,
            'ema_alignment': 'neutral',
            'confirmation_strength': 0,
            'macd_trend': 'neutral'
        }
    
    def _calculate_mtf_consensus(self, mtf_analysis: Dict, timeframes: Dict) -> Dict:
        """计算多时间框架一致性共识"""
        
        # 加权趋势评分
        weighted_trend_score = 0
        weighted_confirmation = 0
        total_weight = 0
        
        bullish_tfs = []
        bearish_tfs = []
        neutral_tfs = []
        
        for tf, weight_config in timeframes.items():
            if tf in mtf_analysis and tf != 'mtf_consensus':
                analysis = mtf_analysis[tf]
                weight = weight_config['weight']
                
                # 累积加权评分
                trend_score = analysis.get('trend_score', 0)
                confirmation = analysis.get('confirmation_strength', 0)
                
                weighted_trend_score += trend_score * weight
                weighted_confirmation += confirmation * weight
                total_weight += weight
                
                # 记录各时间框架趋势
                direction = analysis.get('trend_direction', 'neutral')
                if direction == 'bullish':
                    bullish_tfs.append(tf)
                elif direction == 'bearish':
                    bearish_tfs.append(tf)
                else:
                    neutral_tfs.append(tf)
        
        # 标准化权重
        if total_weight > 0:
            weighted_trend_score /= total_weight
            weighted_confirmation /= total_weight
        
        # 一致性评级
        total_tfs = len(bullish_tfs) + len(bearish_tfs) + len(neutral_tfs)
        if total_tfs == 0:
            consensus_strength = 0
            consensus_direction = 'neutral'
        else:
            bullish_ratio = len(bullish_tfs) / total_tfs
            bearish_ratio = len(bearish_tfs) / total_tfs
            
            if bullish_ratio >= 0.75:
                consensus_strength = 'very_strong'
                consensus_direction = 'bullish'
            elif bullish_ratio >= 0.5:
                consensus_strength = 'moderate'
                consensus_direction = 'bullish'
            elif bearish_ratio >= 0.75:
                consensus_strength = 'very_strong'
                consensus_direction = 'bearish'
            elif bearish_ratio >= 0.5:
                consensus_strength = 'moderate'
                consensus_direction = 'bearish'
            else:
                consensus_strength = 'weak'
                consensus_direction = 'neutral'
        
        return {
            'weighted_trend_score': weighted_trend_score,  # -1 to 1
            'weighted_confirmation': weighted_confirmation, # 0 to 1
            'consensus_direction': consensus_direction,
            'consensus_strength': consensus_strength,
            'bullish_timeframes': bullish_tfs,
            'bearish_timeframes': bearish_tfs,
            'neutral_timeframes': neutral_tfs,
            'alignment_ratio': max(bullish_ratio, bearish_ratio) if total_tfs > 0 else 0
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
        try:
            # 标量（含字符串、布尔等）直接重复铺满
            if pd.api.types.is_scalar(data):
                scalar_value = fill_value if data is None else data
                return pd.Series([scalar_value] * length, index=range(length))

            # 可迭代对象且长度匹配时按原样构造
            if hasattr(data, '__len__') and len(data) == length:
                return pd.Series(data, index=range(length))

        except Exception:
            pass

        # 其他情况退回到填充值
        return pd.Series([fill_value] * length, index=range(length))

    def _apply_signal_cooldown(self, signal_series: pd.Series, periods: int) -> pd.Series:
        """为布尔信号序列施加滚动冷却，防止短时间内重复触发"""
        if periods <= 1 or signal_series.empty:
            return signal_series.astype(bool)

        # 使用滚动最大值检测冷却窗口内是否已有信号
        recent_activation = (
            signal_series.astype(int)
            .rolling(window=periods, min_periods=1)
            .max()
            .shift(1)
            .fillna(0)
            .astype(bool)
        )

        return signal_series.astype(bool) & (~recent_activation)
    
    def calculate_candle_quality(self, dataframe: DataFrame) -> pd.Series:
        """
        计算K线实体/影线比质量指标 (基于ChatGPT建议)
        用于过滤假突破和确认真实突破
        """
        # K线实体
        body = abs(dataframe['close'] - dataframe['open'])
        
        # 上影线
        upper_shadow = dataframe['high'] - dataframe[['close', 'open']].max(axis=1)
        
        # 下影线  
        lower_shadow = dataframe[['close', 'open']].min(axis=1) - dataframe['low']
        
        # 实体/影线比 (避免除零)
        shadow_total = upper_shadow + lower_shadow + 1e-8
        body_ratio = body / shadow_total
        
        return body_ratio
    
    def calculate_unified_risk_factors(self, pair: str = None, dataframe: DataFrame = None,
                                      leverage: int = None, current_atr: float = None) -> dict:
        """
        🧠 统一的多因子风险计算框架 V2.0
        供动态止损和跟踪止损共同使用的核心计算组件

        返回标准化的风险因子集合：
        - asset_type: 币种类型
        - base_risk: 基础风险比例
        - leverage_factor: 杠杆调整因子
        - atr_factor: ATR波动调整因子
        - trend_factor: 趋势强度因子
        - atr_percentile: ATR历史百分位
        """
        factors = {
            'asset_type': 'Others',
            'base_risk': 0.04,
            'leverage_factor': 1.0,
            'atr_factor': 1.0,
            'trend_factor': 1.0,
            'atr_percentile': 1.0,
            'adx_value': 25.0,
            'market_condition': 'normal'
        }
        
        try:
            # 1. 资产类型识别
            ASSET_CONFIG = {
                'BTC': {'base_risk': 0.025, 'atr_multiplier': 3.0},
                'ETH': {'base_risk': 0.030, 'atr_multiplier': 3.2},
                'SOL': {'base_risk': 0.030, 'atr_multiplier': 3.5},
                'BNB': {'base_risk': 0.030, 'atr_multiplier': 3.5},
                'Others': {'base_risk': 0.020, 'atr_multiplier': 4.0}  # 🎯 大幅增加非大币止损空间，避免被扫
            }
            
            if pair:
                pair_upper = pair.upper()
                if 'BTC' in pair_upper and 'BTCDOM' not in pair_upper:
                    factors['asset_type'] = 'BTC'
                elif 'ETH' in pair_upper:
                    factors['asset_type'] = 'ETH'
                elif 'SOL' in pair_upper:
                    factors['asset_type'] = 'SOL'
                elif 'BNB' in pair_upper:
                    factors['asset_type'] = 'BNB'
            
            asset_config = ASSET_CONFIG[factors['asset_type']]
            factors['base_risk'] = asset_config['base_risk']
            
            # 2. 杠杆因子（使用sqrt缓和）
            if leverage and leverage > 0:
                factors['leverage_factor'] = 1.0 / (leverage ** 0.5)
            
            # 3. ATR波动性分析
            if current_atr is not None and current_atr > 0:
                # ATR历史百分位
                if dataframe is not None and 'atr_p' in dataframe.columns and len(dataframe) > 50:
                    historical_atr = dataframe['atr_p'].tail(100).dropna()
                    if len(historical_atr) > 20:
                        atr_median = historical_atr.quantile(0.5)
                        if atr_median > 0:
                            factors['atr_percentile'] = max(0.5, min(current_atr / atr_median, 3.0))
                
                # ATR调整因子
                base_multiplier = asset_config['atr_multiplier']
                factors['atr_factor'] = base_multiplier * (0.8 + 0.4 * factors['atr_percentile'])
            
            # 4. 趋势强度分析
            if dataframe is not None and len(dataframe) > 5:
                current_candle = dataframe.iloc[-1]
                adx = current_candle.get('adx', 25)
                factors['adx_value'] = adx
                
                # 趋势分级
                if adx > 40:
                    factors['trend_factor'] = 1.15
                    factors['market_condition'] = 'strong_trend'
                elif adx > 30:
                    factors['trend_factor'] = 1.08
                    factors['market_condition'] = 'moderate_trend'
                elif adx > 20:
                    factors['trend_factor'] = 1.00
                    factors['market_condition'] = 'normal'
                else:
                    factors['trend_factor'] = 0.85
                    factors['market_condition'] = 'choppy'
            
            logger.debug(
                f"🔧 统一风险因子 {pair}: "
                f"币种={factors['asset_type']} | "
                f"杠杆调整={factors['leverage_factor']:.2f} | "
                f"ATR因子={factors['atr_factor']:.2f} | "
                f"趋势因子={factors['trend_factor']:.2f} | "
                f"市场={factors['market_condition']}"
            )
            
        except Exception as e:
            logger.warning(f"风险因子计算异常 {pair}: {e}")

        return factors
    
    def _get_trade_fee_rates(self, trade: Trade) -> tuple[float, float]:
        """获取开平仓手续费率（若缺失则回退策略默认值）。"""
        default_fee = getattr(self, 'fee', 0.001)
        fee_open = trade.fee_open if trade.fee_open is not None else default_fee
        fee_close = trade.fee_close if trade.fee_close is not None else fee_open
        return fee_open, fee_close

    def _calc_slippage_allowance(self, leverage: float) -> float:
        """根据杠杆推导滑点/缓冲占比。"""
        leverage = max(leverage, 1)
        return self.trailing_min_profit_buffer + max(0.0, (leverage - 1) * self.trailing_slippage_per_leverage)

    @staticmethod
    def _ratio_to_price(entry_price: float, ratio: float, is_short: bool) -> float:
        """将收益率转换回绝对价格，便于日志分析。"""
        if entry_price is None or entry_price <= 0:
            return 0.0
        if not is_short:
            return entry_price * (1 + ratio)
        return entry_price * (1 - ratio)

    @staticmethod
    def _price_ratio(entry_price: float, target_price: float, is_short: bool) -> float:
        """根据目标价格计算相对于开仓价的收益率。"""
        if entry_price is None or entry_price <= 0 or target_price is None or target_price <= 0:
            return 0.0
        if not is_short:
            return (target_price / entry_price) - 1
        return 1 - (target_price / entry_price)

    @staticmethod
    def _account_ratio_from_price(price_ratio: float, leverage: float, buffer_ratio: float) -> float:
        """将价格收益率换算成账户层面的收益率（考虑杠杆与手续费/滑点缓冲）。"""
        leverage = max(leverage, 1e-6)
        return price_ratio * leverage - buffer_ratio

    @staticmethod
    def _price_ratio_from_account(account_ratio: float, leverage: float, buffer_ratio: float) -> float:
        """反向换算：给定账户收益率，推导所需的价格收益率。"""
        leverage = max(leverage, 1e-6)
        return (account_ratio + buffer_ratio) / leverage

    def _finalize_stoploss(self, trade: Trade, stoploss_ratio: float, current_rate: float,
                           pair: str, reason: str, leverage: float,
                           fee_ratio_total: float, slippage_allowance: float) -> float:
        """在返回前统一记录止损对应的价格与账户影响。"""
        if stoploss_ratio is None:
            return None

        entry_price = trade.open_rate or 0.0
        stop_price = self._ratio_to_price(entry_price, stoploss_ratio, trade.is_short)
        buffer_ratio = fee_ratio_total + slippage_allowance
        account_impact = stoploss_ratio * max(leverage, 1e-6) - buffer_ratio

        # 统一以 summary 级别输出，受 verbosity 控制
        self._log_message(
            f"⛔ 止损更新[{reason}] {pair}: price={stop_price:.6f} ratio={stoploss_ratio:.2%} "
            f"leverage={leverage:.1f}x account≈{account_impact:.2%}",
            importance="summary"
        )

        return stoploss_ratio
    
    def check_bull_market_environment(self, dataframe: DataFrame) -> pd.Series:
        """
        简化的牛市环境检测 (ChatGPT建议的EMA排列方案)
        """
        # EMA多头排列
        ema_bullish = (dataframe['ema_8'] > dataframe['ema_21'])
        
        # 如果有ema_50，添加更强的确认
        if 'ema_50' in dataframe.columns:
            ema_strong_bullish = (dataframe['ema_8'] > dataframe['ema_21']) & (dataframe['ema_21'] > dataframe['ema_50'])
            return ema_strong_bullish
        
        return ema_bullish
    
    def check_reversal_signal_invalidation(self, dataframe: DataFrame) -> Dict[str, pd.Series]:
        """
        反指信号失效预警系统 (基于ChatGPT建议)
        检测反指信号可能失效的市场条件
        """
        # 计算价格位置百分位
        price_percentile = (dataframe['close'] - dataframe['low'].rolling(50).min()) / \
                          (dataframe['high'].rolling(50).max() - dataframe['low'].rolling(50).min() + 0.0001)
        
        # BB反指信号失效条件
        bb_reversal_invalid = (
            # 高位极度超买
            (dataframe['rsi_14'] > 80) &
            (price_percentile > 0.95) &
            
            # 成交量不足
            (dataframe['volume_ratio'] < 1.1)
        ) | (
            # 技术反转信号确认
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(2)) &  # MACD持续收缩
            (dataframe['close'] < dataframe['ema_8'])  # 价格跌破短期均线
        )
        
        # MACD反指信号失效条件
        macd_reversal_invalid = (
            # 真正的顶部反转信号
            (dataframe['rsi_14'] > 80) &
            (dataframe['close'] < dataframe['close'].rolling(5).mean()) &  # 价格连续下跌
            (dataframe['volume_ratio'] > 1.5) &  # 放量下跌
            
            # 或EMA排列转空
            (dataframe['ema_8'] < dataframe['ema_21'])
        )
        
        # 通用失效条件
        general_invalid = (
            # 市场情绪由正转负（可根据需要扩展）
            (dataframe['adx'] < 15) |  # 无趋势环境
            (dataframe['atr_p'] > dataframe['atr_p'].rolling(20).quantile(0.9))  # 极端波动
        )
        
        return {
            'bb_reversal_invalid': bb_reversal_invalid,
            'macd_reversal_invalid': macd_reversal_invalid,
            'general_invalid': general_invalid,
            'any_invalid': bb_reversal_invalid | macd_reversal_invalid | general_invalid
        }
    
    def calculate_signal_quality_grade(self, signal_quality_score: float, enter_tag: str = None) -> str:
        """
        基于信号质量评分计算档位等级
        质量评分系统: 1-10分 → high/medium/low confidence
        """
        # 特殊信号类型处理
        if enter_tag and 'Reversal' in enter_tag:
            # 反指信号默认降一档处理
            if signal_quality_score >= 8.5:
                return 'medium_confidence'
            elif signal_quality_score >= 6.0:
                return 'low_confidence'
            else:
                return 'very_low_confidence'
        
        # 标准信号档位映射
        if signal_quality_score >= 8.0:
            return 'high_confidence'
        elif signal_quality_score >= 6.0:
            return 'medium_confidence'
        elif signal_quality_score >= 4.0:
            return 'low_confidence'
        else:
            return 'very_low_confidence'
    
    def calculate_atr_percentile(self, dataframe: DataFrame, current_atr: float, lookback_periods: int = 100) -> float:
        """
        计算当前ATR相对于历史ATR的百分位数排名
        用于判断当前波动性是否异常
        """
        try:
            if 'atr_p' not in dataframe.columns or len(dataframe) < lookback_periods:
                return 1.0  # 默认正常水平
            
            # 获取历史ATR数据
            historical_atr = dataframe['atr_p'].tail(lookback_periods).dropna()
            if len(historical_atr) < 20:
                return 1.0
            
            # 计算50th百分位数（中位数）
            atr_median = historical_atr.quantile(0.5)
            
            # 避免除零错误
            if atr_median <= 0:
                return 1.0
            
            # 当前ATR相对于历史中位数的比值
            atr_ratio = current_atr / atr_median
            
            # 限制在合理范围内 (0.5 - 3.0)
            return max(0.5, min(atr_ratio, 3.0))
            
        except Exception as e:
            logger.debug(f"ATR百分位数计算失败: {e}")
            return 1.0
    
    def calculate_trend_strength_factor(self, dataframe: DataFrame) -> float:
        """
        基于ADX计算趋势强度调整因子
        强趋势给更大止损空间，震荡市收紧止损
        """
        try:
            if len(dataframe) < 5:
                return 1.0
                
            current_candle = dataframe.iloc[-1]
            adx = current_candle.get('adx', 25)
            
            # 趋势强度分级
            if adx > 40:
                # 极强趋势 - 给更大空间
                trend_factor = 1.15
            elif adx > 30:
                # 强趋势 - 适度放宽
                trend_factor = 1.08  
            elif adx > 20:
                # 中等趋势 - 标准
                trend_factor = 1.00
            else:
                # 弱趋势/震荡 - 收紧止损
                trend_factor = 0.85
            
            return trend_factor
            
        except Exception as e:
            logger.debug(f"趋势强度计算失败: {e}")
            return 1.0
    
    def calculate_dynamic_stoploss(self, signal_quality_grade: str, leverage: int, current_atr: float, 
                                 pair: str = None, dataframe: DataFrame = None) -> float:
        """
        🧠 深度优化的动态止损计算系统 V3.1 - 使用统一风险框架
        
        核心理念：在风险控制和趋势跟随之间找到最佳平衡
        
        主公式：
        Dynamic_StopLoss = (Base_Risk × Leverage_Factor) + ATR_Component × (Trend_Factor × Quality_Adj)
        """
        try:
            # 获取统一风险因子
            risk_factors = self.calculate_unified_risk_factors(
                pair=pair,
                dataframe=dataframe,
                leverage=leverage,
                current_atr=current_atr
            )
            
            # 基础止损（币种风险 × 杠杆调整）
            base_stoploss = risk_factors['base_risk'] * risk_factors['leverage_factor']
            
            # ATR动态组件 （加法而非乘法）
            atr_component = 0.0
            if current_atr > 0:
                atr_component = current_atr * risk_factors['atr_factor']
                # 放宽ATR贡献上限，给剧烈行情更多缓冲
                atr_component = min(atr_component, 0.18)
            
            # 额外的趋势确认调整
            trend_adjustment = risk_factors['trend_factor']
            if dataframe is not None and len(dataframe) > 5:
                current_candle = dataframe.iloc[-1]
                ema_8 = current_candle.get('ema_8', 0)
                ema_21 = current_candle.get('ema_21', 0)
                ema_50 = current_candle.get('ema_50', 0)
                
                # EMA排列确认趋势
                if ema_8 > ema_21 > ema_50 or ema_8 < ema_21 < ema_50:
                    trend_adjustment *= 1.1  # 趋势确认，额外10%空间
            
            # 信号质量调整
            QUALITY_ADJUSTMENTS = {
                'high_confidence': 1.2,       # 高质量信号20%更多空间
                'medium_confidence': 1.0,     # 标准
                'low_confidence': 0.85,       # 低质量收紧15%
                'very_low_confidence': 0.7    # 极低质量收紧30%
            }
            quality_adjustment = QUALITY_ADJUSTMENTS.get(signal_quality_grade, 1.0)
            
            # 综合计算（混合加法和乘法）
            # 基础止损 + ATR贡献，然后应用趋势和质量调整
            dynamic_stoploss = base_stoploss + atr_component
            dynamic_stoploss *= trend_adjustment * quality_adjustment
            
            # 🔧 Meme币优化：基于账户止损比例（freqtrade标准）
            # 返回值含义：账户亏损X%时触发止损，对应价格波动 = X% / 杠杆
            # 例如：14x杠杆，账户止损-20% → 价格波动1.43%

            # 根据信号质量设置账户止损范围（逐仓风控）
            QUALITY_ACCOUNT_STOPLOSS = {
                'high_confidence': 0.25,      # 25%账户止损（高质量多给空间）
                'medium_confidence': 0.20,    # 20%账户止损（标准）
                'low_confidence': 0.15,       # 15%账户止损（低质量收紧）
                'very_low_confidence': 0.12   # 12%账户止损（极低质量）
            }

            base_account_stop = QUALITY_ACCOUNT_STOPLOSS.get(signal_quality_grade, 0.20)

            # 波动率调整（ATR高时放宽止损）
            if atr_component > 0.08:  # 高波动环境
                volatility_adjustment = 1.15
            elif atr_component > 0.05:  # 中等波动
                volatility_adjustment = 1.08
            else:  # 低波动
                volatility_adjustment = 1.0

            min_account_stoploss = base_account_stop * volatility_adjustment

            # 应用边界
            final_stoploss = max(dynamic_stoploss, min_account_stoploss)

            # 绝对最大账户止损限制（逐仓保护）
            # 避免单笔交易损失过多保证金
            max_account_loss = 0.35  # 35%账户止损上限
            final_stoploss = min(final_stoploss, max_account_loss)
            
            # 详细日志记录和验证
            self._verify_stoploss_calculation(
                pair=pair,
                leverage=leverage,
                final_stoploss=final_stoploss,
                risk_factors=risk_factors,
                components={
                    'base_stoploss': base_stoploss,
                    'atr_component': atr_component,
                    'trend_adjustment': trend_adjustment,
                    'quality_adjustment': quality_adjustment
                }
            )
            
            self._log_message(
                f"💰 动态止损V3.1 [{leverage}x杠杆] {pair or 'Unknown'}: "
                f"币种={risk_factors['asset_type']} | "
                f"基础={base_stoploss:.1%} | "
                f"ATR贡献={atr_component:.1%} | "
                f"趋势={trend_adjustment:.2f} | "
                f"质量={quality_adjustment:.2f} | "
                f"最终={final_stoploss:.1%}",
                importance="summary"
            )
            
            return final_stoploss
            
        except Exception as e:
            logger.error(f"动态止损计算异常 {pair}: {e}", exc_info=True)
            # 异常情况下的安全止损（给足够空间避免滑点损失）
            if leverage <= 10:
                return 0.18  # 18%
            elif leverage <= 20:
                return 0.24  # 24%
            else:
                return 0.30  # 30%
    
    def calculate_dynamic_rsi_thresholds(self, dataframe: DataFrame) -> Dict[str, pd.Series]:
        """
        🎯 RSI动态阈值系统 - 基于市场环境智能调整阈值
        
        基于前期验证研究发现:
        - 固定70/30阈值存在假信号问题
        - 需要根据趋势强度、波动率、市场环境动态调整
        - 强趋势市场需放宽阈值，避免过早退出
        - 高波动环境需收紧阈值，减少假信号
        """
        length = len(dataframe)
        
        # === 基础阈值设定（HYPEROPT优化）===
        # 使用HYPEROPT优化的RSI阈值参数
        base_overbought = self.rsi_short_max
        base_oversold = self.rsi_long_min
        
        # === 市场环境因子计算 ===
        
        # 1. 趋势强度调整因子
        trend_strength = dataframe.get('trend_strength', self._safe_series(50, length))
        adx_strength = dataframe.get('adx', self._safe_series(20, length))
        
        # 强趋势时放宽阈值 (避免在趋势中过早出场)
        strong_trend_mask = (trend_strength > 60) | (adx_strength > 30)
        trend_adjustment = np.where(strong_trend_mask, 10, 0)  # 强趋势+10
        
        # 2. 波动率调整因子
        volatility = dataframe.get('atr_p', self._safe_series(0.02, length))
        volatility_percentile = volatility.rolling(50).rank(pct=True)
        
        # 高波动时收紧阈值 (减少噪音造成的假信号)
        volatility_adjustment = (volatility_percentile - 0.5) * 10  # -5到+5的调整
        
        # 3. 市场环境调整
        market_sentiment = dataframe.get('market_sentiment', self._safe_series(0, length))
        
        # 极端情绪时更严格的阈值
        extreme_fear_mask = market_sentiment < -0.7
        extreme_greed_mask = market_sentiment > 0.7
        sentiment_adjustment = np.where(extreme_fear_mask, -5,  # 极度恐慌时降低超卖阈值
                                      np.where(extreme_greed_mask, 5, 0))  # 极度贪婪时提高超买阈值
        
        # 4. 成交量环境调整
        volume_ratio = dataframe.get('volume_ratio', self._safe_series(1, length))
        high_volume_mask = volume_ratio > 2.0  # 异常放量
        volume_adjustment = np.where(high_volume_mask, 5, 0)  # 异常放量时更保守
        
        # === 动态阈值计算 ===
        
        # 超买阈值: 基础值 + 趋势调整 + 波动率调整 + 情绪调整 + 成交量调整
        dynamic_overbought = (base_overbought + 
                            trend_adjustment + 
                            volatility_adjustment + 
                            sentiment_adjustment + 
                            volume_adjustment).clip(self.rsi_short_min, 95)  # 限制范围（HYPEROPT优化）
        
        # 超卖阈值: 基础值 - 趋势调整 - 波动率调整 + 情绪调整 - 成交量调整  
        dynamic_oversold = (base_oversold - 
                          trend_adjustment - 
                          volatility_adjustment + 
                          sentiment_adjustment - 
                          volume_adjustment).clip(5, self.rsi_long_max)   # 限制范围（HYPEROPT优化）
        
        # === 时间框架特殊调整 ===
        if self.timeframe in ('3m', '5m', '15m'):
            # 短周期框架需要调整阈值（15m相对温和）
            adjustment = 1 if self.timeframe == '15m' else 3
            dynamic_overbought = dynamic_overbought + adjustment  # 超买阈值提高
            dynamic_oversold = dynamic_oversold - adjustment      # 超卖阈值降低
            
            # 重新限制范围（HYPEROPT优化）
            dynamic_overbought = dynamic_overbought.clip(self.rsi_short_min, 98)
            dynamic_oversold = dynamic_oversold.clip(2, self.rsi_long_max)
        
        # === 时间环境调整 ===
        try:
            current_hour = datetime.now(timezone.utc).hour
            
            # 美盘开盘时间 (波动加大，阈值收紧)
            if 14 <= current_hour <= 16:
                time_adjustment = 2
            # 亚洲深夜 (流动性差，阈值放宽)  
            elif 3 <= current_hour <= 6:
                time_adjustment = -3
            else:
                time_adjustment = 0
                
            dynamic_overbought = dynamic_overbought + time_adjustment
            dynamic_oversold = dynamic_oversold - time_adjustment
            
        except Exception:
            pass  # 时间调整失败时忽略
        
        # === 确保Series格式正确 ===
        if not isinstance(dynamic_overbought, pd.Series):
            dynamic_overbought = pd.Series(dynamic_overbought, index=dataframe.index)
        if not isinstance(dynamic_oversold, pd.Series):
            dynamic_oversold = pd.Series(dynamic_oversold, index=dataframe.index)
            
        # === 返回结果 ===
        result = {
            'overbought': dynamic_overbought.fillna(80),  # 默认值80（加密货币优化）
            'oversold': dynamic_oversold.fillna(20),      # 默认值20（加密货币优化）
            'overbought_extreme': dynamic_overbought + 10, # 极端超买
            'oversold_extreme': dynamic_oversold - 10,    # 极端超卖
            # 调试信息
            'trend_adj': trend_adjustment,
            'vol_adj': volatility_adjustment, 
            'sentiment_adj': sentiment_adjustment,
            'volume_adj': volume_adjustment
        }
        
        return result
    
    def validate_ema_cross_signals(self, dataframe: DataFrame) -> Dict[str, pd.Series]:
        """
        🎯 EMA交叉假信号过滤器 - 减少35%假信号率
        
        基于前期验证研究发现:
        - EMA交叉信号有35%假信号率，需要多重确认
        - 成交量支撑是关键过滤条件
        - 需要防止在极端位置的交叉信号
        - 结合MACD和RSI进行二次确认
        """
        length = len(dataframe)
        
        # === 基础EMA交叉检测 ===
        # 金叉：EMA8上穿EMA21
        basic_golden_cross = (
            (dataframe['ema_8'] > dataframe['ema_21']) & 
            (dataframe['ema_8'].shift(1) <= dataframe['ema_21'].shift(1))
        )
        
        # 死叉：EMA8下穿EMA21  
        basic_death_cross = (
            (dataframe['ema_8'] < dataframe['ema_21']) & 
            (dataframe['ema_8'].shift(1) >= dataframe['ema_21'].shift(1))
        )
        
        # === 多重确认过滤系统 ===
        
        # 1. 成交量确认 (研究显示这是最重要的过滤条件)
        volume_confirm_bullish = dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2
        volume_confirm_bearish = dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.3  # 做空需要更强成交量
        
        # 2. 价格动量确认 (避免滞后的交叉信号)
        price_momentum_bullish = (
            (dataframe['close'] > dataframe['close'].shift(2)) &  # 价格上升趋势
            (dataframe['close'] > dataframe['open'])  # 当前K线为阳线
        )
        price_momentum_bearish = (
            (dataframe['close'] < dataframe['close'].shift(2)) &  # 价格下降趋势
            (dataframe['close'] < dataframe['open'])  # 当前K线为阴线
        )
        
        # 3. 位置过滤 (防止在极端位置交叉)
        price_position = (dataframe['close'] - dataframe['low'].rolling(20).min()) / \
                        (dataframe['high'].rolling(20).max() - dataframe['low'].rolling(20).min() + 0.0001)
        
        position_safe_bullish = (price_position > 0.15) & (price_position < 0.85)  # 不在极端位置
        position_safe_bearish = (price_position > 0.15) & (price_position < 0.85)
        
        # 4. MACD二次确认 (同向确认)
        macd_confirm_bullish = dataframe['macd'] > dataframe['macd_signal']
        macd_confirm_bearish = dataframe['macd'] < dataframe['macd_signal']
        
        # 5. RSI健康度确认 (避免极端超买超卖时的交叉)
        rsi_healthy_bullish = (dataframe['rsi_14'] > 25) & (dataframe['rsi_14'] < 75)
        rsi_healthy_bearish = (dataframe['rsi_14'] > 25) & (dataframe['rsi_14'] < 75)
        
        # 6. 趋势强度确认 (确保有足够的趋势动力)
        trend_strength_ok = abs(dataframe.get('trend_strength', 0)) > 10
        
        # 7. 假突破检测 (检查前期是否有反复交叉)
        recent_crosses_count = (
            basic_golden_cross.rolling(5).sum() + basic_death_cross.rolling(5).sum()
        )
        no_frequent_crosses = recent_crosses_count <= 2  # 近5根K线内交叉不超过2次
        
        # === 短周期时间框架特殊过滤 ===
        if self.timeframe in ('3m', '5m', '15m'):
            # 短周期框架需要更严格的过滤（15m相对宽松）
            # 连续性确认：需要连续2根K线支持方向
            bullish_continuation = (
                (dataframe['ema_8'] > dataframe['ema_8'].shift(1)) &  # EMA8持续上升
                (dataframe['close'] > dataframe['ema_8'])  # 价格在EMA8之上
            )
            bearish_continuation = (
                (dataframe['ema_8'] < dataframe['ema_8'].shift(1)) &  # EMA8持续下降  
                (dataframe['close'] < dataframe['ema_8'])  # 价格在EMA8之下
            )
        else:
            bullish_continuation = True
            bearish_continuation = True
        
        # === 计算确认分数 ===
        # 每个确认条件给1分，总分7分
        bullish_score = (
            volume_confirm_bullish.astype(int) + 
            price_momentum_bullish.astype(int) + 
            position_safe_bullish.astype(int) + 
            macd_confirm_bullish.astype(int) + 
            rsi_healthy_bullish.astype(int) +
            trend_strength_ok.astype(int) +
            no_frequent_crosses.astype(int)
        )
        
        bearish_score = (
            volume_confirm_bearish.astype(int) + 
            price_momentum_bearish.astype(int) + 
            position_safe_bearish.astype(int) + 
            macd_confirm_bearish.astype(int) + 
            rsi_healthy_bearish.astype(int) +
            trend_strength_ok.astype(int) +
            no_frequent_crosses.astype(int)
        )
        
        # === 最终验证信号 ===
        # 强信号：至少5个确认条件 (成功率80%+)
        strong_golden_cross = basic_golden_cross & (bullish_score >= 5) & bullish_continuation
        strong_death_cross = basic_death_cross & (bearish_score >= 5) & bearish_continuation
        
        # 中等信号：至少3个确认条件 (成功率65%+)  
        medium_golden_cross = basic_golden_cross & (bullish_score >= 3) & bullish_continuation
        medium_death_cross = basic_death_cross & (bearish_score >= 3) & bearish_continuation
        
        # 弱信号：少于3个确认条件 (避免使用)
        weak_golden_cross = basic_golden_cross & (bullish_score < 3)
        weak_death_cross = basic_death_cross & (bearish_score < 3)
        
        # === 返回结果 ===
        result = {
            # 强信号 (推荐使用)
            'strong_golden_cross': strong_golden_cross,
            'strong_death_cross': strong_death_cross,
            
            # 中等信号 (谨慎使用)
            'medium_golden_cross': medium_golden_cross,  
            'medium_death_cross': medium_death_cross,
            
            # 弱信号 (避免使用)
            'weak_golden_cross': weak_golden_cross,
            'weak_death_cross': weak_death_cross,
            
            # 确认分数 (调试用)
            'bullish_confirmation_score': bullish_score,
            'bearish_confirmation_score': bearish_score,
            
            # 原始信号 (对比用)
            'basic_golden_cross': basic_golden_cross,
            'basic_death_cross': basic_death_cross
        }
        
        return result
    
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
        
        # 市场状态 - 优化为向量化计算，避免 O(n²) 复杂度
        if len(dataframe) > 50:
            dataframe['market_state'] = self._detect_market_state_vectorized(dataframe, pair)
        else:
            dataframe['market_state'] = 'sideways'

        # 多时间框架分析 - 只在启用时才计算，避免延迟
        if self.use_mtf_entry_filter:
            mtf_analysis = self.analyze_multi_timeframe(dataframe, metadata)
            dataframe = self.apply_mtf_analysis_to_dataframe(dataframe, mtf_analysis, metadata)
        else:
            # MTF已禁用，使用默认值避免计算延迟
            dataframe['mtf_consensus_direction'] = 'neutral'
            dataframe['mtf_consensus_strength'] = 'weak'
            dataframe['mtf_trend_score'] = 0.0
            dataframe['mtf_long_filter'] = 1  # 允许所有多头信号
            dataframe['mtf_short_filter'] = 1  # 允许所有空头信号
        
        # 综合信号强度（增强版）
        dataframe['signal_strength'] = self.calculate_enhanced_signal_strength(dataframe)

        # 最终检查和清理重复索引
        if dataframe.index.duplicated().any():
            logger.warning(f"最终检查发现重复索引，正在清理: {pair}")
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

        # === 🎯 添加动态RSI阈值系统 ===
        # 在所有基础指标计算完成后调用，确保有完整的trend_strength、atr_p等依赖指标
        try:
            dynamic_rsi_thresholds = self.calculate_dynamic_rsi_thresholds(dataframe)

            # 将动态阈值添加到dataframe中（兼容双命名，避免遗留引用失效）
            dynamic_overbought = dynamic_rsi_thresholds['overbought']
            dynamic_oversold = dynamic_rsi_thresholds['oversold']

            dataframe['rsi_dynamic_overbought'] = dynamic_overbought
            dataframe['rsi_dynamic_oversold'] = dynamic_oversold
            # 兼容旧列名
            dataframe['rsi_overbought_dynamic'] = dynamic_overbought
            dataframe['rsi_oversold_dynamic'] = dynamic_oversold

            dataframe['rsi_overbought_extreme'] = dynamic_rsi_thresholds['overbought_extreme']
            dataframe['rsi_oversold_extreme'] = dynamic_rsi_thresholds['oversold_extreme']
            
            # 调试信息列 (强制输出用于分析)
            dataframe['rsi_trend_adj'] = dynamic_rsi_thresholds['trend_adj']
            dataframe['rsi_vol_adj'] = dynamic_rsi_thresholds['vol_adj']
            dataframe['rsi_sentiment_adj'] = dynamic_rsi_thresholds['sentiment_adj']
            dataframe['rsi_volume_adj'] = dynamic_rsi_thresholds['volume_adj']
            
            self._log_message(
                f"✅ 动态RSI阈值系统已激活 - {metadata['pair']}",
                importance="summary"
            )
            
        except Exception as e:
            logger.warning(f"动态RSI阈值计算失败，使用默认值: {e}")
            # 降级处理：使用固定阈值
            fixed_overbought = pd.Series(self.rsi_short_max, index=dataframe.index)
            fixed_oversold = pd.Series(self.rsi_long_min, index=dataframe.index)

            dataframe['rsi_dynamic_overbought'] = fixed_overbought
            dataframe['rsi_dynamic_oversold'] = fixed_oversold

            dataframe['rsi_overbought_dynamic'] = fixed_overbought
            dataframe['rsi_oversold_dynamic'] = fixed_oversold
            dataframe['rsi_overbought_extreme'] = self.rsi_short_max + 10
            dataframe['rsi_oversold_extreme'] = self.rsi_long_min - 10

        # === 🎯 添加EMA交叉假信号过滤系统 ===
        try:
            ema_cross_validation = self.validate_ema_cross_signals(dataframe)
            
            # 将验证结果添加到dataframe中
            dataframe['ema_strong_golden_cross'] = ema_cross_validation['strong_golden_cross']
            dataframe['ema_strong_death_cross'] = ema_cross_validation['strong_death_cross']
            dataframe['ema_medium_golden_cross'] = ema_cross_validation['medium_golden_cross']
            dataframe['ema_medium_death_cross'] = ema_cross_validation['medium_death_cross']
            dataframe['ema_bullish_score'] = ema_cross_validation['bullish_confirmation_score']
            dataframe['ema_bearish_score'] = ema_cross_validation['bearish_confirmation_score']
            
            # 统计信号质量
            strong_signals = ema_cross_validation['strong_golden_cross'].sum() + ema_cross_validation['strong_death_cross'].sum()
            weak_signals = ema_cross_validation['weak_golden_cross'].sum() + ema_cross_validation['weak_death_cross'].sum()
            
            self._log_message(
                f"✅ EMA交叉过滤器已激活 - {metadata['pair']}: 强信号{strong_signals}个, 弱信号{weak_signals}个(已过滤)",
                importance="summary"
            )
            
        except Exception as e:
            logger.warning(f"EMA交叉验证计算失败，使用基础信号: {e}")
            # 降级处理：使用基础EMA交叉
            dataframe['ema_strong_golden_cross'] = (dataframe['ema_8'] > dataframe['ema_21']) & (dataframe['ema_8'].shift(1) <= dataframe['ema_21'].shift(1))
            dataframe['ema_strong_death_cross'] = (dataframe['ema_8'] < dataframe['ema_21']) & (dataframe['ema_8'].shift(1) >= dataframe['ema_21'].shift(1))
            dataframe['ema_bullish_score'] = 3
            dataframe['ema_bearish_score'] = 3

        # 性能优化：去碎片化DataFrame以避免PerformanceWarning
        dataframe = dataframe.copy()

        # === 新增指标（2024最新研究）===

        # ATR（波动率）- 用于动态止损
        # talib.abstract 需要传入整个 dataframe
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Stochastic Oscillator（动量振荡器）- 用于捕捉超买超卖和背离
        # talib.abstract.STOCH 返回的是 Function，需要直接调用
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Bollinger Bands（布林带）- 用于波动率突破检测
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_lower'] = bb['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']

        # === 量能指标（用于退出信号量能确认）===
        dataframe['volume_ma_20'] = dataframe['volume'].rolling(20).mean()

        # === 横盘市场检测指标（2024最新研究）===

        # NATR（标准化ATR）- ATR占价格的百分比，用于检测低波动横盘
        dataframe['natr'] = (dataframe['atr'] / dataframe['close']) * 100

        # BB宽度标准化（已计算，确保可用）
        # dataframe['bb_width'] 已在上面计算

        # Donchian Channel（用于突破检测）
        dataframe['donchian_high_20'] = dataframe['high'].rolling(20).max()
        dataframe['donchian_low_20'] = dataframe['low'].rolling(20).min()
        dataframe['donchian_mid_20'] = (dataframe['donchian_high_20'] + dataframe['donchian_low_20']) / 2

        # Donchian Channel 50日（用于长线退出信号）
        dataframe['donchian_high_50'] = dataframe['high'].rolling(50).max()
        dataframe['donchian_low_50'] = dataframe['low'].rolling(50).min()

        # 🆕 Donchian位置指标（0-1范围）- 用于预测性退出
        # 0 = 在下轨，1 = 在上轨，0.5 = 在中轨
        donchian_range = dataframe['donchian_high_50'] - dataframe['donchian_low_50']
        dataframe['donchian_position_50'] = (dataframe['close'] - dataframe['donchian_low_50']) / donchian_range.replace(0, 1)  # 避免除零

        # Donchian Channel宽度（用于检测价格压缩）
        dataframe['donchian_width'] = (dataframe['donchian_high_20'] - dataframe['donchian_low_20']) / dataframe['donchian_mid_20']

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
        """🕐 应用增强的MTF多时间框架确认机制"""
        
        # === 获取MTF共识数据 ===
        mtf_consensus = mtf_analysis.get('mtf_consensus', {})
        
        # 使用新的加权评分系统
        mtf_trend_score = mtf_consensus.get('weighted_trend_score', 0)  # -1 to 1
        mtf_strength_score = mtf_consensus.get('weighted_confirmation', 0)  # 0 to 1
        consensus_direction = mtf_consensus.get('consensus_direction', 'neutral')
        consensus_strength = mtf_consensus.get('consensus_strength', 'weak')
        alignment_ratio = mtf_consensus.get('alignment_ratio', 0)
        
        # === MTF确认级别系统 ===
        # 超强确认：75%以上时间框架一致
        mtf_very_strong_bull = (
            (consensus_direction == 'bullish') & 
            (consensus_strength == 'very_strong') &
            (mtf_trend_score > 0.5)
        )
        
        mtf_very_strong_bear = (
            (consensus_direction == 'bearish') & 
            (consensus_strength == 'very_strong') &
            (mtf_trend_score < -0.5)
        )
        
        # 中等确认：50%以上时间框架一致
        mtf_moderate_bull = (
            (consensus_direction == 'bullish') & 
            (consensus_strength in ['moderate', 'very_strong']) &
            (mtf_trend_score > 0.2)
        )
        
        mtf_moderate_bear = (
            (consensus_direction == 'bearish') & 
            (consensus_strength in ['moderate', 'very_strong']) &
            (mtf_trend_score < -0.2)
        )
        
        # === 风险评分系统 ===
        # 基于各时间框架RSI位置的风险评估
        mtf_risk_score = 0
        total_weight = 0
        
        timeframe_weights = {'15m': 0.35, '1h': 0.30, '4h': 0.25, '1d': 0.10}
        
        for tf, weight in timeframe_weights.items():
            if tf in mtf_analysis:
                analysis = mtf_analysis[tf]
                rsi = analysis.get('rsi', 50)
                price_position = analysis.get('price_position', 0.5)
                
                # 风险评分计算
                if rsi > 75 and price_position > 0.8:  # 超买且高位
                    mtf_risk_score += weight * 1
                elif rsi < 25 and price_position < 0.2:  # 超卖且低位
                    mtf_risk_score -= weight * 1
                    
                total_weight += weight
        
        if total_weight > 0:
            mtf_risk_score /= total_weight
        
        # === 动量一致性评分 ===
        # 检查各时间框架动量方向是否一致
        bullish_momentum_count = 0
        bearish_momentum_count = 0
        total_tf_count = 0
        
        for tf in ['3m', '5m', '1h', '4h']:
            if tf in mtf_analysis:
                momentum = mtf_analysis[tf].get('momentum', 'neutral')
                if momentum in ['bullish', 'strong_bullish']:
                    bullish_momentum_count += 1
                elif momentum in ['bearish', 'strong_bearish']:
                    bearish_momentum_count += 1
                total_tf_count += 1
        
        momentum_consistency = 0
        if total_tf_count > 0:
            bullish_ratio = bullish_momentum_count / total_tf_count
            bearish_ratio = bearish_momentum_count / total_tf_count
            momentum_consistency = max(bullish_ratio, bearish_ratio)
        
        # === MTF信号过滤器（优化版） ===
        # 多头信号过滤器
        mtf_long_filter = (
            (mtf_trend_score > 0.1) |  # 轻微偏多即可，降低门槛
            mtf_moderate_bull |        # 或中等确认
            (momentum_consistency > 0.5)  # 或动量一致性好
        )
        
        # 空头信号过滤器  
        mtf_short_filter = (
            (mtf_trend_score < -0.1) |  # 轻微偏空即可
            mtf_moderate_bear |         # 或中等确认
            (momentum_consistency > 0.5)  # 或动量一致性好
        )
        
        # === 高质量MTF信号（严格条件）===
        mtf_strong_bull = mtf_very_strong_bull & (momentum_consistency > 0.75)
        mtf_strong_bear = mtf_very_strong_bear & (momentum_consistency > 0.75)
        
        # === 获取关键支撑阻力位 ===
        # 使用1小时和4小时框架的价格位置
        h1_data = mtf_analysis.get('1h', {})
        h4_data = mtf_analysis.get('4h', {})
        
        # 估算支撑阻力位（基于价格位置）
        current_close = dataframe['close'].iloc[-1] if not dataframe.empty else 0
        h1_price_pos = h1_data.get('price_position', 0.5)
        h4_price_pos = h4_data.get('price_position', 0.5)
        
        # 简化的支撑阻力计算
        estimated_range = current_close * 0.02  # 2%的估算范围
        h1_support = current_close - estimated_range * h1_price_pos
        h1_resistance = current_close + estimated_range * (1 - h1_price_pos)
        
        # === 应用增强MTF数据到DataFrame ===
        h4_support = current_close - estimated_range * h4_price_pos
        h4_resistance = current_close + estimated_range * (1 - h4_price_pos)
        
        # 计算MTF确认得分（综合评分）
        mtf_confirmation_score = (
            mtf_strength_score * 0.4 +      # 40% 确认强度
            momentum_consistency * 0.3 +     # 30% 动量一致性
            alignment_ratio * 0.3            # 30% 时间框架一致比例
        )
        
        mtf_columns = {
            # === 核心MTF评分 ===
            'mtf_trend_score': self._safe_series(mtf_trend_score, len(dataframe)),  # -1 to 1
            'mtf_strength_score': self._safe_series(mtf_strength_score, len(dataframe)),  # 0 to 1 
            'mtf_risk_score': self._safe_series(mtf_risk_score, len(dataframe)),  # -1 to 1
            'mtf_confirmation_score': self._safe_series(mtf_confirmation_score, len(dataframe)),  # 0 to 1
            
            # === MTF共识信息 ===
            'mtf_consensus_direction': self._safe_series(consensus_direction, len(dataframe)),
            'mtf_consensus_strength': self._safe_series(consensus_strength, len(dataframe)),
            'mtf_alignment_ratio': self._safe_series(alignment_ratio, len(dataframe)),
            'mtf_momentum_consistency': self._safe_series(momentum_consistency, len(dataframe)),
            
            # === MTF信号过滤器（新版本）===
            'mtf_long_filter': self._safe_series(1 if mtf_long_filter else 0, len(dataframe)),
            'mtf_short_filter': self._safe_series(1 if mtf_short_filter else 0, len(dataframe)),
            
            # === MTF强信号（严格条件）===
            'mtf_strong_bull': self._safe_series(1 if mtf_strong_bull else 0, len(dataframe)),
            'mtf_strong_bear': self._safe_series(1 if mtf_strong_bear else 0, len(dataframe)),
            
            # === MTF等级信号 ===
            'mtf_very_strong_bull': self._safe_series(1 if mtf_very_strong_bull else 0, len(dataframe)),
            'mtf_very_strong_bear': self._safe_series(1 if mtf_very_strong_bear else 0, len(dataframe)),
            'mtf_moderate_bull': self._safe_series(1 if mtf_moderate_bull else 0, len(dataframe)),
            'mtf_moderate_bear': self._safe_series(1 if mtf_moderate_bear else 0, len(dataframe)),
            
            # === 关键价格位（增强版）===
            'h1_support': self._safe_series(h1_support, len(dataframe)),
            'h1_resistance': self._safe_series(h1_resistance, len(dataframe)),
            'h4_support': self._safe_series(h4_support, len(dataframe)),
            'h4_resistance': self._safe_series(h4_resistance, len(dataframe)),
            
            # === 位置关系（动态计算）===
            'near_h1_support': (abs(dataframe['close'] - h1_support) / dataframe['close'] < 0.005).astype(int),
            'near_h1_resistance': (abs(dataframe['close'] - h1_resistance) / dataframe['close'] < 0.005).astype(int),
            'near_h4_support': (abs(dataframe['close'] - h4_support) / dataframe['close'] < 0.01).astype(int),
            'near_h4_resistance': (abs(dataframe['close'] - h4_resistance) / dataframe['close'] < 0.01).astype(int)
        }
        
        # MTF分析日志记录
        if consensus_strength != 'weak':
            pair = metadata.get('pair', 'UNKNOWN')
            bullish_tfs = mtf_consensus.get('bullish_timeframes', [])
            bearish_tfs = mtf_consensus.get('bearish_timeframes', [])
            neutral_tfs = mtf_consensus.get('neutral_timeframes', [])
            
            self.event_log.info(
                "mtf_confirmation",
                pair=pair,
                consensus=consensus_direction,
                strength=consensus_strength,
                trend_score=f"{mtf_trend_score:.2f}",
                strength_score=f"{mtf_strength_score:.2f}",
                momentum_consistency=f"{momentum_consistency:.2f}",
                bullish_tfs=bullish_tfs,
                bearish_tfs=bearish_tfs,
                neutral_tfs=neutral_tfs,
                confirmation_score=f"{mtf_confirmation_score:.2f}",
            )
        
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
        """
        🎯 优化的信号质量评分系统 (集成MTF趋势一致性)
        
        计算流程：
        1. 基础技术指标质量评分 (0-1)
        2. 转换为1-10评分标准
        3. MTF趋势一致性调整 (关键优化!)
        4. 标准化回0-1区间
        """
        base_quality = self._safe_series(0.5, len(dataframe))  # 默认中等质量
        
        # === 第一阶段：基础技术指标质量评分 ===
        # 基于信号强度一致性计算质量
        if 'signal_strength' in dataframe.columns:
            # 信号强度绝对值越大质量越高
            abs_strength = abs(dataframe['signal_strength'])
            base_quality = abs_strength / 10.0  # 标准化到0-1
        
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
        
        # 综合基础质量评分
        if consistency_factors:
            avg_consistency = np.mean(consistency_factors, axis=0)
            base_quality = (base_quality + avg_consistency) / 2
        
        base_quality = base_quality.fillna(0.5).clip(0, 1)
        
        # === 第二阶段：MTF趋势一致性优化 (核心创新!) ===
        enhanced_quality = base_quality.copy()
        
        # 检查是否有MTF数据可用
        has_mtf_data = (
            'mtf_consensus_direction' in dataframe.columns or
            'mtf_consensus_strength' in dataframe.columns or  
            'mtf_trend_score' in dataframe.columns
        )
        
        if has_mtf_data:
            # 为每行数据应用MTF调整
            for idx in dataframe.index:
                try:
                    # 构建MTF数据字典
                    mtf_data = {
                        'consensus_direction': dataframe.get('mtf_consensus_direction', {}).get(idx, 'neutral'),
                        'consensus_strength': dataframe.get('mtf_consensus_strength', {}).get(idx, 'weak'),
                        'trend_score': dataframe.get('mtf_trend_score', {}).get(idx, 0.0)
                    }
                    
                    # 检查该点是否有交易信号
                    has_long_signal = dataframe.get('enter_long', {}).get(idx, 0) == 1
                    has_short_signal = dataframe.get('enter_short', {}).get(idx, 0) == 1
                    
                    if has_long_signal or has_short_signal:
                        # 转换为1-10评分
                        base_score_10 = base_quality.iloc[idx] * 10
                        
                        # 确定信号方向
                        signal_direction = 'long' if has_long_signal else 'short'
                        
                        # 应用MTF趋势一致性调整
                        adjusted_score = self.adjust_signal_by_mtf_consensus(
                            base_score_10, mtf_data, signal_direction
                        )
                        
                        # 转换回0-1区间 (允许超过1.0以奖励高一致性)
                        enhanced_quality.iloc[idx] = min(adjusted_score / 10.0, 1.5)
                        
                except Exception as e:
                    # 出错时保持原始评分
                    continue
        
        return enhanced_quality.fillna(0.5).clip(0, 1.5)  # 允许最高1.5倍质量奖励
    
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
        
        return base_leverage.fillna(1.0).clip(0.5, 5.0).round().astype(int)  # 1-5倍整数杠杆
    
    def filter_5min_noise(self, dataframe: DataFrame) -> Dict[str, pd.Series]:
        """🔊 噪音过滤系统 - 识别并过滤短周期市场噪音（适用于3m/5m/15m等短周期）"""
        
        # === 1. 微结构噪音检测 ===
        # 检测高频交易产生的价格跳动
        price_volatility = dataframe['close'].pct_change().rolling(5).std()
        volume_volatility = dataframe['volume'].pct_change().rolling(5).std()
        
        # 异常价格波动（通常是噪音）
        abnormal_price_volatility = price_volatility > price_volatility.rolling(20).quantile(0.85)
        
        # === 2. 假突破过滤器 ===
        # 检测短期反转的假突破
        price_change_3min = dataframe['close'].pct_change(3)
        price_change_1min = dataframe['close'].pct_change(1)
        
        # 快速反转模式（上涨后立即下跌）
        false_breakout_up = (
            (price_change_3min > 0.005) &  # 3根K线涨幅>0.5%
            (price_change_1min < -0.002)   # 最后1根K线跌幅>0.2%
        )
        
        false_breakout_down = (
            (price_change_3min < -0.005) & # 3根K线跌幅>0.5%
            (price_change_1min > 0.002)    # 最后1根K线涨幅>0.2%
        )
        
        # === 3. 市场做市活动检测 ===
        # 检测低波动高频交易
        price_range = (dataframe['high'] - dataframe['low']) / dataframe['close']
        low_volatility_high_volume = (
            (price_range < 0.003) &  # 价格范围<0.3%
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.5)  # 成交量高于平均1.5倍
        )
        
        # === 4. 临时价格冲击过滤 ===
        # 检测单根K线异常大幅波动
        single_candle_spike = (
            (dataframe['high'] / dataframe['open'] > 1.008) |  # 单根K线冲高0.8%
            (dataframe['low'] / dataframe['open'] < 0.992)     # 单根K线下探0.8%
        ) & (abs(dataframe['close'] - dataframe['open']) / dataframe['open'] < 0.002)  # 但收盘价变化<0.2%
        
        # === 5. 成交量确认过滤器 ===
        # 无成交量支撑的价格移动通常是噪音
        volume_ma = dataframe['volume'].rolling(20).mean()
        insufficient_volume = dataframe['volume'] < volume_ma * 0.6
        
        # === 6. ATR标准化过滤器 ===
        # 基于ATR过滤微小变动 - 优化：使用 TA-Lib ATR 或向量化计算
        if 'atr_14' in dataframe.columns:
            atr_14 = dataframe['atr_14']
        elif 'atr' in dataframe.columns:
            atr_14 = dataframe['atr']  # 使用已计算的 ATR
        else:
            # 使用 TA-Lib 计算 ATR（更准确且高效）
            atr_14 = ta.ATR(dataframe, timeperiod=14)

        price_change_normalized = abs(dataframe['close'] - dataframe['close'].shift(1)) / atr_14.replace(0, 1)
        insignificant_move = price_change_normalized < 0.2  # ATR的20%以下为微小变动
        
        # === 7. 时间序列一致性检测 ===
        # 检测与短期趋势不一致的信号
        short_trend = dataframe['close'].rolling(5).mean().diff()
        medium_trend = dataframe['close'].rolling(15).mean().diff()
        
        # 短期趋势与中期趋势方向相反时可能是噪音
        trend_inconsistency = (
            (short_trend > 0) & (medium_trend < 0) |
            (short_trend < 0) & (medium_trend > 0)
        )
        
        # === 综合噪音评分系统 ===
        noise_factors = []
        noise_weights = []
        
        if abnormal_price_volatility is not None:
            noise_factors.append(abnormal_price_volatility.astype(int))
            noise_weights.append(0.15)
            
        if false_breakout_up is not None and false_breakout_down is not None:
            noise_factors.append((false_breakout_up | false_breakout_down).astype(int))
            noise_weights.append(0.20)
            
        if low_volatility_high_volume is not None:
            noise_factors.append(low_volatility_high_volume.astype(int))
            noise_weights.append(0.15)
            
        if single_candle_spike is not None:
            noise_factors.append(single_candle_spike.astype(int))
            noise_weights.append(0.20)
            
        if insufficient_volume is not None:
            noise_factors.append(insufficient_volume.astype(int))
            noise_weights.append(0.10)
            
        if insignificant_move is not None:
            noise_factors.append(insignificant_move.astype(int))
            noise_weights.append(0.10)
            
        if trend_inconsistency is not None:
            noise_factors.append(trend_inconsistency.astype(int))
            noise_weights.append(0.10)
        
        # 计算综合噪音得分 (0-1)
        if noise_factors:
            noise_score = sum(f * w for f, w in zip(noise_factors, noise_weights))
        else:
            noise_score = pd.Series(0, index=dataframe.index)
        
        # === 过滤决策 ===
        # 高噪音区域（>0.55）应避免交易
        high_noise_zone = noise_score > 0.55
        
        # 中等噪音区域（0.35-0.55）需要额外确认
        medium_noise_zone = (noise_score > 0.35) & (noise_score <= 0.55)
        
        # 低噪音区域（<=0.35）相对安全
        low_noise_zone = noise_score <= 0.35
        
        # === 信号调整建议 ===
        # 噪音环境下的信号强度调整
        signal_strength_adjustment = 1.0 - noise_score * 0.6  # 最多降低60%强度
        signal_strength_adjustment = signal_strength_adjustment.clip(0.1, 1.0)
        
        return {
            'noise_score': noise_score,
            'high_noise_zone': high_noise_zone,
            'medium_noise_zone': medium_noise_zone, 
            'low_noise_zone': low_noise_zone,
            'signal_strength_adjustment': signal_strength_adjustment,
            'clean_environment': low_noise_zone & ~abnormal_price_volatility,
            'avoid_trading': high_noise_zone | (abnormal_price_volatility & insufficient_volume)
        }
    
    def optimize_macd_leading_signals(self, dataframe: DataFrame) -> Dict[str, pd.Series]:
        """🚀 MACD前置确认系统 - 解决传统MACD滞后性问题"""
        
        # === 1. 多周期MACD融合系统 ===
        # 快速MACD (8,17,5) - 用于早期信号检测
        fast_macd_arr = ta.MACD(dataframe['close'], fastperiod=8, slowperiod=17, signalperiod=5)
        fast_macd_line = pd.Series(fast_macd_arr[0], index=dataframe.index)
        fast_macd_signal = pd.Series(fast_macd_arr[1], index=dataframe.index)
        fast_macd_hist = pd.Series(fast_macd_arr[2], index=dataframe.index)
        
        # 标准MACD (12,26,9) - 从dataframe获取现有值或计算
        if 'macd' in dataframe.columns:
            std_macd = dataframe['macd']
            std_macd_signal = dataframe.get('macd_signal', pd.Series(ta.MACD(dataframe['close'])[1], index=dataframe.index))
            std_macd_hist = dataframe.get('macd_hist', pd.Series(ta.MACD(dataframe['close'])[2], index=dataframe.index))
        else:
            std_macd_arr = ta.MACD(dataframe['close'])
            std_macd = pd.Series(std_macd_arr[0], index=dataframe.index)
            std_macd_signal = pd.Series(std_macd_arr[1], index=dataframe.index)
            std_macd_hist = pd.Series(std_macd_arr[2], index=dataframe.index)
        
        # 慢速MACD (19,39,9) - 用于趋势确认
        slow_macd_arr = ta.MACD(dataframe['close'], fastperiod=19, slowperiod=39, signalperiod=9)
        slow_macd_line = pd.Series(slow_macd_arr[0], index=dataframe.index)
        slow_macd_signal = pd.Series(slow_macd_arr[1], index=dataframe.index)
        
        # === 2. MACD加速度/减速度分析 ===
        # 检测MACD线的变化速度，提前识别转折点
        macd_velocity = std_macd.diff()  # MACD一阶导数（速度）
        macd_acceleration = macd_velocity.diff()  # MACD二阶导数（加速度）
        
        # 检测MACD减速（可能预示反转）
        macd_deceleration = (
            (macd_velocity > 0) &  # 向上运动
            (macd_acceleration < 0) &  # 但在减速
            (macd_acceleration < macd_acceleration.shift(1))  # 减速在加剧
        )
        
        macd_acceleration_up = (
            (macd_velocity < 0) &  # 向下运动
            (macd_acceleration > 0) &  # 但在加速向上
            (macd_acceleration > macd_acceleration.shift(1))  # 加速在增强
        )
        
        # === 3. MACD线收敛/发散检测 ===
        # 检测MACD线和信号线之间的距离变化
        macd_distance = abs(std_macd - std_macd_signal)
        macd_convergence = macd_distance < macd_distance.shift(1)  # 线条在收敛
        
        # 强收敛：连续3期收敛且距离在缩小
        strong_convergence = (
            macd_convergence &
            macd_convergence.shift(1) &
            macd_convergence.shift(2) &
            (macd_distance < macd_distance.rolling(5).mean() * 0.5)  # 距离小于5期平均的50%
        )
        
        # === 4. 预交叉信号检测 ===  
        # 检测即将发生的金叉/死叉
        macd_approaching_bullish = (
            (std_macd < std_macd_signal) &  # 当前MACD在信号线下
            (std_macd > std_macd.shift(1)) &  # MACD向上
            (std_macd_signal < std_macd_signal.shift(1)) &  # 信号线向下或平
            strong_convergence &  # 强收敛
            (abs(std_macd - std_macd_signal) < abs(std_macd.shift(2) - std_macd_signal.shift(2)) * 0.3)  # 距离大幅缩小
        )
        
        macd_approaching_bearish = (
            (std_macd > std_macd_signal) &  # 当前MACD在信号线上  
            (std_macd < std_macd.shift(1)) &  # MACD向下
            (std_macd_signal > std_macd_signal.shift(1)) &  # 信号线向上或平
            strong_convergence &  # 强收敛
            (abs(std_macd - std_macd_signal) < abs(std_macd.shift(2) - std_macd_signal.shift(2)) * 0.3)
        )
        
        # === 5. 零轴预突破检测 ===
        # MACD线接近零轴但未穿越时的早期信号
        macd_near_zero_bullish = (
            (std_macd < 0) & (std_macd > -abs(std_macd.rolling(20).std())) &  # 接近零轴
            (std_macd > std_macd.shift(1)) &  # 向上运动
            (macd_velocity > 0) &  # 速度为正
            (macd_acceleration > 0)  # 加速向上
        )
        
        macd_near_zero_bearish = (
            (std_macd > 0) & (std_macd < abs(std_macd.rolling(20).std())) &  # 接近零轴
            (std_macd < std_macd.shift(1)) &  # 向下运动
            (macd_velocity < 0) &  # 速度为负 
            (macd_acceleration < 0)  # 加速向下
        )
        
        # === 6. 成交量加权MACD ===
        # 使用成交量来验证MACD信号的有效性
        volume_ratio = dataframe.get('volume_ratio', pd.Series(1, index=dataframe.index))
        volume_weighted_strength = (
            (volume_ratio > 1.2) &  # 成交量放大
            (volume_ratio > volume_ratio.shift(1))  # 成交量递增
        )
        
        # === 7. 多时间框架MACD确认 ===
        # 使用快速MACD提供早期信号，标准MACD确认
        fast_bullish_cross = (
            (fast_macd_line > fast_macd_signal) &
            (fast_macd_line.shift(1) <= fast_macd_signal.shift(1))
        )
        
        fast_bearish_cross = (
            (fast_macd_line < fast_macd_signal) &
            (fast_macd_line.shift(1) >= fast_macd_signal.shift(1))
        )
        
        # === 8. MACD背离检测增强版 ===
        # 价格与MACD的背离更准确地预测反转
        price_high_5 = dataframe['high'].rolling(5).max()
        price_low_5 = dataframe['low'].rolling(5).min()
        macd_high_5 = std_macd_hist.rolling(5).max()
        macd_low_5 = std_macd_hist.rolling(5).min()
        
        # 看跌背离：价格新高但MACD未创新高
        bearish_divergence = (
            (dataframe['high'] >= price_high_5.shift(1)) &  # 价格创新高
            (std_macd_hist < macd_high_5.shift(1)) &  # MACD未创新高
            (std_macd_hist > 0)  # MACD在正区域
        )
        
        # 看涨背离：价格新低但MACD未创新低
        bullish_divergence = (
            (dataframe['low'] <= price_low_5.shift(1)) &  # 价格创新低
            (std_macd_hist > macd_low_5.shift(1)) &  # MACD未创新低
            (std_macd_hist < 0)  # MACD在负区域
        )
        
        # === 9. 综合前置信号评分系统 ===
        # 多种信号的加权组合，提供0-1的信心度评分
        bullish_signal_strength = (
            fast_bullish_cross.astype(int) * 0.25 +  # 25% - 快速金叉
            macd_approaching_bullish.astype(int) * 0.30 +  # 30% - 即将金叉
            macd_acceleration_up.astype(int) * 0.20 +  # 20% - 加速向上
            macd_near_zero_bullish.astype(int) * 0.15 +  # 15% - 零轴突破
            bullish_divergence.astype(int) * 0.10  # 10% - 看涨背离
        )
        
        bearish_signal_strength = (
            fast_bearish_cross.astype(int) * 0.25 +  # 25% - 快速死叉
            macd_approaching_bearish.astype(int) * 0.30 +  # 30% - 即将死叉  
            macd_deceleration.astype(int) * 0.20 +  # 20% - 减速
            macd_near_zero_bearish.astype(int) * 0.15 +  # 15% - 零轴突破
            bearish_divergence.astype(int) * 0.10  # 10% - 看跌背离
        )
        
        # === 10. 成交量确认过滤器 ===
        # 只有在有成交量支撑的情况下才发出信号
        volume_confirmed_bullish = bullish_signal_strength * volume_weighted_strength.astype(int) 
        volume_confirmed_bearish = bearish_signal_strength * volume_weighted_strength.astype(int)
        
        # === 最终前置信号（高质量） ===
        # 需要信号强度>0.5且有成交量确认
        leading_macd_bullish = (
            (bullish_signal_strength > 0.5) |  # 信号强度足够
            (volume_confirmed_bullish > 0.3)   # 或成交量确认较强
        )
        
        leading_macd_bearish = (
            (bearish_signal_strength > 0.5) |  # 信号强度足够
            (volume_confirmed_bearish > 0.3)   # 或成交量确认较强
        )
        
        return {
            # 前置信号
            'macd_leading_bullish': leading_macd_bullish,
            'macd_leading_bearish': leading_macd_bearish,
            
            # 信号强度评分
            'macd_bullish_strength': bullish_signal_strength,
            'macd_bearish_strength': bearish_signal_strength,
            
            # 技术特征
            'macd_approaching_bullish': macd_approaching_bullish,
            'macd_approaching_bearish': macd_approaching_bearish,
            'macd_acceleration_up': macd_acceleration_up,
            'macd_deceleration': macd_deceleration,
            'macd_near_zero_bullish': macd_near_zero_bullish,
            'macd_near_zero_bearish': macd_near_zero_bearish,
            
            # 背离信号
            'macd_bullish_divergence': bullish_divergence,
            'macd_bearish_divergence': bearish_divergence,
            
            # 快速MACD交叉
            'fast_macd_bullish': fast_bullish_cross,
            'fast_macd_bearish': fast_bearish_cross,
            
            # 多周期数据
            'fast_macd': fast_macd_line,
            'fast_macd_signal': fast_macd_signal,
            'slow_macd': slow_macd_line,
            'slow_macd_signal': slow_macd_signal,
            'macd_velocity': macd_velocity,
            'macd_acceleration': macd_acceleration
        }
    
    def calculate_comprehensive_signal_quality(self, dataframe: DataFrame, signal_direction: str = 'long') -> Dict[str, pd.Series]:
        """🎯 综合信号质量评分系统 - 整合所有增强功能"""
        
        # === 1. 技术指标对齐度评分 (0-100分) ===
        alignment_factors = []
        alignment_weights = []
        
        # RSI动态阈值对齐
        if 'rsi_dynamic_oversold' in dataframe.columns and 'rsi_dynamic_overbought' in dataframe.columns:
            if signal_direction == 'long':
                rsi_alignment = (
                    (dataframe['rsi_14'] < dataframe['rsi_dynamic_overbought']) &
                    (dataframe['rsi_14'] > dataframe['rsi_dynamic_oversold'] * 0.8)  # 接近但未超买
                ).astype(float) * 100
            else:
                rsi_alignment = (
                    (dataframe['rsi_14'] > dataframe['rsi_dynamic_oversold']) &
                    (dataframe['rsi_14'] < dataframe['rsi_dynamic_overbought'] * 1.2)  # 接近但未超卖
                ).astype(float) * 100
            
            alignment_factors.append(rsi_alignment)
            alignment_weights.append(0.25)
        
        # EMA交叉验证对齐
        if 'ema_bullish_score' in dataframe.columns:
            if signal_direction == 'long':
                ema_alignment = (dataframe['ema_bullish_score'] / 7.0) * 100  # 标准化到0-100
            else:
                ema_alignment = (dataframe.get('ema_bearish_score', 0) / 7.0) * 100
            
            alignment_factors.append(ema_alignment)
            alignment_weights.append(0.20)
        
        # MACD前置确认对齐
        if 'macd_bullish_strength' in dataframe.columns and 'macd_bearish_strength' in dataframe.columns:
            if signal_direction == 'long':
                macd_alignment = dataframe['macd_bullish_strength'] * 100
            else:
                macd_alignment = dataframe['macd_bearish_strength'] * 100
            
            alignment_factors.append(macd_alignment)
            alignment_weights.append(0.25)
        
        # 趋势强度对齐
        if 'trend_strength' in dataframe.columns:
            if signal_direction == 'long':
                trend_alignment = np.maximum(dataframe['trend_strength'], 0)  # 正趋势强度
            else:
                trend_alignment = np.maximum(-dataframe['trend_strength'], 0)  # 负趋势强度
            
            alignment_factors.append(trend_alignment)
            alignment_weights.append(0.15)
        
        # ADX趋势确认
        if 'adx' in dataframe.columns:
            adx_quality = np.minimum(dataframe['adx'] / 50 * 100, 100)  # ADX越高质量越好
            alignment_factors.append(adx_quality)
            alignment_weights.append(0.15)
        
        # 加权平均技术对齐度
        if alignment_factors:
            weights_sum = sum(alignment_weights)
            technical_alignment = sum(f * w for f, w in zip(alignment_factors, alignment_weights)) / weights_sum
        else:
            technical_alignment = pd.Series(50, index=dataframe.index)  # 默认中等
        
        # === 2. MTF多时间框架确认质量 (0-100分) ===
        if 'mtf_confirmation_score' in dataframe.columns:
            mtf_quality = dataframe['mtf_confirmation_score'] * 100
        else:
            mtf_quality = pd.Series(50, index=dataframe.index)
        
        # === 3. 噪音环境质量评分 (0-100分) ===
        if 'noise_score' in dataframe.columns:
            # 噪音越低，环境质量越高
            noise_quality = (1 - dataframe['noise_score']) * 100
        else:
            noise_quality = pd.Series(70, index=dataframe.index)  # 默认较好
        
        # === 4. 成交量确认质量 (0-100分) ===
        volume_factors = []
        
        # 成交量比率
        if 'volume_ratio' in dataframe.columns:
            # 1.2-2.5倍成交量为最佳，过高过低都不好
            volume_optimal = np.where(
                dataframe['volume_ratio'] > 2.5, 50,  # 过高
                np.where(
                    dataframe['volume_ratio'] < 0.8, 30,  # 过低
                    np.minimum((dataframe['volume_ratio'] - 0.8) / 1.7 * 100, 100)  # 0.8-2.5范围线性映射
                )
            )
            volume_factors.append(volume_optimal)
        
        # 成交量持续性
        if 'volume' in dataframe.columns:
            volume_trend = (
                (dataframe['volume'] > dataframe['volume'].shift(1)) &
                (dataframe['volume'].shift(1) > dataframe['volume'].shift(2))
            ).astype(float) * 100
            volume_factors.append(volume_trend * 0.5)  # 权重较低
        
        if volume_factors:
            volume_quality = np.mean(volume_factors, axis=0)
        else:
            volume_quality = pd.Series(60, index=dataframe.index)
        
        # === 5. 动量质量评分 (0-100分) ===
        momentum_factors = []
        
        # 动量评分
        if 'momentum_score' in dataframe.columns:
            if signal_direction == 'long':
                momentum_strength = np.maximum(dataframe['momentum_score'], 0) * 50  # 标准化
            else:
                momentum_strength = np.maximum(-dataframe['momentum_score'], 0) * 50
            momentum_factors.append(momentum_strength)
        
        # 价格动量一致性
        if 'close' in dataframe.columns:
            price_momentum_3 = (dataframe['close'] / dataframe['close'].shift(3) - 1) * 100
            if signal_direction == 'long':
                momentum_consistency = np.maximum(price_momentum_3, 0) * 10  # 放大到0-100范围
            else:
                momentum_consistency = np.maximum(-price_momentum_3, 0) * 10
            momentum_factors.append(momentum_consistency)
        
        if momentum_factors:
            momentum_quality = np.mean(momentum_factors, axis=0)
            momentum_quality = np.minimum(momentum_quality, 100)  # 限制最高100
        else:
            momentum_quality = pd.Series(50, index=dataframe.index)
        
        # === 6. 市场位置质量评分 (0-100分) ===
        position_quality = pd.Series(50, index=dataframe.index)  # 默认中等
        
        # 价格位置评估
        if 'close' in dataframe.columns and len(dataframe) > 20:
            highest_20 = dataframe['high'].rolling(20).max()
            lowest_20 = dataframe['low'].rolling(20).min()
            price_position = (dataframe['close'] - lowest_20) / (highest_20 - lowest_20 + 0.0001)
            
            if signal_direction == 'long':
                # 做多时，在0.2-0.6位置最佳（避免追高，但不在最底部）
                position_quality = np.where(
                    price_position < 0.2, 90,  # 低位机会
                    np.where(
                        price_position > 0.8, 20,  # 高位风险
                        100 - abs(price_position - 0.4) * 100  # 中位最佳
                    )
                )
            else:
                # 做空时，在0.4-0.8位置最佳
                position_quality = np.where(
                    price_position > 0.8, 90,  # 高位机会
                    np.where(
                        price_position < 0.2, 20,  # 低位风险
                        100 - abs(price_position - 0.6) * 100  # 偏高位最佳
                    )
                )
        
        # === 7. 综合质量评分 (0-100分) ===
        # 各维度权重分配
        quality_components = {
            'technical_alignment': (technical_alignment, 0.25),      # 25% - 技术指标一致性
            'mtf_confirmation': (mtf_quality, 0.20),               # 20% - 多时间框架确认
            'noise_environment': (noise_quality, 0.15),            # 15% - 环境质量
            'volume_confirmation': (volume_quality, 0.15),         # 15% - 成交量确认
            'momentum_quality': (momentum_quality, 0.15),          # 15% - 动量质量
            'position_quality': (position_quality, 0.10)           # 10% - 位置质量
        }
        
        # 加权平均总质量分数
        total_score = sum(score * weight for score, weight in quality_components.values())
        
        # === 8. 质量等级分类 ===
        quality_grade = pd.Series('C', index=dataframe.index)  # 默认C级
        quality_grade = np.where(total_score >= 85, 'A+',
                        np.where(total_score >= 80, 'A', 
                        np.where(total_score >= 75, 'A-',
                        np.where(total_score >= 70, 'B+',
                        np.where(total_score >= 65, 'B',
                        np.where(total_score >= 60, 'B-',
                        np.where(total_score >= 55, 'C+',
                        np.where(total_score >= 50, 'C',
                        np.where(total_score >= 40, 'C-',
                        np.where(total_score >= 30, 'D', 'F'))))))))))
        
        # === 9. 信号过滤建议 ===
        # 基于质量评分提供过滤建议
        high_quality_signals = total_score >= 75      # A级以上
        medium_quality_signals = total_score >= 60    # B级以上  
        low_quality_signals = total_score < 50        # C级以下
        
        # === 10. 仓位大小建议 ===
        # 根据信号质量建议仓位大小倍数
        position_multiplier = np.where(
            total_score >= 85, 1.5,      # A+级: 1.5倍仓位
            np.where(total_score >= 80, 1.3,  # A级: 1.3倍
            np.where(total_score >= 70, 1.0,  # B级: 标准仓位
            np.where(total_score >= 60, 0.8,  # B-级: 0.8倍
            np.where(total_score >= 50, 0.5,  # C级: 0.5倍
            0.3)))))  # D级及以下: 0.3倍
        
        return {
            # 核心评分
            'signal_quality_score': total_score / 100,  # 标准化到0-1
            'signal_quality_grade': quality_grade,
            'signal_quality_raw': total_score,  # 原始0-100分
            
            # 分项评分
            'technical_alignment_score': technical_alignment,
            'mtf_quality_score': mtf_quality, 
            'noise_quality_score': noise_quality,
            'volume_quality_score': volume_quality,
            'momentum_quality_score': momentum_quality,
            'position_quality_score': position_quality,
            
            # 决策辅助
            'high_quality_signal': high_quality_signals,
            'medium_quality_signal': medium_quality_signals,
            'low_quality_signal': low_quality_signals,
            'position_size_multiplier': position_multiplier,
            
            # 质量等级统计
            'quality_distribution': {
                'A_grade_signals': (quality_grade.isin(['A+', 'A', 'A-'])).sum(),
                'B_grade_signals': (quality_grade.isin(['B+', 'B', 'B-'])).sum(), 
                'C_grade_signals': (quality_grade.isin(['C+', 'C', 'C-'])).sum(),
                'D_grade_signals': (quality_grade.isin(['D', 'F'])).sum()
            }
        }
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """智能入场系统 - 防止追涨杀跌"""

        pair = metadata['pair']

        # 🔧 修复索引对齐问题：重置索引为连续整数，确保所有numpy操作后仍能对齐
        dataframe = dataframe.reset_index(drop=True)

        # === 🔊 5分钟框架噪音过滤（优先级最高）===
        # === 噪音过滤系统 ===
        noise_filters = self.filter_5min_noise(dataframe)

        # 在高噪音环境下完全禁止交易（主要使用的过滤器）
        noise_free_env = ~noise_filters['avoid_trading']
        # 注意：clean_trading_env 变量已定义但未使用，保留以备将来可能的功能扩展
        
        # === 核心防追涨杀跌过滤器 ===
        # 计算价格位置（20根K线相对位置，0=最低 1=最高）
        # 安全除法：添加小值避免除零，保持pandas兼容性
        highest_20 = dataframe['high'].rolling(20).max()
        lowest_20 = dataframe['low'].rolling(20).min()
        price_range_20 = highest_20 - lowest_20
        # 使用pandas Series操作，避免numpy array转换
        price_position = (dataframe['close'] - lowest_20) / (price_range_20 + 0.0001)
        
        # 🚨 动态价格位置守卫 - 在强势时放宽、弱势时收紧
        # 修复：使用pandas方法代替numpy，避免索引对齐问题
        trend_strength_series = self._safe_series(dataframe.get('trend_strength', 0), len(dataframe), 0).astype(float)
        trend_strength_series.index = dataframe.index
        atr_series = self._safe_series(dataframe.get('atr_p', 0.02), len(dataframe), 0.02).astype(float)
        atr_series.index = dataframe.index

        # 🎯 根据币种类型选择参数（蓝筹 vs Meme）
        is_bluechip = pair in self.bluechip_pairs
        overextended_long_pos_cap = self.overextended_long_pos_cap_bluechip if is_bluechip else self.overextended_long_pos_cap_meme
        oversold_short_pos_floor = self.oversold_short_pos_floor_bluechip if is_bluechip else self.oversold_short_pos_floor_meme

        # 使用pandas的clip和where保持Series类型
        long_price_cap = (
            0.72
            + trend_strength_series.clip(0, 80) / 400  # 强趋势可稍放宽
            + (atr_series - 0.015).clip(0, 0.05) * 1.2  # 高波动需要更多空间
        ).clip(0.58, 0.82)
        # 取最小值
        long_price_cap = long_price_cap.where(
            long_price_cap < overextended_long_pos_cap - 0.05,
            overextended_long_pos_cap - 0.05
        )
        not_at_top = price_position < long_price_cap

        # 使用pandas方法保持Series类型
        short_price_floor = (
            0.28
            - trend_strength_series.clip(-80, 0) / 400  # 强烈下跌时允许稍低
            - (atr_series - 0.015).clip(0, 0.05) * 1.0
        ).clip(0.18, 0.45)
        # 取最大值
        short_price_floor = short_price_floor.where(
            short_price_floor > oversold_short_pos_floor + 0.05,
            oversold_short_pos_floor + 0.05
        )
        not_at_bottom = price_position > short_price_floor

        # 额外的极端价格区过滤，防止在高位追多或低位追空
        overextended_bull = (
            (price_position > overextended_long_pos_cap) &
            (dataframe['rsi_14'] > self.overextended_long_rsi_cap) &
            (dataframe['close'] > dataframe['ema_21'] * self.overextended_long_ema_mult) &
            (dataframe['bb_position'] > self.overextended_long_bb_cap)
        )

        oversold_bear = (
            (price_position < oversold_short_pos_floor) &
            (dataframe['rsi_14'] < self.oversold_short_rsi_floor) &
            (dataframe['close'] < dataframe['ema_21'] * self.oversold_short_ema_mult) &
            (dataframe['bb_position'] < self.oversold_short_bb_floor)
        )
        
        # === 🎯 HYPEROPT优化价格位置过滤器 ===
        # 计算价格在历史范围中的分位数（20周期）
        price_percentile_20 = (dataframe['close'] - lowest_20) / (highest_20 - lowest_20 + 0.0001)
        price_percentile_50 = (dataframe['close'] - dataframe['low'].rolling(50).min()) / (dataframe['high'].rolling(50).max() - dataframe['low'].rolling(50).min() + 0.0001)
        
        # 基于HYPEROPT参数的有利区间判断
        in_favorable_long_zone = price_percentile_20 < self.price_percentile_long_max
        in_favorable_short_zone = price_percentile_20 > self.price_percentile_short_min
        
        # 最佳入场区间
        in_best_long_zone = price_percentile_20 < self.price_percentile_long_best
        in_best_short_zone = price_percentile_20 > self.price_percentile_short_best
        
        # 综合市场环境判断
        long_favourable_environment = in_favorable_long_zone & noise_free_env
        short_favourable_environment = in_favorable_short_zone & noise_free_env

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

        # === 🎯 时机选择过滤器：防止追涨杀跌 ===
        # 计算短期涨跌幅，避免在快速拉升时追涨
        price_change_1h = (dataframe['close'] - dataframe['close'].shift(4)) / dataframe['close'].shift(4)  # 4*15min=1小时涨跌幅
        price_change_30m = (dataframe['close'] - dataframe['close'].shift(2)) / dataframe['close'].shift(2)  # 2*15min=30分钟涨跌幅

        # 根据币种类型设置不同的涨幅阈值
        momentum_threshold_long = 0.06 if is_bluechip else 0.08   # 蓝筹6%，meme币8%
        momentum_threshold_short = -0.06 if is_bluechip else -0.08 # 对称的做空阈值

        # 避免追涨：1小时涨幅过大时不做多
        avoid_fomo_long = price_change_1h < momentum_threshold_long
        # 避免追跌：1小时跌幅过大时不做空
        avoid_fomo_short = price_change_1h > momentum_threshold_short

        # 避免在极端短期波动时开仓（30分钟涨跌幅超过4%）
        avoid_extreme_momentum = (abs(price_change_30m) < 0.04)

        timing_filter_long = avoid_fomo_long & avoid_extreme_momentum
        timing_filter_short = avoid_fomo_short & avoid_extreme_momentum

        # 增强的基础环境判断
        basic_env = (
            (dataframe['volume_ratio'] > 0.8) &  # 成交量不能太低
            (dataframe['atr_p'] > 0.001) &       # 波动性基本要求
            sideways_filter &                     # 横盘市场过滤
            rsi_momentum_strong &                # RSI动量未衰竭
            volume_support                       # 成交量支撑
        )
        
        # === 🎯 新增：MTF趋势一致性环境过滤器 ===
        # 趋势对齐的多头环境：MTF确认看多或趋势偏多
        mtf_direction_series = self._safe_series(
            dataframe.get('mtf_consensus_direction', 'neutral'),
            len(dataframe),
            'neutral'
        ).astype(str)
        mtf_direction_series.index = dataframe.index
        mtf_strength_series = self._safe_series(
            dataframe.get('mtf_consensus_strength', 'weak'),
            len(dataframe),
            'weak'
        ).astype(str)
        mtf_strength_series.index = dataframe.index
        mtf_trend_score_series = self._safe_series(
            dataframe.get('mtf_trend_score', 0.0),
            len(dataframe),
            0.0
        ).astype(float)
        mtf_trend_score_series.index = dataframe.index
        mtf_long_filter_series = self._safe_series(
            dataframe.get('mtf_long_filter', 0),
            len(dataframe),
            0
        ).astype(int)
        mtf_long_filter_series.index = dataframe.index
        mtf_short_filter_series = self._safe_series(
            dataframe.get('mtf_short_filter', 0),
            len(dataframe),
            0
        ).astype(int)
        mtf_short_filter_series.index = dataframe.index

        trend_aligned_long_env = (
            basic_env &
            timing_filter_long &  # 🎯 新增：时机选择过滤器
            (
                (mtf_direction_series == 'bullish') |
                ((mtf_strength_series.isin(['moderate', 'strong', 'very_strong'])) & (mtf_trend_score_series > -0.05)) |
                (mtf_trend_score_series > 0.1) |
                (mtf_long_filter_series == 1)
            )
        )

        # 趋势对齐的空头环境：MTF确认看空或趋势偏空
        trend_aligned_short_env = (
            basic_env &
            timing_filter_short &  # 🎯 新增：时机选择过滤器
            (
                (mtf_direction_series == 'bearish') |
                ((mtf_strength_series.isin(['moderate', 'strong', 'very_strong'])) & (mtf_trend_score_series < 0.05)) |
                (mtf_trend_score_series < -0.1) |
                (mtf_short_filter_series == 1)
            )
        )

        if not self.use_mtf_entry_filter:
            mtf_long_filter_series = pd.Series(1, index=dataframe.index)
            mtf_short_filter_series = pd.Series(1, index=dataframe.index)
            dataframe['mtf_long_filter'] = 1
            dataframe['mtf_short_filter'] = 1
            trend_aligned_long_env = basic_env & timing_filter_long   # 🎯 保持时机过滤器
            trend_aligned_short_env = basic_env & timing_filter_short # 🎯 保持时机过滤器
        
        # 强趋势环境：用于禁用逆势信号
        very_strong_bull_env = (mtf_direction_series == 'bullish') & (mtf_strength_series == 'very_strong')
        
        very_strong_bear_env = (mtf_direction_series == 'bearish') & (mtf_strength_series == 'very_strong')
        
        # 🚨 修复：定义缺失的环境变量（之前未定义导致60+信号失效）
        # 做多有利环境：趋势不过度弱势 + 情绪不过度悲观
        long_favourable_environment = (
            basic_env &  # 基础环境良好
            (dataframe['trend_strength'] > -40) &  # 趋势不过度弱势（放宽）
            (dataframe.get('market_sentiment', 0) > -0.8) &  # 情绪不过度悲观（放宽）
            (dataframe['rsi_14'] > 25) &
            not_at_top
        )
        
        # 做空有利环境：趋势不过度强势 + 情绪不过度乐观  
        short_favourable_environment = (
            basic_env &  # 基础环境良好
            (dataframe['trend_strength'] < 40) &   # 趋势不过度强势（放宽）
            (dataframe.get('market_sentiment', 0) < 0.8) &   # 情绪不过度乐观（放宽）
            (dataframe['rsi_14'] < 75) &
            not_at_bottom
        )
        
        # === 🌍 市场状态感知系统 ===
        market_regime_data = self._enhanced_market_regime_detection(dataframe)
        current_regime = market_regime_data['regime']
        regime_confidence = market_regime_data['confidence']
        signals_advice = market_regime_data['signals_advice']

        # 记录市场状态到dataframe（用于后续分析）
        dataframe.loc[:, 'market_regime'] = current_regime
        dataframe.loc[:, 'regime_confidence'] = regime_confidence
        
        self._log_message(
            f"📊 市场状态识别 {metadata.get('pair', '')}: "
            f"{current_regime} (置信度:{regime_confidence:.1%}) | "
            f"推荐信号:{signals_advice.get('recommended_signals', [])} | "
            f"避免信号:{signals_advice.get('avoid_signals', [])}",
            importance="verbose"
        )

        recommended_signals = set(signals_advice.get('recommended_signals', []))

        allow_relaxed_long_env = (
            (regime_confidence < 0.45) or
            ('RSI_Trend_Confirmation' in recommended_signals) or
            ('EMA_Golden_Cross' in recommended_signals)
        )
        allow_relaxed_short_env = (
            (regime_confidence < 0.45) or
            ('RSI_Overbought_Fall' in recommended_signals) or
            ('MACD_Bearish' in recommended_signals)
        )

        noise_strength_series = noise_filters.get('signal_strength_adjustment', 1.0)
        if isinstance(noise_strength_series, pd.Series):
            noise_strength_series = noise_strength_series.reindex(dataframe.index, fill_value=1.0).astype(float)
        else:
            noise_strength_series = pd.Series([float(noise_strength_series)] * len(dataframe), index=dataframe.index)
        noise_relaxed_support = noise_strength_series > 0.7
        noise_ok_relaxed = noise_strength_series > 0.55

        long_env_strict = (
            trend_aligned_long_env &
            (~very_strong_bear_env) &
            (~overextended_bull) &
            not_at_top &
            (noise_free_env | noise_relaxed_support)
        )
        long_env_relaxed = (
            basic_env &
            (~very_strong_bear_env) &
            (~overextended_bull) &
            not_at_top &
            (noise_free_env | (allow_relaxed_long_env & noise_ok_relaxed))
        )

        short_env_strict = (
            trend_aligned_short_env &
            (~very_strong_bull_env) &
            (~oversold_bear) &
            not_at_bottom &
            (noise_free_env | noise_relaxed_support)
        )
        short_env_relaxed = (
            basic_env &
            (~very_strong_bull_env) &
            (~oversold_bear) &
            not_at_bottom &
            (noise_free_env | (allow_relaxed_short_env & noise_ok_relaxed))
        )

        # === 💰 智能市场适应性信号 ===

        # === 🛡️ 强势下跌保护（防止在强趋势逆向时被拉爆）===
        # 对称于做空的强势上涨保护（第7067-7071行）
        strong_downtrend_protection = ~(
            (dataframe['ema_8'] < dataframe['ema_21'] * 0.98) &  # EMA8明显低于EMA21
            (dataframe['adx'] > 30) &  # 强趋势
            (dataframe['close'] < dataframe['ema_50'])  # 价格在中期均线之下
        )

        # 🎯 Signal 1: RSI超卖反弹（增强动态版）
        # === 使用新的动态RSI阈值系统 ===
        # 基于趋势强度、波动率、市场情绪等多重因子计算的智能阈值
        dynamic_oversold = dataframe.get(
            'rsi_dynamic_oversold',
            dataframe.get('rsi_oversold_dynamic', 25)
        )
        dynamic_overbought = dataframe.get(
            'rsi_dynamic_overbought',
            dataframe.get('rsi_overbought_dynamic', 75)
        )
        
        # === 多重确认机制（增强版）===
        rsi_condition = (dataframe['rsi_14'] < dynamic_oversold)
        rsi_momentum = (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2))  # 连续2期上升
        
        # === 趋势确认：只在上升趋势或横盘中做多 ===
        trend_confirmation = (
            (dataframe['ema_8'] >= dataframe['ema_21']) |  # 多头排列
            (dataframe['adx'] < 25)  # 或横盘环境
        )
        
        # === 成交量确认：突破需要成交量支撑 ===
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.95
        )

        # === 强度确认：ADX显示趋势开始形成 ===
        strength_confirmation = (
            (dataframe['adx'] > 20) &  # 最低强度要求
            (dataframe['adx'] > dataframe['adx'].shift(2))  # ADX上升
        )

        # === 背离检测：避免在顶背离时入场 ===
        no_bearish_divergence = ~dataframe.get('bearish_divergence', False).astype(bool)

        trend_basis = (
            (dataframe['ema_8'] > dataframe['ema_21']) |
            ((dataframe['close'] > dataframe['ema_21']) & (dataframe['ema_8'] > dataframe.get('ema_13', dataframe['ema_21'])))
        )

        adx_support = dataframe['adx'] > 18

        price_breakout = (
            (dataframe['close'] > dataframe['high'].rolling(3).max().shift(1)) |
            ((dataframe['close'] > dataframe.get('ema_13', dataframe['ema_21'])) & (dataframe['close'] > dataframe['close'].shift(1)))
        )

        rsi_trend_confirmation_core = (
            trend_basis &
            adx_support &
            (dataframe['rsi_14'] > dynamic_oversold + 3) &
            price_breakout &
            volume_confirmation &
            no_bearish_divergence &
            long_env_strict
        )

        rsi_rebound_signal = (
            rsi_condition &
            trend_basis &
            strength_confirmation &
            (dataframe['close'] >= dataframe['ema_21'] * 0.985) &
            (dataframe['close'] <= dataframe['ema_21'] * 1.03) &
            no_bearish_divergence &
            long_env_relaxed &
            (dataframe['volume_ratio'] > 0.85)
        )

        # 🎯 修复：正确的RSI反弹确认逻辑
        # 正确的RSI趋势确认：RSI从低位(<30)反弹并突破30
        rsi_midtrend_reset = (
            (dataframe['rsi_14'].shift(1) < dynamic_oversold) &  # 前一根K线RSI在超卖区
            (dataframe['rsi_14'] > dynamic_oversold) &  # 当前K线RSI突破超卖区
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &  # RSI向上
            trend_basis &
            long_env_relaxed &
            (dataframe['close'] > dataframe['ema_21'] * 0.99) &
            (dataframe['close'] < dataframe['ema_21'] * 1.04) &
            no_bearish_divergence
        )

        rsi_fast_breakout = (
            long_env_relaxed &
            (noise_strength_series > 0.55) &
            not_at_top &
            (dataframe['close'] > dataframe.get('ema_8', dataframe['close'])) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['rsi_14'].shift(1) < dynamic_oversold + 4) &
            (dataframe['rsi_14'] > dynamic_oversold + 0.5) &
            (mtf_trend_score_series > -0.05)
        )

        # === 🎯 RSI信号额外验证层：恢复严格过滤（基于2024-2025最佳实践）===
        # 修复策略：恢复关键过滤器，防止高位入场和弱趋势触发
        # 参考来源：Multi-Filter RSI Momentum Confirmation Trading Strategy (2024)

        # K线形态过滤：实体占比>50%（过滤弱势K线）
        candle_body = abs(dataframe['close'] - dataframe['open'])
        candle_range = dataframe['high'] - dataframe['low']
        strong_candle = candle_body > (candle_range * 0.5)

        enhanced_market_check = (
            not_at_top &  # ✅ 恢复：不在高位入场
            (dataframe['adx'] > 25) &  # ✅ 提高到行业标准（原18）
            timing_filter_long &  # 保留：防止1小时内涨幅过大（6-8%）
            (dataframe['close'] > dataframe['ema_50'] * 0.98) &  # ✅ 新增：价格接近中期均线
            (dataframe['rsi_14'] < 70) &  # ✅ 收紧：75→70
            (dataframe['volume'] > dataframe['volume_ma_20'] * 1.5) &  # ✅ 成交量确认（1.5倍均量）
            strong_candle  # ✅ K线形态过滤（实体>50%）
        )

        rsi_trend_confirmation = (
            rsi_trend_confirmation_core |
            rsi_rebound_signal |
            rsi_midtrend_reset |
            rsi_fast_breakout
        ) & enhanced_market_check & strong_downtrend_protection  # ✅ 添加下跌保护
        dataframe.loc[rsi_trend_confirmation, 'enter_long'] = 1
        dataframe.loc[rsi_trend_confirmation, 'enter_tag'] = 'RSI_Trend_Confirmation'
        
        # 🎯 Signal 2: EMA金叉信号 - 已删除
        # 原因：数据分析显示平均胜率仅33.0%，平均收益-0.72%（亏损）
        # 平均入场位置71.6%（偏高），特别是ETH上胜率仅28.6%
        # 结论：EMA金叉是典型的滞后指标，趋势确立后才触发，已错过最佳入场点
        # 策略：后续考虑用"EMA即将金叉"（距离<1%）的领先型信号替代

        # 定义MACD删除后缺失的变量
        allow_relaxed_breakout = False  # BB反指信号控制（MACD相关变量）
        not_too_far_from_200ema = (dataframe['close'] < dataframe['ema_200'] * 1.15)  # 不超过200EMA的15%

        # 🚫 布林带突破跟随信号（已禁用）
        # 原因：假突破率过高，53次交易亏损31.34%
        # 根据ChatGPT分析，此信号在震荡市中容易被反向拉爆
        # 暂时完全禁用，等主策略稳定后再考虑重新设计

        # bb_breakthrough_follow = (
        #     # 原突破信号逻辑已注释禁用
        #     False  # 完全禁用此信号
        # )
        # dataframe.loc[bb_breakthrough_follow, 'enter_long'] = 1
        # dataframe.loc[bb_breakthrough_follow, 'enter_tag'] = 'BB_Breakthrough_Follow'
        
        # Signal 5 已删除 - Simple_Breakout容易产生假突破信号
        
        # === 📉 简化的做空信号 ===
        
        # 🎯 Signal 1: RSI超买回落（增强动态版）
        # === 使用新的动态RSI阈值系统 ===
        # 基于趋势强度、波动率、市场情绪等多重因子计算的智能阈值
        # dynamic_overbought已在上面定义，这里直接使用
        
        # === 多重确认机制 - 简化版本 ===
        # 数据分析：RSI做空信号当前0触发，过滤太严格
        # 修复策略：大幅简化条件，允许更容易触发
        rsi_condition = (dataframe['rsi_14'] > dynamic_overbought)
        rsi_momentum = (dataframe['rsi_14'] < dataframe['rsi_14'].shift(1))  # 🎯 放宽：shift(2)→shift(1)

        # === 趋势确认：放宽要求 ===
        trend_confirmation = (
            (dataframe['ema_8'] <= dataframe['ema_21']) |  # 空头排列
            (dataframe['adx'] < 30)  # 🎯 放宽：25→30，更容易触发
        )

        # === 背离检测：避免在底背离时入场 ===
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)

        # 🔧 新增：强势上涨保护 - 避免在强势上涨趋势中被反向拉爆
        strong_uptrend_protection = ~(
            (dataframe['ema_8'] > dataframe['ema_21'] * 1.02) &  # EMA8明显高于EMA21
            (dataframe['adx'] > 30) &  # 强趋势
            (dataframe['close'] > dataframe['ema_50'])  # 价格在中期均线之上
        )

        rsi_overbought_fall = (
            rsi_condition &
            rsi_momentum &
            trend_confirmation &
            # 🎯 移除：price_confirmation（不要求价格已经下跌）
            # 🎯 移除：volume_confirmation（不要求成交量）
            # 🎯 移除：strength_confirmation（不要求ADX上升）
            no_bullish_divergence &
            not_at_bottom &  # 防止在底部追空
            strong_uptrend_protection &  # 🔧 新增：强势上涨保护
            (short_env_strict | (short_env_relaxed & allow_relaxed_short_env)) &
            (~oversold_bear)
        )
        rsi_fast_short = (
            short_env_relaxed &
            (noise_strength_series > 0.55) &
            not_at_bottom &
            (dataframe['close'] < dataframe.get('ema_8', dataframe['close'])) &
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['rsi_14'].shift(1) > dynamic_overbought - 4) &
            (dataframe['rsi_14'] < dynamic_overbought - 0.5) &
            (mtf_trend_score_series < 0.05)
        )

        # 🔧 新增：做空位置检查 - 确保在相对高位才做空
        at_high_position = price_position > 0.55  # 至少在55%以上位置才做空

        rsi_overbought_fall = (
            (rsi_overbought_fall | rsi_fast_short) &
            not_at_bottom &
            at_high_position  # 🔧 新增：高位检查，避免低位做空
        )
        # === 📊 信号质量评分系统 ===
        rsi_long_score = self._calculate_signal_quality_score(
            dataframe, rsi_trend_confirmation, 'RSI_Trend_Confirmation'
        )
        rsi_short_score = self._calculate_signal_quality_score(
            dataframe, rsi_overbought_fall, 'RSI_Overbought_Fall'
        )
        
        # === 📊 RSI信号质量过滤（移除避免逻辑限制）===
        # 所有高质量信号都允许触发，不受市场状态限制
        
        # RSI做多信号 - 提高质量要求 🎯
        high_quality_long = rsi_trend_confirmation & (rsi_long_score >= 6.5)  # 提高阈值防止假信号

        # RSI做空信号 - 提高质量要求 🎯
        high_quality_short = rsi_overbought_fall & (rsi_short_score >= 7.5)  # 🔧 优化：从6.5提高到7.5，减少低质量做空

        # 市场状态奖励：在推荐的市场环境中降低质量要求
        if 'RSI_Trend_Confirmation' in recommended_signals:
            regime_bonus_long = rsi_trend_confirmation & (rsi_long_score >= 5.5)  # 提高推荐环境要求 🎯
            high_quality_long = high_quality_long | regime_bonus_long
            
        if 'RSI_Overbought_Fall' in recommended_signals:
            regime_bonus_short = rsi_overbought_fall & (rsi_short_score >= 5.5)  # 提高推荐环境要求 🎯
            high_quality_short = high_quality_short | regime_bonus_short

        dataframe.loc[high_quality_long, 'enter_long'] = 1
        dataframe.loc[high_quality_long, 'enter_tag'] = 'RSI_Trend_Confirmation'
        dataframe.loc[high_quality_long, 'signal_quality'] = rsi_long_score
        dataframe.loc[high_quality_long, 'market_regime_bonus'] = 'RSI_Trend_Confirmation' in recommended_signals

        dataframe.loc[high_quality_short, 'enter_short'] = 1
        dataframe.loc[high_quality_short, 'enter_tag'] = 'RSI_Overbought_Fall'
        dataframe.loc[high_quality_short, 'signal_quality'] = rsi_short_score
        dataframe.loc[high_quality_short, 'market_regime_bonus'] = 'RSI_Overbought_Fall' in recommended_signals
        
        # 🎯 Signal 2: EMA死叉信号（增强过滤版）
        # === 使用EMA交叉过滤器的死叉信号 ===
        # 对称的做空信号，与金叉做多相对应

        strong_ema_death = dataframe.get('ema_strong_death_cross', False)

        # ✅ 只使用强死叉信号（成功率80%+），删除中等信号
        validated_death_cross = strong_ema_death

        # === 🛡️ 急跌保护：检测短期急跌（防止追空到底部）===
        # 计算1小时（4根15分钟K线）内的跌幅
        recent_decline_pct = (dataframe['close'] / dataframe['close'].shift(4) - 1)
        no_panic_sell = recent_decline_pct > -0.08  # 1小时内跌幅<8%

        # === 🎯 EMA死叉信号额外验证层：防止假突破 ===
        # 增强的空头市场状态验证
        enhanced_bearish_check = (
            not_at_bottom &                                           # 基础位置检查
            (dataframe['adx'] > 30) &                                 # ✅ 提高趋势强度要求：25→30
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2) &  # 成交量确认
            timing_filter_short &                                     # 使用时机过滤器防止追跌
            (dataframe['close'] < dataframe['ema_50']) &              # 确保在中期趋势之下
            (dataframe['rsi_14'] > 40) &                             # ✅ 提高RSI下限：30→40
            (dataframe['rsi_14'] < 70) &                             # ✅ 添加RSI上限，避免在下跌后期开空
            (dataframe['close'] < dataframe['ema_21']) &              # 确保在短期趋势之下
            (dataframe['ema_8'] < dataframe['ema_21']) &              # 确保短期EMA在长期EMA之下
            no_panic_sell                                             # ✅ 急跌保护
        )

        # 只使用强信号 + 增强验证
        ema_death_cross = validated_death_cross & enhanced_bearish_check

        # 应用基础环境过滤和噪音过滤（对称的空头过滤）
        ema_death_cross = ema_death_cross & (short_env_strict | (short_env_relaxed & allow_relaxed_short_env))
        
        dataframe.loc[ema_death_cross, 'enter_short'] = 1
        dataframe.loc[ema_death_cross, 'enter_tag'] = 'EMA_Death_Cross_Filtered'
        
        
        # 🎯 Signal 3: MACD看跌信号（增强前置确认版）
        # === MACD基础信号（传统+前置） ===
        macd_death_cross = (
            (dataframe['macd'] < dataframe['macd_signal']) & 
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))
        )
        macd_hist_negative = (
            (dataframe['macd_hist'] < 0) & 
            (dataframe['macd_hist'].shift(1) >= 0)
        )
        
        # === 🛡️ MACD做空防假信号系统 ===
        # 1. 检测是否在明显底部区域（避免在低位追空）
        price_in_bottom_20pct = (
            (dataframe['close'] < dataframe['low'].rolling(50).quantile(0.2)) |  # 在50日低点的前20%
            (dataframe['rsi_14'] < 30) |  # RSI超卖
            (dataframe['bb_position'] < 0.2)  # 在布林带下轨区域
        )
        
        # 2. MACD需要在正值区域或刚转负（避免在低位杀跌）
        macd_not_oversold = (
            (dataframe['macd'] > 0) |  # MACD在零轴上方
            ((dataframe['macd'] < 0) & (dataframe['macd'] > dataframe['macd'].rolling(20).quantile(0.7)))  # 或在负值区域的高位
        )
        
        # 3. 价格不能距离200EMA太远（避免杀跌）
        not_too_far_below_200ema = (
            (dataframe['close'] > dataframe['ema_200'] * 0.85)  # 不低于200EMA的15%
        )
        
        # 4. 趋势环境必须支持做空
        trend_supports_short = (
            (dataframe['ema_8'] < dataframe['ema_21']) &  # 短期趋势向下
            (dataframe['close'] < dataframe['ema_34'])    # 价格在中期均线下方
        )
        
        # === 🎯 简化MACD做空条件（真实死叉确认版）===
        # ✅ 修改：移除预判性逻辑，只保留真实死叉（基于MACD最佳实践2024）
        # 参考：MACD是滞后指标，预判容易假信号，应等待真实死叉确认
        macd_basic_signal = (
            # ✅ 真实死叉确认
            (dataframe['macd'] < dataframe['macd_signal']) &  # 真实死叉
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1)) &  # 刚发生死叉
            (dataframe['macd'] > 0)  # 在零轴上方（顶部区域）
        )
        
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
        # ✅ 修改：提高成交量要求到1.5倍（基于MACD最佳实践2024）
        volume_confirmation = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # ✅ 提高：0.95→1.5
            (dataframe['volume'] > dataframe['volume'].shift(1) * 0.9)  # 成交量保持
        )

        # 4. 强度确认：ADX显示趋势强化
        strength_confirmation = (
            (dataframe['adx'] > 20) &                     # 有一定趋势强度
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
        
        # === 🎯 最终MACD看跌信号（预判性顶部反转版） ===
        macd_bearish = (
            macd_basic_signal &

            # === 顶部环境确认 ===
            (short_env_strict | (short_env_relaxed & allow_relaxed_short_env)) &
            (
                (dataframe['close'] > dataframe['high'].rolling(50).quantile(0.8)) |  # 在50日高点的前20%
                (dataframe['rsi_14'] > 65)  # 或RSI偏高
            ) &  # 在顶部区域确认
            not_too_far_from_200ema &  # 不距离200EMA太远

            # === MTF趋势支持确认 ===
            # 额外的顶部确认过滤
            (dataframe['rsi_14'] > 35) &  # RSI不能太低
            (dataframe['close'] > dataframe['close'].rolling(50).mean()) &  # 价格在50日均线上方
            (dataframe['volume_ratio'] > 0.75) &  # 有基本的成交量支撑

            # 🛡️ 启用防护系统：避免在底部追空
            (~price_in_bottom_20pct)  # 不在底部20%区域（RSI<30 或 BB下轨 或 价格低位）
        )
        
        # === 📊 MACD信号质量评分 ===
        macd_score = self._calculate_macd_signal_quality(dataframe, macd_bearish, 'MACD_Bearish')
        
        # === 📊 MACD信号质量过滤（提高质量要求）===
        # ✅ 修改：提高质量阈值，减少低质量信号（基于2024最佳实践）
        high_quality_macd = macd_bearish & (macd_score >= 7.5)  # ✅ 提高：6.5→7.5

        # 市场状态奖励：在强下跌趋势中降低MACD要求（但仍高于原阈值）
        if 'MACD_Bearish' in recommended_signals:
            regime_bonus_macd = macd_bearish & (macd_score >= 6.5)  # ✅ 提高：5.5→6.5
            high_quality_macd = high_quality_macd | regime_bonus_macd

        # 反指信号失效预警检测（在此处计算以便两个反指信号都能使用）
        invalidation_signals = self.check_reversal_signal_invalidation(dataframe)

        # 强牛市环境检测（ChatGPT建议的三重确认）
        strong_bull_market = (
            (dataframe['ema_8'] > dataframe['ema_21']) & 
            (dataframe['ema_21'] > dataframe.get('ema_50', dataframe['ema_21'])) &  # EMA多头排列
            (dataframe['adx'] > 25) &                                               # 趋势强度确认
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))             # MACD柱状图向上翻转
        )

        # 仅在非强牛市环境中保留做空信号，避免与反指信号冲突
        macd_short_entries = high_quality_macd & (~strong_bull_market) & (~oversold_bear) & not_at_bottom
        dataframe.loc[macd_short_entries, 'enter_short'] = 1
        dataframe.loc[macd_short_entries, 'enter_tag'] = 'MACD_Bearish'
        dataframe.loc[macd_short_entries, 'signal_quality'] = macd_score
        dataframe.loc[macd_short_entries, 'market_regime_bonus'] = 'MACD_Bearish' in recommended_signals

        # 🔄 MACD_Bearish 反指信号（仅强牛市中使用）
        # 当MACD死叉在强牛市中出现时，反向做多
        # 根据币种类型选择反指位置上限
        reversal_pos_cap = self.reversal_pos_cap_bluechip if is_bluechip else self.reversal_pos_cap_meme

        macd_reversal_candidates = high_quality_macd & strong_bull_market
        macd_bearish_reversal = (
            macd_reversal_candidates &
            # 安全过滤条件
            (dataframe['rsi_14'] < 80) &                                           # 避免极度超买
            (dataframe['volume_ratio'] > 1.0) &                                   # 成交量支撑
            noise_free_env &                                                       # 噪音过滤

            # 避免在真正的顶部反转时触发
            (dataframe['close'] > dataframe['close'].rolling(10).mean()) &       # 价格仍在短期均线上方

            # 反指信号失效预警过滤
            (~invalidation_signals['macd_reversal_invalid']) &                   # 排除MACD反指失效场景
            (price_position < reversal_pos_cap)                                  # 避免在极端高位反指追多
        )

        # 反指信号做多（原来做空改为做多）
        dataframe.loc[macd_bearish_reversal, 'enter_long'] = 1
        dataframe.loc[macd_bearish_reversal, 'enter_tag'] = 'MACD_Bearish_Reversal'
        dataframe.loc[macd_bearish_reversal, 'signal_quality'] = macd_score
        
        # 🎯 Signal 4: 布林带假反压真突破信号（反指改造）
        # 原本的"反压"信号在牛市中频繁失效，改为利用假反压后的真突破
        
        # 计算K线质量和牛市环境
        candle_quality = self.calculate_candle_quality(dataframe)
        bull_environment = self.check_bull_market_environment(dataframe)
        
        bb_fake_rejection_breakout = (
            # 1. 原反压触发条件（检测到"假反压"）
            (dataframe['close'] >= dataframe['bb_upper'] * 0.995) &  # 接近上轨
            (dataframe['close'].shift(1) < dataframe['close'].shift(2)) &  # 前一根K线有回落迹象
            
            # 2. 突破确认条件（ChatGPT建议）
            (dataframe['close'] > dataframe['high'].shift(1)) &      # 突破前高
            (dataframe['close'] > dataframe['bb_upper']) &           # 确实突破上轨
            
            # 3. K线质量确认
            (candle_quality >= 1.0) &                               # 实体占主导
            
            # 4. 市场环境确认
            bull_environment &                                       # 牛市或上升趋势
            (dataframe['volume_ratio'] > 1.2) &                     # 放量突破
            (dataframe['rsi_14'] >= 50) & (dataframe['rsi_14'] <= 80) &  # RSI健康区间
            
            # 5. 环境过滤（避免极端情况）
            (long_env_strict | (long_env_relaxed & allow_relaxed_breakout)) &
            (~(dataframe['rsi_14'] > 80)) &                         # 避免极度超买
            
            # 6. 反指信号失效预警过滤
            (~invalidation_signals['bb_reversal_invalid']) &        # 排除BB反指失效场景
            (price_position.shift(1) < 0.75) &  # 确保突破非高位连续上行（固定75%上限）
            (~overextended_bull)
        )
        
        # 反指信号做多（原来做空改为做多）
        dataframe.loc[bb_fake_rejection_breakout, 'enter_long'] = 1
        dataframe.loc[bb_fake_rejection_breakout, 'enter_tag'] = 'BB_Fake_Rejection_Breakout'
        
        # 🚀 Signal 5: 强趋势跟随信号（新增 - 利用MTF强趋势获得高胜率）
        # === 强多头趋势跟随信号 ===
        # 关键指标缓冲，避免重复读取
        momentum_exhaustion = dataframe.get(
            'momentum_exhaustion_score',
            self._safe_series(0.0, len(dataframe))
        )
        momentum_score = dataframe.get(
            'momentum_score',
            self._safe_series(0.0, len(dataframe))
        )
        trend_strength_series = dataframe.get(
            'trend_strength',
            self._safe_series(0.0, len(dataframe))
        )
        market_sentiment_series = dataframe.get(
            'market_sentiment',
            self._safe_series(0.0, len(dataframe))
        )
        orderbook_buy_pressure = dataframe.get(
            'ob_buy_pressure',
            self._safe_series(0.5, len(dataframe))
        )
        orderbook_sell_pressure = dataframe.get(
            'ob_sell_pressure',
            self._safe_series(0.5, len(dataframe))
        )
        orderbook_liquidity = dataframe.get(
            'ob_liquidity_score',
            self._safe_series(0.4, len(dataframe))
        )
        price_acceleration = dataframe.get(
            'price_acceleration_rate',
            self._safe_series(0.0, len(dataframe))
        )

        mtf_direction_series = dataframe.get('mtf_consensus_direction')
        if isinstance(mtf_direction_series, pd.Series):
            mtf_direction_series = mtf_direction_series.fillna('neutral').astype(str)
        else:
            mtf_direction_series = pd.Series(['neutral'] * len(dataframe), index=dataframe.index)

        mtf_strength_series = dataframe.get('mtf_consensus_strength')
        if isinstance(mtf_strength_series, pd.Series):
            mtf_strength_series = mtf_strength_series.fillna('weak').astype(str)
        else:
            mtf_strength_series = pd.Series(['weak'] * len(dataframe), index=dataframe.index)

        mtf_direction_series = dataframe.get('mtf_consensus_direction')
        if isinstance(mtf_direction_series, pd.Series):
            mtf_direction_series = mtf_direction_series.fillna('neutral').astype(str)
        else:
            mtf_direction_series = pd.Series(['neutral'] * len(dataframe), index=dataframe.index)

        mtf_strength_series = dataframe.get('mtf_consensus_strength')
        if isinstance(mtf_strength_series, pd.Series):
            mtf_strength_series = mtf_strength_series.fillna('weak').astype(str)
        else:
            mtf_strength_series = pd.Series(['weak'] * len(dataframe), index=dataframe.index)
        mtf_direction_series = dataframe.get(
            'mtf_consensus_direction',
            self._safe_series('neutral', len(dataframe), 'neutral')
        )
        if not isinstance(mtf_direction_series, pd.Series):
            mtf_direction_series = pd.Series([mtf_direction_series] * len(dataframe), index=range(len(dataframe)))
        mtf_direction_series = mtf_direction_series.fillna('neutral').astype(str)

        mtf_strength_series = dataframe.get(
            'mtf_consensus_strength',
            self._safe_series('weak', len(dataframe), 'weak')
        )
        if not isinstance(mtf_strength_series, pd.Series):
            mtf_strength_series = pd.Series([mtf_strength_series] * len(dataframe), index=range(len(dataframe)))
        mtf_strength_series = mtf_strength_series.fillna('weak').astype(str)
        mtf_direction_series = dataframe.get(
            'mtf_consensus_direction',
            self._safe_series('neutral', len(dataframe), 'neutral')
        )
        mtf_strength_series = dataframe.get(
            'mtf_consensus_strength',
            self._safe_series('weak', len(dataframe), 'weak')
        )

        # === 强多头趋势跟随（升级版） ===
        # 根据币种类型选择强趋势位置上限
        strong_bullish_pos_cap = self.strong_bullish_pos_cap_bluechip if is_bluechip else self.strong_bullish_pos_cap_meme

        strong_bullish_base = (
            very_strong_bull_env &
            (dataframe.get('mtf_consensus_direction', '') == 'bullish') &
            (dataframe.get('mtf_consensus_strength', '') == 'very_strong') &
            (dataframe['close'] > dataframe['ema_21']) &
            (dataframe['ema_8'] > dataframe['ema_21']) &
            (dataframe['rsi_14'] > 45) & (dataframe['rsi_14'] < 72) &
            (dataframe['volume_ratio'] > 0.9) & (dataframe['volume_ratio'] < 2.6) &
            (price_position > 0.20) &
            (price_position < strong_bullish_pos_cap) &
            (momentum_exhaustion < 0.55) &
            (momentum_score > 0.15) & (momentum_score < 1.0) &
            (trend_strength_series > 35) & (trend_strength_series < 85) &
            (market_sentiment_series > -0.5) &
            (orderbook_buy_pressure > 0.55) &
            (orderbook_liquidity > 0.35) &
            (price_acceleration > -0.05) & (price_acceleration < 0.06) &
            basic_env &
            noise_free_env &
            trend_aligned_long_env &
            (~overextended_bull) &
            not_at_top
        )
        strong_bullish_follow = self._apply_signal_cooldown(
            strong_bullish_base.astype(bool),
            self.strong_signal_cooldown_bars
        )

        # === 强空头趋势跟随（升级版） ===
        strong_bearish_base = (
            very_strong_bear_env &
            (dataframe.get('mtf_consensus_direction', '') == 'bearish') &
            (dataframe.get('mtf_consensus_strength', '') == 'very_strong') &
            (dataframe['close'] < dataframe['ema_21']) &
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['rsi_14'] > 40) & (dataframe['rsi_14'] < 65) &
            (dataframe['volume_ratio'] > 0.9) & (dataframe['volume_ratio'] < 2.4) &
            (price_position > max(self.strong_bearish_pos_floor, 0.25)) &
            (price_position < 0.85) &
            (momentum_exhaustion < 0.45) &
            (momentum_score < -0.15) & (momentum_score > -1.0) &
            (trend_strength_series < -35) & (trend_strength_series > -85) &
            (market_sentiment_series < 0.5) &
            (orderbook_sell_pressure > 0.55) &
            (orderbook_liquidity > 0.35) &
            (price_acceleration < 0.04) & (price_acceleration > -0.10) &
            basic_env &
            noise_free_env &
            trend_aligned_short_env &
            (~oversold_bear) &
            not_at_bottom
        )
        strong_bearish_follow = self._apply_signal_cooldown(
            strong_bearish_base.astype(bool),
            self.strong_signal_cooldown_bars
        )

        # 应用强趋势跟随信号（高质量评分）
        dataframe.loc[strong_bullish_follow, 'enter_long'] = 1
        dataframe.loc[strong_bullish_follow, 'enter_tag'] = 'Strong_Bullish_Follow'
        dataframe.loc[strong_bullish_follow, 'signal_quality'] = 8.8  # 高质量评分

        dataframe.loc[strong_bearish_follow, 'enter_short'] = 1
        dataframe.loc[strong_bearish_follow, 'enter_tag'] = 'Strong_Bearish_Follow'
        dataframe.loc[strong_bearish_follow, 'signal_quality'] = 8.8  # 高质量评分
        
        # 强趋势跟随信号日志
        if strong_bullish_follow.any():
            self.event_log.info(
                "trend_follow_long",
                pair=metadata['pair'],
                signals=int(strong_bullish_follow.sum()),
            )
        if strong_bearish_follow.any():
            self.event_log.info(
                "trend_follow_short",
                pair=metadata['pair'],
                signals=int(strong_bearish_follow.sum()),
            )

        # ================================
        # 🆕 新增专业做多信号（基于2025年最佳实践）
        # ================================

        # 🎯 Signal 7: 成交量背离做多 🌟🌟
        # === 价格下跌但成交量递减 - 卖压衰竭信号 ===
        volume_lookback_long = 5

        # 价格连续下跌
        price_falling = (
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['close'].shift(1) < dataframe['close'].shift(2)) &
            (dataframe['close'] < dataframe['close'].shift(volume_lookback_long) * 0.99)  # 5天跌幅>1%
        )

        # 成交量递减
        volume_declining_long = (
            (dataframe['volume'] < dataframe['volume'].shift(1)) &
            (dataframe['volume'].shift(1) < dataframe['volume'].shift(2)) &
            (dataframe['volume'] < dataframe['volume'].rolling(volume_lookback_long).mean() * 0.85)
        )

        # 成交量背离做多
        volume_divergence_long = (
            price_falling &
            volume_declining_long &

            # 价格必须在相对低位
            (price_position < 0.45) &
            (price_position > 0.10) &

            # RSI确认超卖
            (dataframe['rsi_14'] < 40) &
            (dataframe['rsi_14'] > 20) &

            # 趋势确认：下跌趋势末期
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['adx'] > 15) &

            # 环境过滤
            long_env_relaxed &
            not_at_top
        )

        dataframe.loc[volume_divergence_long, 'enter_long'] = 1
        dataframe.loc[volume_divergence_long, 'enter_tag'] = 'Volume_Divergence_Long'

        # ================================
        # 做空信号开始
        # ================================

        # 🎯 Signal 6: RSI反弹做空 - 填补"低吸"型做空空白（新增）
        # === 对称RSI超卖做多的"低吸"做空信号 ===
        # 逻辑：在下跌趋势的反弹中做空，而非在顶部追空
        rsi_rebound_short = (
            # === 下跌趋势或空头环境 ===
            (
                (dataframe['ema_8'] < dataframe['ema_21']) |  # 空头排列
                (dataframe['close'] < dataframe['ema_50'])  # 或价格在中期均线下方
            ) &

            # === RSI从低位反弹到阻力区（关键！）===
            (dataframe['rsi_14'].shift(3) < 40) &  # 3根K线前RSI在低位
            (dataframe['rsi_14'] > 50) &  # 现在反弹到50以上（阻力区）
            (dataframe['rsi_14'] < 65) &  # 但还没到极端超买

            # === 价格反弹到关键阻力位 ===
            (
                # 接近EMA21阻力（下跌趋势的回调目标）
                ((dataframe['close'] > dataframe['ema_21'] * 0.98) &
                 (dataframe['close'] < dataframe['ema_21'] * 1.02)) |

                # 或接近EMA50阻力（中期均线压力）
                ((dataframe['close'] > dataframe['ema_50'] * 0.98) &
                 (dataframe['close'] < dataframe['ema_50'] * 1.02))
            ) &

            # === 反弹动能开始衰竭 ===
            (
                (dataframe['rsi_14'] < dataframe['rsi_14'].shift(1)) |  # RSI开始回落
                (dataframe['volume'] < dataframe['volume'].rolling(10).mean())  # 或成交量萎缩
            ) &

            # === 价格位置确认（在反弹中，不在底部）===
            (price_position > 0.35) &  # 不在极端低位（避免接飞刀）
            (price_position < 0.75) &  # 也不在高位

            # === 环境过滤 ===
            short_env_relaxed &
            not_at_bottom &
            (~oversold_bear)  # 不在极度超卖状态
        )

        dataframe.loc[rsi_rebound_short, 'enter_short'] = 1
        dataframe.loc[rsi_rebound_short, 'enter_tag'] = 'RSI_Rebound_Short'

        # 🎯 Signal 7: MACD反指做空 - 对称MACD_Bearish_Reversal（新增）
        # === 在强熊市中MACD金叉时反向做空 ===
        # 根据币种类型选择反指位置下限
        reversal_pos_floor_short = self.reversal_pos_cap_bluechip if is_bluechip else self.reversal_pos_cap_meme
        reversal_pos_floor_short = 1.0 - reversal_pos_floor_short  # 转换为下限（0.12-0.15）

        # 强熊市环境检测
        strong_bear_market = (
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['ema_21'] < dataframe.get('ema_50', dataframe['ema_21'])) &  # EMA空头排列
            (dataframe['adx'] > 25) &  # 趋势强度确认
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))  # MACD柱状图向下
        )

        # 检测MACD金叉（本应做多）
        macd_golden_candidates = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1)) &
            (dataframe['macd'] > dataframe['macd'].shift(1))
        )

        macd_golden_reversal_short = (
            macd_golden_candidates &
            strong_bear_market &

            # 安全过滤条件
            (dataframe['rsi_14'] > 20) &  # 避免极度超卖
            (dataframe['volume_ratio'] > 1.0) &  # 成交量支撑
            noise_free_env &  # 噪音过滤

            # 避免在真正的底部反转时触发
            (dataframe['close'] < dataframe['close'].rolling(10).mean()) &  # 价格仍在短期均线下方

            # 反指信号失效预警过滤（复用现有的）
            (~invalidation_signals.get('macd_reversal_invalid', False)) &
            (price_position > reversal_pos_floor_short)  # 避免在极端低位反指追空
        )

        # 反指信号做空（原来做多改为做空）
        dataframe.loc[macd_golden_reversal_short, 'enter_short'] = 1
        dataframe.loc[macd_golden_reversal_short, 'enter_tag'] = 'MACD_Golden_Reversal_Short'

        # ❌ Signal 8: BB突破向下做空 - 已删除
        # 原因：在超卖位置追空，胜率40%，总亏损-36.797 USDT
        # 根据Bollinger Bands最佳实践（2024）：BB在横盘市场易产生whipsaw
        # 在超卖区做空违背"低买高卖"原则，容易遇到技术性反弹
        # 决策：删除此信号，专注于高质量的反转和趋势跟随信号

        # 🎯 Signal 9: RSI熊市背离做空（新增）
        # === 价格创新高但RSI创新低 - 经典反转信号 ===
        lookback_divergence = 14  # 背离检测回溯周期

        # 检测价格创新高（添加 fillna 避免 NaN 导致的对齐问题）
        price_new_high = (
            dataframe['high'] >= dataframe['high'].rolling(lookback_divergence).max().shift(1).fillna(dataframe['high'])
        ).astype(bool)

        # 检测RSI创新低（或未能创新高）
        rsi_lower_high = (
            dataframe['rsi_14'] < dataframe['rsi_14'].rolling(lookback_divergence).max().shift(1).fillna(100)
        ).astype(bool)

        # 熊市背离确认
        rsi_bearish_divergence_short = (
            price_new_high &
            rsi_lower_high &

            # RSI必须在超买区或接近超买区
            (dataframe['rsi_14'] > 55) &
            (dataframe['rsi_14'] < 85) &

            # 价格在相对高位
            (price_position > 0.50) &

            # 趋势条件：上升趋势中的背离最有效
            (dataframe['ema_8'] > dataframe['ema_21']) &

            # 成交量确认
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.8) &

            # 环境过滤
            short_env_relaxed &
            not_at_bottom &
            noise_free_env
        )

        dataframe.loc[rsi_bearish_divergence_short, 'enter_short'] = 1
        dataframe.loc[rsi_bearish_divergence_short, 'enter_tag'] = 'RSI_Bearish_Divergence_Short'

        # 🎯 Signal 10: 成交量背离做空（新增）
        # === 价格上涨但成交量递减 - 上涨乏力信号 ===
        volume_lookback = 5

        # 价格连续上涨
        price_rising = (
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &
            (dataframe['close'] > dataframe['close'].shift(volume_lookback) * 1.01)  # 5天涨幅>1%
        )

        # 成交量递减
        volume_declining = (
            (dataframe['volume'] < dataframe['volume'].shift(1)) &
            (dataframe['volume'].shift(1) < dataframe['volume'].shift(2)) &
            (dataframe['volume'] < dataframe['volume'].rolling(volume_lookback).mean() * 0.85)  # 成交量低于5日均量85%
        )

        # 成交量背离做空
        volume_divergence_short = (
            price_rising &
            volume_declining &

            # 价格必须在相对高位
            (price_position > 0.55) &
            (price_position < 0.90) &

            # RSI确认超买
            (dataframe['rsi_14'] > 60) &
            (dataframe['rsi_14'] < 80) &

            # 趋势确认：上升趋势末期
            (dataframe['ema_8'] > dataframe['ema_21']) &
            (dataframe['adx'] > 15) &  # 有一定趋势强度

            # 环境过滤
            short_env_relaxed &
            not_at_bottom
        )

        dataframe.loc[volume_divergence_short, 'enter_short'] = 1
        dataframe.loc[volume_divergence_short, 'enter_tag'] = 'Volume_Divergence_Short'


        
        # ==============================
        # 🚨 新增：智能仓位权重系统 - 基于信号质量动态调整
        # ==============================
        
        # 1. 信号质量评分系统
        dataframe['signal_quality_score'] = self._calculate_signal_quality(dataframe)
        dataframe['position_weight'] = self._calculate_position_weight(dataframe)
        dataframe['leverage_multiplier'] = self._calculate_leverage_multiplier(dataframe)

        # --- 入场信心评分（多指标融合）---
        series_len = len(dataframe)
        momentum_series = self._safe_series(dataframe.get('momentum_score', 0.0), series_len, 0.0).astype(float)
        volume_series = self._safe_series(dataframe.get('volume_ratio', 1.0), series_len, 1.0).astype(float)
        trend_series = self._safe_series(dataframe.get('trend_strength', 0.0), series_len, 0.0).astype(float)
        mtf_trend_series = self._safe_series(dataframe.get('mtf_trend_score', 0.0), series_len, 0.0).astype(float)
        price_acc_series = self._safe_series(dataframe.get('price_acceleration', 0.0), series_len, 0.0).astype(float)
        ob_buy_series = self._safe_series(dataframe.get('orderbook_buy_pressure', 0.5), series_len, 0.5).astype(float).clip(0, 1)
        ob_sell_series = self._safe_series(dataframe.get('orderbook_sell_pressure', 0.5), series_len, 0.5).astype(float).clip(0, 1)
        ob_liquidity_series = self._safe_series(dataframe.get('orderbook_liquidity', 0.4), series_len, 0.4).astype(float).clip(0, 1)

        norm_momentum_long = ((momentum_series.clip(-0.6, 0.6) + 0.6) / 1.2).clip(0, 1)
        norm_momentum_short = ((-momentum_series.clip(-0.6, 0.6) + 0.6) / 1.2).clip(0, 1)

        norm_volume = ((volume_series - 0.6) / 1.4).clip(0, 1)

        norm_trend_long = ((trend_series - (-20)) / (80 - (-20))).clip(0, 1)
        norm_trend_short = ((-trend_series - (-20)) / (80 - (-20))).clip(0, 1)

        norm_mtf_long = ((mtf_trend_series - (-0.3)) / (1.0 - (-0.3))).clip(0, 1)
        norm_mtf_short = ((-mtf_trend_series - (-0.3)) / (1.0 - (-0.3))).clip(0, 1)

        norm_price_long = ((price_acc_series - (-0.02)) / (0.06 - (-0.02))).clip(0, 1)
        norm_price_short = ((-price_acc_series - (-0.02)) / (0.06 - (-0.02))).clip(0, 1)

        entry_confidence_long = (
            0.28 * norm_momentum_long +
            0.18 * norm_volume +
            0.18 * norm_trend_long +
            0.16 * norm_mtf_long +
            0.10 * norm_price_long +
            0.10 * ob_buy_series +
            0.05 * ob_liquidity_series
        ).clip(0, 1)

        entry_confidence_short = (
            0.28 * norm_momentum_short +
            0.18 * norm_volume +
            0.18 * norm_trend_short +
            0.16 * norm_mtf_short +
            0.10 * norm_price_short +
            0.10 * ob_sell_series +
            0.05 * ob_liquidity_series
        ).clip(0, 1)

        # 根据交易风格动态调整阈值
        style = getattr(self.style_manager, 'current_style', 'stable') if hasattr(self, 'style_manager') else 'stable'
        long_threshold = float(self.entry_confidence_threshold_long)
        short_threshold = float(self.entry_confidence_threshold_short)

        if style == 'aggressive':
            long_threshold -= 0.07
            short_threshold -= 0.07
        elif style == 'sideways':
            long_threshold += 0.05
            short_threshold += 0.05

        long_threshold = float(np.clip(long_threshold, 0.4, 0.8))
        short_threshold = float(np.clip(short_threshold, 0.4, 0.8))

        dataframe['entry_confidence_long'] = entry_confidence_long
        dataframe['entry_confidence_short'] = entry_confidence_short

        long_conf_mask = entry_confidence_long >= long_threshold
        short_conf_mask = entry_confidence_short >= short_threshold

        dropped_long = (dataframe['enter_long'] == 1) & (~long_conf_mask)
        dropped_short = (dataframe['enter_short'] == 1) & (~short_conf_mask)

        if dropped_long.any():
            dataframe.loc[dropped_long, 'enter_long'] = 0
            dataframe.loc[dropped_long, 'enter_tag'] = ''
        if dropped_short.any():
            dataframe.loc[dropped_short, 'enter_short'] = 0
            dataframe.loc[dropped_short, 'enter_tag'] = ''

        if (dropped_long.any() or dropped_short.any()) and self.event_log:
            self.event_log.info(
                "entry_confidence_filter",
                pair=metadata['pair'],
                long_filtered=int(dropped_long.sum()),
                short_filtered=int(dropped_short.sum()),
                long_threshold=f"{long_threshold:.2f}",
                short_threshold=f"{short_threshold:.2f}"
            )

        # 统计各类信号数量
        total_long_signals = dataframe['enter_long'].sum()
        total_short_signals = dataframe['enter_short'].sum()
        
        # 统计环境条件激活率
        env_basic_rate = basic_env.sum() / len(dataframe) * 100
        env_long_rate = long_favourable_environment.sum() / len(dataframe) * 100  
        env_short_rate = short_favourable_environment.sum() / len(dataframe) * 100
        
        # 检测是否有信号被激活
        if total_long_signals > 0 or total_short_signals > 0:
            self.event_log.info(
                "signal_activity",
                pair=metadata['pair'],
                long_signals=int(total_long_signals),
                short_signals=int(total_short_signals),
                total=int(total_long_signals + total_short_signals),
                env_basic=f"{env_basic_rate:.1f}%",
                env_long=f"{env_long_rate:.1f}%",
                env_short=f"{env_short_rate:.1f}%",
            )

        # 如果没有信号，报告详细诊断
        if total_long_signals == 0 and total_short_signals == 0 and self.enable_signal_inactive_logging:
            now = datetime.now(timezone.utc)
            last_log = self._last_signal_inactive_log.get(pair)
            if last_log is None or (now - last_log).total_seconds() >= self.signal_inactive_log_interval:
                trend_strength_value = dataframe['trend_strength'].iloc[-1] if 'trend_strength' in dataframe.columns else 0.0
                self.event_log.warning(
                    "signal_inactive",
                    pair=metadata['pair'],
                    env_basic_block=f"{100 - env_basic_rate:.1f}%",
                    env_long_block=f"{100 - env_long_rate:.1f}%",
                    env_short_block=f"{100 - env_short_rate:.1f}%",
                    rsi=f"{dataframe['rsi_14'].iloc[-1]:.1f}" if 'rsi_14' in dataframe.columns else "n/a",
                    trend_strength=f"{trend_strength_value:.1f}",
                )
                self._last_signal_inactive_log[pair] = now
        
        return dataframe
    
    
    def _log_entry_signal(self, pair: str, side: str, candle: pd.Series) -> None:
        """输出精简的入场信号日志。"""

        try:
            entry_tag = candle.get('enter_tag', 'UNKNOWN')
            signal_strength = candle.get('signal_strength')
            trend_strength = candle.get('trend_strength')
            adx_value = candle.get('adx')
            volume_ratio = candle.get('volume_ratio')
            atr_percent = candle.get('atr_p')

            risk_score = 0
            if isinstance(adx_value, (int, float, np.integer, np.floating)):
                adx_float = float(adx_value)
                if adx_float >= 25:
                    risk_score += 1
                elif adx_float < 15:
                    risk_score -= 1

            if isinstance(volume_ratio, (int, float, np.integer, np.floating)):
                volume_float = float(volume_ratio)
                if volume_float >= 1.2:
                    risk_score += 1
                elif volume_float < 0.8:
                    risk_score -= 1

            if isinstance(atr_percent, (int, float, np.integer, np.floating)):
                atr_float = float(atr_percent)
                if atr_float <= 0.02:
                    risk_score += 1
                elif atr_float > 0.05:
                    risk_score -= 1

            risk_level = 'low' if risk_score >= 2 else 'medium' if risk_score >= 0 else 'high'

            try:
                max_risk_pct = float(self.max_risk_per_trade) * 100
            except Exception:
                max_risk_pct = None

            self.event_log.info(
                "entry_signal",
                importance="summary",
                pair=pair,
                side=side,
                tag=entry_tag,
                strength=f"{float(signal_strength):.2f}" if isinstance(signal_strength, (int, float, np.integer, np.floating)) else None,
                trend=f"{float(trend_strength):.0f}" if isinstance(trend_strength, (int, float, np.integer, np.floating)) else None,
                adx=f"{float(adx_value):.1f}" if isinstance(adx_value, (int, float, np.integer, np.floating)) else None,
                volume=f"{float(volume_ratio):.2f}" if isinstance(volume_ratio, (int, float, np.integer, np.floating)) else None,
                atr=f"{float(atr_percent)*100:.2f}%" if isinstance(atr_percent, (int, float, np.integer, np.floating)) else None,
                risk_level=risk_level,
                risk_budget=f"{max_risk_pct:.1f}%" if max_risk_pct is not None else None,
            )
        except Exception as exc:
            logger.debug(f"记录入场信号失败 {pair}: {exc}")

    def _log_enhanced_entry_decision(self, pair: str, dataframe: DataFrame, current_data, direction: str):
        """兼容旧调用，记录多头入场。"""
        side = 'long' if direction and direction.upper().startswith('L') else 'short'
        self._log_entry_signal(pair, side, current_data)

    def _log_short_entry_decision(self, pair: str, dataframe: DataFrame, current_data):
        """兼容旧调用，记录空头入场。"""
        self._log_entry_signal(pair, 'short', current_data)
    
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
            history = getattr(self, 'trade_history', [])
            if not history:
                return []

            window = max(1, int(window_size)) if window_size else len(history)
            return history[-window:]
        except Exception as exc:
            logger.warning(f"获取交易历史失败: {exc}")
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
        default_state = {
            'trend_strength': 0.0,
            'volatility_level': 0.0,
            'market_state': getattr(self.style_manager, 'current_style', 'stable') if hasattr(self, 'style_manager') else 'unknown',
            'regime_stability': 0.0
        }

        try:
            if not hasattr(self, 'dp') or self.dp is None:
                return default_state

            pairs = self.dp.current_whitelist()
            if not pairs:
                return default_state

            trend_values: List[float] = []
            volatility_values: List[float] = []
            regime_votes: List[str] = []

            for pair in pairs:
                dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
                if dataframe.empty:
                    continue

                current_row = dataframe.iloc[-1]
                trend = current_row.get('trend_strength')
                if isinstance(trend, (int, float, np.integer, np.floating)):
                    trend_values.append(float(trend))

                atr_p = current_row.get('atr_p')
                if isinstance(atr_p, (int, float, np.integer, np.floating)):
                    volatility_values.append(float(atr_p))

                if hasattr(self, 'style_manager'):
                    try:
                        vote = self.style_manager.classify_market_regime(dataframe)
                        regime_votes.append(vote)
                    except Exception:
                        continue

            if not trend_values and not regime_votes:
                return default_state

            avg_trend = float(np.mean(trend_values)) if trend_values else 0.0
            avg_volatility = float(np.mean(volatility_values)) if volatility_values else 0.0

            if regime_votes:
                vote_counter = Counter(regime_votes)
                dominant_regime, dominant_count = vote_counter.most_common(1)[0]
                stability = dominant_count / len(regime_votes)
            else:
                dominant_regime = default_state['market_state']
                stability = 0.0

            monitoring_state = {
                'trend_strength': avg_trend,
                'volatility_level': avg_volatility,
                'market_state': dominant_regime,
                'regime_stability': stability
            }

            # 维护历史记录供后续分析
            if hasattr(self, 'market_regime_history'):
                self.market_regime_history.append(dominant_regime)
                if len(self.market_regime_history) > 200:
                    self.market_regime_history = self.market_regime_history[-200:]

            if hasattr(self, 'volatility_regime_history'):
                self.volatility_regime_history.append(avg_volatility)
                if len(self.volatility_regime_history) > 200:
                    self.volatility_regime_history = self.volatility_regime_history[-200:]

            return monitoring_state

        except Exception as exc:
            logger.debug(f"市场状态监控失败: {exc}")
            return default_state
    
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
                # 注意：leverage_multiplier 是 @property，不能直接赋值
                # 应该通过 style_manager 调整杠杆范围
                # 这里仅记录需要降低杠杆的意图
                adjustments.append('should_reduce_leverage')
                logger.info(f"检测到高波动率 {metrics['volatility']:.2%}，建议降低杠杆")
            
            # 基于回撤的调整
            if metrics['max_drawdown'] < -0.1:
                # 启用更严格的风险管理
                # Note: drawdown_protection is now a HYPEROPT parameter, cannot modify in place
                drawdown_protection_adjusted = self.drawdown_protection * 0.8
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
            from freqtrade.exchange import timeframe_to_prev_date
            from pandas import Timestamp
            
            entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            
            # 确保索引是datetime类型
            if hasattr(dataframe.index, 'to_pydatetime'):
                # 将entry_date转换为Timestamp以便比较
                entry_timestamp = Timestamp(entry_date)
                entry_candles = dataframe[dataframe.index <= entry_timestamp]
            else:
                # 如果索引不是datetime，使用位置索引
                entry_candles = dataframe.tail(20)  # 获取最近20根K线作为备选
            
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
    
    def _calculate_atr_multiplier(self, entry_atr_p: float, current_candle: dict, enter_tag: str, leverage: int = None) -> float:
        """
        计算ATR倍数 - 核心参数，决定止损给予的波动空间
        基于杠杆、信号类型和市场环境动态调整
        """
        # === 0. 杠杆基础倍数（最重要的调整）===
        if leverage:
            if leverage <= 3:
                base_multiplier = 3.0    # 低杠杆：宽松止损
            elif leverage <= 6:
                base_multiplier = 2.0    # 中低杠杆：标准止损
            elif leverage <= 10:
                base_multiplier = 1.5    # 中杠杆：收紧止损
            elif leverage <= 15:
                base_multiplier = 1.0    # 高杠杆：严格止损
            else:  # 16-20x
                base_multiplier = 0.7    # 极高杠杆：超严格止损
        else:
            # 默认值（无杠杆信息时）
            base_multiplier = 2.8
        
        # === 1. 信号类型调整（在杠杆基础上微调）===
        signal_adjustments = {
            'RSI_Trend_Confirmation': 0.9,    # RSI信号相对可靠，可稍微收紧
            'RSI_Overbought_Fall': 0.9,    
            'MACD_Bearish': 1.1,           # MACD信号容易假突破，需要放宽
            'MACD_Bullish': 1.1,
            'EMA_Golden_Cross': 1.0,       # 趋势信号，标准调整
            'EMA_Death_Cross': 1.0,
        }
        
        signal_factor = signal_adjustments.get(enter_tag, 1.0)
        multiplier = base_multiplier * signal_factor
        
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
        if not self.enable_profit_protection:
            return None  # 显式关闭盈利保护，避免形成伪跟踪止损

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
        
        self._log_message(
            f"🛡️ ATR止损 {pair} [{trade.enter_tag}]: "
            f"盈利{current_profit:.1%} | "
            f"持仓{hours_held:.1f}h | "
            f"开仓ATR{entry_atr_p:.3f} | "
            f"ATR倍数{base_atr_multiplier:.1f} | "
            f"时间衰减{time_decay_factor:.2f} | "
            f"趋势调整{trend_adjustment:.2f} | "
            f"最终止损{final_stoploss:.3f}",
            importance="summary"
        )

    def _assess_trend_state(self, current_candle, is_long: bool = True) -> str:
        """
        🎯 趋势状态判断 - 用于动态调整止损策略

        基于 ADX + EMA_50 + DI 综合判断当前趋势强度和方向

        返回值：
        - STRONG_UPTREND/STRONG_DOWNTREND: 强趋势（ADX>30，价格远离EMA_50，DI方向明确）
        - MODERATE_UPTREND/MODERATE_DOWNTREND: 中等趋势（ADX>25，价格在EMA_50上方）
        - CHOPPY: 震荡市场（ADX<25），容易whipsaw
        - TREND_BROKEN: 趋势破坏（价格跌破EMA_50）

        用途：
        - 强趋势 → 给最大止损空间（3.5 ATR），延长确认时间（60分钟）
        - 中等趋势 → 标准空间（3.0 ATR），标准确认（30分钟）
        - 震荡市场 → 收紧空间（2.5 ATR），快速确认（15分钟）
        - 趋势破坏 → 立即退出（0分钟确认）
        """
        adx = current_candle.get('adx', 20)
        price = current_candle['close']
        ema_50 = current_candle.get('ema_50', price)
        plus_di = current_candle.get('plus_di', 20)
        minus_di = current_candle.get('minus_di', 20)

        if is_long:
            # 多头趋势判断
            if adx > 30 and price > ema_50 * 1.002 and plus_di > minus_di:
                return 'STRONG_UPTREND'  # 强上升趋势：ADX强劲 + 价格远离EMA + DI确认
            elif adx > 25 and price > ema_50:
                return 'MODERATE_UPTREND'  # 中等上升趋势：ADX中等 + 价格在EMA上方
            elif adx < 25:
                return 'CHOPPY'  # 震荡市场：ADX弱，无明确趋势
            elif price < ema_50 * 0.998:
                return 'TREND_BROKEN'  # 趋势破坏：价格跌破EMA_50
        else:
            # 空头趋势判断
            if adx > 30 and price < ema_50 * 0.998 and minus_di > plus_di:
                return 'STRONG_DOWNTREND'  # 强下降趋势
            elif adx > 25 and price < ema_50:
                return 'MODERATE_DOWNTREND'  # 中等下降趋势
            elif adx < 25:
                return 'CHOPPY'  # 震荡市场
            elif price > ema_50 * 1.002:
                return 'TREND_BROKEN'  # 趋势破坏：价格突破EMA_50

        return 'UNCERTAIN'  # 不确定状态

    def _calculate_smart_trailing_stop(
        self,
        trade: Trade,
        current_profit: float,
        entry_confidence: float,
        dataframe: DataFrame
    ) -> Optional[float]:
        """
        🎯 基于信心的智能跟踪止损计算

        Returns:
            None: 不触发跟踪止损
            float: 跟踪止损比例（负值）
        """
        pair = trade.pair
        trade_key = f"{pair}_{trade.id}"

        try:
            # === 1. 确定信心等级和参数 ===
            if entry_confidence <= self.confidence_threshold_low:
                confidence_level = "low"
                activation_threshold = self.trailing_activation_low_confidence
                distance_multiplier = self.trailing_distance_low_confidence
            elif entry_confidence <= self.confidence_threshold_dca:
                confidence_level = "mid"
                activation_threshold = self.trailing_activation_mid_confidence
                distance_multiplier = self.trailing_distance_mid_confidence
            else:
                confidence_level = "high"
                activation_threshold = self.trailing_activation_high_confidence
                distance_multiplier = self.trailing_distance_high_confidence

            # === 2. 检查是否达到激活条件 ===
            if current_profit < activation_threshold:
                return None  # 未达到激活利润

            # === 3. 计算多因子动态跟踪距离 ===
            # 3.1 获取风险因子
            current_leverage = getattr(self, '_current_leverage', {}).get(pair, 10)
            current_atr = dataframe['atr_p'].iloc[-1] if 'atr_p' in dataframe.columns and len(dataframe) > 0 else 0.02

            risk_factors = self.calculate_unified_risk_factors(
                pair=pair,
                dataframe=dataframe,
                leverage=current_leverage,
                current_atr=current_atr
            )

            # 3.2 币种基础距离
            base_distances = {
                'BTC': 0.025,  # BTC 2.5%
                'ETH': 0.030,  # ETH 3%
                'SOL': 0.035,  # SOL 3.5%
                'BNB': 0.035,  # BNB 3.5%
                'Others': 0.040  # 其他 4%
            }
            base_distance = base_distances.get(risk_factors['asset_type'], 0.04)

            # 3.3 ATR波动因子（0.7-1.5）
            atr_multiplier = risk_factors.get('atr_factor', 1.0)
            if atr_multiplier > 1.5:  # 高波动
                atr_factor = 1.3
            elif atr_multiplier < 0.7:  # 低波动
                atr_factor = 0.8
            else:
                atr_factor = 1.0

            # 3.4 趋势强度因子（0.8-1.2）
            trend_factor_raw = risk_factors.get('trend_factor', 1.0)
            if trend_factor_raw > 1.2:  # 强趋势给更多空间
                trend_factor = 1.15
            elif trend_factor_raw < 0.8:  # 弱趋势更紧密跟踪
                trend_factor = 0.9
            else:
                trend_factor = 1.0

            # 3.5 杠杆因子（0.8-1.2）
            leverage_factor_raw = risk_factors.get('leverage_factor', 1.0)
            if current_leverage >= 15:  # 高杠杆需要更大空间避免滑点
                leverage_factor = 1.2
            elif current_leverage <= 5:  # 低杠杆可以更紧
                leverage_factor = 0.9
            else:
                leverage_factor = 1.0

            # 3.6 检查是否完成partial_exit
            exits_completed = False
            if hasattr(self, '_profit_exits') and trade_key in self._profit_exits:
                completed_levels = self._profit_exits[trade_key].get('completed_levels', [])
                exits_completed = len(completed_levels) >= 3

            # partial_exit完成后收紧因子
            partial_exit_factor = self.trailing_tighten_after_exits if exits_completed else 1.0

            # 3.7 计算最终跟踪距离
            trailing_distance = (
                base_distance
                * atr_factor
                * trend_factor
                * leverage_factor
                * distance_multiplier
                * partial_exit_factor
            )

            # 限制范围：1.5% - 10%
            trailing_distance = max(0.015, min(0.10, trailing_distance))

            # === 4. 初始化或更新状态 ===
            if trade_key not in self._trailing_stop_state:
                self._trailing_stop_state[trade_key] = {
                    'peak_profit': current_profit,
                    'exits_completed': exits_completed,
                    'last_distance': trailing_distance,
                    'activated': True
                }
                # 记录激活事件
                self._log_trailing_stop_event(
                    "activated", pair,
                    confidence=entry_confidence,
                    current_profit=current_profit,
                    activation=activation_threshold,
                    distance=trailing_distance
                )
            else:
                state = self._trailing_stop_state[trade_key]

                # 更新峰值利润
                if current_profit > state['peak_profit']:
                    old_peak = state['peak_profit']
                    state['peak_profit'] = current_profit
                    self._log_trailing_stop_event(
                        "peak_updated", pair,
                        old_peak=old_peak,
                        new_peak=current_profit
                    )

                # 检测partial_exit完成状态变化
                if exits_completed and not state['exits_completed']:
                    old_distance = state['last_distance']
                    state['exits_completed'] = True
                    state['last_distance'] = trailing_distance
                    self._log_trailing_stop_event(
                        "adjusted", pair,
                        old_distance=old_distance,
                        new_distance=trailing_distance,
                        reason="partial_exit完成"
                    )
                else:
                    state['last_distance'] = trailing_distance

            # === 5. 计算止损价格 ===
            peak_profit = self._trailing_stop_state[trade_key]['peak_profit']
            stop_loss_ratio = peak_profit - trailing_distance

            # === 6. 检查是否触发止损 ===
            if current_profit <= stop_loss_ratio:
                drawdown = peak_profit - current_profit
                self._log_trailing_stop_event(
                    "triggered", pair,
                    peak_profit=peak_profit,
                    current_profit=current_profit,
                    drawdown=drawdown,
                    distance=trailing_distance
                )
                return -abs(stop_loss_ratio)  # 返回负值止损

            return None  # 未触发

        except Exception as e:
            logger.error(f"智能跟踪止损计算失败 {pair}: {e}")
            return None

    def _log_trailing_stop_event(self, event_type: str, pair: str, **kwargs) -> None:
        """
        🔊 记录跟踪止损事件

        Args:
            event_type: 事件类型 (activated, adjusted, peak_updated, triggered)
            pair: 交易对
            **kwargs: 其他参数
        """
        try:
            if event_type == "activated":
                self._log_message(
                    f"🎯 跟踪止损激活 {pair}: "
                    f"信心={kwargs.get('confidence', 0):.2f}, "
                    f"当前利润={kwargs.get('current_profit', 0):.1%}, "
                    f"激活点={kwargs.get('activation', 0):.1%}, "
                    f"跟踪距离={kwargs.get('distance', 0):.1%}",
                    importance="summary"
                )
            elif event_type == "adjusted":
                self._log_message(
                    f"🔧 跟踪距离调整 {pair}: "
                    f"{kwargs.get('old_distance', 0):.1%} → {kwargs.get('new_distance', 0):.1%}, "
                    f"原因={kwargs.get('reason', 'unknown')}",
                    importance="verbose"
                )
            elif event_type == "peak_updated":
                self._log_message(
                    f"📈 峰值利润更新 {pair}: "
                    f"{kwargs.get('old_peak', 0):.1%} → {kwargs.get('new_peak', 0):.1%}",
                    importance="verbose"
                )
            elif event_type == "triggered":
                self._log_message(
                    f"⛔ 跟踪止损触发 {pair}: "
                    f"峰值={kwargs.get('peak_profit', 0):.1%}, "
                    f"当前={kwargs.get('current_profit', 0):.1%}, "
                    f"回撤={kwargs.get('drawdown', 0):.1%}, "
                    f"距离={kwargs.get('distance', 0):.1%}",
                    importance="summary"
                )
        except Exception as e:
            logger.error(f"记录跟踪止损事件失败 {pair}: {e}")

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float,
                       after_fill: bool = False, **kwargs) -> Optional[float]:
        """
        🎯 智能跟踪止损系统

        核心逻辑：
        - 仅在盈利时启用（符合NoStoploss哲学）
        - 基于入场信心三级分类（低/中/高）
        - 多因子动态距离计算（币种+ATR+趋势+杠杆+信心+partial_exit）
        - 与profit_protection协作保护利润
        """
        try:
            # === 1. 功能开关检查 ===
            if not self.enable_trailing_stop:
                return None

            # === 2. 仅在盈利时启用 ===
            if self.trailing_only_in_profit and current_profit <= 0:
                # 如果当前亏损，清除跟踪状态（防止从盈利回撤到亏损时仍被跟踪止损）
                trade_key = f"{pair}_{trade.id}"
                if hasattr(self, '_trailing_stop_state') and trade_key in self._trailing_stop_state:
                    del self._trailing_stop_state[trade_key]
                    self._log_trailing_stop_event("deactivated", pair,
                                                  reason="profit_negative",
                                                  current_profit=f"{current_profit:.2%}")
                return None

            # === 3. 获取dataframe ===
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return None

            # === 4. 获取入场信心 ===
            entry_confidence = self._get_entry_confidence(trade)

            # === 5. 计算智能跟踪止损 ===
            trailing_stop = self._calculate_smart_trailing_stop(
                trade, current_profit, entry_confidence, dataframe
            )

            # === 6. 与profit_protection配合 ===
            if self.enable_profit_protection and trailing_stop is not None:
                trade_key = f"{pair}_{trade.id}"

                # 检查是否有profit_protection触发
                if hasattr(self, '_profit_protection') and trade_key in self._profit_protection:
                    # 检查是否完成所有3批止盈
                    if hasattr(self, '_profit_exits') and trade_key in self._profit_exits:
                        completed_levels = self._profit_exits[trade_key].get('completed_levels', [])
                        if len(completed_levels) >= 3:
                            # 计算profit_protection止损
                            peak_profit = self._profit_protection[trade_key]['peak_profit']
                            drawdown_threshold = peak_profit * self.profit_drawdown_threshold

                            if current_profit < drawdown_threshold:
                                # profit_protection触发
                                exited_ratios = self._profit_exits[trade_key].get('exited_ratios', [])
                                remaining_ratio = 1.0 - sum(exited_ratios)
                                profit_protection_stop = -(1.0 - remaining_ratio)  # 计算止损比例

                                # 根据模式选择
                                if self.trailing_mode == "cooperative":
                                    # 取更宽松的（对用户更有利）
                                    return max(trailing_stop, profit_protection_stop) if profit_protection_stop is not None else trailing_stop
                                else:  # aggressive
                                    # 取更严格的
                                    return min(trailing_stop, profit_protection_stop) if profit_protection_stop is not None else trailing_stop

            return trailing_stop

        except Exception as e:
            logger.error(f"智能跟踪止损错误 {pair}: {e}")
            return None
    
    def _calculate_reversal_signal_stoploss(self, enter_tag: str, signal_quality_grade: str, 
                                          current_leverage: int, current_atr: float, 
                                          hours_held: float, current_profit: float, 
                                          current_candle: pd.Series, pair: str, 
                                          dataframe: DataFrame = None) -> Optional[float]:
        """
        🔄 反指信号专用止损优化策略
        基于ChatGPT建议的反指信号风险管理原则
        """
        try:
            # === 1. 反指信号基础风险参数 ===
            # 反指信号风险更高，需要更紧的止损
            base_risk_multiplier = {
                'high_confidence': 0.7,      # 即使高质量也要比普通信号更紧
                'medium_confidence': 0.5,    # 中等质量大幅收紧
                'low_confidence': 0.3,       # 低质量极度保守
                'very_low_confidence': 0.2   # 超低质量几乎不容忍风险
            }
            
            # === 2. 时间衰减加速机制 ===
            # 反指信号如果短期内不见效，应该快速止损
            if hours_held <= 1:
                time_factor = 1.0          # 第一小时正常
            elif hours_held <= 3:
                time_factor = 0.8          # 1-3小时收紧20%
            elif hours_held <= 6:
                time_factor = 0.6          # 3-6小时收紧40%
            else:
                time_factor = 0.4          # 6小时后收紧60%
            
            # === 3. 信号类型特殊处理 ===
            signal_specific_factor = 1.0
            
            if 'MACD_Bearish_Reversal' in enter_tag:
                # MACD反指信号：需要确认趋势反转
                macd = current_candle.get('macd', 0)
                macd_signal = current_candle.get('macd_signal', 0)
                
                if macd > macd_signal:
                    signal_specific_factor = 0.8  # MACD金叉确认，稍微放宽
                else:
                    signal_specific_factor = 1.2  # MACD未确认，收紧止损
                    
            elif 'BB_Fake_Rejection_Breakout' in enter_tag:
                # 布林带假反压真突破：需要成交量确认
                volume_ratio = current_candle.get('volume_ratio', 1.0)
                bb_upper = current_candle.get('bb_upperband', 0)
                close = current_candle.get('close', 0)
                
                if volume_ratio > 1.5 and close > bb_upper:
                    signal_specific_factor = 0.9  # 成交量确认且价格突破，稍微放宽
                else:
                    signal_specific_factor = 1.3  # 缺乏确认，严格止损
            
            # === 4. 盈亏状态调整 ===
            profit_factor = 1.0
            if current_profit > 0.02:       # 盈利超过2%
                profit_factor = 1.2         # 可以稍微放宽止损
            elif current_profit > 0:       # 小幅盈利
                profit_factor = 1.0         # 维持正常止损
            elif current_profit > -0.01:   # 小幅亏损
                profit_factor = 0.9         # 稍微收紧
            else:                          # 较大亏损
                profit_factor = 0.7         # 大幅收紧，快速止损
            
            # === 5. 计算最终反指止损 ===
            base_stoploss = self.calculate_dynamic_stoploss(
                signal_quality_grade, current_leverage, current_atr, pair, dataframe
            )
            
            # 应用所有反指专用调整因子
            reversal_multiplier = (
                base_risk_multiplier[signal_quality_grade] *
                time_factor *
                signal_specific_factor *
                profit_factor
            )
            
            reversal_stoploss_value = base_stoploss * reversal_multiplier
            
            # === 6. 高波动环境额外调整 ===
            volatility_state = current_candle.get('volatility_state', 50)
            if volatility_state >= 75:  # 高波动环境下反指信号更加保守
                volatility_protection = 0.8 if volatility_state >= 90 else 0.9
                reversal_stoploss_value *= volatility_protection
                
            # === 7. 安全边界检查 ===
            # 反指信号最小止损不能低于0.5%，最大不超过4%
            reversal_stoploss_value = max(0.005, min(0.04, reversal_stoploss_value))
            
            final_stoploss = -reversal_stoploss_value  # 做多信号，负值止损
            
            # === 7. 详细日志记录 ===
            self._log_message(
                f"反指止损 {pair}: {enter_tag}({signal_quality_grade}) "
                f"基础={base_stoploss:.3f} 时间因子={time_factor:.2f} "
                f"信号因子={signal_specific_factor:.2f} 盈亏因子={profit_factor:.2f} "
                f"最终={final_stoploss:.3f}",
                importance="verbose"
            )
            
            return final_stoploss
            
        except Exception as e:
            logger.error(f"反指信号止损计算错误 {pair}: {e}")
            # 反指信号紧急止损使用更保守的2%
            return -0.02
    
    def _apply_high_volatility_adjustments(self, base_stoploss: float, current_candle: pd.Series, 
                                         signal_quality_grade: str, hours_held: float, 
                                         current_leverage: int, pair: str) -> float:
        """
        🌪️ 高波动环境动态调整机制
        在极端市场条件下自动调整止损策略以保护资金
        """
        try:
            # === 1. 获取波动性指标 ===
            volatility_state = current_candle.get('volatility_state', 50)
            current_atr = current_candle.get('atr_p', 0.02)
            bb_squeeze = current_candle.get('bb_squeeze', 0)
            
            # === 2. 高波动环境识别 ===
            high_volatility_threshold = 75  # 波动状态超过75%视为高波动
            extreme_volatility_threshold = 90  # 超过90%视为极端波动
            
            # ATR异常检测：当前ATR超过20日均值的1.5倍
            recent_atr_mean = current_candle.get('atr_p', 0.02)  # 简化处理
            atr_spike = current_atr > (recent_atr_mean * 1.5)
            
            # === 3. 波动环境分级调整 ===
            volatility_adjustment_factor = 1.0
            adjustment_reason = "正常波动"
            
            if volatility_state >= extreme_volatility_threshold or atr_spike:
                # 极端波动：大幅收紧止损
                adjustment_reason = "极端波动"
                if signal_quality_grade == 'high_confidence':
                    volatility_adjustment_factor = 0.6   # 高质量信号收紧40%
                elif signal_quality_grade == 'medium_confidence':
                    volatility_adjustment_factor = 0.4   # 中等质量信号收紧60%
                else:
                    volatility_adjustment_factor = 0.3   # 低质量信号收紧70%
                    
            elif volatility_state >= high_volatility_threshold:
                # 高波动：适度收紧止损
                adjustment_reason = "高波动"
                if signal_quality_grade == 'high_confidence':
                    volatility_adjustment_factor = 0.8   # 高质量信号收紧20%
                elif signal_quality_grade == 'medium_confidence':
                    volatility_adjustment_factor = 0.7   # 中等质量信号收紧30%
                else:
                    volatility_adjustment_factor = 0.6   # 低质量信号收紧40%
            
            # === 4. 布林带挤压特殊处理 ===
            if bb_squeeze > 0.8:  # 布林带严重挤压
                # 挤压后往往伴随剧烈波动，预防性收紧
                volatility_adjustment_factor *= 0.8
                adjustment_reason += "+布林挤压"
            
            # === 5. 杠杆风险叠加调整 ===
            if current_leverage >= 15 and volatility_state >= high_volatility_threshold:
                # 高杠杆+高波动：额外风险保护
                volatility_adjustment_factor *= 0.9
                adjustment_reason += "+高杠杆风险"
            
            # === 6. 时间维度调整 ===
            if hours_held > 12 and volatility_state >= high_volatility_threshold:
                # 长期持仓在高波动环境中的额外保护
                volatility_adjustment_factor *= 0.9
                adjustment_reason += "+长期持仓"
            
            # === 7. 计算最终调整后止损 ===
            adjusted_stoploss = base_stoploss * volatility_adjustment_factor
            
            # === 8. 边界检查 ===
            # 高波动环境下，最小止损不能低于1%
            min_volatility_stoploss = 0.01 if volatility_state >= high_volatility_threshold else 0.005
            adjusted_stoploss = max(min_volatility_stoploss, min(0.08, adjusted_stoploss))
            
            # === 9. 日志记录 ===
            if adjusted_stoploss != base_stoploss:
                self._log_message(
                    f"高波动调整 {pair}: {adjustment_reason} "
                    f"波动状态={volatility_state:.0f}% ATR={current_atr:.4f} "
                    f"调整因子={volatility_adjustment_factor:.2f} "
                    f"基础={base_stoploss:.3f}→调整后={adjusted_stoploss:.3f}",
                    importance="verbose"
                )
            
            return adjusted_stoploss
            
        except Exception as e:
            logger.error(f"高波动调整计算错误 {pair}: {e}")
            # 出错时返回更保守的止损
            return min(base_stoploss, 0.03)
    
    def _check_24h_time_stoploss(self, trade: Trade, current_time: datetime, hours_held: float, 
                               current_profit: float, signal_quality_grade: str, 
                               enter_tag: str, pair: str) -> Optional[float]:
        """
        ⏰ 24小时时间止损监控系统
        确保持仓不会无限期套牢，根据时间和盈亏状况触发强制平仓
        """
        try:
            # === 1. 时间阈值定义 ===
            # 不同信号质量的最大持仓时间
            max_hold_hours = {
                'high_confidence': 36,      # 高质量信号允许持仓36小时
                'medium_confidence': 24,    # 中等质量信号24小时
                'low_confidence': 12,       # 低质量信号12小时
                'very_low_confidence': 6    # 极低质量信号6小时
            }
            
            # 反指信号时间限制更严格
            if 'Reversal' in enter_tag:
                max_hold_hours = {
                    'high_confidence': 24,      # 反指高质量24小时
                    'medium_confidence': 12,    # 反指中等质量12小时
                    'low_confidence': 6,        # 反指低质量6小时
                    'very_low_confidence': 3    # 反指极低质量3小时
                }
            
            signal_max_hours = max_hold_hours.get(signal_quality_grade, 24)
            
            # === 2. 时间阶段检查 ===
            # 不同时间段的处理逻辑
            
            # 2.1 早期阶段（前25%时间）：只有极端亏损才触发时间止损
            early_stage_hours = signal_max_hours * 0.25
            if hours_held <= early_stage_hours:
                if current_profit < -0.05:  # 亏损超过5%立即止损
                    logger.warning(f"早期时间止损 {pair}: 持仓{hours_held:.1f}h, 亏损{current_profit:.2%}")
                    return -0.02  # 立即2%止损
                return None  # 否则不干预
            
            # 2.2 中期阶段（25%-75%时间）：中等亏损开始干预
            mid_stage_hours = signal_max_hours * 0.75
            if hours_held <= mid_stage_hours:
                if current_profit < -0.02:  # 亏损超过2%触发
                    # 根据亏损程度调整止损紧密度
                    if current_profit < -0.04:    # 亏损4%以上紧急止损
                        time_stoploss = -0.015
                    else:                        # 亏损2-4%适度收紧
                        time_stoploss = -0.025
                    
                    logger.warning(f"中期时间止损 {pair}: 持仓{hours_held:.1f}h, 亏损{current_profit:.2%}, 止损{time_stoploss:.3f}")
                    return time_stoploss
                return None
            
            # === 3. 临近最大持仓时间（75%-100%）：强制清理 ===
            if hours_held >= signal_max_hours * 0.75:
                
                # 3.1 盈利处理：降低止损让利润奔跑
                if current_profit > 0.01:  # 盈利超过1%
                    if current_profit > 0.05:    # 大幅盈利，可以容忍一些回撤
                        time_stoploss = -0.02
                    else:                       # 小幅盈利，保护部分利润
                        time_stoploss = -0.015
                        
                    self._log_message(
                        f"盈利时间保护 {pair}: 持仓{hours_held:.1f}h, 盈利{current_profit:.2%}, 保护止损{time_stoploss:.3f}",
                        importance="summary"
                    )
                    return time_stoploss
                
                # 3.2 小幅亏损：渐进收紧
                elif current_profit > -0.02:
                    time_stoploss = -0.01   # 小幅亏损1%止损
                    logger.warning(f"后期小亏损止损 {pair}: 持仓{hours_held:.1f}h, 亏损{current_profit:.2%}")
                    return time_stoploss
                
                # 3.3 较大亏损：立即止损
                else:
                    time_stoploss = -0.005  # 大幅亏损0.5%快速止损
                    logger.warning(f"后期大亏损紧急止损 {pair}: 持仓{hours_held:.1f}h, 亏损{current_profit:.2%}")
                    return time_stoploss
            
            # === 4. 超时强制平仓 ===
            if hours_held >= signal_max_hours:
                # 无论盈亏，超时必须平仓
                if current_profit > 0:
                    time_stoploss = -0.005  # 盈利时给一点空间
                    logger.warning(f"超时盈利平仓 {pair}: 持仓{hours_held:.1f}h, 盈利{current_profit:.2%}")
                else:
                    time_stoploss = 0.001   # 亏损时立即市价平仓
                    logger.warning(f"超时亏损强制平仓 {pair}: 持仓{hours_held:.1f}h, 亏损{current_profit:.2%}")
                
                return time_stoploss
            
            # 没有触发任何时间止损条件
            return None
            
        except Exception as e:
            logger.error(f"24小时时间止损计算错误 {pair}: {e}")
            # 出错时检查是否超过24小时，强制平仓
            if hours_held >= 24:
                return -0.01
            return None
    
    def _check_emergency_circuit_breaker(self, current_profit: float, hours_held: float, 
                                       current_candle: pd.Series, trade: Trade, 
                                       pair: str) -> Optional[float]:
        """
        🚨 紧急熔断机制
        在极端情况下立即强制止损，保护账户免受灾难性损失
        """
        try:
            # === 1. 极端亏损熔断 ===
            # 单笔交易亏损超过10%立即熔断
            if current_profit <= -0.10:
                logger.critical(f"🚨极端亏损熔断 {pair}: 亏损{current_profit:.2%}, 立即止损!")
                return 0.001  # 立即市价止损
            
            # === 2. 快速亏损熔断 ===
            # 短时间内大幅亏损（1小时内亏损超过5%）
            if hours_held <= 1 and current_profit <= -0.05:
                logger.critical(f"🚨快速亏损熔断 {pair}: {hours_held:.1f}小时内亏损{current_profit:.2%}, 立即止损!")
                return 0.001
            
            # === 3. 极端波动熔断 ===
            volatility_state = current_candle.get('volatility_state', 50)
            current_atr = current_candle.get('atr_p', 0.02)
            
            # 波动率超过95%且持仓亏损超过3%
            if volatility_state >= 95 and current_profit <= -0.03:
                logger.critical(f"🚨极端波动熔断 {pair}: 波动{volatility_state:.0f}% 亏损{current_profit:.2%}, 立即止损!")
                return 0.001
            
            # ATR异常飙升（当前ATR超过5%）且持仓亏损
            if current_atr >= 0.05 and current_profit <= -0.02:
                logger.critical(f"🚨ATR异常熔断 {pair}: ATR={current_atr:.2%} 亏损{current_profit:.2%}, 立即止损!")
                return 0.001
            
            # === 4. 连续止损熔断 ===
            # 检查是否有连续多次止损记录（需要从交易历史判断）
            # 这里简化处理：如果亏损超过8%且持仓超过30分钟，视为可能的连续失败
            if current_profit <= -0.08 and hours_held >= 0.5:
                logger.critical(f"🚨大幅亏损熔断 {pair}: 持仓{hours_held:.1f}h 亏损{current_profit:.2%}, 紧急止损!")
                return 0.002  # 0.2%快速止损
            
            # === 5. 市场崩盘熔断 ===
            # 检查多个技术指标同时恶化
            rsi = current_candle.get('rsi_14', 50)
            macd = current_candle.get('macd', 0)
            bb_position = current_candle.get('bb_position', 0.5)
            
            # 极端超卖+MACD深度负值+价格击穿布林带下轨+持仓亏损
            market_crash_conditions = (
                rsi <= 15 and           # 极端超卖
                macd <= -0.02 and       # MACD深度负值
                bb_position <= 0.1 and # 价格远低于布林带下轨
                current_profit <= -0.04 # 亏损超过4%
            )
            
            if market_crash_conditions:
                logger.critical(f"🚨市场崩盘熔断 {pair}: RSI={rsi:.1f} MACD={macd:.4f} 亏损{current_profit:.2%}, 紧急避险!")
                return 0.001
            
            # === 6. 长期套牢熔断 ===
            # 持仓超过48小时且仍有显著亏损
            if hours_held >= 48 and current_profit <= -0.03:
                logger.critical(f"🚨长期套牢熔断 {pair}: 持仓{hours_held:.1f}h 亏损{current_profit:.2%}, 强制清理!")
                return 0.001
            
            # 没有触发任何熔断条件
            return None
            
        except Exception as e:
            logger.error(f"紧急熔断检查错误 {pair}: {e}")
            # 如果熔断系统本身出错且亏损严重，保守止损
            if current_profit <= -0.08:
                return 0.005
            return None
    
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
                if signal_type in ['RSI_Trend_Confirmation']:
                    if rsi < 25:
                        score += 2    # 深度超卖，机会大
                    elif rsi < 30:
                        score += 1.5  # 正常超卖
                elif signal_type in ['RSI_Overbought_Fall']:
                    if rsi > 75:
                        score += 2    # 深度超买，风险大
                    elif rsi > 70:
                        score += 1.5  # 正常超买
                elif signal_type in ['EMA_Death_Cross_Filtered']:  # 🎯 新增EMA死叉评分
                    if rsi > 40:  # RSI在中性区域，适合做空
                        score += 1.5
                    elif rsi > 50:
                        score += 2  # RSI偏高，更适合做空
                elif signal_type in ['EMA_Golden_Cross_Filtered']:  # 🎯 新增EMA金叉评分
                    if rsi < 60:  # RSI在中性区域，适合做多
                        score += 1.5
                    elif rsi < 50:
                        score += 2  # RSI偏低，更适合做多
                
                # === 2. 趋势强度与方向 (0-2分) ===
                adx = current_data.get('adx', 25)
                ema_8 = current_data.get('ema_8', 0)
                ema_21 = current_data.get('ema_21', 0)
                
                if adx > 30:  # 强趋势
                    if signal_type in ['RSI_Trend_Confirmation'] and ema_8 > ema_21:
                        score += 2  # 上升趋势中的超卖，高质量
                    elif signal_type in ['RSI_Overbought_Fall'] and ema_8 < ema_21:
                        score += 2  # 下跌趋势中的超买，高质量
                    elif signal_type in ['EMA_Death_Cross_Filtered'] and ema_8 < ema_21:
                        score += 2  # 🎯 强下跌趋势中的EMA死叉，高质量
                    elif signal_type in ['EMA_Golden_Cross_Filtered'] and ema_8 > ema_21:
                        score += 2  # 🎯 强上升趋势中的EMA金叉，高质量
                    else:
                        score += 0.5  # 逆势信号，质量一般
                elif 20 < adx <= 30:  # 中等趋势
                    if signal_type in ['EMA_Death_Cross_Filtered', 'EMA_Golden_Cross_Filtered']:
                        score += 1.5  # 🎯 EMA信号在中等趋势中表现较好
                    else:
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

                if signal_type in ['RSI_Trend_Confirmation', 'EMA_Golden_Cross_Filtered'] and no_bearish_div:
                    score += 1  # 🎯 做多信号无顶背离
                elif signal_type in ['RSI_Overbought_Fall', 'EMA_Death_Cross_Filtered'] and no_bullish_div:
                    score += 1  # 🎯 做空信号无底背离

                # === 6. 市场环境加分 (0-0.5分) ===
                price_position = current_data.get('price_position', 0.5)
                if signal_type in ['RSI_Trend_Confirmation', 'EMA_Golden_Cross_Filtered'] and 0.2 < price_position < 0.7:
                    score += 0.5  # 🎯 做多信号不在极端位置
                elif signal_type in ['RSI_Overbought_Fall', 'EMA_Death_Cross_Filtered'] and 0.3 < price_position < 0.8:
                    score += 0.5  # 🎯 做空信号不在极端位置
                
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
        基于市场状态给出交易建议（避免逻辑已禁用，所有信号均可使用）
        """
        advice = {
            'recommended_signals': [],
            'avoid_signals': [],  # 🚀 永远为空，所有信号都允许
            'risk_adjustment': 1.0,
            'position_size_multiplier': 1.0
        }
        
        # 基于不同市场状态给出推荐信号（但不再避免任何信号）
        if 'TRENDING_UPTREND' in regime:
            advice['recommended_signals'] = ['RSI_Trend_Confirmation', 'EMA_Golden_Cross']
            advice['position_size_multiplier'] = 1.2
            
        elif 'TRENDING_DOWNTREND' in regime:
            advice['recommended_signals'] = ['RSI_Overbought_Fall', 'MACD_Bearish', 'EMA_Death_Cross']
            advice['position_size_multiplier'] = 1.2
            
        elif 'SIDEWAYS' in regime:
            if volatility_regime == 'LOW':
                advice['recommended_signals'] = ['RSI_Trend_Confirmation', 'RSI_Overbought_Fall', 'BB_Fake_Rejection_Breakout']  # 移除BB_Breakthrough_Follow
            else:
                advice['recommended_signals'] = ['BB_Lower_Bounce', 'BB_Fake_Rejection_Breakout']  # 使用反指信号替代
            advice['position_size_multiplier'] = 0.8
            
        elif 'VOLATILE' in regime:
            advice['recommended_signals'] = ['BB_Lower_Bounce', 'BB_Upper_Rejection']  # 波动市场中布林带表现最好
            advice['risk_adjustment'] = 1.5
            advice['position_size_multiplier'] = 0.6
            
        # 位置调整（不再避免信号，只调整仓位）
        if position_regime in ['NEAR_HIGH']:
            advice['position_size_multiplier'] *= 0.8
        elif position_regime in ['NEAR_LOW']:
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
        
        # 市场状态奖励：推荐信号额外奖励（避免惩罚已禁用）
        regime_bonus = 1.0
        if entry_tag in signals_advice.get('recommended_signals', []):
            regime_bonus = 1.2  # 推荐信号+20%杠杆
        # 🚀 避免逻辑已禁用 - 不再对任何信号进行惩罚
        
        return quality_bonus * regime_bonus
    
    def adjust_signal_by_mtf_consensus(self, base_quality: float, mtf_data: dict, signal_direction: str) -> float:
        """
        🎯 基于MTF趋势一致性调整信号质量评分 (核心功能)
        
        根据多时间框架趋势一致性，对信号质量进行动态调整：
        - MTF趋势一致：质量提升 (+20% ~ +50%)
        - MTF趋势冲突：质量降低 (-30% ~ -60%)
        - MTF趋势中性：轻微调整 (±10%)
        
        Args:
            base_quality: 基础信号质量评分 (1-10)
            mtf_data: MTF趋势数据字典 {consensus_direction, consensus_strength, trend_score}
            signal_direction: 信号方向 ('long' | 'short')
            
        Returns:
            float: 调整后的信号质量评分 (0.5-15.0)
        """
        if base_quality <= 0:
            return 0.5  # 最低保护值
            
        # 提取MTF数据
        mtf_direction = mtf_data.get('consensus_direction', 'neutral')
        mtf_strength = mtf_data.get('consensus_strength', 'weak')  
        mtf_score = mtf_data.get('trend_score', 0.0)  # -1到1之间
        
        # 计算趋势一致性系数
        alignment_multiplier = 1.0
        
        # === 主要逻辑：趋势方向一致性检查 ===
        if signal_direction == 'long':
            if mtf_direction == 'bullish':
                # 做多信号与看多趋势一致
                if mtf_strength == 'very_strong':
                    alignment_multiplier = 1.5   # +50% 强烈看多时的做多信号
                elif mtf_strength == 'strong':
                    alignment_multiplier = 1.35  # +35% 
                elif mtf_strength == 'moderate':
                    alignment_multiplier = 1.2   # +20%
                else:  # weak
                    alignment_multiplier = 1.1   # +10%
                    
            elif mtf_direction == 'bearish':
                # 做多信号与看空趋势冲突 (这就是之前的BUG!)
                if mtf_strength == 'very_strong':
                    alignment_multiplier = 0.3   # -70% 强烈看空时禁止做多
                elif mtf_strength == 'strong':
                    alignment_multiplier = 0.4   # -60%
                elif mtf_strength == 'moderate':
                    alignment_multiplier = 0.6   # -40%
                else:  # weak
                    alignment_multiplier = 0.8   # -20%
                    
            else:  # neutral
                # MTF中性，基于trend_score微调
                if mtf_score > 0.1:
                    alignment_multiplier = 1.1   # 轻微偏多
                elif mtf_score < -0.1:
                    alignment_multiplier = 0.9   # 轻微偏空
                # else: 保持1.0
                
        elif signal_direction == 'short':
            if mtf_direction == 'bearish':
                # 做空信号与看空趋势一致
                if mtf_strength == 'very_strong':
                    alignment_multiplier = 1.5   # +50%
                elif mtf_strength == 'strong':
                    alignment_multiplier = 1.35  # +35%
                elif mtf_strength == 'moderate':
                    alignment_multiplier = 1.2   # +20%
                else:  # weak
                    alignment_multiplier = 1.1   # +10%
                    
            elif mtf_direction == 'bullish':
                # 做空信号与看多趋势冲突 (同样的BUG!)
                if mtf_strength == 'very_strong':
                    alignment_multiplier = 0.3   # -70% 强烈看多时禁止做空
                elif mtf_strength == 'strong':
                    alignment_multiplier = 0.4   # -60%
                elif mtf_strength == 'moderate':
                    alignment_multiplier = 0.6   # -40%
                else:  # weak
                    alignment_multiplier = 0.8   # -20%
                    
            else:  # neutral
                # MTF中性，基于trend_score微调
                if mtf_score < -0.1:
                    alignment_multiplier = 1.1   # 轻微偏空
                elif mtf_score > 0.1:
                    alignment_multiplier = 0.9   # 轻微偏多
        
        # 应用调整并限制范围
        adjusted_quality = base_quality * alignment_multiplier
        
        # 最终限制：0.5 - 15.0 (允许超过10分以奖励高一致性信号)
        return max(0.5, min(15.0, adjusted_quality))
    
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
        
        # 信号可靠性映射（优化权重配置）
        signal_reliability = {
            'RSI_Trend_Confirmation': 1.2,                # RSI信号相对可靠
            'RSI_Overbought_Fall': 1.2,
            'EMA_Golden_Cross_Filtered': 1.3,          # EMA金叉信号最可靠
            'EMA_Death_Cross_Filtered': 1.1,           # EMA死叉稍保守（分化处理）
            'MACD_Bearish': 1.0,                       # MACD信号保守
            'MACD_Bullish': 1.0,
            'BB_Lower_Bounce': 1.4,                    # BB信号权重提升至1.4
            'BB_Upper_Rejection': 1.4,                 # BB信号权重提升至1.4
            # 其他EMA相关信号（分化处理）
            'Strong_Bullish_Follow': 1.2,              # 强趋势跟随中等权重
            'Strong_Bearish_Follow': 1.2,
        }
        
        base_multiplier = signal_reliability.get(entry_tag, 1.0)
        
        # 市场推荐奖励（避免惩罚已禁用）
        if entry_tag in signals_advice.get('recommended_signals', []):
            base_multiplier *= 1.1  # 额外+10%
        # 🚀 避免逻辑已禁用 - 不再对任何信号进行惩罚
        
        return base_multiplier
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        🎯 多时间框架退出信号 - 技术面确认版

        设计思路（对标入场信号的多时间框架结构）：
        - 入场：15m (60%) + 1h (40%) 双周期确认
        - 退出：同样使用 15m + 1h，但更敏感（避免"吃不到完整趋势"）

        退出层次：
        1. 强反转：15m + 1h 都确认反转 → 立即退出
        2. 趋势减弱：15m 反转 + 1h 趋势减弱 → 谨慎退出
        3. 极端信号：15m 极端超买/超卖 → 快速响应

        exit_tag 细分（便于回测分析）：
        - mtf_strong_reversal: 双时间框架确认强反转
        - mtf_trend_weak: 趋势减弱（15m反转+1h减弱）
        - 15m_extreme: 15m极端信号（RSI>80或<20）
        """
        if dataframe.empty:
            return dataframe

        # 初始化退出信号
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''

        pair = metadata.get('pair', 'UNKNOWN')

        # === 🚀 新增：快速预警信号检测（逐行计算）===
        # 动量衰减
        macd_falling = (
            (dataframe['macd'] < dataframe['macd'].shift(1)) &
            (dataframe['macd'].shift(1) < dataframe['macd'].shift(2))
        )

        # RSI背离简化版
        rsi_bearish_div = (
            (dataframe['close'] > dataframe['close'].shift(5)) &  # 价格上涨
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(5))   # RSI下降
        )
        rsi_bullish_div = (
            (dataframe['close'] < dataframe['close'].shift(5)) &  # 价格下跌
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(5))   # RSI上升
        )

        # 跌破/突破EMA21
        break_ema21_down = (dataframe['close'].shift(1) > dataframe['ema_21'].shift(1)) & (dataframe['close'] < dataframe['ema_21'])
        break_ema21_up = (dataframe['close'].shift(1) < dataframe['ema_21'].shift(1)) & (dataframe['close'] > dataframe['ema_21'])

        # 反向大成交量
        high_vol_selloff = (dataframe['volume_ratio'] > 1.5) & (dataframe['close'] < dataframe['close'].shift(1))
        high_vol_rally = (dataframe['volume_ratio'] > 1.5) & (dataframe['close'] > dataframe['close'].shift(1))

        # 综合预警评分（多头预警）
        warning_long = (
            macd_falling.astype(int) * 25 +
            rsi_bearish_div.astype(int) * 30 +
            break_ema21_down.astype(int) * 20 +
            high_vol_selloff.astype(int) * 25
        )

        # 综合预警评分（空头预警）
        warning_short = (
            macd_falling.astype(int) * 25 +
            rsi_bullish_div.astype(int) * 30 +
            break_ema21_up.astype(int) * 20 +
            high_vol_rally.astype(int) * 25
        )

        # 预警等级
        has_strong_warning_long = warning_long > 50   # 强预警
        has_moderate_warning_long = warning_long > 30  # 中度预警
        has_strong_warning_short = warning_short > 50
        has_moderate_warning_short = warning_short > 30

        # === 🚀 新增：5m快速反转检测（用于紧急出场）===
        # 5m框架的快速反转信号（不等待1h确认）
        fast_reversal_long = (
            (dataframe['rsi_14'] > 70) & # RSI超买
            (dataframe['close'] < dataframe['ema_21']) &  # 跌破EMA21
            (dataframe['macd'] < dataframe['macd_signal'])  # MACD死叉
        )

        fast_reversal_short = (
            (dataframe['rsi_14'] < 30) &  # RSI超卖
            (dataframe['close'] > dataframe['ema_21']) &  # 突破EMA21
            (dataframe['macd'] > dataframe['macd_signal'])  # MACD金叉
        )

        # === 🚀 新增：强趋势保护过滤器 ===
        # ADX>35 表示强趋势，需要更强的反转信号才出场
        in_strong_trend = dataframe['adx'] > 35

        try:
            # === 获取 1h 时间框架数据 ===
            df_1h = self.dp.get_pair_dataframe(pair, '1h')

            if df_1h.empty or len(df_1h) < 50:
                logger.warning(f"退出信号: {pair} 1h数据不足，仅使用15m")
                use_mtf = False
            else:
                use_mtf = True

                # 计算 1h 时间框架指标（如果不存在）
                # 🔧 修复：使用 pandas_ta (pta) 而不是 talib.abstract
                if 'rsi_14' not in df_1h.columns:
                    df_1h['rsi_14'] = pta.rsi(df_1h['close'], length=14)
                if 'ema_21' not in df_1h.columns:
                    df_1h['ema_21'] = pta.ema(df_1h['close'], length=21)
                if 'ema_50' not in df_1h.columns:
                    df_1h['ema_50'] = pta.ema(df_1h['close'], length=50)
                if 'macd' not in df_1h.columns or 'macd_signal' not in df_1h.columns:
                    macd = pta.macd(df_1h['close'], fast=12, slow=26, signal=9)
                    df_1h['macd'] = macd['MACD_12_26_9']
                    df_1h['macd_signal'] = macd['MACDs_12_26_9']
                if 'adx' not in df_1h.columns:
                    adx_data = pta.adx(df_1h['high'], df_1h['low'], df_1h['close'], length=14)
                    df_1h['adx'] = adx_data['ADX_14']

                # 获取 1h 最新数据
                h1_current = df_1h.iloc[-1]
                h1_prev = df_1h.iloc[-2] if len(df_1h) > 1 else h1_current

                # === 1h 趋势分析 ===
                h1_close = h1_current['close']
                h1_ema_21 = h1_current['ema_21']
                h1_ema_50 = h1_current['ema_50']
                h1_macd = h1_current['macd']
                h1_macd_signal = h1_current['macd_signal']
                h1_adx = h1_current['adx']
                h1_adx_prev = h1_prev['adx']

                # 1h 多头趋势反转：价格跌破EMA21 + MACD死叉
                h1_bullish_reversal = (
                    (h1_close < h1_ema_21) &  # 跌破短期均线
                    (h1_macd < h1_macd_signal)  # MACD死叉
                )

                # 1h 空头趋势反转：价格突破EMA21 + MACD金叉
                h1_bearish_reversal = (
                    (h1_close > h1_ema_21) &
                    (h1_macd > h1_macd_signal)
                )

                # 1h 趋势减弱：ADX连续下降
                h1_adx_prev2 = df_1h.iloc[-3]['adx'] if len(df_1h) > 2 else h1_adx_prev
                h1_trend_weakening = (h1_adx < h1_adx_prev) & (h1_adx_prev < h1_adx_prev2)

        except Exception as e:
            logger.warning(f"退出信号: {pair} 获取1h数据失败: {e}，仅使用15m")
            use_mtf = False
            h1_bullish_reversal = False
            h1_bearish_reversal = False
            h1_trend_weakening = False

        # === 15m 时间框架分析（主时间框架）===

        # 0. 趋势前置条件（避免横盘市场误触发）
        # 做多退出：要求之前有上涨趋势（最近20根K线RSI曾>55 或 价格显著高于EMA50）
        rsi_20_max = dataframe['rsi_14'].rolling(20).max()
        price_above_ema50_pct = ((dataframe['close'] - dataframe['ema_50']) / dataframe['ema_50'] * 100).rolling(20).max()
        long_trend_existed = (rsi_20_max > 55) | (price_above_ema50_pct > 2)  # RSI曾>55 或 曾高于EMA50超过2%

        # 做空退出：要求之前有下跌趋势（最近20根K线RSI曾<45 或 价格显著低于EMA50）
        rsi_20_min = dataframe['rsi_14'].rolling(20).min()
        price_below_ema50_pct = ((dataframe['ema_50'] - dataframe['close']) / dataframe['ema_50'] * 100).rolling(20).max()
        short_trend_existed = (rsi_20_min < 45) | (price_below_ema50_pct > 2)  # RSI曾<45 或 曾低于EMA50超过2%

        # === 横盘市场检测（2024最新研究 - 三重确认）===
        # 1. ADX<20: 无明确趋势
        # 2. BB宽度<0.04: 价格压缩（调整后的阈值，根据实际测试优化）
        # 3. NATR<2.0%: 低波动率
        is_sideways = (
            (dataframe['adx'] < 20) &  # ADX<20表示无趋势
            (dataframe['bb_width'] < 0.04) &  # BB宽度<0.04表示价格压缩（实际测试后调整）
            (dataframe['natr'] < 2.0)  # NATR<2.0%表示低波动
        )

        # === 突破确认（Donchian Channel + 量能）===
        # 只在价格突破关键位置 + 大量能时才允许退出信号

        # 多头退出：价格跌破20日低点（向下突破）
        price_breakdown_low = dataframe['close'] < dataframe['donchian_low_20'].shift(1)

        # 空头退出：价格突破20日高点（向上突破）
        price_breakout_high = dataframe['close'] > dataframe['donchian_high_20'].shift(1)

        # 量能激增确认（1.8倍平均量）
        volume_surge = dataframe['volume'] > (dataframe['volume_ma_20'] * 1.8)

        # 突破确认（价格突破 + 量能激增）
        breakdown_confirmed = price_breakdown_low & volume_surge  # 多头退出确认
        breakout_confirmed = price_breakout_high & volume_surge  # 空头退出确认

        # 1. RSI 分析
        # 1a. RSI 极端值
        rsi_extreme_overbought = dataframe['rsi_14'] > 80  # 极度超买
        rsi_extreme_oversold = dataframe['rsi_14'] < 20  # 极度超卖

        # 1b. RSI 从高位/低位回落（动态确认真反转）
        rsi_5_max = dataframe['rsi_14'].rolling(5).max()  # 最近5根K线RSI最大值
        rsi_5_min = dataframe['rsi_14'].rolling(5).min()  # 最近5根K线RSI最小值
        rsi_falling_from_high = (dataframe['rsi_14'] < rsi_5_max - 5) & (rsi_5_max > 65)  # 从高位回落>5点
        rsi_rising_from_low = (dataframe['rsi_14'] > rsi_5_min + 5) & (rsi_5_min < 35)  # 从低位回升>5点

        # === 新增：基于2024最新研究的信号 ===

        # 2a. ATR 动态止损信号（长线优化：提高到2.0×ATR）
        peak_price_20 = dataframe['high'].rolling(20).max()
        pullback_from_peak = peak_price_20 - dataframe['close']
        atr_exit_signal = pullback_from_peak > (2.0 * dataframe['atr'])  # 回撤 > 2.0*ATR（长线）

        # 2b. Stochastic 超买/超卖 + 回落
        stoch_overbought = dataframe['stoch_k'] > 75
        stoch_oversold = dataframe['stoch_k'] < 25
        stoch_falling = (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) & (dataframe['stoch_k'].shift(1) < dataframe['stoch_k'].shift(2))
        stoch_rising = (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1)) & (dataframe['stoch_k'].shift(1) > dataframe['stoch_k'].shift(2))

        # 2c. Bollinger Bands 突破回落
        price_above_bb_upper = dataframe['close'] > dataframe['bb_upper']
        price_back_from_upper = (dataframe['close'] < dataframe['bb_upper']) & (dataframe['close'].shift(1) > dataframe['bb_upper'].shift(1))
        price_below_bb_lower = dataframe['close'] < dataframe['bb_lower']
        price_back_from_lower = (dataframe['close'] > dataframe['bb_lower']) & (dataframe['close'].shift(1) < dataframe['bb_lower'].shift(1))

        # 2. MACD 死叉/金叉（连续确认，不是单根）
        macd_bearish = (
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) < dataframe['macd_signal'].shift(1))
        )
        macd_bullish = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) > dataframe['macd_signal'].shift(1))
        )

        # 3. 价格跌破/突破 EMA21（短期趋势反转）+ 连续确认
        price_below_ema21 = dataframe['close'] < dataframe['ema_21']
        price_above_ema21 = dataframe['close'] > dataframe['ema_21']

        # 价格连续下跌/上涨（2根K线）
        price_falling = dataframe['close'] < dataframe['close'].shift(1)
        price_rising = dataframe['close'] > dataframe['close'].shift(1)

        # 4. 量能确认（提高阈值）
        volume_confirm = dataframe['volume'] > dataframe['volume_ma_20'] * 1.2

        # ======================================================================
        # === 🎯 专业顶部/底部识别系统（基于2024最新技术分析研究）===
        # ======================================================================

        # === 1. 动量背离检测 ⚡ 最强信号（提前2-5根K线预警）===

        # 1a. 寻找最近的价格高点/低点（最近20根K线内）
        price_peaks = dataframe['high'].rolling(20).max()  # 20根K线最高价
        price_troughs = dataframe['low'].rolling(20).min()  # 20根K线最低价

        # 检测新高/新低
        making_new_high = dataframe['high'] >= price_peaks.shift(1)  # 创新高
        making_new_low = dataframe['low'] <= price_troughs.shift(1)  # 创新低

        # 1b. RSI背离检测
        rsi_peaks = dataframe['rsi_14'].rolling(10).max()  # 最近10根K线RSI最高值
        rsi_troughs = dataframe['rsi_14'].rolling(10).min()  # 最近10根K线RSI最低值

        # 看跌背离（多头到顶）：价格创新高，但RSI未创新高
        rsi_bearish_divergence = (
            making_new_high &  # 价格创新高
            (dataframe['rsi_14'] < rsi_peaks.shift(5) - 5) &  # 但RSI比5根K线前的峰值低5点以上
            (rsi_peaks.shift(5) > 65)  # 且之前RSI在超买区
        )

        # 看涨背离（空头到底）：价格创新低，但RSI未创新低
        rsi_bullish_divergence = (
            making_new_low &  # 价格创新低
            (dataframe['rsi_14'] > rsi_troughs.shift(5) + 5) &  # 但RSI比5根K线前的谷底高5点以上
            (rsi_troughs.shift(5) < 35)  # 且之前RSI在超卖区
        )

        # 1c. MACD柱状图背离检测
        macd_hist = dataframe['macd'] - dataframe['macd_signal']  # MACD柱状图
        macd_hist_peaks = macd_hist.rolling(10).max()
        macd_hist_troughs = macd_hist.rolling(10).min()

        # MACD看跌背离：价格创新高，但MACD柱状图未创新高
        macd_bearish_divergence = (
            making_new_high &
            (macd_hist < macd_hist_peaks.shift(5)) &  # MACD柱状图未创新高
            (macd_hist_peaks.shift(5) > 0)  # 之前MACD在正值区
        )

        # MACD看涨背离：价格创新低，但MACD柱状图未创新低
        macd_bullish_divergence = (
            making_new_low &
            (macd_hist > macd_hist_troughs.shift(5)) &  # MACD柱状图未创新低
            (macd_hist_troughs.shift(5) < 0)  # 之前MACD在负值区
        )

        # === 2. 双顶/三顶形态检测 📍 ===

        # 2a. 识别最近的高点（局部峰值）
        # 高点定义：比前后2根K线的高价都高
        local_peak = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['high'] > dataframe['high'].shift(2)) &
            (dataframe['high'] > dataframe['high'].shift(-1)) &
            (dataframe['high'] > dataframe['high'].shift(-2))
        )

        # 2b. 识别最近的低点（局部谷底）
        local_trough = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['low'] < dataframe['low'].shift(2)) &
            (dataframe['low'] < dataframe['low'].shift(-1)) &
            (dataframe['low'] < dataframe['low'].shift(-2))
        )

        # 2c. 双顶/三顶检测：最近20根K线内，高点在±0.5%范围内出现2-3次
        # 统计最近高点与20根K线最高价的距离
        distance_to_peak_pct = ((dataframe['high'] - price_peaks.shift(1)) / price_peaks.shift(1) * 100).abs()

        # 最近20根K线内，接近最高点（±0.5%）的次数
        near_peak_count = (distance_to_peak_pct < 0.5).rolling(20).sum()

        # 双顶/三顶信号：2-3次测试顶部但无法突破
        double_triple_top = (
            (near_peak_count >= 2) &  # 至少2次接近顶部
            (near_peak_count <= 3) &  # 最多3次（避免横盘）
            (dataframe['high'] < price_peaks.shift(1))  # 当前未能突破前高
        )

        # 2d. 双底/三底检测（对称逻辑）
        distance_to_trough_pct = ((dataframe['low'] - price_troughs.shift(1)) / price_troughs.shift(1) * 100).abs()
        near_trough_count = (distance_to_trough_pct < 0.5).rolling(20).sum()

        double_triple_bottom = (
            (near_trough_count >= 2) &
            (near_trough_count <= 3) &
            (dataframe['low'] > price_troughs.shift(1))  # 当前未能跌破前低
        )

        # === 3. 量能衰竭检测 📉 ===

        # 3a. 多次冲高但量能递减（多头到顶）
        volume_ma_5 = dataframe['volume'].rolling(5).mean()
        volume_declining = (
            (dataframe['volume'] < volume_ma_5.shift(5) * 0.8) &  # 当前量能比5根K线前少20%
            (dataframe['high'] >= dataframe['high'].shift(5) * 0.995)  # 但价格仍在高位（±0.5%）
        )

        # 3b. 多次下探但量能递减（空头到底）
        volume_declining_bottom = (
            (dataframe['volume'] < volume_ma_5.shift(5) * 0.8) &
            (dataframe['low'] <= dataframe['low'].shift(5) * 1.005)
        )

        # === 4. 综合顶部/底部信号 🎯 ===

        # 多头到顶信号（3个条件中至少满足2个）
        top_signal_count = (
            rsi_bearish_divergence.astype(int) +
            macd_bearish_divergence.astype(int) +
            (double_triple_top & volume_declining).astype(int)
        )

        trend_top_signal = (
            long_trend_existed &  # 必须有上涨趋势
            ~is_sideways &  # 不在横盘
            (top_signal_count >= 2)  # 至少2个到顶信号
        )

        # 空头到底信号（对称逻辑）
        bottom_signal_count = (
            rsi_bullish_divergence.astype(int) +
            macd_bullish_divergence.astype(int) +
            (double_triple_bottom & volume_declining_bottom).astype(int)
        )

        trend_bottom_signal = (
            short_trend_existed &  # 必须有下跌趋势
            ~is_sideways &  # 不在横盘
            (bottom_signal_count >= 2)  # 至少2个到底信号
        )

        # === 多头退出信号组合（只在上涨趋势后触发）===

        # 层次0：🎯 趋势到顶识别（最高优先级）- 专业顶部信号
        exit_long_top = (
            trend_top_signal &  # 到顶信号（背离+双顶+量能衰竭，至少2个）
            volume_confirm  # 量能确认
        )
        dataframe.loc[exit_long_top, 'exit_long'] = 1
        dataframe.loc[exit_long_top, 'exit_tag'] = 'trend_top_reversal'

        # 层次1：双时间框架强反转 - 15m + 1h 都确认反转（长线优化：去掉4h避免延迟）
        if use_mtf:
            exit_long_strong = (
                long_trend_existed &  # 前置条件：曾有上涨趋势
                ~is_sideways &  # 横盘过滤：不在横盘市场触发
                (macd_bearish & price_below_ema21 & price_falling) &  # 15m反转+价格下跌
                h1_bullish_reversal &  # 1h反转
                volume_confirm &  # 量能确认
                ~exit_long_top  # 排除已标记的到顶信号
            )
            dataframe.loc[exit_long_strong, 'exit_long'] = 1
            dataframe.loc[exit_long_strong, 'exit_tag'] = 'mtf_strong_reversal'

        # 层次2：趋势减弱 - 15m反转 + 1h趋势减弱 + RSI从高位回落
        if use_mtf:
            exit_long_weak = (
                long_trend_existed &  # 前置条件：曾有上涨趋势
                ~is_sideways &  # 横盘过滤：不在横盘市场触发
                (macd_bearish & price_below_ema21 & price_falling) &  # 15m反转+价格下跌
                rsi_falling_from_high &  # RSI从高位回落（关键：动量反转确认）
                h1_trend_weakening &  # 1h趋势减弱
                volume_confirm &  # 量能确认
                ~exit_long_strong &  # 排除已标记的强信号
                ~exit_long_top  # 排除已标记的到顶信号
            )
            dataframe.loc[exit_long_weak, 'exit_long'] = 1
            dataframe.loc[exit_long_weak, 'exit_tag'] = 'mtf_trend_weak'

        # ❌ 删除原层次3：RSI>80极端信号（对长线策略太敏感，短期回调会误触发）

        # 层次3：多指标趋势确认（ATR + Stochastic + Bollinger）
        # 🆕 长线优化：要求至少3个指标同时确认
        atr_signal_long = long_trend_existed & atr_exit_signal  # ATR动态止损（2.0×ATR）
        stoch_signal_long = long_trend_existed & stoch_overbought & stoch_falling  # Stochastic超买回落
        bb_signal_long = long_trend_existed & price_back_from_upper & volume_confirm  # BB突破回落

        signal_count_long = atr_signal_long.astype(int) + stoch_signal_long.astype(int) + bb_signal_long.astype(int)

        exit_long_advanced = (
            ~is_sideways &  # 横盘过滤：不在横盘市场触发
            (signal_count_long >= 3) &  # 🆕 至少3个指标确认（长线优化：提高门槛）
            ~dataframe['exit_tag'].isin(['trend_top_reversal', 'mtf_strong_reversal', 'mtf_trend_weak'])
        )
        dataframe.loc[exit_long_advanced, 'exit_long'] = 1
        dataframe.loc[exit_long_advanced, 'exit_tag'] = 'multi_indicator_confirm'

        # 层次4：Donchian 50日最后防线 - 价格跌破长期支撑位
        # ✅ 增强版：添加反转确认，防止假跌破

        # 1. 价格跌破50日低点（需要站稳，不只是触碰）
        price_breakdown_50 = (
            (dataframe['close'] < dataframe['donchian_low_50'].shift(1)) &  # 当前跌破
            (dataframe['close'] < dataframe['donchian_low_50'].shift(1) * 0.998)  # 站稳2个点位
        )

        # 2. 反转确认条件
        breakdown_confirmed = (
            (dataframe['rsi_14'] < 35) &  # RSI弱势（真实反转）
            (dataframe['adx'] > 25) &  # ✅ 提高ADX要求：15→25（确保有强趋势）
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.3)  # ✅ 成交量确认（1.3倍均量）
        )

        # 3. 价格动量确认（连续下跌）
        price_momentum_down = (
            (dataframe['close'] < dataframe['close'].shift(1)) &  # 当前K线下跌
            (dataframe['close'] < dataframe['close'].shift(2))    # 连续2根K线下跌
        )

        exit_long_breakdown = (
            long_trend_existed &  # 前置条件：曾有上涨趋势
            ~is_sideways &  # 横盘过滤
            price_breakdown_50 &  # 跌破50日低点
            breakdown_confirmed &  # ✅ 反转确认（RSI+ADX+成交量）
            price_momentum_down &  # ✅ 价格动量确认
            ~dataframe['exit_tag'].isin(['trend_top_reversal', 'mtf_strong_reversal', 'mtf_trend_weak', 'multi_indicator_confirm'])
        )
        dataframe.loc[exit_long_breakdown, 'exit_long'] = 1
        dataframe.loc[exit_long_breakdown, 'exit_tag'] = 'donchian_50_breakdown'

        # === 空头退出信号组合（只在下跌趋势后触发）===

        # 层次0：🎯 趋势到底识别（最高优先级）- 专业底部信号
        exit_short_bottom = (
            trend_bottom_signal &  # 到底信号（背离+双底+量能衰竭，至少2个）
            volume_confirm  # 量能确认
        )
        dataframe.loc[exit_short_bottom, 'exit_short'] = 1
        dataframe.loc[exit_short_bottom, 'exit_tag'] = 'trend_bottom_reversal'

        # 层次1：双时间框架强反转 - 15m + 1h 都确认反转（长线优化：去掉4h避免延迟）
        if use_mtf:
            exit_short_strong = (
                short_trend_existed &  # 前置条件：曾有下跌趋势
                ~is_sideways &  # 横盘过滤：不在横盘市场触发
                (macd_bullish & price_above_ema21 & price_rising) &  # 15m反转+价格上涨
                h1_bearish_reversal &  # 1h反转
                volume_confirm &  # 量能确认
                ~exit_short_bottom  # 排除已标记的到底信号
            )
            dataframe.loc[exit_short_strong, 'exit_short'] = 1
            dataframe.loc[exit_short_strong, 'exit_tag'] = 'mtf_strong_reversal'

        # 层次2：趋势减弱 - 15m反转 + 1h趋势减弱 + RSI从低位回升
        if use_mtf:
            exit_short_weak = (
                short_trend_existed &  # 前置条件：曾有下跌趋势
                ~is_sideways &  # 横盘过滤：不在横盘市场触发
                (macd_bullish & price_above_ema21 & price_rising) &  # 15m反转+价格上涨
                rsi_rising_from_low &  # RSI从低位回升（关键：动量反转确认）
                h1_trend_weakening &  # 1h趋势减弱
                volume_confirm &  # 量能确认
                ~exit_short_strong &  # 排除已标记的强信号
                ~exit_short_bottom  # 排除已标记的到底信号
            )
            dataframe.loc[exit_short_weak, 'exit_short'] = 1
            dataframe.loc[exit_short_weak, 'exit_tag'] = 'mtf_trend_weak'

        # ❌ 删除原层次3：RSI<20极端信号（对长线策略太敏感，短期反弹会误触发）

        # 层次3：多指标趋势确认（ATR + Stochastic + Bollinger）
        # 🆕 长线优化：要求至少3个指标同时确认 + ATR倍数提高到2.0
        valley_price_20 = dataframe['low'].rolling(20).min()
        bounce_from_valley = dataframe['close'] - valley_price_20
        atr_exit_signal_short = bounce_from_valley > (2.0 * dataframe['atr'])  # 🆕 2.0×ATR（长线）

        atr_signal_short = short_trend_existed & atr_exit_signal_short
        stoch_signal_short = short_trend_existed & stoch_oversold & stoch_rising
        bb_signal_short = short_trend_existed & price_back_from_lower & volume_confirm

        signal_count_short = atr_signal_short.astype(int) + stoch_signal_short.astype(int) + bb_signal_short.astype(int)

        exit_short_advanced = (
            ~is_sideways &  # 横盘过滤：不在横盘市场触发
            (signal_count_short >= 3) &  # 🆕 至少3个指标确认（长线优化：提高门槛）
            ~dataframe['exit_tag'].isin(['trend_bottom_reversal', 'mtf_strong_reversal', 'mtf_trend_weak'])
        )
        dataframe.loc[exit_short_advanced, 'exit_short'] = 1
        dataframe.loc[exit_short_advanced, 'exit_tag'] = 'multi_indicator_confirm'

        # 层次4：Donchian 50日最后防线 - 价格突破长期阻力位
        # ✅ 增强版：添加反转确认，防止假突破

        # 1. 价格突破50日高点（需要站稳，不只是触碰）
        price_breakout_50 = (
            (dataframe['close'] > dataframe['donchian_high_50'].shift(1)) &  # 当前突破
            (dataframe['close'] > dataframe['donchian_high_50'].shift(1) * 1.002)  # 站稳2个点位
        )

        # 2. 反转确认条件
        breakout_confirmed = (
            (dataframe['rsi_14'] > 65) &  # RSI强势（真实反转）
            (dataframe['adx'] > 25) &  # ✅ 提高ADX要求：15→25（确保有强趋势）
            (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.3)  # ✅ 成交量确认（1.3倍均量）
        )

        # 3. 价格动量确认（连续上涨）
        price_momentum_up = (
            (dataframe['close'] > dataframe['close'].shift(1)) &  # 当前K线上涨
            (dataframe['close'] > dataframe['close'].shift(2))    # 连续2根K线上涨
        )

        exit_short_breakout = (
            short_trend_existed &  # 前置条件：曾有下跌趋势
            ~is_sideways &  # 横盘过滤
            price_breakout_50 &  # 突破50日高点
            breakout_confirmed &  # ✅ 反转确认（RSI+ADX+成交量）
            price_momentum_up &  # ✅ 价格动量确认
            ~dataframe['exit_tag'].isin(['trend_bottom_reversal', 'mtf_strong_reversal', 'mtf_trend_weak', 'multi_indicator_confirm'])
        )
        dataframe.loc[exit_short_breakout, 'exit_short'] = 1
        dataframe.loc[exit_short_breakout, 'exit_tag'] = 'donchian_50_breakout'

        # === 🚀 新增：分级出场系统（激进保护型）===
        # 根据预警等级和趋势状态，创建不同敏感度的出场信号

        # 🚨 级别1：紧急出场（强预警 + 快速反转）
        # 适用场景：高浮盈时，出现强烈反转信号，立即出场保护利润
        emergency_exit_long = (
            has_strong_warning_long &  # 预警等级>50
            fast_reversal_long &  # 5m快速反转
            (dataframe['rsi_14'] > 65)  # RSI确认超买
        )

        emergency_exit_short = (
            has_strong_warning_short &  # 预警等级>50
            fast_reversal_short &  # 5m快速反转
            (dataframe['rsi_14'] < 35)  # RSI确认超卖
        )

        # 紧急出场信号优先级最高，覆盖之前的标记
        dataframe.loc[emergency_exit_long, 'exit_long'] = 1
        dataframe.loc[emergency_exit_long, 'exit_tag'] = 'emergency_profit_protect'

        dataframe.loc[emergency_exit_short, 'exit_short'] = 1
        dataframe.loc[emergency_exit_short, 'exit_tag'] = 'emergency_profit_protect'

        # ⚠️ 级别2：警戒出场（中度预警 + 5m反转）
        # 适用场景：中等浮盈时，5m反转但不需要1h确认
        cautious_exit_long = (
            has_moderate_warning_long &  # 预警等级>30
            fast_reversal_long &  # 5m反转
            (dataframe['close'] < dataframe['ema_21']) &  # 跌破EMA21
            ~emergency_exit_long &  # 排除已标记的紧急出场
            ~dataframe['exit_long'].astype(bool)  # 排除已有的出场信号
        )

        cautious_exit_short = (
            has_moderate_warning_short &  # 预警等级>30
            fast_reversal_short &  # 5m反转
            (dataframe['close'] > dataframe['ema_21']) &  # 突破EMA21
            ~emergency_exit_short &  # 排除已标记的紧急出场
            ~dataframe['exit_short'].astype(bool)  # 排除已有的出场信号
        )

        dataframe.loc[cautious_exit_long, 'exit_long'] = 1
        dataframe.loc[cautious_exit_long, 'exit_tag'] = 'cautious_early_exit'

        dataframe.loc[cautious_exit_short, 'exit_short'] = 1
        dataframe.loc[cautious_exit_short, 'exit_tag'] = 'cautious_early_exit'

        # 📊 强趋势保护：在ADX>35的强趋势中，提高出场门槛
        # 防止在强趋势中被轻易震出
        # 如果在强趋势中，取消级别2的警戒出场（保护趋势持仓）
        in_strong_uptrend = in_strong_trend & (dataframe['close'] > dataframe['ema_50'])
        in_strong_downtrend = in_strong_trend & (dataframe['close'] < dataframe['ema_50'])

        # 强趋势中取消警戒出场（但保留紧急出场）
        cancel_cautious_long = in_strong_uptrend & (dataframe['exit_tag'] == 'cautious_early_exit')
        cancel_cautious_short = in_strong_downtrend & (dataframe['exit_tag'] == 'cautious_early_exit')

        dataframe.loc[cancel_cautious_long, 'exit_long'] = 0
        dataframe.loc[cancel_cautious_long, 'exit_tag'] = ''

        dataframe.loc[cancel_cautious_short, 'exit_short'] = 0
        dataframe.loc[cancel_cautious_short, 'exit_tag'] = ''

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
            # 获取币种风险乘数（主流币放大，其余显著缩小）
            coin_risk_multiplier = self.COIN_RISK_MULTIPLIERS.get(
                coin_risk_tier,
                self.COIN_RISK_MULTIPLIERS.get('medium_risk', 0.3)
            )

            if self.enforce_small_stake_for_non_bluechips:
                if coin_risk_tier != 'mainstream':
                    coin_risk_multiplier = min(coin_risk_multiplier, self.non_bluechip_stake_multiplier)
            else:
                if coin_risk_tier != 'mainstream':
                    coin_risk_multiplier = max(coin_risk_multiplier, 1.0)
            
            # 计算动态仓位大小
            position_size_ratio = self.calculate_position_size(current_rate, market_state, pair)

            # 获取账户余额
            available_balance = self.wallets.get_free(self.config['stake_currency'])

            # 🎯 应用 tradable_balance_ratio 设置
            # 用户可以通过config.json的tradable_balance_ratio控制交易资金比例
            tradable_ratio = self.config.get('tradable_balance_ratio', 0.99)  # 默认99%
            tradable_balance = available_balance * tradable_ratio

            # === 应用币种风险乘数到仓位计算 ===
            # 🎯 修改资金分配：使用更多资金进行初始交易
            # 初始仓位用50%资金，预留50%给DCA
            dca_reserved_balance = tradable_balance * 0.50  # 50%预留给DCA
            initial_balance_for_trade = tradable_balance * 0.50  # 50%用于初始仓位
            
            # 基础仓位计算（基于预留后的资金）
            base_calculated_stake = initial_balance_for_trade * position_size_ratio
            
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
            risk_tier_labels = {
                'low_risk': 'low',
                'medium_risk': 'medium',
                'high_risk': 'high'
            }

            self.event_log.info(
                "stake_calculation",
                pair=pair,
                market_state=market_state,
                balance=f"{available_balance:.2f}",
                tradable_ratio=f"{tradable_ratio:.0%}",
                tradable_balance=f"{tradable_balance:.2f}",
                dca_reserved=f"{dca_reserved_balance:.2f}",
                initial_pool=f"{initial_balance_for_trade:.2f}",
                risk_tier=coin_risk_tier,
                risk_label=risk_tier_labels.get(coin_risk_tier, coin_risk_tier),
                position_ratio=f"{position_size_ratio:.2%}",
                base_position=f"{base_calculated_stake:.2f}",
                risk_multiplier=f"{coin_risk_multiplier:.2f}",
                adjusted_position=f"{calculated_stake:.2f}",
                leverage=f"{int(dynamic_leverage)}x",
                final_amount=f"{final_stake:.2f}",
                # 🔧 修复Bug #3: 防止除零
                expected_quantity=f"{final_stake / max(current_rate, 1e-8):.6f}",
                decision_time=str(current_time),
            )
            
            # 重要：设置策略的当前杠杆（供Freqtrade使用）
            if hasattr(self, '_current_leverage'):
                self._current_leverage[pair] = int(dynamic_leverage)
            else:
                self._current_leverage = {pair: int(dynamic_leverage)}
            
            # 记录详细的风险计算日志
            self._log_risk_calculation_details(pair, {
                'current_price': current_rate,
                'planned_position': position_size_ratio,
                'stoploss_level': abs(self.stoploss),
                'leverage': dynamic_leverage,
                'market_state': market_state,
                'volatility': volatility,
                'side': side,
                'entry_tag': entry_tag,
            }, {
                'risk_amount': final_stake * abs(self.stoploss),
                'risk_percentage': (final_stake * abs(self.stoploss)) / tradable_balance,
                'max_loss': final_stake * abs(self.stoploss),
                'adjusted_position': position_size_ratio,
                'suggested_leverage': dynamic_leverage,
                'risk_rating': self._calculate_risk_rating(final_stake * abs(self.stoploss) / tradable_balance),
                'rating_reason': f'基于{market_state}市场状态和{volatility*100:.1f}%波动率的综合评估'
            })
            
            return final_stake
            
        except Exception as e:
            logger.error(f"仓位计算失败: {e}")
            return proposed_stake

    def _get_entry_confidence(self, trade: Trade) -> float:
        """
        获取交易的入场信心分数

        从开仓时的K线中读取 entry_confidence_long 或 entry_confidence_short
        用于判断是否允许DCA和如何执行分批止盈

        Args:
            trade: 交易对象

        Returns:
            float: 入场信心分数 (0-1)，默认0.7
        """
        try:
            pair = trade.pair
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            if dataframe.empty:
                return 0.7  # 默认中等信心

            # 找到最接近开仓时间的K线
            entry_time = trade.open_date_utc.replace(tzinfo=None)
            dataframe_times = pd.to_datetime(dataframe.index).tz_localize(None)
            time_diff = abs(dataframe_times - entry_time)

            # 使用 argmin() 获取最小值的位置索引
            closest_pos = time_diff.argmin()
            closest_idx = dataframe.index[closest_pos]

            # 根据交易方向读取对应的信心分数
            if not trade.is_short:
                entry_confidence = dataframe.loc[closest_idx, 'entry_confidence_long']
            else:
                entry_confidence = dataframe.loc[closest_idx, 'entry_confidence_short']

            return float(entry_confidence) if not pd.isna(entry_confidence) else 0.7

        except Exception as e:
            logger.warning(f"获取entry_confidence失败 {trade.pair}: {e}")
            return 0.7  # 默认中等信心

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """🎯 2025科学DCA系统 + 动态分批止盈"""

        # === 🎯 固定分批止盈系统（解决账本混乱问题）===
        if current_profit > 0:
            # 获取交易对和入场信号
            pair = trade.pair
            enter_tag = trade.enter_tag or 'default'

            # 从开仓时获取 entry_confidence（使用新的辅助函数）
            entry_confidence = self._get_entry_confidence(trade)

            # 跟踪已退出的批次
            if not hasattr(self, '_profit_exits'):
                self._profit_exits = {}

            trade_key = f"{pair}_{trade.id}"

            # ✅ 首次初始化：固定止盈目标和比例（只计算一次）
            if trade_key not in self._profit_exits:
                # 获取开仓时的市场指标
                try:
                    dataframe_now, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    if not dataframe_now.empty:
                        current_candle = dataframe_now.iloc[-1]
                        adx = current_candle.get('adx', 25)
                        atr = current_candle.get('atr', current_candle.get('atr_p', 0.02))
                        atr_pct = atr / max(current_rate, 1e-8)
                    else:
                        adx = 25
                        atr_pct = 0.02
                except:
                    adx = 25
                    atr_pct = 0.02

                # 1. 计算固定止盈目标
                signal_multiplier = self.SIGNAL_PROFIT_MULTIPLIERS.get(enter_tag, self.SIGNAL_PROFIT_MULTIPLIERS['default'])
                base_targets = [0.10, 0.20, 0.35]
                atr_multipliers = [2.0, 3.0, 4.5]
                atr_bonus = [atr_pct * mult for mult in atr_multipliers]
                fixed_targets = [
                    max(base, base + atr_bonus[i]) * signal_multiplier
                    for i, base in enumerate(base_targets)
                ]

                # 2. 计算固定退出比例
                if entry_confidence > 0.8:
                    base_ratios = [0.10, 0.15, 0.20]
                elif entry_confidence > 0.65:
                    base_ratios = [0.15, 0.20, 0.25]
                elif entry_confidence > 0.55:
                    base_ratios = [0.20, 0.25, 0.30]
                else:
                    base_ratios = [0.30, 0.35, 0.35]

                # 根据开仓时的市场状态调整
                if adx > 35:
                    trend_adjustment = -0.05
                elif adx < 20:
                    trend_adjustment = 0.05
                else:
                    trend_adjustment = 0.0

                if atr_pct > 0.03:
                    volatility_adjustment = 0.05
                elif atr_pct < 0.015:
                    volatility_adjustment = -0.05
                else:
                    volatility_adjustment = 0.0

                total_adjustment = trend_adjustment + volatility_adjustment
                fixed_ratios = [
                    max(0.05, min(0.40, ratio + total_adjustment))
                    for ratio in base_ratios
                ]

                # 保存固定参数
                self._profit_exits[trade_key] = {
                    'targets': fixed_targets,              # ✅ 固定目标
                    'ratios': fixed_ratios,                # ✅ 固定比例
                    'completed_levels': [],
                    'actual_exited_stake': 0.0             # ✅ 记录实际退出的金额
                }

            # 使用已保存的固定参数
            fixed_targets = self._profit_exits[trade_key]['targets']
            fixed_ratios = self._profit_exits[trade_key]['ratios']

            # 检查各批次
            for level, (target, ratio) in enumerate(zip(fixed_targets, fixed_ratios), 1):
                if level in self._profit_exits[trade_key]['completed_levels']:
                    continue

                if current_profit >= target:
                    # 触发退出
                    self._profit_exits[trade_key]['completed_levels'].append(level)

                    # === 🔒 低信心全清仓机制 ===
                    if self.low_confidence_full_exit and entry_confidence <= self.confidence_threshold_low and level == 3:
                        # 计算剩余仓位
                        already_exited_stake = self._profit_exits[trade_key]['actual_exited_stake']
                        remaining_stake = trade.stake_amount - already_exited_stake

                        # 记录本次退出金额
                        self._profit_exits[trade_key]['actual_exited_stake'] += remaining_stake

                        if self.enable_dca_logging:
                            logger.info(
                                f"🔒 低信心全清仓 L{level} {pair}: "
                                f"信心={entry_confidence:.2f}, 利润={current_profit:.1%}, "
                                f"已退={already_exited_stake:.2f}, 清仓={remaining_stake:.2f}（全部剩余）"
                            )

                        return -remaining_stake
                    else:
                        # 正常分批退出
                        stake_to_exit = trade.stake_amount * ratio

                        # ✅ 记录实际退出金额
                        self._profit_exits[trade_key]['actual_exited_stake'] += stake_to_exit

                        if self.enable_dca_logging:
                            logger.info(
                                f"🎯 分批止盈 L{level} {pair}: "
                                f"利润={current_profit:.1%}, 目标={target:.1%}, "
                                f"退出={ratio:.0%}, 金额={stake_to_exit:.2f}"
                            )

                        return -stake_to_exit

        # === 🛡️ 利润回撤保护系统（中高信心交易） ===
        if current_profit > 0 and self.enable_profit_protection:
            # 获取入场信心（如果还没获取）
            if 'entry_confidence' not in locals():
                entry_confidence = self._get_entry_confidence(trade)

            # 只对中高信心交易启用利润保护
            if entry_confidence > self.confidence_threshold_low:
                pair = trade.pair
                trade_key = f"{pair}_{trade.id}"

                # 初始化利润保护跟踪
                if not hasattr(self, '_profit_protection'):
                    self._profit_protection = {}

                if trade_key not in self._profit_protection:
                    self._profit_protection[trade_key] = {
                        'peak_profit': current_profit,
                        'all_exits_completed': False
                    }

                # 检查是否完成所有3批止盈
                if hasattr(self, '_profit_exits') and trade_key in self._profit_exits:
                    completed_levels = self._profit_exits[trade_key].get('completed_levels', [])
                    all_completed = len(completed_levels) >= 3
                    self._profit_protection[trade_key]['all_exits_completed'] = all_completed

                    # 只有在完成所有分批止盈后才启用回撤保护
                    if all_completed:
                        # 更新峰值利润
                        if current_profit > self._profit_protection[trade_key]['peak_profit']:
                            self._profit_protection[trade_key]['peak_profit'] = current_profit

                        peak_profit = self._profit_protection[trade_key]['peak_profit']
                        drawdown_threshold = peak_profit * self.profit_drawdown_threshold

                        # 检查是否回撤超过阈值
                        if current_profit < drawdown_threshold:
                            # 触发利润保护，清仓所有剩余仓位
                            # ✅ 使用实际退出金额计算剩余
                            already_exited_stake = self._profit_exits[trade_key].get('actual_exited_stake', 0.0)
                            remaining_stake = trade.stake_amount - already_exited_stake

                            if self.enable_dca_logging:
                                logger.info(
                                    f"🛡️ 利润回撤保护触发 {pair}: "
                                    f"峰值={peak_profit:.1%}, 当前={current_profit:.1%}, "
                                    f"回撤={(peak_profit - current_profit)/peak_profit:.1%}, "
                                    f"阈值={self.profit_drawdown_threshold:.0%}, "
                                    f"已退={already_exited_stake:.2f}, 清仓={remaining_stake:.2f}（剩余）"
                                )

                            # 返回负数表示清仓全部剩余仓位
                            return -remaining_stake

        # === 🔧 DCA功能开关检查 ===
        if not self.enable_dca:
            return None
        
        # 检查DCA次数限制
        if trade.nr_of_successful_entries >= self.max_dca_orders:
            return None
        
        # === 🔬 第一阶段：科学位置验证 ===
        optimal_dca_result = self._calculate_optimal_dca_position(trade, current_rate, current_profit)
        
        if not optimal_dca_result['should_dca']:
            if self.enable_dca_logging:
                self.event_log.info(
                    "dca_skipped",
                    pair=trade.pair,
                    reason=optimal_dca_result['reason'],
                )
            return None
        
        # === 🎯 第二阶段：科学仓位计算 ===
        dca_details = self._calculate_scientific_dca_amount(
            trade, current_rate, optimal_dca_result
        )

        if dca_details is None:
            return None

        dca_amount = dca_details['amount']

        if self.enable_dca_logging:
            self.event_log.info(
                "dca_triggered",
                importance="summary",
                pair=trade.pair,
                level=optimal_dca_result['dca_level'],
                target_price=f"{optimal_dca_result['target_price']:.6f}",
                drawdown=f"{abs(current_profit):.1%}",
                optimal_spacing=f"{optimal_dca_result['optimal_spacing']:.1%}",
                volatility_factor=f"{optimal_dca_result['volatility_factor']:.1%}",
                amount=f"{dca_amount:.2f}",
                progression=f"1.5^{optimal_dca_result['dca_level']-1}",
                avg_improvement=f"{dca_details['avg_improvement']:.2%}",
                signals=",".join(optimal_dca_result.get('signal_tags', [])),
                price_deviation=f"{optimal_dca_result['price_deviation']:.2%}",
                safety_hint=f"{optimal_dca_result['safety_hint']:.2f}",
            )

        return dca_amount
    
    def _calculate_scientific_dca_amount(self, trade: Trade, current_rate: float,
                                       dca_result: dict) -> Optional[dict]:
        """🔬 基于几何级数 + 位置质量的智能DCA金额计算"""

        dca_level = dca_result['dca_level']
        base_amount = trade.stake_amount

        # === 几何级数核心公式 ===
        geometric_multiplier = 1.5 ** (dca_level - 1)

        # === 🎯 位置质量评估系统 ===
        position_quality_score = self._evaluate_position_quality(trade, current_rate, dca_result)

        # === 波动率自适应调整 ===
        volatility_factor = dca_result['volatility_factor']
        volatility_adj = 1.0 + min(0.4, volatility_factor * 8)  # 缓和波动率影响，最高+40%

        # === 位置质量奖励系统 ===
        if position_quality_score >= 8.5:
            quality_bonus = 1.8
            if self.enable_dca_logging:
                self.event_log.info(
                    "dca_quality",
                    pair=trade.pair,
                    quality=f"{position_quality_score:.1f}",
                    tier="excellent",
                )
        elif position_quality_score >= 7.0:
            quality_bonus = 1.4
            if self.enable_dca_logging:
                self.event_log.info(
                    "dca_quality",
                    pair=trade.pair,
                    quality=f"{position_quality_score:.1f}",
                    tier="strong",
                )
        elif position_quality_score >= 5.5:
            quality_bonus = 1.1
        else:
            quality_bonus = 0.7

        # === 基础金额计算 ===
        scientific_amount = base_amount * geometric_multiplier * volatility_adj * quality_bonus

        # === 风险限制 ===
        max_single_multiplier = min(3.0, 1.5 + position_quality_score * 0.2)
        max_single_dca = base_amount * max_single_multiplier

        available_balance = self.wallets.get_free(self.config['stake_currency'])
        # 🎯 应用 tradable_balance_ratio
        tradable_ratio = self.config.get('tradable_balance_ratio', 0.99)
        tradable_balance = available_balance * tradable_ratio

        portfolio_limit_pct = min(0.25, 0.12 + position_quality_score * 0.015)
        portfolio_limit = tradable_balance * portfolio_limit_pct

        # 🔧 修复风险#1: 添加DCA金额硬性上限
        # 防止几何级数乘以质量奖励导致单次DCA过大
        absolute_max_dca = base_amount * 10  # 单次DCA不超过初始仓位的10倍

        final_amount = min(scientific_amount, max_single_dca, portfolio_limit, absolute_max_dca)

        # === 最小有效性检查 ===
        min_meaningful = base_amount * 0.3
        if final_amount < min_meaningful:
            if self.enable_dca_logging:
                self.event_log.info(
                    "dca_cancelled_small",
                    pair=trade.pair,
                    amount=f"{final_amount:.2f}",
                    minimum=f"{min_meaningful:.2f}",
                )
            return None

        # === 预计均价改善检查 ===
        existing_units = abs(trade.amount) if getattr(trade, 'amount', 0) else base_amount / max(trade.open_rate, 1e-8)
        existing_cost = existing_units * trade.open_rate

        new_units = final_amount / max(current_rate, 1e-8)
        total_units = existing_units + new_units
        projected_avg = (existing_cost + final_amount) / max(total_units, 1e-8)

        if not trade.is_short:
            improvement = (trade.open_rate - projected_avg) / max(trade.open_rate, 1e-8)
            improvement_required = 0.004  # 至少改善0.4%
            improvement_valid = improvement > improvement_required
        else:
            improvement = (projected_avg - trade.open_rate) / max(trade.open_rate, 1e-8)
            improvement_required = 0.004
            improvement_valid = improvement > improvement_required

        if not improvement_valid:
            if self.enable_dca_logging:
                self.event_log.info(
                    "dca_cancelled_no_improve",
                    pair=trade.pair,
                    projected_avg=f"{projected_avg:.6f}",
                    current_avg=f"{trade.open_rate:.6f}",
                    improvement=f"{improvement:.3%}",
                    required=f"{improvement_required:.3%}",
                )
            return None

        # === 记录金额拆解 ===
        if self.enable_dca_logging:
            self.event_log.info(
                "dca_amount_calc",
                pair=trade.pair,
                quality=f"{position_quality_score:.1f}",
                quality_bonus=f"{quality_bonus:.1f}x",
                geometric=f"1.5^{dca_level-1}={geometric_multiplier:.1f}x",
                volatility_adj=f"{volatility_adj:.1f}x",
                final_multiplier=f"{final_amount / base_amount:.1f}x",
                max_single=f"{max_single_multiplier:.1f}x",
                portfolio_limit=f"{portfolio_limit_pct:.1%}",
            )

        return {
            'amount': final_amount,
            'projected_avg': projected_avg,
            'avg_improvement': improvement,
            'new_units': new_units,
            'portfolio_limit': portfolio_limit,
        }

    def _check_dca_rescue_possibility(self, trade: Trade, current_rate: float,
                                       current_profit: float, dynamic_stoploss_value: float,
                                       dataframe) -> dict:
        """
        🚑 DCA救援可行性检查
        在动态止损即将触发时，评估是否值得通过DCA尝试救援
        返回: {'can_rescue': bool, 'signal_strength': float, 'reason': str}
        """
        try:
            # === 基础条件检查 ===
            dca_level = trade.nr_of_successful_entries + 1
            max_dca = self.max_dca_orders

            # 已达最大DCA次数
            if dca_level > max_dca:
                return {'can_rescue': False, 'signal_strength': 0, 'reason': 'dca_limit'}

            # 亏损过深不救援（超过止损位1.5倍）
            if abs(current_profit) > abs(dynamic_stoploss_value) * 1.5:
                return {'can_rescue': False, 'signal_strength': 0, 'reason': 'loss_too_deep'}

            # === 检查DCA基础可行性 ===
            dca_result = self._calculate_optimal_dca_position(trade, current_rate, current_profit)
            if not dca_result.get('should_dca', False):
                return {'can_rescue': False, 'signal_strength': 0, 'reason': 'dca_not_viable'}

            # === 技术信号评分系统 ===
            current_candle = dataframe.iloc[-1]
            signal_score = 0

            # RSI超卖/超买 (+3分)
            rsi = current_candle.get('rsi', 50)
            if not trade.is_short and rsi < 30:
                signal_score += 3
            elif trade.is_short and rsi > 70:
                signal_score += 3

            # 价格接近EMA支撑/阻力 (+2分)
            ema_200 = current_candle.get('ema_200', 0)
            if ema_200 > 0:
                price_to_ema = abs(current_rate - ema_200) / ema_200
                if price_to_ema < 0.01:  # 1%范围内
                    signal_score += 2

            # 成交量激增 (+2分)
            volume = current_candle.get('volume', 0)
            volume_ma = current_candle.get('volume_ma_20', 0)
            if volume_ma > 0 and volume > volume_ma * 1.5:
                signal_score += 2

            # 趋势稳定 (+1分)
            adx = current_candle.get('adx', 0)
            if 20 <= adx <= 40:
                signal_score += 1

            # MACD背离 (+2分)
            macd = current_candle.get('macd', 0)
            macd_signal = current_candle.get('macdsignal', 0)
            if not trade.is_short and macd > macd_signal:
                signal_score += 2
            elif trade.is_short and macd < macd_signal:
                signal_score += 2

            # === 判断是否救援 ===
            # 需要至少5分才能救援
            can_rescue = signal_score >= 5

            return {
                'can_rescue': can_rescue,
                'signal_strength': signal_score,
                'reason': f"score_{signal_score}" if can_rescue else 'insufficient_signals'
            }

        except Exception:
            return {'can_rescue': False, 'signal_strength': 0, 'reason': 'error'}

    
    def _evaluate_position_quality(self, trade: Trade, current_rate: float, dca_result: dict) -> float:
        """
        🎯 位置质量评估系统 - 评估当前是否为绝佳加仓时机
        
        评分标准 (0-10分)：
        - 技术指标确认度
        - 关键支撑阻力位
        - 市场恐慌/贪婪程度  
        - 成交量确认
        - 趋势一致性
        """
        try:
            dataframe = self.get_dataframe_with_indicators(trade.pair, self.timeframe)
            if dataframe.empty:
                return 5.0  # 默认中等质量
                
            current_data = dataframe.iloc[-1]
            entry_price = trade.open_rate
            
            # 初始化评分
            quality_score = 5.0  # 基础分
            
            # === 📊 技术指标确认 (0-2.5分) ===
            rsi = current_data.get('rsi_14', 50)
            volume_ratio = current_data.get('volume_ratio', 1.0)
            
            if not trade.is_short:  # 做多头寸
                # RSI极度超卖奖励
                if rsi <= 25:
                    quality_score += 1.0  # 极度超卖+1分
                elif rsi <= 35:
                    quality_score += 0.7  # 超卖+0.7分
                elif rsi <= 45:
                    quality_score += 0.3  # 偏低+0.3分
                else:
                    quality_score -= 0.5  # RSI不够低扣分
                    
            else:  # 做空头寸  
                # RSI极度超买奖励
                if rsi >= 75:
                    quality_score += 1.0
                elif rsi >= 65:
                    quality_score += 0.7
                elif rsi >= 55:
                    quality_score += 0.3
                else:
                    quality_score -= 0.5
            
            # === 📈 成交量恐慌确认 (0-1.5分) ===
            if volume_ratio >= 2.5:
                quality_score += 1.2  # 极度恐慌成交量
            elif volume_ratio >= 1.8:
                quality_score += 0.8  # 高成交量
            elif volume_ratio >= 1.2:
                quality_score += 0.4  # 正常放量
            else:
                quality_score -= 0.3  # 成交量不足扣分
                
            # === 🎯 关键技术位置 (0-2分) ===
            # 检查是否接近重要支撑阻力位
            ema_200 = current_data.get('ema_200', current_rate)
            ema_50 = current_data.get('ema_50', current_rate)
            
            # 200日均线支撑/阻力奖励
            distance_200ema = abs(current_rate - ema_200) / ema_200
            if distance_200ema <= 0.02:  # 距离200日均线2%内
                quality_score += 1.5  # 关键位置大奖励
            elif distance_200ema <= 0.05:  # 距离200日均线5%内
                quality_score += 0.8
                
            # === 📉 回撤深度奖励 (0-1.5分) ===
            current_drawdown = abs((current_rate - entry_price) / entry_price)
            if current_drawdown >= 0.15:  # 15%以上回撤
                quality_score += 1.2  # 深度回撤，抄底机会
            elif current_drawdown >= 0.10:  # 10%以上回撤
                quality_score += 0.8
            elif current_drawdown >= 0.05:  # 5%以上回撤  
                quality_score += 0.4
                
            # === 🛡️ 安全系数提示奖励 (0-0.5分) ===
            safety_hint = dca_result.get('safety_hint', 1.5)
            if safety_hint >= 3.0:
                quality_score += 0.5  # 极高安全系数
            elif safety_hint >= 2.5:
                quality_score += 0.3
            elif safety_hint >= 2.0:
                quality_score += 0.1
                
            # 限制评分范围 0-10
            quality_score = max(0, min(10, quality_score))
            
        except Exception as e:
            logger.warning(f"位置质量评估失败: {e}")
            quality_score = 5.0  # 出错时返回中等质量
        
        return quality_score
    
    def _calculate_optimal_dca_position(self, trade: Trade, current_rate: float, current_profit: float) -> dict:
        """
        基于价格偏差与技术确认的DCA入场判定

        优化：只有高信心交易（confidence > 0.75）才允许DCA
        逻辑：高信心说明方向对，回调时逢低加码；低/中信心不值得追加投入
        """

        # === 🎯 步骤1：入场信心检查（最高优先级） ===
        entry_confidence = self._get_entry_confidence(trade)
        if entry_confidence <= self.confidence_threshold_dca:
            return {
                'should_dca': False,
                'reason': f'仅高信心交易允许DCA（当前信心={entry_confidence:.2f} ≤ {self.confidence_threshold_dca}）'
            }

        # === 步骤2：基础参数获取 ===
        entry_price = trade.open_rate
        current_drawdown = abs(current_profit)
        dca_level = trade.nr_of_successful_entries + 1

        dataframe = self.get_dataframe_with_indicators(trade.pair, self.timeframe)
        if dataframe.empty:
            return {'should_dca': False, 'reason': '指标数据不足，跳过DCA'}

        current_data = dataframe.iloc[-1]
        prev_data = dataframe.iloc[-2] if len(dataframe) > 1 else current_data

        atr_value = current_data.get('atr', current_data.get('atr_p', 0.02))
        volatility_factor = (atr_value / entry_price) if entry_price else 0.015

        # === 步骤3：回撤范围检查（放宽后：1%-20%） ===
        min_drawdown = self.dca_min_drawdown  # 1%（原1.5%）
        if current_drawdown < min_drawdown:
            return {'should_dca': False, 'reason': f'回撤不足 {current_drawdown:.1%} < {min_drawdown:.1%}'}

        base_spacing = max(0.015, volatility_factor * 2.5)
        optimal_spacings = []
        # 🔧 修复Bug #1: 生成足够的spacing，避免数组越界
        # max_dca_orders=5，所以需要至少6个元素（dca_level可能是1-6）
        for level in range(1, self.max_dca_orders + 2):  # +2确保足够
            spacing = base_spacing * (1.0 + (level - 1) * 0.35)
            optimal_spacings.append(spacing)

        # 改用 >= 检查，防止越界
        if dca_level >= len(optimal_spacings):
            return {'should_dca': False, 'reason': f'超出最大DCA级别{len(optimal_spacings)-1}'}

        spacing = optimal_spacings[dca_level - 1]
        if not trade.is_short:
            target_price = entry_price * (1 - spacing)
        else:
            target_price = entry_price * (1 + spacing)

        # === 步骤4：价格窗口检查（放宽后：±2%/10%） ===
        upper_tolerance = self.dca_price_tolerance_upper  # 2%（原0.8%）
        lower_tolerance = self.dca_price_tolerance_lower  # 10%（原5%）
        price_deviation = (current_rate - target_price) / target_price

        price_vs_entry = (current_rate - entry_price) / entry_price
        if not trade.is_short:
            valid_price_window = (-lower_tolerance <= price_deviation <= upper_tolerance) and price_vs_entry < -0.01
        else:
            valid_price_window = (-upper_tolerance <= price_deviation <= lower_tolerance) and price_vs_entry > 0.01

        if not valid_price_window:
            return {
                'should_dca': False,
                'reason': f'价格偏离目标区间 {price_deviation:.1%}',
            }

        # === 🔧 删除趋势强度检查（简化条件，提高触发率） ===
        # 原代码检查 trend_strength ±60，现已删除
        # 理由：高信心交易已经过入场时的多时间框架确认，无需重复检查趋势

        signals: list[str] = []
        rsi_value = current_data.get('rsi_14', 50)
        ema50 = current_data.get('ema_50', entry_price)
        ema200 = current_data.get('ema_200', entry_price)
        volume_ratio = current_data.get('volume_ratio', 1.0)
        momentum_score = current_data.get('momentum_score', 0.0)
        adx_value = current_data.get('adx', 20)

        if not trade.is_short:
            if rsi_value <= 35:
                signals.append('rsi_oversold')
            if current_rate <= ema50 * 0.995:
                signals.append('ema50_support')
            if current_rate <= ema200 * 1.005:
                signals.append('ema200_zone')
            if volume_ratio >= 1.3:
                signals.append('volume_spike')
            if momentum_score > -0.2:
                signals.append('momentum_stabilizing')
        else:
            if rsi_value >= 65:
                signals.append('rsi_overbought')
            if current_rate >= ema50 * 1.005:
                signals.append('ema50_resistance')
            if current_rate >= ema200 * 0.995:
                signals.append('ema200_zone')
            if volume_ratio >= 1.3:
                signals.append('volume_spike')
            if momentum_score < 0.2:
                signals.append('momentum_turning')

        if adx_value < 12:
            signals.append('low_trend')

        # === 步骤5：技术信号检查（放宽后：首次0个，后续1个） ===
        min_signals_required = self.dca_min_signals_first if dca_level == 1 else self.dca_min_signals_after
        # 0个信号表示不检查，直接通过
        if min_signals_required > 0 and len(signals) < min_signals_required:
            return {
                'should_dca': False,
                'reason': f'技术确认不足（需要{min_signals_required}个，当前{len(signals)}个）',
            }

        distance_to_target = abs(entry_price - target_price) / max(entry_price, 1e-8)
        safety_hint = distance_to_target / max(0.01, current_drawdown)

        # === 步骤6：最大回撤检查（放宽后：20%） ===
        max_allowed_drawdown = self.dca_max_drawdown  # 20%（原15%）
        if current_drawdown > max_allowed_drawdown:
            return {
                'should_dca': False,
                'reason': f'超过最大允许回撤 {current_drawdown:.1%} > {max_allowed_drawdown:.1%}',
            }

        return {
            'should_dca': True,
            'reason': '条件满足',
            'dca_level': dca_level,
            'optimal_spacing': spacing,
            'target_price': target_price,
            'volatility_factor': volatility_factor,
            'price_deviation': price_deviation,
            'signal_tags': signals,
            'safety_hint': safety_hint,
        }
    
            
    def _calculate_smart_dca_amount(self, trade: Trade, dca_decision: dict, 
                                  current_data: dict, market_state: str) -> float:
        """计算智能DCA金额 - 根据杠杆、信心度和风险动态调整"""
        
        try:
            # 基础DCA金额
            base_amount = trade.stake_amount
            entry_count = trade.nr_of_successful_entries + 1
            
            # === 获取杠杆和预设倍数 ===
            leverage = 10  # 默认值
            preset_multiplier = 1.2  # 默认倍数
            
            if hasattr(self, '_trade_targets') and trade.pair in self._trade_targets:
                targets = self._trade_targets[trade.pair]
                leverage = targets.get('leverage', 10)
                leverage_params = targets.get('leverage_params', {})
                preset_multiplier = leverage_params.get('dca', {}).get('multiplier', 1.2)
                
                # 查找当前DCA级别的预设倍数
                dca_levels = leverage_params.get('dca', {}).get('price_levels', [])
                for dca_level in dca_levels:
                    if dca_level['level'] == entry_count:
                        preset_multiplier = dca_level.get('amount_multiplier', preset_multiplier)
                        self._log_message(
                            f"💰 使用预设DCA倍数: {preset_multiplier:.1f}x (杠杆{leverage}x)",
                            importance="summary"
                        )
                        break
            
            # === 根据DCA类型调整基础倍数（结合杠杆）===
            # 高杠杆时降低DCA倍数，避免过度暴露
            leverage_factor = 1.0 if leverage <= 5 else 0.8 if leverage <= 10 else 0.6
            
            dca_type_multipliers = {
                'OVERSOLD_REVERSAL_DCA': 1.5 * leverage_factor,
                'OVERBOUGHT_REJECTION_DCA': 1.5 * leverage_factor,
                'SUPPORT_LEVEL_DCA': 1.3 * leverage_factor,
                'RESISTANCE_LEVEL_DCA': 1.3 * leverage_factor,
                'TREND_CONTINUATION_DCA': 1.2 * leverage_factor,
                'TREND_CONTINUATION_DCA_SHORT': 1.2 * leverage_factor,
                'VOLUME_CONFIRMED_DCA': 1.1 * leverage_factor
            }
            
            # 优先使用预设倍数，否则使用类型倍数
            if hasattr(self, '_trade_targets') and trade.pair in self._trade_targets:
                type_multiplier = preset_multiplier
                self.event_log.info(
                    "dca_multiplier_preset",
                    pair=trade.pair,
                    multiplier=f"{type_multiplier:.1f}x",
                )
            else:
                type_multiplier = dca_type_multipliers.get(dca_decision['dca_type'], 1.0 * leverage_factor)
                self.event_log.info(
                    "dca_multiplier_dynamic",
                    pair=trade.pair,
                    multiplier=f"{type_multiplier:.1f}x",
                    leverage_adjustment=f"{leverage_factor:.1f}x",
                )
            
            # === 根据信心度调整 ===
            confidence_multiplier = 0.5 + (dca_decision['confidence'] * 0.8)  # 0.5-1.3倍
            
            # === 根据市场状态调整（高杠杆时更保守）===
            market_base_multipliers = {
                'strong_uptrend': 1.4,
                'strong_downtrend': 1.4,
                'mild_uptrend': 1.2,
                'mild_downtrend': 1.2,
                'sideways': 1.0,
                'volatile': 0.7,
                'consolidation': 1.1
            }
            # 高杠杆时降低市场乘数
            market_multiplier = market_base_multipliers.get(market_state, 1.0) * leverage_factor
            
            # === 根据加仓次数递减（高杠杆时更严格）===
            # 后续加仓应该更保守，高杠杆时衰减更快
            leverage_decay_factor = 0.15 if leverage <= 5 else 0.20 if leverage <= 10 else 0.25
            entry_decay = max(0.4, 1.0 - (entry_count - 1) * leverage_decay_factor)
            
            # === 综合计算DCA金额 ===
            total_multiplier = (type_multiplier * confidence_multiplier * 
                              market_multiplier * entry_decay)
            
            calculated_dca = base_amount * total_multiplier

            self.event_log.info(
                "dca_amount_breakdown",
                pair=trade.pair,
                base_amount=f"{base_amount:.2f}",
                type_multiplier=f"{type_multiplier:.2f}x",
                confidence_multiplier=f"{confidence_multiplier:.2f}x",
                market_multiplier=f"{market_multiplier:.2f}x",
                decay_multiplier=f"{entry_decay:.2f}x",
                leverage_factor=f"{leverage_factor:.2f}x",
                total_multiplier=f"{total_multiplier:.2f}x",
                calculated_amount=f"{calculated_dca:.2f}",
            )
            
            # === 应用限制 ===
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            # 🎯 应用 tradable_balance_ratio
            tradable_ratio = self.config.get('tradable_balance_ratio', 0.99)
            tradable_balance = available_balance * tradable_ratio

            # === 杠杆调整的DCA限制 ===
            # 高杠杆时更严格的DCA限制
            leverage_max_ratios = {
                'low': 0.15 * leverage_factor,      # 低风险
                'medium': 0.10 * leverage_factor,   # 中等风险
                'high': 0.05 * leverage_factor      # 高风险
            }

            max_ratio = leverage_max_ratios.get(dca_decision['risk_level'], 0.05 * leverage_factor)
            max_dca_amount = tradable_balance * max_ratio
            
            final_dca = min(calculated_dca, max_dca_amount)
            
            # 确保最小金额
            min_stake = getattr(self, 'minimal_roi', {}).get('minimal_stake', 10)
            final_dca = max(min_stake, final_dca)
            
            self.event_log.info(
                "dca_amount_final",
                pair=trade.pair,
                amount=f"{final_dca:.2f}",
                limit=f"{max_dca_amount:.2f}",
                leverage=f"{leverage}x",
            )
            
            return final_dca
            
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
            # 🎯 应用 tradable_balance_ratio
            tradable_ratio = self.config.get('tradable_balance_ratio', 0.99)
            tradable_balance = available_balance * tradable_ratio

            total_exposure = trade.stake_amount + dca_amount
            exposure_ratio = total_exposure / tradable_balance

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
            
            # 5. 最小金额检查（不足则自动提升至阈值，而非拒绝）
            ratio = getattr(self, 'min_meaningful_dca_ratio', 0.2)
            min_meaningful_dca = trade.stake_amount * max(0.0, float(ratio))
            if risk_check['adjusted_amount'] < min_meaningful_dca:
                risk_check['risk_factors'].append('提升到最小有效金额')
                risk_check['adjusted_amount'] = min_meaningful_dca
                risk_check['reason'] = f'提升DCA金额至最小有效金额${min_meaningful_dca:.2f}'
            
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
            
            self._log_message(dca_log, importance="summary")
            
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
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        🎯 Custom Exit - 已禁用

        退出完全由 populate_exit_trend 的多时间框架技术指标控制
        保留此函数仅为框架兼容性
        """
        return None

    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        ✅ 交易退出确认 - 综合功能版

        功能:
        1. 开仓保护期机制 - 防止过早退出
        2. 微盈利保护 - 保护小额利润
        3. 连胜连败统计 - 跟踪交易状态
        4. 内存清理 - 清理跟踪数据防止内存泄漏

        Args:
            pair: 交易对
            trade: 交易对象
            order_type: 订单类型
            amount: 数量
            rate: 价格
            time_in_force: 时间有效性
            exit_reason: 退出原因
            current_time: 当前时间

        Returns:
            bool: True=允许退出, False=拒绝退出
        """
        try:
            # === 1. 开仓保护期检查 ===
            if exit_reason == 'exit_signal':
                # 计算开仓时间差（使用时间框架动态调整保护期）
                timeframe_minutes = {
                    '1m': 1, '3m': 3, '5m': 5, '15m': 15,
                    '30m': 30, '1h': 60, '4h': 240, '1d': 1440
                }.get(self.timeframe, 15)  # 默认15分钟

                # 动态保护期：根据时间框架调整
                # 较小时间框架需要更多K线保护，较大时间框架需要更长时间保护
                if timeframe_minutes <= 5:
                    base_protection = timeframe_minutes * 8  # 8根K线
                elif timeframe_minutes <= 30:
                    base_protection = timeframe_minutes * 6  # 6根K线
                else:
                    base_protection = timeframe_minutes * 4  # 4根K线

                # === 2025升级：浮亏时延长保护期 ===
                # 给趋势更多恢复时间，避免在回调时被震出
                profit_ratio_temp = trade.calc_profit_ratio(rate)
                if profit_ratio_temp < 0:  # 浮亏时延长50%
                    protection_period_minutes = base_protection * 1.5
                else:
                    protection_period_minutes = base_protection

                # 计算开仓时长
                time_since_open = (current_time - trade.open_date_utc).total_seconds() / 60

                # 在保护期内拒绝智能交叉退出
                if time_since_open < protection_period_minutes:
                    self._log_message(
                        f"🛡️ {pair} 开仓保护期内 ({time_since_open:.1f}/{protection_period_minutes:.0f}分钟)，拒绝exit_signal退出",
                        importance="summary"
                    )
                    return False

                # 超出保护期，但如果是微盈利状态，给予额外保护
                profit_ratio_check = trade.calc_profit_ratio(rate)
                if 0 < profit_ratio_check < 0.005:  # 0-0.5%的微盈利
                    self._log_message(
                        f"🛡️ {pair} 微盈利保护 ({profit_ratio_check:.2%})，拒绝exit_signal退出",
                        importance="summary"
                    )
                    return False

                # === 2025智能退出决策：基于价格涨幅的差异化保护 ===
                # 核心思想：价格涨得越多，越不应该因为小回调就退出
                # 场景：100u开仓 → 涨到130u（涨30%）→ 回调到120u（涨20%）→ 不应该退出
                try:
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    if dataframe.empty:
                        return True  # 无数据时允许退出

                    current_candle = dataframe.iloc[-1]

                    # 计算纯价格涨幅（不考虑杠杆、手续费）
                    entry_price = trade.open_rate
                    current_price = rate
                    peak_price = getattr(trade, 'max_rate', current_price)  # 最高价格

                    # 🔧 修复Bug #3: 添加除零保护
                    if entry_price == 0 or current_price == 0 or peak_price == 0:
                        return True  # 价格异常时允许退出

                    # 做多：计算价格涨幅
                    if not trade.is_short:
                        current_gain = (current_price / entry_price - 1) * 100  # 当前涨幅
                        peak_gain = (peak_price / entry_price - 1) * 100  # 峰值涨幅
                        pullback_from_peak = (peak_price - current_price) / peak_price * 100  # 从峰值回撤
                    # 做空：计算价格跌幅
                    else:
                        current_gain = (entry_price / current_price - 1) * 100  # 当前跌幅
                        peak_gain = (entry_price / peak_price - 1) * 100  # 峰值跌幅
                        pullback_from_peak = (current_price - peak_price) / current_price * 100  # 从峰值反弹

                    # === 策略1：大涨幅保护（当前涨幅>15%） ===
                    if current_gain > 15:
                        # 峰值涨幅更高时，允许更大回撤
                        if peak_gain > 25:  # 曾经涨过25%
                            # 回撤<10%时不退出（保护大利润）
                            if pullback_from_peak < 10:
                                self._log_message(
                                    f"🛡️ {pair} 大涨幅保护 (开仓{entry_price:.2f}, 当前{current_price:.2f}, 涨{current_gain:.1f}%, 峰值涨{peak_gain:.1f}%, 回撤{pullback_from_peak:.1f}%<10%), 不退出",
                                    importance="summary"
                                )
                                return False
                        elif peak_gain > 20:  # 曾经涨过20%
                            # 回撤<8%时不退出
                            if pullback_from_peak < 8:
                                self._log_message(
                                    f"🛡️ {pair} 大涨幅保护 (涨{current_gain:.1f}%, 峰值涨{peak_gain:.1f}%, 回撤{pullback_from_peak:.1f}%<8%), 不退出",
                                    importance="summary"
                                )
                                return False
                        else:
                            # 普通情况：回撤<5%不退出
                            if pullback_from_peak < 5:
                                self._log_message(
                                    f"🛡️ {pair} 涨幅保护 (涨{current_gain:.1f}%, 回撤{pullback_from_peak:.1f}%<5%), 不退出",
                                    importance="summary"
                                )
                                return False

                    # === 策略2：中等涨幅（5-15%） ===
                    elif 5 < current_gain <= 15:
                        # 回撤<4%时不退出（保护中等利润）
                        if pullback_from_peak < 4:
                            self._log_message(
                                f"🛡️ {pair} 中等涨幅保护 (涨{current_gain:.1f}%, 回撤{pullback_from_peak:.1f}%<4%), 不退出",
                                importance="summary"
                            )
                            return False

                    # === 策略3：小涨幅（2-5%） ===
                    elif 2 < current_gain <= 5:
                        # 回撤<2%时不退出
                        if pullback_from_peak < 2:
                            self._log_message(
                                f"🛡️ {pair} 小涨幅保护 (涨{current_gain:.1f}%, 回撤{pullback_from_peak:.1f}%<2%), 不退出",
                                importance="summary"
                            )
                            return False

                    # === 策略4：微涨幅/平手（-2% ~ 2%） ===
                    elif -2 < current_gain <= 2:
                        # 需要价格确实在连续下跌才退出（连续3根K线）
                        if len(dataframe) >= 3:
                            close_1 = dataframe.iloc[-1]['close']
                            close_2 = dataframe.iloc[-2]['close']
                            close_3 = dataframe.iloc[-3]['close']

                            # 如果价格并未连续下跌，给更多时间
                            if not (close_1 < close_2 < close_3):
                                self._log_message(
                                    f"🛡️ {pair} 微涨幅保护 (涨{current_gain:.1f}%), 价格未连续下跌, 不退出",
                                    importance="summary"
                                )
                                return False

                    # === 策略5：浮亏（<-2%） ===
                    elif current_gain < -2:
                        # 浮亏时要求价格真的破位才退出

                        # 检查是否破位EMA50
                        if 'ema_50' in current_candle:
                            ema_50 = current_candle['ema_50']
                            if not trade.is_short and current_price > ema_50:
                                self._log_message(
                                    f"🛡️ {pair} 浮亏保护 (跌{abs(current_gain):.1f}%), 价格仍在EMA50({ema_50:.2f})上方, 不退出",
                                    importance="summary"
                                )
                                return False
                            elif trade.is_short and current_price < ema_50:
                                self._log_message(
                                    f"🛡️ {pair} 浮亏保护 (做空跌{abs(current_gain):.1f}%), 价格仍在EMA50下方, 不退出",
                                    importance="summary"
                                )
                                return False

                        # 检查量能（低量下跌可能是假突破）
                        volume = current_candle.get('volume', 0)
                        volume_ma = current_candle.get('volume_ma_20', 1)
                        volume_ratio = volume / volume_ma if volume_ma > 0 else 0

                        if volume_ratio < 0.8:  # 缩量下跌
                            self._log_message(
                                f"🛡️ {pair} 浮亏保护 (跌{abs(current_gain):.1f}%), 缩量下跌({volume_ratio:.2f}x), 不退出",
                                importance="summary"
                            )
                            return False

                except Exception as e:
                    logger.warning(f"智能退出决策失败 {pair}: {e}")
                    # 出错时允许退出（避免卡住）
                    pass

            # === 2. 计算交易盈亏并更新统计 ===
            profit_ratio = trade.calc_profit_ratio(rate)

            # 更新连胜连败计数
            if profit_ratio > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self._log_message(
                    f"🏆 {pair} 盈利交易，连胜: {self.consecutive_wins}",
                    importance="summary"
                )
            else:
                self.consecutive_wins = 0
                self.consecutive_losses += 1
                self._log_message(
                    f"❌ {pair} 亏损交易，连败: {self.consecutive_losses}",
                    importance="summary"
                )

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

            # 保持历史记录在合理范围内（最多保留500条）
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]

            # === 3. 清理跟踪数据（防止内存泄漏） ===
            trade_key = f"{pair}_{trade.id}"

            # 清理分批止盈跟踪数据
            if hasattr(self, '_profit_taking_tracker') and trade_key in self._profit_taking_tracker:
                del self._profit_taking_tracker[trade_key]

            # 🔧 修复Bug #2: 清理新增的分批止盈和利润保护跟踪数据
            if hasattr(self, '_profit_exits') and trade_key in self._profit_exits:
                del self._profit_exits[trade_key]

            if hasattr(self, '_profit_protection') and trade_key in self._profit_protection:
                del self._profit_protection[trade_key]

            # 清理智能跟踪止损状态
            if hasattr(self, '_trailing_stop_state') and trade_key in self._trailing_stop_state:
                del self._trailing_stop_state[trade_key]

            # 注意：不清理 _current_leverage[pair]，因为同一交易对可能有多个交易

            self._log_message(f"🧹 清理交易数据 {pair}: 跟踪数据已清理", importance="summary")

        except Exception as e:
            logger.error(f"交易退出确认失败 {pair}: {e}")

        return True  # 确认退出
    
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

            leverage = getattr(trade, 'leverage', None) or getattr(self, '_current_leverage', {}).get(pair, 1)
            leverage = max(leverage, 1)
            fee_open_rate, fee_close_rate = self._get_trade_fee_rates(trade)
            fee_ratio_total = (fee_open_rate + fee_close_rate) * leverage
            slippage_allowance = self._calc_slippage_allowance(leverage)
            buffer_ratio = fee_ratio_total + slippage_allowance
            
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
            base_targets = {
                'level_1': {'target': base_distance * 0.6, 'close': 0.25, 'desc': '快速获利'},
                'level_2': {'target': base_distance * 1.0, 'close': 0.35, 'desc': '主要获利'},
                'level_3': {'target': base_distance * 1.6, 'close': 0.25, 'desc': '趋势延伸'},
                'level_4': {'target': base_distance * 2.5, 'close': 0.15, 'desc': '超预期收益'}
            }

            targets: dict[str, dict[str, float]] = {}

            # 计算实际价格目标
            for level_name, base_data in base_targets.items():
                target_profit = base_data.get('target')
                close_pct = base_data.get('close', 0.0)
                if target_profit is None or close_pct <= 0:
                    continue

                if not trade.is_short:
                    target_price = trade.open_rate * (1 + target_profit)
                else:
                    target_price = trade.open_rate * (1 - target_profit)

                price_ratio = self._price_ratio(trade.open_rate, target_price, trade.is_short)
                account_ratio = self._account_ratio_from_price(price_ratio, leverage, buffer_ratio)

                if account_ratio <= 0:
                    # 若扣除费用后无净利润，则跳过该档
                    continue

                targets[level_name] = {
                    'price': target_price,
                    'close': close_pct,
                    'desc': base_data.get('desc', ''),
                    'price_ratio': price_ratio,
                    'target_profit': account_ratio,
                    'profit_pct': account_ratio * 100,
                    'price_pct': price_ratio * 100
                }

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
                
                current_data = dataframe.iloc[-1]

                # 记录入场信号概况
                normalized_side = side.lower() if isinstance(side, str) else ''
                if normalized_side in ('buy', 'long'):
                    trade_side = 'long'
                elif normalized_side in ('sell', 'short'):
                    trade_side = 'short'
                else:
                    trade_side = 'long'
                    logger.warning(f"未知的交易方向 {side}, 默认按long处理: {pair}")

                self._log_entry_signal(pair, trade_side, current_data)

                # === 🎯 预计算价格目标和风控参数 ===
                # 获取当前杠杆
                current_leverage = getattr(self, '_current_leverage', {}).get(pair, 10)

                # 注意：跟踪止损参数现在在 custom_stoploss 中动态计算，不需要预设

                # 计算ATR价格值
                atr_value = current_atr_p * rate
                
                # 获取所有杠杆调整后的参数
                is_short_trade = normalized_side in ('sell', 'short')

                leverage_params = self.calculate_leverage_adjusted_params(
                    leverage=current_leverage,
                    atr_value=atr_value,
                    entry_price=rate,
                    is_short=is_short_trade
                )
                
                # 存储到交易元数据（供后续使用）
                if not hasattr(self, '_trade_targets'):
                    self._trade_targets = {}
                
                self._trade_targets[pair] = {
                    'entry_price': rate,
                    'entry_time': current_time,
                    'leverage': current_leverage,
                    'side': side,
                    'leverage_params': leverage_params,
                    'stop_loss_price': leverage_params['stop_loss']['price'],
                    'take_profit_prices': leverage_params['take_profit'],
                    'dca_levels': leverage_params['dca']['price_levels'],
                    'trailing_activation': leverage_params['trailing_stop']['activation_price']
                }
                
                # === 🔥 详细日志输出（显示所有价格目标）===
                self._log_trade_entry_targets(pair, rate, leverage_params)
            
            
            self._log_message(
                f"交易确认通过: {pair} {side} {amount} @ {rate}",
                importance="summary"
            )
            return True
            
        except Exception as e:
            logger.error(f"交易确认失败: {e}")
            return False

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
                return float(min(2, int(max_leverage)))
            
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
            # 🔧 修复: 使用实际市场状态而非硬编码'sideways'
            market_state = regime.lower() if regime else 'sideways'
            base_leverage = self.calculate_leverage(market_state, volatility, pair, current_time)
            
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
            
            final_leverage = max(1, int(safe_leverage))  # 最低1倍整数杠杆
            
            # === 🔧 修复: 同步杠杆数据流，确保与custom_stoploss一致 ===
            if not hasattr(self, '_current_leverage'):
                self._current_leverage = {}
            self._current_leverage[pair] = final_leverage
            
            # === 8. 详细日志 ===
            self._log_message(
                f"🎯 智能杠杆 {pair} [{entry_tag}]: "
                f"基础{base_leverage}x × "
                f"状态{regime_multiplier:.2f} × "
                f"信号{signal_multiplier:.2f} × "
                f"质量{signal_quality_bonus:.2f} = "
                f"{calculated_leverage:.1f}x → {final_leverage}x | "
                f"市场:{regime} ({regime_confidence:.1%})",
                importance="summary"
            )
            
            return float(final_leverage)
            
        except Exception as e:
            logger.error(f"杠杆计算失败 {pair}: {e}")
            return float(min(2, int(max_leverage)))  # 出错时返回安全整数杠杆
    
    def leverage_update_callback(self, trade: Trade, **kwargs):
        """杠杆更新回调"""
        try:
            new_leverage = kwargs.get('new_leverage')
            if new_leverage is None:
                new_leverage = getattr(trade, 'leverage', None)
            if new_leverage is None:
                return

            try:
                target_leverage = int(round(float(new_leverage)))
            except Exception:
                return

            if not hasattr(self, '_current_leverage'):
                self._current_leverage = {}

            previous = self._current_leverage.get(trade.pair)
            if previous == target_leverage:
                return

            self._current_leverage[trade.pair] = target_leverage
            self.event_log.info(
                "leverage_update",
                importance="summary",
                pair=trade.pair,
                leverage=f"{target_leverage}x",
            )
        except Exception as exc:
            logger.warning(f"杠杆更新回调失败 {trade.pair}: {exc}")
    
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
    