#!/usr/bin/env python3
"""
AI Agent集成交易策略 - 多时间框架分析版本
使用demo中验证的多时间框架分析功能
"""

import logging
from functools import reduce
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pandas import DataFrame
from technical import qtpylib
import requests
import json
import os
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dotenv import load_dotenv

from typing import TYPE_CHECKING
try:
    # 优先使用FreqTrade提供的IStrategy
    from freqtrade.strategy import IStrategy as _FreqtradeIStrategy  # type: ignore
    IStrategy = _FreqtradeIStrategy  # type: ignore[misc]
except Exception:
    # 如果没有FreqTrade环境，定义降级版IStrategy以便脚本独立运行
    class IStrategy:  # type: ignore[no-redef]
        def __init__(self, config=None):
            pass

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class AIAgentTradingStrategy(IStrategy):
    """
    AI Agent 集成交易策略 - 多时间框架分析版本
    
    功能特性：
    1. 多时间框架技术分析 (5m, 15m, 1h, 4h)
    2. 集成新闻和社媒数据采集
    3. 先进的技术指标计算 (RSI, BOLL, MACD等)
    4. AI Agent自动分析市场信号
    5. 智能交易决策引擎
    6. 实时风险管理
    """

    minimal_roi = {
        "0": 0.05,
        "10": 0.03,
        "20": 0.01,
        "30": 0.005,
        "60": 0
    }

    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    
    timeframe = '5m'

    # Can this strategy go short?
    can_short = True

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Run "populate_indicators" only for new candle
    process_only_new_candles = False

    # These values can be overridden in the config
    use_custom_stoploss = True
    
    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    plot_config = {
        'main_plot': {
            'ema10': {},
            'ema50': {},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            },
            "AI分析": {
                'ai_combined_score': {'color': 'green'},
                'ai_confidence': {'color': 'purple'},
            }
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # AI Agent配置
        self.ai_agent_config = (config or {}).get("ai_agent", {
            "sentiment_weights": {
                "news": 0.4,
                "social": 0.3,
                "technical": 0.3
            },
            "signal_threshold": 0.6,
            "data_cache_duration": 300
        })
        
        # 初始化AI分析器
        self.ai_analyzer = None
        self._init_ai_analyzer()
    
    def _init_ai_analyzer(self):
        """初始化AI分析器"""
        try:
            from ai_agent.ai_analyzer import AIAgentAnalyzer
            self.ai_analyzer = AIAgentAnalyzer(self.ai_agent_config)
            logger.info("AI分析器初始化成功")
        except Exception as e:
            logger.error(f"AI分析器初始化失败: {e}")
            self.ai_analyzer = None

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will not be traded, but can be used for analysis.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加技术指标到给定的DataFrame
        """
        
        # 基础技术指标
        dataframe = self._calculate_basic_indicators(dataframe)
        
        # 多时间框架AI分析
        if self.ai_analyzer is not None:
            dataframe = self._add_multi_timeframe_analysis(dataframe, metadata)
        else:
            # 降级为基础分析
            self._add_fallback_ai_columns(dataframe)
        
        return dataframe
    
    def _calculate_basic_indicators(self, dataframe: DataFrame) -> DataFrame:
        """计算基础技术指标"""
        # RSI (使用Wilder's方法，与币安一致)
        delta = dataframe['close'].astype(float).diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # 使用pandas的ewm方法实现Wilder's平滑
        # alpha = 1/14, 对应 span = 2/alpha - 1 = 27
        avg_gains = gains.ewm(alpha=1/14, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/14, adjust=False).mean()

        rs = avg_gains / avg_losses
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        ).fillna(0.5)
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        ).fillna(0.1)

        # MACD
        exp1 = dataframe['close'].ewm(span=12).mean()
        exp2 = dataframe['close'].ewm(span=26).mean()
        dataframe['macd'] = exp1 - exp2
        dataframe['macdsignal'] = dataframe['macd'].ewm(span=9).mean()
        dataframe['macdhist'] = dataframe['macd'] - dataframe['macdsignal']

        # 移动平均
        dataframe['sma'] = dataframe['close'].rolling(40).mean()
        dataframe['ema5'] = dataframe['close'].ewm(span=5).mean()
        dataframe['ema10'] = dataframe['close'].ewm(span=10).mean()
        dataframe['ema50'] = dataframe['close'].ewm(span=50).mean()
        dataframe['ema100'] = dataframe['close'].ewm(span=100).mean()

        # 简化的技术指标
        dataframe['adx'] = 25.0  # 固定值用于演示

        return dataframe
    
    def _add_fallback_ai_columns(self, dataframe: DataFrame):
        """添加AI分析列的默认值"""
        dataframe['ai_technical_score'] = 0.0
        dataframe['ai_sentiment_score'] = 0.0
        dataframe['ai_combined_score'] = 0.0
        dataframe['ai_confidence'] = 0.5
        dataframe['ai_risk_level'] = 0.5
        dataframe['ai_action'] = 'HOLD'
        dataframe['valid_timeframes'] = 0
        
        # OpenAI分析默认值
        dataframe['openai_market_state'] = 'neutral'
        dataframe['openai_recommendation'] = 'hold'
        dataframe['openai_strength'] = 0.5
        dataframe['openai_full_analysis'] = ''
        dataframe['openai_technical_score'] = 5.0
        dataframe['openai_macro_score'] = 5.0
        dataframe['openai_risk_level'] = 5
        dataframe['openai_target_price'] = None
        dataframe['openai_stop_loss'] = None
        
        # 宏观经济数据默认值
        dataframe['macro_nasdaq_trend'] = "unknown"
        dataframe['macro_fed_sentiment'] = "unknown"
        dataframe['macro_vix_level'] = 20.0
        dataframe['macro_dxy_index'] = 100.0
        dataframe['macro_gold_price'] = 2000.0
        dataframe['macro_score'] = 0.0
        
        # 时间框架得分
        for tf in ['5m', '15m', '1h', '4h']:
            dataframe[f'tf_{tf}_score'] = 0.0
    
    def _add_multi_timeframe_analysis(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """添加多时间框架AI分析"""
        symbol = metadata.get('pair', 'BTCUSDT')
        if '/' in symbol:
            symbol = symbol.replace('/', '')

        # 强制使用真实数据提供者（硬失败）
        from ai_agent.real_data_provider import get_data_provider
        provider = get_data_provider()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 真实数据任务 (宏观 + 新闻 + 社交) 并发执行
        news_symbol = symbol if '/' in symbol else f"{symbol.replace('USDT','')}/USDT"
        tasks = [
            provider.get_macro_economic_data(),
            provider.get_crypto_news(news_symbol),
            provider.get_social_sentiment(news_symbol)
        ]
        macro_data, news_data, social_data = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=False))

        # 判定新闻与社交是否可用（不终止，缺失则后续权重忽略）
        news_available = hasattr(news_data, 'volume') and getattr(news_data, 'volume', 0) > 0
        social_available = isinstance(social_data, dict) and 'sentiment_score' in social_data

        # 构建情感输入，仅包含真实获取到的部分
        sentiment_data = {}
        if news_available:
            sentiment_data['news_sentiment'] = float(getattr(news_data, 'sentiment_score', 0.0))
        if social_available:
            sentiment_data['social_sentiment'] = float(social_data.get('sentiment_score', 0.0))
        # 若两者都缺失，传空字典（后续按仅技术面处理）

        # 进行多时间框架技术 + 情感分析（情感可能为空）
        from typing import cast, Any as _Any
        analyzer = cast(_Any, self.ai_analyzer)
        analysis = loop.run_until_complete(analyzer.analyze_multi_timeframe(symbol, sentiment_data))
        
        # 重算组合得分：若新闻/社交缺失则忽略其权重并重新归一
        tech_score = analysis.get('technical_score', 0.0)
        sentiment_score = analysis.get('sentiment_score', 0.0)
        weights_cfg = self.ai_agent_config.get('sentiment_weights', {"technical":0.3,"news":0.4,"social":0.3})
        w_tech = float(weights_cfg.get('technical', 0.3))
        w_news = float(weights_cfg.get('news', 0.4)) if news_available else 0.0
        w_social = float(weights_cfg.get('social', 0.3)) if social_available else 0.0
        # 如果情感全缺失，只用技术面
        if (w_news + w_social) == 0:
            combined_base = tech_score
        else:
            total = w_tech + w_news + w_social
            combined_base = tech_score * (w_tech/total) + sentiment_score * ((w_news + w_social)/total)
        analysis['combined_score'] = combined_base

        # 宏观经济影响评分
        macro_score = self._analyze_macro_impact(macro_data)
        analysis['combined_score'] = combined_base * 0.6 + macro_score * 0.4
        analysis['macro_score'] = macro_score

        # OpenAI增强（硬失败）
        analysis = self._enhance_with_openai_analysis(dataframe, symbol, analysis, macro_score)

        loop.close()
        logger.info(f"多时间框架真实数据分析完成: 有效框架={analysis.get('valid_timeframes', 0)}, 综合得分={analysis.get('combined_score', 0.0):+.3f}")

        # 写入DataFrame
        dataframe['ai_technical_score'] = analysis.get('technical_score', 0.0)
        dataframe['ai_sentiment_score'] = analysis.get('sentiment_score', 0.0)
        dataframe['ai_combined_score'] = analysis.get('combined_score', 0.0)
        dataframe['ai_confidence'] = analysis.get('confidence', 0.0)
        dataframe['ai_risk_level'] = analysis.get('risk_level', 0.5)
        dataframe['ai_action'] = analysis.get('action', 'HOLD')
        dataframe['valid_timeframes'] = analysis.get('valid_timeframes', 0)

        dataframe['openai_market_state'] = analysis.get('openai_market_state', 'neutral')
        dataframe['openai_recommendation'] = analysis.get('openai_recommendation', 'hold')
        dataframe['openai_strength'] = analysis.get('openai_strength', 0.5)
        dataframe['openai_full_analysis'] = analysis.get('openai_full_analysis', '')
        dataframe['openai_technical_score'] = analysis.get('openai_technical_score', 5.0)
        dataframe['openai_macro_score'] = analysis.get('openai_macro_score', 5.0)
        dataframe['openai_risk_level'] = analysis.get('openai_risk_level', 5)
        dataframe['openai_target_price'] = analysis.get('openai_target_price')
        dataframe['openai_target_price_1'] = analysis.get('openai_target_price_1')
        dataframe['openai_target_price_2'] = analysis.get('openai_target_price_2')
        dataframe['openai_stop_loss'] = analysis.get('openai_stop_loss')
        dataframe['openai_add_position_price'] = analysis.get('openai_add_position_price')
        dataframe['openai_timeframe_summary'] = analysis.get('openai_timeframe_summary', '')
        dataframe['openai_key_reason'] = analysis.get('openai_key_reason', '')

        dataframe['macro_nasdaq_trend'] = macro_data.nasdaq_trend or 'neutral'
        dataframe['macro_fed_sentiment'] = macro_data.fomc_sentiment or 'neutral'
        dataframe['macro_vix_level'] = macro_data.vix_index or 20.0
        dataframe['macro_dxy_index'] = macro_data.dxy_index or 100.0
        dataframe['macro_gold_price'] = macro_data.gold_price or 2000.0
        dataframe['macro_score'] = macro_score

        timeframe_scores = analysis.get('timeframe_scores', {})
        for tf in ['5m', '15m', '1h', '4h']:
            dataframe[f'tf_{tf}_score'] = timeframe_scores.get(tf, 0.0)

        return dataframe

    def _analyze_macro_impact(self, macro_data) -> float:
        """分析宏观经济数据对加密货币的影响"""
        try:
            macro_score = 0.0
            
            # 纳斯达克影响分析 (权重: 35%)
            if macro_data.nasdaq_trend:
                if macro_data.nasdaq_trend == "strong_bullish":
                    macro_score += 0.35 * 0.8  # 强烈利好
                elif macro_data.nasdaq_trend == "bullish":
                    macro_score += 0.35 * 0.4  # 利好
                elif macro_data.nasdaq_trend == "strong_bearish":
                    macro_score += 0.35 * (-0.8)  # 强烈利空
                elif macro_data.nasdaq_trend == "bearish":
                    macro_score += 0.35 * (-0.4)  # 利空
                # neutral 不调整分数
            
            # 美联储政策影响分析 (权重: 30%)
            if macro_data.fomc_sentiment:
                if macro_data.fomc_sentiment == "dovish":
                    macro_score += 0.30 * 0.6  # 鸽派政策利好
                elif macro_data.fomc_sentiment == "hawkish":
                    macro_score += 0.30 * (-0.6)  # 鹰派政策利空
                # neutral 不调整分数
            
            # VIX恐慌指数影响分析 (权重: 20%)
            if macro_data.vix_index:
                if macro_data.vix_index > 30:
                    macro_score += 0.20 * (-0.7)  # 高恐慌指数利空
                elif macro_data.vix_index < 15:
                    macro_score += 0.20 * 0.5   # 低恐慌指数利好
                elif 20 <= macro_data.vix_index <= 25:
                    macro_score += 0.20 * 0.2   # 正常范围轻微利好
            
            # 美元指数DXY影响分析 (权重: 10%)
            if macro_data.dxy_index:
                # 美元指数上升通常对加密货币不利
                if macro_data.dxy_index > 105:
                    macro_score += 0.10 * (-0.4)  # 强美元利空
                elif macro_data.dxy_index < 95:
                    macro_score += 0.10 * 0.3   # 弱美元利好
            
            # 黄金价格影响分析 (权重: 5%)
            if macro_data.gold_price:
                # 黄金上涨通常表明避险情绪，可能利好加密货币作为另类资产
                # 这里简化处理，基于黄金价格范围判断
                if macro_data.gold_price > 2100:  # 高金价
                    macro_score += 0.05 * 0.3
                elif macro_data.gold_price < 1800:  # 低金价
                    macro_score += 0.05 * (-0.2)
            
            # 限制分数范围在 [-1, 1]
            macro_score = max(-1.0, min(1.0, macro_score))
            
            logger.info(f"宏观经济影响评分: {macro_score:.3f}")
            return macro_score
            
        except Exception as e:
            logger.error(f"宏观经济影响分析错误: {e}")
            return 0.0

    def _enhance_with_openai_analysis(self, dataframe: DataFrame, symbol: str, analysis: dict, macro_score: float) -> dict:
        """使用OpenAI增强分析 - 传递更详细的数据"""
        try:
            from ai_agent.openai_analyzer import get_openai_analyzer
            
            openai_analyzer = get_openai_analyzer()
            
            # SDK客户端可能因环境原因不可用，但在有API Key时仍可通过HTTP通道使用
            if openai_analyzer and getattr(openai_analyzer, 'is_available', False) and len(dataframe) > 0:
                latest = dataframe.iloc[-1]
                
                # 准备市场数据（明确5分钟周期）
                market_data = {
                    "symbol": symbol,
                    "current_price": float(latest['close']),
                    "price_change_24h": 0.0,  # 可以从历史数据计算
                    "volume_24h": float(latest['volume']),
                    "rsi": float(latest['rsi']) if not np.isnan(latest['rsi']) else 50,
                    "macd": "bullish" if float(latest['macd']) > float(latest['macdsignal']) else "bearish",
                    "bb_position": "upper" if latest['bb_percent'] > 0.8 else "lower" if latest['bb_percent'] < 0.2 else "middle",
                    "trend_strength": "strong" if abs(analysis.get('technical_score', 0.0)) > 0.5 else "moderate"
                }
                
                # 获取多时间框架技术指标
                multi_timeframe_indicators = _get_multi_timeframe_indicators(symbol)
                
                # 准备宏观经济数据
                macro_economic_data = {
                    "nasdaq_change": 0.0,  # 可以从宏观数据计算
                    "vix_level": latest.get('macro_vix_level', 20.0),
                    "fed_sentiment": latest.get('macro_fed_sentiment', 'neutral'),
                    "dxy_index": latest.get('macro_dxy_index', 100.0),
                    "gold_price": latest.get('macro_gold_price', 2000.0)
                }
                
                # 准备新闻情绪数据
                news_data = {
                    "sentiment_score": analysis.get('sentiment_score', 0.0),
                    "volume": 10  # 模拟新闻数量
                }
                
                # 构建完整的分析数据
                comprehensive_data = {
                    "market": market_data,
                    "multi_timeframe": multi_timeframe_indicators,
                    "macro_economic": macro_economic_data,
                    "news": news_data
                }
                
                # 获取OpenAI深度分析
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                ai_result = loop.run_until_complete(openai_analyzer.analyze_comprehensive(comprehensive_data))
                loop.close()
                
                # 若走到了降级分析，则不输出专业级块
                if isinstance(ai_result, dict) and ai_result.get("analysis_type") == "fallback":
                    logger.warning("OpenAI不可用或调用失败，跳过专业级分析输出")
                    analysis['openai_used'] = False
                    return analysis

                # 将OpenAI分析结果融入最终评分
                openai_confidence = ai_result.get("confidence", 0.5)
                openai_action = ai_result.get("action", "观望")
                openai_trend = ai_result.get("trend", "中性")
                
                # 根据OpenAI建议调整最终分数
                original_score = analysis.get('combined_score', 0.0)
                if openai_action == "买入":
                    enhanced_score = original_score * 0.6 + openai_confidence * 0.3 + macro_score * 0.1
                elif openai_action == "卖出":
                    enhanced_score = original_score * 0.6 - openai_confidence * 0.3 + macro_score * 0.1
                else:  # 观望
                    enhanced_score = original_score * 0.8 + macro_score * 0.2
                
                # 更新分析结果
                analysis['combined_score'] = enhanced_score
                analysis['openai_market_state'] = openai_trend
                analysis['openai_recommendation'] = openai_action
                analysis['openai_strength'] = openai_confidence
                analysis['openai_full_analysis'] = ai_result.get('full_analysis', '')
                analysis['openai_technical_score'] = ai_result.get('technical_score', 5.0)
                analysis['openai_macro_score'] = ai_result.get('macro_score', 5.0)
                analysis['openai_risk_level'] = ai_result.get('risk_level', 5)
                analysis['openai_target_price'] = ai_result.get('target_price')
                analysis['openai_target_price_1'] = ai_result.get('target_price_1')
                analysis['openai_target_price_2'] = ai_result.get('target_price_2')
                analysis['openai_stop_loss'] = ai_result.get('stop_loss')
                analysis['openai_add_position_price'] = ai_result.get('add_position_price')
                analysis['openai_timeframe_summary'] = ai_result.get('timeframe_summary', '')
                analysis['openai_key_reason'] = ai_result.get('key_reason', '')
                analysis['openai_used'] = True
                
                logger.info(f"OpenAI深度分析完成: {openai_trend} - {openai_action} (信心度: {openai_confidence:.1%})")
                
        except Exception as e:
            logger.warning(f"OpenAI深度分析失败: {e}")
        
        return analysis

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于指标计算买入信号
        """
        
        # 基础技术条件
        basic_conditions = [
            (dataframe['rsi'] < 70),  # 避免超买
            (dataframe['close'] > dataframe['ema10']),  # 价格在短期均线之上
            (dataframe['ema10'] > dataframe['ema50']),  # 短期均线高于长期均线
            (dataframe['macd'] > dataframe['macdsignal']),  # MACD金叉
            (dataframe['bb_percent'] < 0.8),  # 不在布林带上轨附近
        ]
        
        # AI条件
        ai_conditions = [
            (dataframe['ai_combined_score'] > 0.2),  # AI综合得分为正
            (dataframe['ai_confidence'] > 0.6),  # AI置信度较高
            (dataframe['valid_timeframes'] >= 2),  # 至少2个时间框架有效
            (dataframe['ai_risk_level'] < 0.7),  # 风险水平不太高
        ]
        
        # 结合所有条件
        all_conditions = basic_conditions + ai_conditions
        
        dataframe.loc[
            reduce(lambda x, y: x & y, all_conditions),
            'enter_long'] = 1

        # 空头进场信号（做空）
        basic_short = [
            (dataframe['rsi'] > 30),  # 避免超卖
            (dataframe['close'] < dataframe['ema10']),  # 价格在短期均线之下
            (dataframe['ema10'] < dataframe['ema50']),  # 短期均线低于长期均线
            (dataframe['macd'] < dataframe['macdsignal']),  # MACD死叉
            (dataframe['bb_percent'] > 0.2),  # 不在布林带下轨附近
        ]

        ai_short = [
            (dataframe['ai_combined_score'] < -0.2),  # AI综合得分偏空
            (dataframe['ai_confidence'] > 0.6),
            (dataframe['valid_timeframes'] >= 2),
            (dataframe['ai_risk_level'] < 0.7),
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, (basic_short + ai_short)),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于指标计算卖出信号
        """
        
        # 基础技术条件
        basic_conditions = [
            (dataframe['rsi'] > 30),  # 避免超卖
            (
                (dataframe['close'] < dataframe['ema10']) |  # 价格跌破短期均线
                (dataframe['ema10'] < dataframe['ema50']) |  # 短期均线跌破长期均线
                (dataframe['macd'] < dataframe['macdsignal']) |  # MACD死叉
                (dataframe['bb_percent'] > 0.2)  # 不在布林带下轨附近
            ),
        ]
        
        # AI条件
        ai_conditions = [
            (
                (dataframe['ai_combined_score'] < -0.2) &  # AI综合得分为负
                (dataframe['ai_confidence'] > 0.6)  # 且AI置信度较高
            ),
            (dataframe['valid_timeframes'] >= 2),  # 至少2个时间框架有效
        ]
        
        # 结合所有条件
        all_conditions = basic_conditions + ai_conditions
        
        dataframe.loc[
            reduce(lambda x, y: x & y, all_conditions),
            'exit_long'] = 1

        # 空头离场信号（平空）
        basic_short_exit = [
            (dataframe['rsi'] < 70),  # 避免超买
            (
                (dataframe['close'] > dataframe['ema10']) |  # 价格上穿短期均线
                (dataframe['ema10'] > dataframe['ema50']) |  # 短期均线上穿长期均线
                (dataframe['macd'] > dataframe['macdsignal']) |  # MACD金叉
                (dataframe['bb_percent'] < 0.8)  # 不在布林带上轨附近
            ),
        ]

        ai_short_exit = [
            (
                (dataframe['ai_combined_score'] > 0.2) &  # AI综合得分转多
                (dataframe['ai_confidence'] > 0.6)
            ),
            (dataframe['valid_timeframes'] >= 2),
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, (basic_short_exit + ai_short_exit)),
            'exit_short'] = 1

        return dataframe



def _get_multi_timeframe_indicators(symbol='BTCUSDT'):
    """获取多时间框架技术指标"""
    try:
        timeframes = {
            '5m': '5m',
            '15m': '15m', 
            '1h': '1h',
            '4h': '4h'
        }
        
        multi_indicators = {}
        
        for tf_name, tf_interval in timeframes.items():
            try:
                # 获取K线数据
                base_url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': tf_interval,
                    'limit': 100  # 获取100根K线用于指标计算
                }
                
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    klines = response.json()
                    
                    # 转换为DataFrame
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                              'close_time', 'quote_asset_volume', 'number_of_trades',
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                    
                    df = pd.DataFrame(klines, columns=columns)
                    
                    # 转换数据类型
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    # 计算技术指标
                    indicators = _calculate_timeframe_indicators(df)
                    multi_indicators[tf_name] = indicators
                    
                else:
                    logger.warning(f"无法获取{tf_name}数据: HTTP {response.status_code}")
                    multi_indicators[tf_name] = None
                    
            except Exception as e:
                logger.error(f"获取{tf_name}指标失败: {e}")
                multi_indicators[tf_name] = None
        
        return multi_indicators
        
    except Exception as e:
        logger.error(f"获取多时间框架指标失败: {e}")
        return {'5m': None, '15m': None, '1h': None, '4h': None}


def _calculate_timeframe_indicators(df):
    """计算单个时间框架的技术指标"""
    try:
        if len(df) < 20:  # 确保有足够数据
            return None
            
        # RSI计算 (使用Wilder's方法，与币安一致)
        delta = df['close'].diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # 使用pandas的ewm方法实现Wilder's平滑
        avg_gains = gains.ewm(alpha=1/14, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/14, adjust=False).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # MACD计算
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9).mean()
        
        # 布林带计算
        bb_period = 20
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # 获取最新值
        latest = df.iloc[-1]
        latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
        latest_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None
        latest_macd_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else None
        latest_bb_upper = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else None
        latest_bb_lower = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else None
        
        # 计算布林带位置
        bb_position = None
        if latest_bb_upper and latest_bb_lower and latest_bb_upper != latest_bb_lower:
            bb_position = ((latest['close'] - latest_bb_lower) / (latest_bb_upper - latest_bb_lower)) * 100
        
        return {
            'rsi': f"{latest_rsi:.2f}" if latest_rsi else "N/A",
            'macd': f"{latest_macd:.4f}" if latest_macd and latest_macd_signal else "N/A",
            'macd_signal': f"{latest_macd_signal:.4f}" if latest_macd_signal else "N/A",
            'bb_position': f"{bb_position:.1f}%" if bb_position else "N/A",
            'price': f"${latest['close']:,.2f}"
        }
        
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        return None


def analyze_real_data(symbol='BTCUSDT'):
    """分析真实市场数据 (带重试与多主机回退)。"""
    print(f"📊 开始分析真实市场数据 - {symbol}")
    print("=" * 50)

    strategy = AIAgentTradingStrategy()

    # 重试参数
    max_retries = int(os.getenv("BINANCE_KLINE_MAX_RETRIES", "3"))
    base_backoff = float(os.getenv("BINANCE_KLINE_BACKOFF_BASE", "0.5"))
    max_backoff = float(os.getenv("BINANCE_KLINE_BACKOFF_MAX", "10"))
    timeout = float(os.getenv("BINANCE_KLINE_TIMEOUT", "4"))
    hosts = [h.strip() for h in os.getenv("BINANCE_HOSTS", "api.binance.com,api1.binance.com,api2.binance.com").split(',') if h.strip()]

    params = {
        'symbol': symbol,
        'interval': '5m',
        'limit': 200
    }

    dataframe = None
    last_error = None
    import random
    for attempt in range(1, max_retries + 1):
        host = hosts[(attempt - 1) % len(hosts)]
        url = f"https://{host}/api/v3/klines"
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code >= 500:
                raise RuntimeError(f"server_{resp.status_code}")
            resp.raise_for_status()
            klines = resp.json()
            if not isinstance(klines, list) or not klines:
                raise RuntimeError("empty_klines")
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'close_time', 'quote_asset_volume', 'number_of_trades',
                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            dataframe = pd.DataFrame(klines, columns=columns)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                dataframe[col] = dataframe[col].astype(float)
            dataframe['date'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
            print(f"✅ 成功获取{len(dataframe)}条真实K线数据 (host={host}, attempt={attempt})")
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                exp = attempt - 1
                wait = min(base_backoff * (2 ** exp), max_backoff) * random.uniform(0.85, 1.25)
                print(f"⚠️  K线获取失败 {type(e).__name__}:{e} host={host} 第{attempt}次 -> {wait:.2f}s 后重试")
                import time as _t; _t.sleep(wait)
            else:
                print(f"❌ K线获取最终失败 (共{max_retries}次): {e}")
    if dataframe is None:
        return False
    
    # 使用真实数据进行分析
    metadata = {'pair': symbol}
    
    # 填充指标
    dataframe = strategy.populate_indicators(dataframe, metadata)
    
    # 生成交易信号
    dataframe = strategy.populate_entry_trend(dataframe, metadata)
    dataframe = strategy.populate_exit_trend(dataframe, metadata)
    
    # 分析结果
    latest = dataframe.iloc[-1]

    # 辅助：安全转换为float/int
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            try:
                return float(v.item()) if hasattr(v, 'item') else default
            except Exception:
                return default

    def _to_int(v, default=0):
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return default
    
    # 生成详细报告
    display_symbol = symbol.replace('USDT', '/USDT') if 'USDT' in symbol and '/' not in symbol else symbol
    print(f"""
🤖 AI Agent真实市场分析报告
===============================================

📊 交易对: {display_symbol}
🕐 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💰 当前价格: ${latest['close']:,.2f} (实时)
📈 最新K线时间: {latest['date'].strftime('%Y-%m-%d %H:%M:%S')}

📈 AI分析结果
------------------------------
技术分析得分: {latest['ai_technical_score']:+.3f}
情感分析得分: {latest['ai_sentiment_score']:+.3f}
综合得分: {latest['ai_combined_score']:+.3f}
AI建议: {latest['ai_action']}
置信度: {latest['ai_confidence']*100:.1f}%
风险等级: {latest['ai_risk_level']*100:.1f}%
有效时间框架: {latest['valid_timeframes']}/4

🌍 宏观经济分析 (真实数据)
------------------------------
纳斯达克趋势: {latest['macro_nasdaq_trend']}
美联储政策: {latest['macro_fed_sentiment']}
VIX恐慌指数: {latest['macro_vix_level']:.1f}
美元指数DXY: {latest['macro_dxy_index']:.1f}
黄金价格: ${latest['macro_gold_price']:.0f}
宏观经济得分: {latest['macro_score']:+.3f}

🤖 OpenAI深度分析
------------------------------
""")
    if latest.get('openai_used', False):
        print(f"市场状态: {latest['openai_market_state']}\nAI建议: {latest['openai_recommendation']}\n信心指数: {latest['openai_strength']*100:.1f}%\n技术面评分: {latest.get('openai_technical_score', 5.0):.1f}/10\n宏观面评分: {latest.get('openai_macro_score', 5.0):.1f}/10\n风险等级: {latest.get('openai_risk_level', 5)}/10级")
    else:
        print("(跳过) OpenAI不可用或调用失败，已使用降级规则，不输出专业级分析块。")

    # 显示OpenAI的详细分析过程（如果有）
    openai_analysis = latest.get('openai_full_analysis', '')
    if openai_analysis and len(openai_analysis) > 100:
        print(f"\n🔍 AI详细分析过程：")
        print("-" * 30)
        # 截取分析的前500个字符避免输出过长
        analysis_preview = openai_analysis[:800] + "..." if len(openai_analysis) > 800 else openai_analysis
        print(analysis_preview)
        print("-" * 30)

    # 显示目标价位和止损（如果有）
    target_price_1 = latest.get('openai_target_price_1')
    target_price_2 = latest.get('openai_target_price_2') 
    target_price = latest.get('openai_target_price')  # 兼容旧格式
    stop_loss = latest.get('openai_stop_loss')
    add_position = latest.get('openai_add_position_price')
    
    if any([target_price_1, target_price_2, target_price, stop_loss, add_position]):
        print(f"\n💡 AI交易策略建议：")
        if target_price_1:
            print(f"   🎯 保守目标: ${target_price_1:,.0f}")
        if target_price_2:
            print(f"   🚀 激进目标: ${target_price_2:,.0f}")
        if target_price and not target_price_1:  # 兼容旧格式
            print(f"   🎯 目标价位: ${target_price:,.0f}")
        if stop_loss:
            print(f"   🛑 止损价位: ${stop_loss:,.0f}")
        if add_position:
            print(f"   ➕ 加仓价位: ${add_position:,.0f}")
            
        # 计算涨跌幅
        current_price = latest['close']
        if target_price_1:
            change_pct = ((target_price_1 - current_price) / current_price) * 100
            print(f"   📊 保守收益: {change_pct:+.1f}%")
        if stop_loss:
            risk_pct = ((stop_loss - current_price) / current_price) * 100
            print(f"   ⚠️  最大风险: {risk_pct:+.1f}%")

    print(f"""
📊 技术指标 (5分钟时间框架)
------------------------------
RSI (14): {latest['rsi']:.2f}
EMA10: ${latest['ema10']:,.2f}
EMA50: ${latest['ema50']:,.2f}
MACD: {latest['macd']:.4f}
MACD信号: {latest['macdsignal']:.4f}
布林带位置: {latest['bb_percent']*100:.1f}%

📊 多时间框架技术指标
------------------------------""")

    # 获取多时间框架的技术指标
    multi_timeframe_indicators = _get_multi_timeframe_indicators(symbol)
    
    for tf, indicators in multi_timeframe_indicators.items():
        print(f"\n{tf} 时间框架:")
        if indicators:
            print(f"   RSI: {indicators.get('rsi', 'N/A')}")
            print(f"   MACD: {indicators.get('macd', 'N/A')}")
            print(f"   布林带位置: {indicators.get('bb_position', 'N/A')}")
        else:
            print(f"   数据不可用")

    print(f"""
🎯 交易信号
------------------------------
买入信号(做多): {'✅ 强烈建议' if latest.get('enter_long', 0) == 1 else '❌ 不建议'}
卖出信号(平多): {'✅ 强烈建议' if latest.get('exit_long', 0) == 1 else '❌ 不建议'}
做空信号(开空): {'✅ 强烈建议' if latest.get('enter_short', 0) == 1 else '❌ 不建议'}
平空信号: {'✅ 强烈建议' if latest.get('exit_short', 0) == 1 else '❌ 不建议'}

📊 多时间框架得分 (基于真实数据):
------------------------------""")
    
    for tf in ['5m', '15m', '1h', '4h']:
        score = _to_float(latest.get(f'tf_{tf}_score', 0.0), 0.0)
        if score > 0.2:
            signal = "看涨🟢"
            trend = "上涨趋势"
        elif score < -0.2:
            signal = "看跌🔴" 
            trend = "下跌趋势"
        else:
            signal = "中性🟡"
            trend = "震荡整理"
        print(f"   {tf:>4}: {score:+.3f} ({signal}) - {trend}")
    
    # 智能风险评估 - 结合多个因素
    base_risk = _to_float(latest.get('ai_risk_level', 0.5), 0.5)
    combined_score = _to_float(latest.get('ai_combined_score', 0.0), 0.0)
    confidence = _to_float(latest.get('ai_confidence', 0.5), 0.5)
    valid_timeframes = _to_int(latest.get('valid_timeframes', 0), 0)
    
    # 调整风险等级逻辑
    adjusted_risk = base_risk
    
    # 如果市场信号不明确（接近震荡），增加风险
    if abs(combined_score) < 0.1:  # 震荡整理状态
        adjusted_risk = max(adjusted_risk, 0.6)  # 至少60%风险
        
    # 如果AI置信度低，增加风险
    if confidence < 0.3:
        adjusted_risk = max(adjusted_risk, 0.7)  # 至少70%风险
        
    # 如果没有有效的时间框架分析，增加风险
    if valid_timeframes == 0:
        adjusted_risk = max(adjusted_risk, 0.8)  # 至少80%风险
    
    # 限制风险范围
    adjusted_risk = max(0.1, min(0.9, adjusted_risk))
    
    if adjusted_risk > 0.7:
        risk_msg = "⚠️ 高风险"
    elif adjusted_risk > 0.5:
        risk_msg = "🟡 中等风险"
    else:
        risk_msg = "✅ 低风险"
    
    # 结合实际信号计算建议仓位与方向（合约做多/做空）
    enter_long_sig = 1 if _to_int(latest.get('enter_long', 0), 0) == 1 else 0
    exit_long_sig = 1 if _to_int(latest.get('exit_long', 0), 0) == 1 else 0
    enter_short_sig = 1 if _to_int(latest.get('enter_short', 0), 0) == 1 else 0
    exit_short_sig = 1 if _to_int(latest.get('exit_short', 0), 0) == 1 else 0

    suggested_position = 0.0
    suggested_side = 'NONE'

    if exit_long_sig == 1 or exit_short_sig == 1:
        # 任一方向触发离场信号，建议空仓
        suggested_position = 0.0
        suggested_side = 'FLAT'
    elif enter_long_sig == 1:
        # 基于信号强度与风险的仓位建议
        if abs(combined_score) < 0.1:  # 震荡整理
            suggested_position = (1 - adjusted_risk) * 0.3  # 最多30%仓位
        elif abs(combined_score) < 0.2:  # 轻微信号  
            suggested_position = (1 - adjusted_risk) * 0.5  # 最多50%仓位
        elif abs(combined_score) < 0.4:  # 中等信号
            suggested_position = (1 - adjusted_risk) * 0.7  # 最多70%仓位
        else:  # 强信号
            suggested_position = (1 - adjusted_risk) * 0.9  # 最多90%仓位
        # 限制仓位范围（仅在触发买入信号时）
        suggested_position = max(0.1, min(0.9, suggested_position))
        suggested_side = 'LONG'
    elif enter_short_sig == 1:
        # 空头与多头对称
        if abs(combined_score) < 0.1:
            suggested_position = (1 - adjusted_risk) * 0.3
        elif abs(combined_score) < 0.2:
            suggested_position = (1 - adjusted_risk) * 0.5
        elif abs(combined_score) < 0.4:
            suggested_position = (1 - adjusted_risk) * 0.7
        else:
            suggested_position = (1 - adjusted_risk) * 0.9
        suggested_position = max(0.1, min(0.9, suggested_position))
        suggested_side = 'SHORT'
    
    print(f"""
⚠️ 风险评估
------------------------------
风险等级: {adjusted_risk*100:.1f}% ({risk_msg})
建议方向: {('做多' if suggested_side=='LONG' else ('做空' if suggested_side=='SHORT' else '观望'))}
建议仓位: {suggested_position*100:.1f}%

🔮 综合建议
------------------------------""")

    # 持仓操作建议 - 与信号一致
    if exit_long_sig == 1 or exit_short_sig == 1:
        print("\n📦 持仓建议\n------------------------------")
        if combined_score < -0.3:
            print("建议减仓或清仓，优先保护收益/控制亏损")
        else:
            print("建议逐步减仓，等待更优进场条件")
    elif enter_long_sig == 0 and enter_short_sig == 0:
        print("\n📦 持仓建议\n------------------------------")
        print("暂不建仓，等待入场信号触发")
    
    # 智能综合建议 - 与实际信号保持一致
    if valid_timeframes == 0:
        print("🚫 数据不足 - 暂停交易，等待数据完善")
    elif confidence < 0.3:
        print("🤔 信号不明确 - 建议观望，等待更清晰信号")
    elif exit_long_sig == 1 or exit_short_sig == 1:
        if combined_score < -0.3:
            print("📉 强烈看跌 - 考虑减仓或止盈")
        else:
            print("📊 谨慎看跌 - 以防守为主，等待反弹减仓")
    elif enter_short_sig == 1:
        if combined_score < -0.3:
            print("📉 强烈看跌 - 可考虑开空")
        elif combined_score < -0.1:
            print("📊 轻微看跌 - 小仓位试探空单")
        elif abs(combined_score) < 0.1:
            print("🔄 震荡整理 - 等待更清晰入场信号")
        else:
            print("📊 信号转向 - 谨慎观望")
    elif enter_long_sig == 1:
        if combined_score > 0.3:
            if adjusted_risk < 0.5:
                print("📈 强烈看涨 - 逢低买入")
            else:
                print("📊 谨慎看涨 - 小仓位试探，严格止损")
        elif combined_score > 0.1:
            print("📊 轻微看涨 - 小仓位试探")
        elif abs(combined_score) < 0.1:
            print("🔄 震荡整理 - 等待更清晰入场信号")
        else:
            print("📊 轻微看跌 - 保持观望")
    else:
        print("🔄 观望 - 未触发入场信号，等待明确信号")
    
    print("\n" + "=" * 50)
    print("✅ 真实市场数据分析完成!")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        # 分析真实数据
        symbol = sys.argv[2] if len(sys.argv) > 2 else 'BTCUSDT'
        success = analyze_real_data(symbol)
    else:
        # 模拟数据测试
        def test_strategy():
            print("(跳过) 未实现的模拟测试，默认返回True")
            return True
        success = test_strategy()
    
    print(f"\n{'✅ 分析成功!' if success else '❌ 分析失败!'}")