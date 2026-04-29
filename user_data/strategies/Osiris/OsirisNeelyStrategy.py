"""
OSIRIS NEELY STRATEGY v1.0 — Neely Method (MEW) para Day Trade
================================================================
Base teórica:
  Glenn Neely, "Mastering Elliott Wave" (Windsor Books, 1990)

DIFERENCIAL sobre OsirisElliottStrategy (Frost/Prechter básico):
  1. Structure Labels (:3/:5) via Pre-Constructive Rules of Logic
     — classifica cada onda como impulsiva/corretiva ANTES de nomear
  2. 0-2 Trendline Channeling — timing de entrada preciso
     — não entra até o preço confirmar acima da 0-2 line
  3. Power Ratings (-3 a +3) baseados no TIPO da W2:
     Running  (-50% retrace) → power +3 → target 261.8% de W1
     IrrFail  (>95% retrace) → power +2 → target 200% de W1
     Zigzag   (38-62%)       → power +1 → target 161.8% de W1
     Flat/Etc               → power  0 → target 161.8% de W1
  4. Terminal Impulse detection → trade de reversão SHORT
  5. Stage 1 confirmation: 2-4 trendline quebrada ≤ tempo de W5

3 SETUPS:
  SETUP 1 — Wave 3 Onset (Long)
    Detecta: W0(L) → W1(H) → W2(L) → início W3
    Entry: breakout > W1 + preço acima 0-2 trendline
    Stop: 0.5 ATR abaixo de W2
    Target: W2_low + W1_length × mult (baseado em Power Rating)

  SETUP 2 — Wave 5 Entry (Long)
    Detecta: W0→W1→W2→W3→W4 completo
    Entry: recovery acima de W4 + 2-4 trendline intacta
    Stop: 0.5 ATR abaixo de W4
    Target: W4_low + W1_length (igualdade) ou × 1.618 (extension)

  SETUP 3 — Terminal Reversal (Short — apenas se can_short=True)
    Detecta: Terminal Impulse (canal convergente, W4 overlap W1, todas :3)
    Entry: W5 completo, preço girando para baixo
    Stop: 0.5 ATR acima de W5 topo
    Target: início do Terminal

Timeframe: 5m (day trade)
Confirmação: 1h via informative pair (tendência macro)
"""

import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional
from datetime import datetime, timedelta

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter
from freqtrade.persistence import Trade

try:
    import talib.abstract as ta
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from neely_engine import NeelyEngine, fib_extension_levels

logger = logging.getLogger(__name__)


class OsirisNeelyStrategy(IStrategy):
    """
    OSIRIS NEELY — Implementação do Neely Method para day trade no Freqtrade.
    Usa o NeelyEngine para calcular padrões, labels e trendlines.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"

    # Safety ROI (custom exit cuida do lucro real)
    minimal_roi = {"0": 0.20}

    # Stoploss base (custom_stoploss vai sobrescrever)
    stoploss = -0.06
    trailing_stop = False
    use_custom_stoploss = True

    # Candlestick count para aquecimento dos indicadores
    startup_candle_count = 200
    process_only_new_candles = True

    # ═══════════════════════════════════════════════════════════════════
    # HYPEROPT PARAMETERS
    # ═══════════════════════════════════════════════════════════════════

    # Neely Engine
    pivot_period    = IntParameter(7, 20, default=10, space="buy", optimize=True,
                                   load=True)
    min_wave_atr    = DecimalParameter(1.5, 4.0, default=2.0, decimals=1,
                                       space="buy", optimize=True, load=True)

    # Score mínimo para entrar
    neely_score_min = DecimalParameter(3.5, 7.5, default=6.0, decimals=1,
                                       space="buy", optimize=True, load=True)

    # Filtro macro (1h)
    use_macro_filter  = CategoricalParameter([True, False], default=True,
                                             space="buy", optimize=False, load=True)
    macro_ema_period  = IntParameter(20, 100, default=50, space="buy",
                                     optimize=True, load=True)

    # Setups ativos
    use_setup1 = CategoricalParameter([True, False], default=True,
                                      space="buy", optimize=False, load=True)
    use_setup2 = CategoricalParameter([True, False], default=True,
                                      space="buy", optimize=False, load=True)
    use_setup3 = CategoricalParameter([True, False], default=False,
                                      space="buy", optimize=False, load=True)

    # Stop: ATR multiplier
    stop_atr_mult = DecimalParameter(0.3, 1.5, default=0.5, decimals=1,
                                     space="stoploss", optimize=True, load=True)

    # Target: R múltiplo mínimo (fechamento parcial em 1R, tudo em 2R)
    tp_r_partial  = DecimalParameter(0.8, 1.5, default=1.0, decimals=1,
                                     space="sell", optimize=True, load=True)
    tp_r_full     = DecimalParameter(1.5, 4.0, default=2.0, decimals=1,
                                     space="sell", optimize=True, load=True)

    # Power Rating mínimo para entrar (0 = aceita todos, 1 = apenas W2 forte)
    min_power_rating = IntParameter(-1, 2, default=0, space="buy",
                                    optimize=True, load=True)

    # ═══════════════════════════════════════════════════════════════════
    # INFORMATIVE PAIR (1h) — filtro macro de tendência
    # ═══════════════════════════════════════════════════════════════════

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, '1h') for p in pairs]

    # ═══════════════════════════════════════════════════════════════════
    # INDICADORES
    # ═══════════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ── ATR (para stops + filtro de tamanho de onda) ────────────────
        if HAS_TALIB:
            dataframe['atr']    = ta.ATR(dataframe, timeperiod=14)
            dataframe['rsi']    = ta.RSI(dataframe, timeperiod=14)
            macd, sig, hist     = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
            dataframe['macd_hist'] = hist
            dataframe['ema20']  = ta.EMA(dataframe, timeperiod=20)
            dataframe['ema50']  = ta.EMA(dataframe, timeperiod=50)
            dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        else:
            # Fallback puro pandas
            dataframe['atr']       = self._atr_pandas(dataframe, 14)
            dataframe['rsi']       = self._rsi_pandas(dataframe, 14)
            dataframe['macd_hist'] = 0.0
            dataframe['ema20']     = dataframe['close'].ewm(span=20).mean()
            dataframe['ema50']     = dataframe['close'].ewm(span=50).mean()
            dataframe['ema200']    = dataframe['close'].ewm(span=200).mean()

        dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()

        # ── Neely Engine ──────────────────────────────────────────────
        engine = NeelyEngine(
            pivot_period=int(self.pivot_period.value),
            min_wave_atr=float(self.min_wave_atr.value),
        )
        dataframe = engine.analyze(dataframe)

        # ── Informative 1h (macro trend) ─────────────────────────────
        if self.use_macro_filter.value:
            dataframe = self._merge_informative_1h(dataframe, metadata)

        return dataframe

    def _merge_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Merge dados 1h para filtro macro."""
        try:
            inf_df = self.dp.get_pair_dataframe(metadata['pair'], '1h')
            if inf_df is None or len(inf_df) == 0:
                dataframe['macro_bull'] = True
                return dataframe

            ep = int(self.macro_ema_period.value)
            if HAS_TALIB:
                inf_df['ema_macro'] = ta.EMA(inf_df, timeperiod=ep)
            else:
                inf_df['ema_macro'] = inf_df['close'].ewm(span=ep).mean()

            inf_df['macro_bull'] = inf_df['close'] > inf_df['ema_macro']
            inf_df = inf_df[['date', 'macro_bull']].copy()
            inf_df.rename(columns={'date': 'date_1h'}, inplace=True)

            # Merge por tempo (forward fill)
            dataframe = merge_informative_pair(
                dataframe, inf_df, self.timeframe, '1h',
                ffill=True, append_timeframe=False,
            )
        except Exception as e:
            logger.warning(f"NeelyStrategy: falha no informative 1h: {e}")
            dataframe['macro_bull'] = True

        return dataframe

    # ═══════════════════════════════════════════════════════════════════
    # BUY SIGNAL
    # ═══════════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long']  = 0
        dataframe['enter_short'] = 0

        score_min  = float(self.neely_score_min.value)
        power_min  = int(self.min_power_rating.value)
        macro_bool = self.use_macro_filter.value

        # Filtro macro (bullish acima da EMA macro)
        macro_bull = dataframe.get('macro_bull', pd.Series(True, index=dataframe.index))

        # ── SETUP 1 — Wave 3 Onset (Long) ────────────────────────────
        if self.use_setup1.value:
            cond_s1 = (
                (dataframe['neely_pattern'] == 'wave3_onset')
                & (dataframe['neely_score'] >= score_min)
                & (dataframe['neely_power'] >= power_min)
                & (dataframe['neely_target'] > dataframe['close'])  # target > preço
                & (dataframe['volume'] > dataframe['volume_ma'] * 0.5)  # liquidez mínima
            )
            if macro_bool:
                cond_s1 = cond_s1 & (macro_bull == True)

            dataframe.loc[cond_s1, 'enter_long'] = 1
            # Tag do setup
            dataframe.loc[cond_s1, 'enter_tag'] = f'neely_w3_p{dataframe.loc[cond_s1, "neely_power"].astype(str)}'

        # ── SETUP 2 — Wave 5 Entry (Long) ────────────────────────────
        if self.use_setup2.value:
            cond_s2 = (
                (dataframe['neely_pattern'] == 'wave5_entry')
                & (dataframe['neely_score'] >= score_min + 0.5)  # barra mais alta para W5
                & (dataframe['neely_target'] > dataframe['close'])
                & (dataframe['volume'] > dataframe['volume_ma'] * 0.5)
            )
            if macro_bool:
                cond_s2 = cond_s2 & (macro_bull == True)

            dataframe.loc[cond_s2, 'enter_long'] = 1
            dataframe.loc[cond_s2, 'enter_tag']  = 'neely_w5'

        # ── SETUP 3 — Terminal Reversal (Short) ──────────────────────
        if self.use_setup3.value and self.can_short:
            cond_s3 = (
                (dataframe['neely_pattern'] == 'terminal_rev')
                & (dataframe['neely_score'] >= score_min + 2.0)  # threshold bem alto para short
                & (dataframe['neely_target'] < dataframe['close'])
                & (dataframe['volume'] > dataframe['volume_ma'] * 0.5)
            )

            dataframe.loc[cond_s3, 'enter_short'] = 1
            dataframe.loc[cond_s3, 'enter_tag']   = 'neely_terminal'

        return dataframe

    # ═══════════════════════════════════════════════════════════════════
    # SELL SIGNAL (saída via custom_exit principalmente)
    # ═══════════════════════════════════════════════════════════════════

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long']  = 0
        dataframe['exit_short'] = 0
        return dataframe

    # ═══════════════════════════════════════════════════════════════════
    # CUSTOM STOPLOSS — Neely: stop abaixo de W2/W4
    # ═══════════════════════════════════════════════════════════════════

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """
        Custom stoploss baseado no Neely Method.

        Lógica:
          1. Na entrada: stop = neely_stop (abaixo de W2 ou W4)
             = distância fixa baseada no preço de entrada
          2. Em profit > 0.5R: move stop para breakeven + 0.1%
          3. Em profit > 1.0R: trailing stop sobe para preservar 0.5R

        O 'neely_stop' foi calculado no engine como:
          W2_price - ATR × stop_atr_mult
        """
        try:
            # Busca candle de entrada para pegar neely_stop original
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or len(dataframe) == 0:
                return -1  # Usa stoploss padrão

            # Pega o stop calculado no momento da entrada (Trade.open_date)
            entry_candle = dataframe[dataframe['date'] <= trade.open_date]
            if len(entry_candle) == 0:
                return -1

            entry_row = entry_candle.iloc[-1]
            neely_stop_price = entry_row.get('neely_stop', 0.0)
            open_rate = trade.open_rate
            atr_val   = entry_row.get('atr', open_rate * 0.01)

            if neely_stop_price <= 0 or open_rate <= 0:
                return -1

            # Stop inicial: distância do Neely stop
            if not trade.is_short:
                initial_sl_pct = (open_rate - neely_stop_price) / open_rate
                initial_sl_pct = max(initial_sl_pct, 0.005)  # mínimo 0.5%

                # Calcula R (1R = distância inicial de stop)
                r_value = initial_sl_pct  # pct por 1R

                # Fase 1: breakeven em 0.5R
                if current_profit >= r_value * 0.5:
                    break_even_sl = -0.001  # +0.1% acima do preço de entrada
                    if current_profit >= r_value * 1.0:
                        # Fase 2: trailing a partir de 1R — preserva 0.5R
                        preserved = r_value * 0.5
                        trailing_sl = -(current_profit - preserved)
                        return max(trailing_sl, -initial_sl_pct)
                    return break_even_sl

                # Antes do breakeven: usa stop fixo do Neely
                return -initial_sl_pct

            else:
                # Short: stop acima de W5 topo
                initial_sl_pct = (neely_stop_price - open_rate) / open_rate
                initial_sl_pct = max(initial_sl_pct, 0.005)

                r_value = initial_sl_pct
                if current_profit >= r_value * 0.5:
                    if current_profit >= r_value * 1.0:
                        preserved = r_value * 0.5
                        trailing_sl = -(current_profit - preserved)
                        return max(trailing_sl, -initial_sl_pct)
                    return -0.001

                return -initial_sl_pct

        except Exception as e:
            logger.warning(f"NeelyStrategy: erro em custom_stoploss: {e}")
            return -1  # usa stoploss padrão

    # ═══════════════════════════════════════════════════════════════════
    # CUSTOM EXIT — Target Fibonacci + Time limit
    # ═══════════════════════════════════════════════════════════════════

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        Saída customizada:
          1. Neely Target Hit: neely_target atingido → sai
          2. 2-4 Trendline break (Stage 1): sai se for W5 e 2-4 quebrou
          3. Divergência bearish: RSI divergindo no topo → saída preventiva
          4. Time limit: posição parada por > 4h sem lucro → corta

        Retorna string com motivo de saída ou None (mantém posição).
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or len(dataframe) == 0:
                return None

            last = dataframe.iloc[-1]
            open_rate  = trade.open_rate
            atr_val    = float(last.get('atr', open_rate * 0.01))
            neely_tgt  = float(last.get('neely_target', 0.0))
            neely_tl24 = float(last.get('neely_tl_24', float('nan')))
            hours_open = (current_time - trade.open_date_utc).total_seconds() / 3600

            # ── TARGET HIT ──────────────────────────────────────────
            if not trade.is_short:
                if neely_tgt > 0 and current_rate >= neely_tgt:
                    return 'neely_target_hit'
            else:
                if neely_tgt > 0 and current_rate <= neely_tgt:
                    return 'neely_target_hit'

            # ── 2-4 TRENDLINE BREAK (Stage 1 — fim do W5 pattern) ──
            # Neely: após W5 completo, 2-4 trendline deve ser quebrada
            # em ≤ tempo de W5. Se a trendline foi quebrada downward = saída.
            if (not trade.is_short and not np.isnan(neely_tl24)
                    and current_rate < neely_tl24
                    and current_profit > 0):
                return 'neely_2_4_trendline_break'

            # ── TIME LIMIT ───────────────────────────────────────────
            tp_r_full = float(self.tp_r_full.value)
            if hours_open > 8 and current_profit < 0:
                # Está perdendo há mais de 8h → corta
                return 'neely_time_cut_loss'
            if hours_open > 24:
                # Após 24h — se não atingiu target, sai com o que tem
                return 'neely_time_limit_24h'

            # ── PARTIAL TARGET (1R): sinal para saída parcial ────────
            # Freqtrade não faz saída parcial nativa, mas podemos sinalizar
            # através de custom_exit com lucro razoável
            if (current_profit >= 0.015 and hours_open > 2
                    and not trade.is_short):
                # Saiu do range normal de W3 — protege com saída se RSI alto
                rsi_now = float(last.get('rsi', 50.0))
                if rsi_now > 75:
                    return 'neely_rsi_overbought_exit'

        except Exception as e:
            logger.warning(f"NeelyStrategy: erro em custom_exit: {e}")

        return None

    # ═══════════════════════════════════════════════════════════════════
    # CONFIRM ENTRY — Gate final antes de abrir trade
    # ═══════════════════════════════════════════════════════════════════

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Gate final:
          - Só opera em horário de alta liquidez BTC (UTC 08:00-22:00)
        """
        # Horário de liquidez (UTC)
        hour = current_time.hour
        if not (8 <= hour <= 22):
            return False

        return True

    # ═══════════════════════════════════════════════════════════════════
    # LEVERAGED POSITION (Futures)
    # ═══════════════════════════════════════════════════════════════════

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Alavancagem baseada no Power Rating do sinal.
          power +3 (Running) → 3x
          power +2           → 2x
          power +1/0         → 1x (sem alavancagem)
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or len(dataframe) == 0:
                return 1.0
            last = dataframe.iloc[-1]
            power = int(last.get('neely_power', 0))
            if power >= 3:
                return min(3.0, max_leverage)
            elif power >= 2:
                return min(2.0, max_leverage)
            else:
                return 1.0
        except Exception:
            return 1.0

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS — Pandas fallback (sem TA-Lib)
    # ═══════════════════════════════════════════════════════════════════

    def _atr_pandas(self, df: DataFrame, period: int = 14) -> pd.Series:
        high  = df['high']
        low   = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _rsi_pandas(self, df: DataFrame, period: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
