"""
NeelyEngine — Motor Neely Method (MEW) para Day Trading
=================================================================
Base:  Glenn Neely, "Mastering Elliott Wave" (Windsor Books, 1990)

Inovações sobre Elliott clássico (Frost/Prechter):
  1. Structure Labels (:3 = corretivo, :5 = impulsivo) via Pre-Constructive
     Rules of Logic — classificados pelo RETRACE da onda SEGUINTE
  2. 0-2 Trendline channeling — timing preciso de entrada
  3. Power Ratings (-3 a +3) — calibra target pelo tipo da W2
  4. Terminal Pattern detection — trade de reversão
  5. Stage 1 + Stage 2 post-pattern confirmation
  6. Fibonacci Internal + External targets

Módulo standalone — sem dependência de Freqtrade.
Usado por OsirisNeelyStrategy.py via NeelyEngine().analyze(df)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class StructLabel(str, Enum):
    """
    Neely Structure Labels — MEW Cap. 3/7.
    :5 = impulsivo (trending), :3 = corretivo.
    """
    IMPULSE   = ':5'   # Trending + Terminal
    CORRECTIVE = ':3'  # Zigzag, Flat, Triangle, Complex
    UNKNOWN   = ':?'   # Ainda não determinado


class W2Type(str, Enum):
    """
    Tipo da Onda 2 — determina Power Rating e target de W3.
    Neely MEW Cap. 10 (Advanced Logic Rules).
    """
    RUNNING      = 'running'       # Retrace < 50% — power +3
    ZIGZAG_DEEP  = 'zigzag_deep'   # Retrace 50-61.8% — power +1
    COMMON_FLAT  = 'common_flat'   # Retrace 61.8-80% — power  0
    IRREGULAR    = 'irregular'     # Retrace 80-95% — power  0
    IRR_FAILURE  = 'irr_failure'   # Retrace 95-100%+ — power +2
    UNKNOWN      = 'unknown'       # Não classificado


class PatternType(str, Enum):
    """Padrão detectado."""
    WAVE3_ONSET    = 'wave3_onset'    # Setup A: início da Onda 3
    WAVE5_ENTRY    = 'wave5_entry'    # Setup B: entrada na Onda 5
    TERMINAL_REV   = 'terminal_rev'   # Setup C: reversão pós-Terminal
    RUNNING_CORR   = 'running_corr'   # Setup D: Running Correction (power+3)
    NONE           = 'none'


@dataclass
class Pivot:
    """Ponto de reversão confirmado."""
    bar_idx: int
    direction: str   # 'H' (high) ou 'L' (low)
    price: float
    confirmed_at: int = 0
    label: StructLabel = StructLabel.UNKNOWN
    label_confidence: float = 0.0
    label_confirmed_at: Optional[int] = None
    retrace_from_prev: float = 0.0   # % de retracement da onda anterior


@dataclass
class WavePattern:
    """Padrão de ondas detectado em uma barra específica."""
    bar_idx: int
    pattern: PatternType
    score: float                    # 0-10 — confiança geral
    w2_type: W2Type = W2Type.UNKNOWN
    power_rating: int = 0           # -3 a +3 (Neely Power Ratings)

    # Pivôs do padrão
    w0_price: float = 0.0           # Origem de W1
    w1_price: float = 0.0           # Fim de W1 (topo)
    w2_price: float = 0.0           # Fim de W2 (fundo)
    w3_price: float = 0.0           # Fim de W3 (só Setup B)
    w4_price: float = 0.0           # Fim de W4 (só Setup B)

    # Trendlines (Neely Channeling)
    tl_0_2_slope: float = 0.0       # Inclinação da 0-2 trendline
    tl_0_2_at_bar: float = 0.0      # Valor da 0-2 trendline na barra atual
    tl_2_4_slope: float = 0.0       # Inclinação da 2-4 trendline
    tl_2_4_at_bar: float = 0.0      # Valor da 2-4 trendline na barra atual

    # Targets (Fibonacci Internal + External)
    target_w3_161: float = 0.0      # Target W3 = W2_low + W1 × 1.618
    target_w3_261: float = 0.0      # Target W3 = W2_low + W1 × 2.618 (Running)
    target_w5_eq:  float = 0.0      # Target W5 = W4_low + W1_length (igualdade)
    target_w5_161: float = 0.0      # Target W5 = W2→W4 + W1 × 1.618

    # Stop
    stop_price: float = 0.0         # Stop abaixo de W2 (ou W4)

    # Terminal metadata
    is_terminal_suspect: bool = False
    terminal_channel_broken: bool = False

    # Post-pattern confirmation
    stage1_confirmed: bool = False  # 2-4 trendline quebrada em ≤ W5 tempo
    stage2_confirmed: bool = False  # Retrace até zona de W4


# ═══════════════════════════════════════════════════════════════════════════
# NEELY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class NeelyEngine:
    """
    Motor Neely Method.

    Uso:
        engine = NeelyEngine(pivot_period=10, min_wave_atr=2.0)
        result_df = engine.analyze(df)
        # result_df tem colunas: neely_pattern, neely_score, neely_target, etc.
    """

    # Power Ratings → Multiplicadores de W3 (Neely MEW Cap. 10)
    POWER_TO_TARGET = {
        3:  2.618,  # Running Correction     → W3 ≥ 261.8% de W1
        2:  2.0,    # Irregular Failure       → W3 ≈ 200% de W1
        1:  1.618,  # Zigzag profundo         → W3 = 161.8% de W1
        0:  1.618,  # Flat comum / Irregular  → W3 = 161.8% de W1
        -1: 1.272,  # Fraco                   → W3 mínimo
    }

    def __init__(
        self,
        pivot_period: int = 10,
        min_wave_atr: float = 2.0,     # W1 mínimo = 2 × ATR (evita ruído)
        w2_min_retrace: float = 0.236, # Neely: W2 nunca retrace menos que 23.6%
        w2_max_retrace: float = 0.990, # Neely: W2 não pode retrace 100%+ de W1 (R1)
        lookahead: int = 0,            # 0 = sem lookahead (real-time)
    ):
        self.pivot_period = pivot_period
        self.min_wave_atr = min_wave_atr
        self.w2_min_retrace = w2_min_retrace
        self.w2_max_retrace = w2_max_retrace
        self.lookahead = lookahead

    # ───────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ───────────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa o DataFrame e adiciona colunas Neely.

        Input: OHLCV DataFrame (Freqtrade padrão)
        Output: mesmo DF + colunas:
          neely_pattern       — str: 'wave3_onset' | 'wave5_entry' | 'terminal_rev' | 'none'
          neely_score         — float 0-10
          neely_power         — int -3..+3
          neely_target        — float: target price
          neely_stop          — float: stop price
          neely_tl_02         — float: valor da 0-2 trendline na barra
          neely_tl_24         — float: valor da 2-4 trendline na barra
          neely_w2_type       — str: tipo da W2
          neely_stage1        — bool: Stage 1 confirmado
          neely_terminal      — bool: suspeita de Terminal
        """
        df = df.copy()

        # Inicializa colunas
        n = len(df)
        df['neely_pattern']  = 'none'
        df['neely_score']    = 0.0
        df['neely_power']    = 0
        df['neely_target']   = 0.0
        df['neely_stop']     = 0.0
        df['neely_tl_02']    = np.nan
        df['neely_tl_24']    = np.nan
        df['neely_tl_02_slope'] = np.nan
        df['neely_tl_24_slope'] = np.nan
        df['neely_w2_type']  = 'unknown'
        df['neely_stage1']   = False
        df['neely_terminal'] = False
        df['neely_w2_price'] = np.nan
        df['neely_w1_price'] = np.nan

        # ATR para filtro de ruído
        atr = self._compute_atr(df)

        # Detecta pivôs
        pivots = self._detect_pivots(df)
        if len(pivots) < 4:
            return df

        # Assign Structure Labels (Pre-Constructive Rules of Logic)
        pivots = self._assign_structure_labels(pivots)

        # Cada pivot (W2/W4/W5-terminal) só pode gerar UM sinal na vida toda.
        # Chave: ('w3'|'w5'|'term', bar_idx_do_pivot)
        signaled_pivots: set = set()

        # Staleness: padrão com pivô âncora mais antigo que N barras é ignorado
        max_wave_age = 80   # ~6.5h em 5m

        # Para cada barra, analisa os pivôs disponíveis
        for bar_idx in range(50, n):
            avail = [p for p in pivots if p.confirmed_at <= bar_idx]
            if len(avail) < 4:
                continue

            cur_price  = float(df['close'].iloc[bar_idx])
            cur_rsi    = self._get_val(df, 'rsi', bar_idx, 50.0)
            cur_macd   = self._get_val(df, 'macd_hist', bar_idx, 0.0)
            cur_atr    = float(atr.iloc[bar_idx]) if not np.isnan(atr.iloc[bar_idx]) else 0.001

            best_pattern: Optional[WavePattern] = None

            # ── Setup A: Wave 3 Onset ──────────────────────────────────
            w2_bar = self._anchor_bar(avail, 'wave3')
            if (w2_bar is not None
                    and bar_idx - w2_bar <= max_wave_age
                    and ('w3', w2_bar) not in signaled_pivots):

                pa = self._setup_wave3(avail, bar_idx, cur_price, cur_rsi, cur_macd, cur_atr)
                if pa and (best_pattern is None or pa.score > best_pattern.score):
                    best_pattern = pa

            # ── Setup B: Wave 5 Entry ──────────────────────────────────
            w4_bar = self._anchor_bar(avail, 'wave5')
            if (w4_bar is not None
                    and bar_idx - w4_bar <= max_wave_age
                    and ('w5', w4_bar) not in signaled_pivots):

                pb = self._setup_wave5(avail, bar_idx, cur_price, cur_rsi, cur_macd, cur_atr)
                if pb and (best_pattern is None or pb.score > best_pattern.score):
                    best_pattern = pb

            # ── Setup C: Terminal Reversal ─────────────────────────────
            term_anchor = self._anchor_bar(avail, 'term')
            if (term_anchor is not None
                    and bar_idx - term_anchor <= max_wave_age
                    and ('term', term_anchor) not in signaled_pivots):

                pc = self._setup_terminal_reversal(avail, bar_idx, cur_price, cur_atr, df)
                if pc and (best_pattern is None or pc.score > best_pattern.score):
                    best_pattern = pc

            if best_pattern and best_pattern.score >= 3.0:
                # Registra pivot como sinalizado (sem reentrada)
                if best_pattern.pattern == PatternType.WAVE3_ONSET and w2_bar is not None:
                    signaled_pivots.add(('w3', w2_bar))
                elif best_pattern.pattern == PatternType.WAVE5_ENTRY and w4_bar is not None:
                    signaled_pivots.add(('w5', w4_bar))
                elif best_pattern.pattern == PatternType.TERMINAL_REV and term_anchor is not None:
                    signaled_pivots.add(('term', term_anchor))

                ix = df.index[bar_idx]
                df.at[ix, 'neely_pattern']  = best_pattern.pattern.value
                df.at[ix, 'neely_score']    = round(best_pattern.score, 2)
                df.at[ix, 'neely_power']    = best_pattern.power_rating
                df.at[ix, 'neely_target']   = (
                    best_pattern.target_w3_261
                    or best_pattern.target_w3_161
                    or best_pattern.target_w5_161
                    or best_pattern.target_w5_eq
                )
                df.at[ix, 'neely_stop']     = best_pattern.stop_price
                df.at[ix, 'neely_tl_02']    = best_pattern.tl_0_2_at_bar
                df.at[ix, 'neely_tl_24']    = best_pattern.tl_2_4_at_bar
                df.at[ix, 'neely_tl_02_slope'] = best_pattern.tl_0_2_slope
                df.at[ix, 'neely_tl_24_slope'] = best_pattern.tl_2_4_slope
                df.at[ix, 'neely_w2_type']  = best_pattern.w2_type.value
                df.at[ix, 'neely_stage1']   = best_pattern.stage1_confirmed
                df.at[ix, 'neely_terminal'] = best_pattern.is_terminal_suspect
                df.at[ix, 'neely_w2_price'] = best_pattern.w2_price
                df.at[ix, 'neely_w1_price'] = best_pattern.w1_price

        return df

    def _anchor_bar(self, pivots: list, setup_type: str) -> Optional[int]:
        """
        Retorna o bar_idx do pivot âncora para deduplicação:
          wave3 → bar_idx do W2 bottom
          wave5 → bar_idx do W4 bottom
          term  → bar_idx do W5 topo (Terminal)
        """
        if setup_type == 'wave3':
            _, _, w2l = self._find_L_H_L(pivots)
            return w2l.bar_idx if w2l else None
        elif setup_type == 'wave5':
            _, _, _, _, w4l = self._find_L_H_L_H_L(pivots)
            return w4l.bar_idx if w4l else None
        elif setup_type == 'term':
            result = self._find_H_L_H_L_H(pivots)
            if result and result[5]:
                return result[5].bar_idx
        return None

    # ───────────────────────────────────────────────────────────────────
    # PIVOT DETECTION
    # ───────────────────────────────────────────────────────────────────

    def _detect_pivots(self, df: pd.DataFrame) -> list[Pivot]:
        """
        Detecta pivôs usando rolling max/min com confirmação causal.
        Um pivô só existe após a janela futura completa do pivot_period.
        """
        p = self.pivot_period
        highs = df['high']
        lows  = df['low']
        n = len(df)

        pivots: list[Pivot] = []

        for i in range(p, n - p):
            # Pivô de alta: high[i] é máximo das últimas 2p+1 barras
            hi_win = highs.iloc[i - p:i + p + 1]
            lo_win = lows.iloc[i - p:i + p + 1]

            if len(hi_win) > 0 and float(highs.iloc[i]) == float(hi_win.max()):
                pivots.append(
                    Pivot(
                        bar_idx=i,
                        direction='H',
                        price=float(highs.iloc[i]),
                        confirmed_at=i + p,
                    )
                )

            if len(lo_win) > 0 and float(lows.iloc[i]) == float(lo_win.min()):
                pivots.append(
                    Pivot(
                        bar_idx=i,
                        direction='L',
                        price=float(lows.iloc[i]),
                        confirmed_at=i + p,
                    )
                )

        # Ordena por bar_idx
        pivots.sort(key=lambda p: p.bar_idx)

        # Remove duplicatas do mesmo bar_idx (manter H e L separados)
        cleaned: list[Pivot] = []
        for pv in pivots:
            if cleaned and cleaned[-1].bar_idx == pv.bar_idx and cleaned[-1].direction == pv.direction:
                # Mesmo bar, mesmo tipo — manter o mais extremo
                if pv.direction == 'H' and pv.price > cleaned[-1].price:
                    cleaned[-1] = pv
                elif pv.direction == 'L' and pv.price < cleaned[-1].price:
                    cleaned[-1] = pv
            else:
                cleaned.append(pv)

        return cleaned

    # ───────────────────────────────────────────────────────────────────
    # STRUCTURE LABELS (Pre-Constructive Rules of Logic)
    # ───────────────────────────────────────────────────────────────────

    def _assign_structure_labels(self, pivots: list[Pivot]) -> list[Pivot]:
        """
        Neely MEW Cap. 3 — Pre-Constructive Rules of Logic.

        Cada onda é classificada como :3 ou :5 baseado no RETRACE da onda SEGUINTE.

        Regras:
          Retrace 0-61.8%     → onda anterior = :5 (impulsiva, trend continuation)
          Retrace 61.8-100%   → ambíguo (dependente do contexto)
          Retrace 100-161.8%  → onda anterior = :3 (corretiva)
          Retrace 161.8%+     → onda anterior = :3 forte (ou x-wave)

        A onda i é o movimento do pivot[i-1] ao pivot[i].
        O retrace é definido pela onda i+1 (pivot[i] → pivot[i+1]).
        """
        if len(pivots) < 3:
            return pivots

        for i in range(0, len(pivots) - 2):
            wave_start  = pivots[i].price
            wave_end    = pivots[i + 1].price
            next_end    = pivots[i + 2].price

            wave_length = abs(wave_end - wave_start)
            if wave_length == 0:
                continue

            # Retrace da onda i+1 em relação à onda i
            next_retracement = abs(next_end - wave_end)
            retrace_ratio = next_retracement / wave_length

            # Assign structure label e confidence
            if retrace_ratio <= 0.382:
                pivots[i + 1].label = StructLabel.IMPULSE
                pivots[i + 1].label_confidence = 0.95
            elif retrace_ratio <= 0.618:
                pivots[i + 1].label = StructLabel.IMPULSE
                pivots[i + 1].label_confidence = 0.75
            elif retrace_ratio <= 0.786:
                # Zona ambígua — mais contexto necessário
                pivots[i + 1].label = StructLabel.UNKNOWN
                pivots[i + 1].label_confidence = 0.40
            elif retrace_ratio <= 1.0:
                pivots[i + 1].label = StructLabel.CORRECTIVE
                pivots[i + 1].label_confidence = 0.60
            elif retrace_ratio <= 1.618:
                pivots[i + 1].label = StructLabel.CORRECTIVE
                pivots[i + 1].label_confidence = 0.80
            else:
                # >161.8% — corretivo forte ou x-wave de pattern complexo
                pivots[i + 1].label = StructLabel.CORRECTIVE
                pivots[i + 1].label_confidence = 0.90

            pivots[i + 1].retrace_from_prev = retrace_ratio
            pivots[i + 1].label_confirmed_at = pivots[i + 2].confirmed_at

        return pivots

    def _resolved_label(self, pivot: Pivot, bar_idx: int) -> tuple[StructLabel, float]:
        label_ready = pivot.label_confirmed_at is not None and pivot.label_confirmed_at <= bar_idx
        if not label_ready:
            return StructLabel.UNKNOWN, 0.0
        return pivot.label, pivot.label_confidence

    # ───────────────────────────────────────────────────────────────────
    # SETUP A — WAVE 3 ONSET
    # ───────────────────────────────────────────────────────────────────

    def _setup_wave3(
        self,
        pivots: list[Pivot],
        bar_idx: int,
        cur_price: float,
        cur_rsi: float,
        cur_macd: float,
        cur_atr: float,
    ) -> Optional[WavePattern]:
        """
        Neely Wave 3 Setup — entrada no início da onda 3.

        Padrão buscado: W0(L) → W1(H) → W2(L) → preço girando para cima

        Validações Neely (além das 3 regras clássicas):
          1. W1 tem label :5 (retracement da W2 < 61.8%) → W1 foi verdadeiramente impulsiva
          2. W2 retrace entre 23.6% e 99% de W1 (R1: não 100%+)
          3. 0-2 Trendline confirma: preço acima da linha ao virar
          4. W2 classificada por tipo → Power Rating → Target calibrado
          5. W1 length > ATR × min_wave_atr (não é ruído)

        Score (máx ~10):
          R1 (violável = descarta):  obrigatório
          W1 label :5:              +1.5
          W2 fib zone (38.2-61.8%): +2.0
          Breakout > W1:            +2.0
          0-2 trendline acima:      +1.0
          MACD virando +:           +1.0
          RSI < 55 em W2:           +1.0
          W1 comprimento robusto:   +1.0
          Estrutura W1 forte (:5 conf > 0.7): +0.5
        """
        # Busca W0(L) → W1(H) → W2(L) nos pivôs mais recentes
        w0, w1h, w2l = self._find_L_H_L(pivots)
        if w0 is None:
            return None

        w0_p  = w0.price
        w1_p  = w1h.price
        w2_p  = w2l.price
        w1_len = w1_p - w0_p

        # W1 deve ser alta (bullish setup)
        if w1_len <= 0:
            return None

        # R1 — W2 não pode retrace 100%+ de W1 (INVIOLÁVEL)
        retrace_w2 = (w1_p - w2_p) / w1_len
        if w2_p <= w0_p:
            return None  # Violou R1

        # W1 mínimo significativo (filtro de ruído)
        if w1_len < cur_atr * self.min_wave_atr:
            return None

        # W2 retrace mínimo
        if retrace_w2 < self.w2_min_retrace:
            return None

        # Máximo de retrace (muito fundo = não é W2 válido)
        if retrace_w2 > self.w2_max_retrace:
            return None

        score = 0.0

        # W1 label :5 (Structure Label assignment)
        w1_label, w1_confidence = self._resolved_label(w1h, bar_idx)
        if w1_label == StructLabel.IMPULSE:
            score += 1.5 * w1_confidence
        elif w1_label == StructLabel.UNKNOWN:
            score += 0.5

        # W2 fib zone — 38.2%-61.8% é clássico; 23.6%-38.2% indica Running (mais poderoso)
        if 0.236 <= retrace_w2 <= 0.382:
            score += 2.0  # Running zone — power +3
        elif 0.382 < retrace_w2 <= 0.618:
            score += 2.0  # Ideal Neely zone
        elif 0.618 < retrace_w2 <= 0.786:
            score += 1.0  # Aceitável
        else:
            score += 0.5  # Profundo mas válido

        # Breakout: preço acima do topo de W1
        if cur_price >= w1_p:
            score += 2.0
        elif cur_price >= w1_p * 0.995:
            score += 1.2  # Quase no breakout
        elif cur_price >= w1_p * 0.985:
            score += 0.5

        # Confirma que retrace de W2 < 61.8% → W2 foi rasa = Running Correction
        # Isso é a Neely 0-2 trendline implícita
        tl_02_at_bar = self._trendline_value(w0.bar_idx, w0_p, w2l.bar_idx, w2_p, bar_idx)
        tl_02_slope = (
            (w2_p - w0_p) / (w2l.bar_idx - w0.bar_idx)
            if w2l.bar_idx != w0.bar_idx else 0.0
        )
        if cur_price > tl_02_at_bar:
            score += 1.0  # Preço acima da 0-2 line = W2 confirmada

        # MACD hist virando positivo
        if cur_macd > 0:
            score += 1.0
        elif cur_macd > -0.0001:
            score += 0.3

        # RSI razoável em W2 bottom (não sobrecomprado)
        if cur_rsi < 45:
            score += 1.0
        elif cur_rsi < 55:
            score += 0.5

        # W1 comprimento robusto (acima de 3× ATR = onda real)
        if w1_len >= cur_atr * 3:
            score += 1.0
        elif w1_len >= cur_atr * 2:
            score += 0.5

        # ── Classificar W2 → Power Rating ─────────────────────────────
        w2_type, power = self._classify_wave2(retrace_w2, w1h, w2l)

        # ── Target calibrado pelo Power Rating ────────────────────────
        mult = self.POWER_TO_TARGET.get(power, 1.618)
        target_161 = w2_p + w1_len * 1.618
        target_mult = w2_p + w1_len * mult

        # Stop: abaixo do fundo de W2 com margem de segurança (Neely: ATR × 0.5)
        stop = w2_p - cur_atr * 0.5

        # ── 0-2 e 2-4 trendlines ──────────────────────────────────────
        tl_24 = np.nan  # Não temos W4 ainda neste setup

        # ── Terminal suspeito? ─────────────────────────────────────────
        # Se W1 retraced menos que 38.2% e preço sobe devagar = pode ser Terminal
        is_terminal = (w1_label == StructLabel.CORRECTIVE)

        if score < 3.0:
            return None

        return WavePattern(
            bar_idx=bar_idx,
            pattern=PatternType.WAVE3_ONSET,
            score=min(score, 10.0),
            w2_type=w2_type,
            power_rating=power,
            w0_price=w0_p,
            w1_price=w1_p,
            w2_price=w2_p,
            tl_0_2_slope=tl_02_slope,
            tl_0_2_at_bar=tl_02_at_bar,
            tl_2_4_at_bar=tl_24,
            target_w3_161=target_161,
            target_w3_261=target_mult,
            stop_price=stop,
            is_terminal_suspect=is_terminal,
        )

    # ───────────────────────────────────────────────────────────────────
    # SETUP B — WAVE 5 ENTRY
    # ───────────────────────────────────────────────────────────────────

    def _setup_wave5(
        self,
        pivots: list[Pivot],
        bar_idx: int,
        cur_price: float,
        cur_rsi: float,
        cur_macd: float,
        cur_atr: float,
    ) -> Optional[WavePattern]:
        """
        Neely Wave 5 Setup — entrada na onda 5 após W4 completa.

        Padrão: W0(L) → W1(H) → W2(L) → W3(H) → W4(L)

        Validações Neely:
         1. R2: W3 > W1 em comprimento (onda 3 não é a mais curta)
         2. R3: W4 não entra no território de preço de W1 (overlap proibido)
         3. W4 retrace < 61.8% de W3 (normalmente)
         4. Alternação W2 vs W4 (tempo ou complexity diferente)
         5. 2-4 Trendline não quebrada pela onda corrente

        Score:
          R2 (W3 > W1):           +1.5
          R3 (no overlap):         +1.5
          W4 fib zone:             +2.0
          Alternação W2/W4:        +1.0
          2-4 trendline intacta:   +1.5
          Preço acima W4:          +1.5
          Divergência RSI:         +1.0
          Stage 1 confirmação:     +1.0 (2-4 quebrada depois de W5)
        """
        # Busca W0(L)→W1(H)→W2(L)→W3(H)→W4(L)
        w0, w1h, w2l, w3h, w4l = self._find_L_H_L_H_L(pivots)
        if w0 is None:
            return None

        w0_p  = w0.price
        w1_p  = w1h.price
        w2_p  = w2l.price
        w3_p  = w3h.price
        w4_p  = w4l.price

        w1_len = w1_p - w0_p
        w3_len = w3_p - w2_p

        if w1_len <= 0 or w3_len <= 0:
            return None

        # R2: W3 > W1 (não-negociável)
        if w3_len <= w1_len:
            return None

        # R3: W4 não entra no território de W1 top
        if w4_p <= w1_p:
            return None  # Overlap proibido (exceto Terminal)

        # W4 retrace de W3
        w3_top = w3_p
        w4_retrace = (w3_top - w4_p) / w3_len

        # W4 muito profundo = suspeita de erro no count
        if w4_retrace > 0.618:
            return None

        score = 0.0

        # R2 check (já validado — bonificação)
        score += 1.5

        # R3 check (já validado — bonificação)
        score += 1.5

        # W4 fib zone (Neely: W4 normalmente 23.6-38.2% de W3)
        if 0.236 <= w4_retrace <= 0.382:
            score += 2.0
        elif 0.382 < w4_retrace <= 0.50:
            score += 1.5
        elif 0.50 < w4_retrace <= 0.618:
            score += 0.5

        # Alternação W2 vs W4 (duração em barras deve ser diferente)
        w2_duration = w2l.bar_idx - w1h.bar_idx
        w4_duration = w4l.bar_idx - w3h.bar_idx
        if w2_duration > 0 and w4_duration > 0:
            ratio = min(w2_duration, w4_duration) / max(w2_duration, w4_duration)
            if ratio < 0.618:  # Bem diferentes — boa alternação
                score += 1.0
            elif ratio < 0.80:
                score += 0.5

        # 2-4 Trendline: preço atual acima da linha (W5 ainda não quebrou)
        tl_24_at_bar = self._trendline_value(w2l.bar_idx, w2_p, w4l.bar_idx, w4_p, bar_idx)
        tl_24_slope = (
            (w4_p - w2_p) / (w4l.bar_idx - w2l.bar_idx)
            if w4l.bar_idx != w2l.bar_idx else 0.0
        )
        if not np.isnan(tl_24_at_bar) and cur_price >= tl_24_at_bar:
            score += 1.5

        # Preço acima do fundo de W4 (entrando em W5)
        if cur_price > w4_p:
            score += 1.5

        # Divergência RSI: RSI em W4 < RSI em W2 (fraqueza bear = apoio bull W5)
        # Proxy: RSI atual < 55 (sem acesso direto ao RSI de W4, usa contexto)
        if cur_rsi < 55:
            score += 0.5
        if cur_rsi < 45:
            score += 0.5

        # ── Targets de W5 ─────────────────────────────────────────────
        # Neely: W5 frequentemente = W1 (igualdade) ou 61.8% de W1
        # Externo: W5 pode estender até 161.8% de W1→W3
        target_eq  = w4_p + w1_len           # W5 = W1 (igualdade)
        target_161 = w4_p + w1_len * 1.618   # 5th Extension

        # Stop: abaixo do fundo de W4
        stop = w4_p - cur_atr * 0.5

        # 0-2 trendline (para referência)
        tl_02_at_bar = self._trendline_value(w0.bar_idx, w0_p, w2l.bar_idx, w2_p, bar_idx)
        tl_02_slope = (
            (w2_p - w0_p) / (w2l.bar_idx - w0.bar_idx)
            if w2l.bar_idx != w0.bar_idx else 0.0
        )

        if score < 4.0:
            return None

        return WavePattern(
            bar_idx=bar_idx,
            pattern=PatternType.WAVE5_ENTRY,
            score=min(score, 10.0),
            w2_type=W2Type.UNKNOWN,
            power_rating=0,
            w0_price=w0_p,
            w1_price=w1_p,
            w2_price=w2_p,
            w3_price=w3_p,
            w4_price=w4_p,
            tl_0_2_slope=tl_02_slope,
            tl_0_2_at_bar=tl_02_at_bar,
            tl_2_4_slope=tl_24_slope,
            tl_2_4_at_bar=tl_24_at_bar,
            target_w5_eq=target_eq,
            target_w5_161=target_161,
            stop_price=stop,
        )

    # ───────────────────────────────────────────────────────────────────
    # SETUP C — TERMINAL REVERSAL
    # ───────────────────────────────────────────────────────────────────

    def _setup_terminal_reversal(
        self,
        pivots: list[Pivot],
        bar_idx: int,
        cur_price: float,
        cur_atr: float,
        df: pd.DataFrame,
    ) -> Optional[WavePattern]:
        """
        Neely Terminal Impulse Reversal (MEW Cap. 11).

        Terminal (Diagonal Triangle) = todas as ondas são :3, canal convergente,
        W4 sobrepõe W1, cada onda menor que a anterior.

        Após o Terminal a correção DEVE retraçar todo o padrão em ≤50% do tempo
        consumido pelo Terminal.

        Setup SHORT: detecta Terminal (ou seus suspeits) e entra na reversão.

        Sinais:
          1. 5 ondas com canal convergente (cada high menor que anterior em uptrend)
          2. W4 preço sobrepõe range de W1 (distingue Terminal de Trending)
          3. Todas as ondas são :3 em Structure Labels
          4. W5 quebrou a 1-3 trendline brevemente e voltou

        Score:
          Canal convergente:          +2.5
          W4 overlap W1:              +2.0
          Todas ondas :3:             +2.0
          W5 quebrou 1-3 line:        +1.5
          RSI divergência bearish:    +1.0
          Volume diminuindo:          +1.0
        """
        # Busca 5 ondas mais recentes: W1→W2→W3→W4→W5
        pattern = self._find_H_L_H_L_H(pivots)  # Downtrend: L→H→L→H→L
        if pattern is None:
            return None

        w0l, w1h, w2l, w3h, w4l, w5h = pattern
        if w5h is None:
            return None

        # Canal convergente: cada pico menor que anterior (Terminal uptrend)
        w1_p = w1h.price
        w3_p = w3h.price
        w5_p = w5h.price
        w2_p = w2l.price
        w4_p = w4l.price

        score = 0.0

        # Canal convergente (highs decrescentes em Terminal uptrend)
        # Em Terminal: W3 < W1, W5 < W3... ou W5 >W3 mas canal ainda converge
        if w5_p < w1_p * 1.05:  # W5 não muito maior que W1
            score += 1.0

        # W4 deve sobrepor território de W1 (key differentiator de Terminal)
        w1_bottom = w0l.price if w0l else w2l.price
        if w4_p <= w1_p:  # W4 cai abaixo do topo de W1 = overlap
            score += 2.0

        # Todas as ondas :3 (Terminal = ondas corretivas)
        terminal_labels = [
            self._resolved_label(w1h, bar_idx)[0],
            self._resolved_label(w2l, bar_idx)[0],
            self._resolved_label(w3h, bar_idx)[0],
            self._resolved_label(w4l, bar_idx)[0],
        ]
        corr_count = sum(1 for lb in terminal_labels if lb == StructLabel.CORRECTIVE)
        score += corr_count * 0.5  # até +2.0

        # Canal 1-3 (deve ser quebrado por W5 brevemente)
        tl_13_at_w5 = self._trendline_value(w1h.bar_idx, w1_p, w3h.bar_idx, w3_p, w5h.bar_idx)
        if not np.isnan(tl_13_at_w5) and w5_p > tl_13_at_w5:
            score += 1.5  # W5 quebrou 1-3 line = Terminal confirmado

        # Preço atual abaixo do pico de W5 (reversão começando)
        if cur_price < w5_p * 0.998:
            score += 1.0

        # Reversal direction: curprice caindo do topo = SHORT setup
        # Para simplificar, detectamos apenas reversão de alta → baixa
        stage1 = False
        if hasattr(self, '_check_stage1_terminal'):
            stage1 = self._check_stage1_terminal(w5h, bar_idx, df)

        if score < 4.0:
            return None

        # Target: reversal para o início do Terminal (W0)
        w0_p = w0l.price if w0l else w2_p - 0.01
        stop = w5_p + cur_atr * 0.5

        return WavePattern(
            bar_idx=bar_idx,
            pattern=PatternType.TERMINAL_REV,
            score=min(score, 10.0),
            w0_price=w0_p,
            w1_price=w1_p,
            w2_price=w2_p,
            w3_price=w3_p,
            w4_price=w4_p,
            target_w3_161=w0_p,   # Target = começo do Terminal
            stop_price=stop,
            is_terminal_suspect=True,
            stage1_confirmed=stage1,
        )

    # ───────────────────────────────────────────────────────────────────
    # WAVE 2 CLASSIFICATION → POWER RATING
    # ───────────────────────────────────────────────────────────────────

    def _classify_wave2(
        self, retrace_pct: float, w1h: Pivot, w2l: Pivot
    ) -> tuple[W2Type, int]:
        """
        Classifica a W2 pelo tipo de correção e retorna o Power Rating.

        Neely MEW Cap. 10 — Advanced Logic Rules:
          Running Correction     : retrace < 38.2%   → power +3 (W3 = 261.8%+ de W1)
          Zigzag deep / Common   : retrace 38.2-61.8% → power +1 (W3 = 161.8% de W1)
          Common Flat            : retrace 61.8-78.6% → power  0
          Irregular              : retrace 78.6-95%   → power  0
          Irregular Failure      : retrace 95-100%    → power +2 (W3 poderosa)
        """
        if retrace_pct < 0.382:
            return W2Type.RUNNING, 3
        elif retrace_pct < 0.618:
            return W2Type.ZIGZAG_DEEP, 1
        elif retrace_pct < 0.786:
            return W2Type.COMMON_FLAT, 0
        elif retrace_pct < 0.950:
            return W2Type.IRREGULAR, 0
        else:
            return W2Type.IRR_FAILURE, 2

    # ───────────────────────────────────────────────────────────────────
    # HELPERS — Pivot Pattern Search
    # ───────────────────────────────────────────────────────────────────

    def _find_L_H_L(self, pivots: list[Pivot]) -> tuple:
        """Busca o padrão L→H→L mais recente nos pivôs."""
        w0, w1h, w2l = None, None, None
        for i in range(len(pivots) - 1, -1, -1):
            pv = pivots[i]
            if pv.direction == 'L' and w2l is None:
                w2l = pv
            elif pv.direction == 'H' and w2l is not None and w1h is None:
                w1h = pv
            elif pv.direction == 'L' and w1h is not None and w0 is None:
                w0 = pv
                break
        return w0, w1h, w2l

    def _find_L_H_L_H_L(self, pivots: list[Pivot]) -> tuple:
        """Busca L→H→L→H→L mais recente."""
        w0, w1h, w2l, w3h, w4l = None, None, None, None, None
        for i in range(len(pivots) - 1, -1, -1):
            pv = pivots[i]
            if pv.direction == 'L' and w4l is None:
                w4l = pv
            elif pv.direction == 'H' and w4l is not None and w3h is None:
                w3h = pv
            elif pv.direction == 'L' and w3h is not None and w2l is None:
                w2l = pv
            elif pv.direction == 'H' and w2l is not None and w1h is None:
                w1h = pv
            elif pv.direction == 'L' and w1h is not None and w0 is None:
                w0 = pv
                break
        return w0, w1h, w2l, w3h, w4l

    def _find_H_L_H_L_H(self, pivots: list[Pivot]) -> Optional[tuple]:
        """Busca L→H→L→H→L→H mais recente (Terminal uptrend + peak)."""
        result = [None] * 6
        needed = ['L', 'H', 'L', 'H', 'L', 'H']
        pos = 5
        for i in range(len(pivots) - 1, -1, -1):
            pv = pivots[i]
            if pv.direction == needed[pos]:
                result[pos] = pv
                pos -= 1
                if pos < 0:
                    break
        if result[0] is None or result[5] is None:
            return None
        return tuple(result)

    # ───────────────────────────────────────────────────────────────────
    # HELPERS — Channeling + Fibonacci
    # ───────────────────────────────────────────────────────────────────

    def _trendline_value(
        self, bar_a: int, price_a: float, bar_b: int, price_b: float, bar_x: int
    ) -> float:
        """
        Calcula o valor de uma trendline (line through A, B) na barra bar_x.
        """
        if bar_b == bar_a:
            return float('nan')
        slope = (price_b - price_a) / (bar_b - bar_a)
        return price_a + slope * (bar_x - bar_a)

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ATR (Average True Range)."""
        high  = df['high']
        low   = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _get_val(self, df: pd.DataFrame, col: str, idx: int, default: float) -> float:
        """Pega valor de uma coluna com fallback."""
        if col not in df.columns:
            return default
        val = df[col].iloc[idx]
        try:
            fval = float(val)
            return default if np.isnan(fval) else fval
        except (TypeError, ValueError):
            return default


# ═══════════════════════════════════════════════════════════════════════════
# FIBONACCI UTILS (usados na estratégia)
# ═══════════════════════════════════════════════════════════════════════════

def fib_retracement_levels(start: float, end: float) -> dict:
    """
    Níveis de retracement Fibonacci de (start → end).
    Retorna dict {ratio: price_level}.
    """
    move = end - start
    return {
        0.0:   end,
        0.236: end - move * 0.236,
        0.382: end - move * 0.382,
        0.500: end - move * 0.500,
        0.618: end - move * 0.618,
        0.786: end - move * 0.786,
        1.000: start,
    }


def fib_extension_levels(start: float, end: float, retrace: float) -> dict:
    """
    Níveis de extensão Fibonacci projetados do (retrace) a partir do movimento (start→end).
    Retorna dict {ratio: price_level}.
    """
    move = end - start
    return {
        1.000: retrace + move * 1.000,
        1.272: retrace + move * 1.272,
        1.618: retrace + move * 1.618,
        2.000: retrace + move * 2.000,
        2.618: retrace + move * 2.618,
    }
