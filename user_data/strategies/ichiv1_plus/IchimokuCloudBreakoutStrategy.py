from datetime import datetime

from pandas import DataFrame

from freqtrade.strategy import IntParameter, IStrategy


class IchimokuCloudBreakoutStrategy(IStrategy):
    """
    Ichimoku Kinko Hyo cloud breakout strategy.

    What this strategy does:
    - Computes the full Ichimoku system:
        - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        - Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 bars forward
        - Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 bars forward
        - Chikou Span: Close shifted 26 bars back
    - Enters long when:
        (a) Price is above both Senkou Span A and Span B (above the cloud).
        (b) Tenkan-sen is above Kijun-sen (short-term trend aligns with long-term).
        (c) Chikou Span is above price from 26 bars ago (past confirms current strength).
    - Enters short under mirrored conditions (below cloud, Tenkan < Kijun).
    - Exits when Tenkan crosses back through Kijun against the trade direction.

    Why it may work:
    Ichimoku is a comprehensive Japanese technical system that incorporates multiple
    timeframes of price information into a single chart. The cloud acts as a
    dynamic support/resistance zone. A price above a bullish cloud (Span A > Span B)
    signals that the medium-term trend is up, and requiring all three component
    confirmations reduces false entries significantly. It is widely used and
    the logic can be understood and defended.

    Expected failure modes:
    - Ichimoku requires significant bar history (52+ bars minimum); early data is noisy.
    - The system is slower to enter and exit than price-action-only approaches.
    - In choppy markets, price oscillates through the cloud, producing repeated signals.

    Retail-friendly because:
    - All components are based on simple price math (no complex transforms).
    - The visual cloud representation makes it easy to explain entries on a chart.
    - 4h timeframe keeps trade frequency manageable.
    """

    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "4h"
    startup_candle_count: int = 130
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 0.20}
    stoploss = -0.07
    trailing_stop = False

    tenkan_period = IntParameter(7, 15, default=9, space="buy", optimize=True, load=True)
    kijun_period = IntParameter(20, 35, default=26, space="buy", optimize=True, load=True)
    senkou_b_period = IntParameter(44, 60, default=52, space="buy", optimize=True, load=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tenkan": {"color": "blue"},
            "kijun": {"color": "red"},
            "span_a": {"color": "rgba(0,255,0,0.2)"},
            "span_b": {"color": "rgba(255,0,0,0.2)"},
        },
        "subplots": {},
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tenkan_p = int(self.tenkan_period.value)
        kijun_p = int(self.kijun_period.value)
        senkou_b_p = int(self.senkou_b_period.value)
        displacement = kijun_p  # standard: displace forward by kijun_period bars

        # Tenkan-sen: (N-period high + N-period low) / 2
        dataframe["tenkan"] = (
            dataframe["high"].rolling(tenkan_p).max() + dataframe["low"].rolling(tenkan_p).min()
        ) / 2

        # Kijun-sen: (M-period high + M-period low) / 2
        dataframe["kijun"] = (
            dataframe["high"].rolling(kijun_p).max() + dataframe["low"].rolling(kijun_p).min()
        ) / 2

        # Senkou Span A: (Tenkan + Kijun) / 2, shifted forward by displacement
        span_a_raw = (dataframe["tenkan"] + dataframe["kijun"]) / 2
        dataframe["span_a"] = span_a_raw.shift(displacement)

        # Senkou Span B: (senkou_b_p high + senkou_b_p low) / 2, shifted forward
        span_b_raw = (
            dataframe["high"].rolling(senkou_b_p).max()
            + dataframe["low"].rolling(senkou_b_p).min()
        ) / 2
        dataframe["span_b"] = span_b_raw.shift(displacement)

        # Causal Chikou reference: compare current close against close from
        # displacement bars ago (no future-candle access).
        dataframe["chikou"] = dataframe["close"].shift(displacement)

        # Cloud boundaries (using current cloud values = the span values computed
        # from displacement bars ago, which is what is "current" in the cloud)
        # For trading signals, we use the span values that correspond to NOW
        # (i.e., span computed from data displacement bars ago, now visible at t=0)
        dataframe["cloud_top"] = dataframe[["span_a", "span_b"]].max(axis=1)
        dataframe["cloud_bot"] = dataframe[["span_a", "span_b"]].min(axis=1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: price above cloud, Tenkan above Kijun, current close above
        # displacement-bars-ago close.
        above_cloud = (dataframe["close"] > dataframe["cloud_top"]) & (
            dataframe["cloud_top"].notna()
        )
        tenkan_above_kijun = dataframe["tenkan"] > dataframe["kijun"]
        chikou_bullish = dataframe["close"] > dataframe["chikou"]

        long_setup = (
            above_cloud & tenkan_above_kijun & chikou_bullish & (dataframe["volume"] > 0)
        )

        # Short: price below cloud, Tenkan below Kijun, current close below
        # displacement-bars-ago close.
        below_cloud = (dataframe["close"] < dataframe["cloud_bot"]) & (
            dataframe["cloud_bot"].notna()
        )
        tenkan_below_kijun = dataframe["tenkan"] < dataframe["kijun"]
        chikou_bearish = dataframe["close"] < dataframe["chikou"]

        short_setup = (
            below_cloud & tenkan_below_kijun & chikou_bearish & (dataframe["volume"] > 0)
        )

        dataframe.loc[long_setup, ["enter_long", "enter_tag"]] = (1, "ichimoku_cloud_long")
        dataframe.loc[short_setup, ["enter_short", "enter_tag"]] = (1, "ichimoku_cloud_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when Tenkan crosses below Kijun (short-term trend reversal)
        exit_long = (dataframe["tenkan"] < dataframe["kijun"]) & (
            dataframe["tenkan"].shift(1) >= dataframe["kijun"].shift(1)
        )
        # Exit short when Tenkan crosses above Kijun
        exit_short = (dataframe["tenkan"] > dataframe["kijun"]) & (
            dataframe["tenkan"].shift(1) <= dataframe["kijun"].shift(1)
        )

        dataframe.loc[exit_long & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (
            1,
            "ichimoku_exit_long",
        )
        dataframe.loc[exit_short & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (
            1,
            "ichimoku_exit_short",
        )
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(2.0, max_leverage)