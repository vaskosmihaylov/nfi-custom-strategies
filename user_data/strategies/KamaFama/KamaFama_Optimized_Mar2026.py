from functools import reduce

from pandas import DataFrame

from freqtrade.strategy import IntParameter

from KamaFama_2_20250115 import KamaFama_2_20250115


class KamaFama_2_20250115_OptMar2026(KamaFama_2_20250115):
    sell_fastx = IntParameter(45, 100, default=78, space="sell", optimize=True)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""

        buy = (
            (dataframe["kama"] > dataframe["fama"])
            & (dataframe["fama"] > dataframe["mama"] * 0.981)
            & (dataframe["r_14"] < -66.0)
            & (dataframe["mama_diff"] < -0.025)
            & (dataframe["cti"] < -0.78)
            & (dataframe["close"].rolling(48).max() >= dataframe["close"] * 1.05)
            & (dataframe["close"].rolling(288).max() >= dataframe["close"] * 1.125)
            & (dataframe["rsi_84"] < 56)
            & (dataframe["rsi_112"] < 56)
            & (dataframe["volume"] > 0)
        )
        conditions.append(buy)
        dataframe.loc[buy, "enter_tag"] += "buy"

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe
