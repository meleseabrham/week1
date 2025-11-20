"""Run TA-Lib and PyNance based analytics on cached price data."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import talib as ta
from pynance.tech import movave as pn_movave
from pynance.tech import simple as pn_simple

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

PRICE_DIR = Path("data/processed/prices")
TECH_DIR = Path("data/processed/technical_metrics")
FIG_DIR = Path("reports/figures")

TECH_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.use("Agg")


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Close"])


def compute_talib_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values
    df["SMA_20"] = ta.SMA(close, timeperiod=20)
    df["EMA_50"] = ta.EMA(close, timeperiod=50)
    df["RSI_14"] = ta.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = ta.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist
    df["ATR_14"] = ta.ATR(
        df["High"].values, df["Low"].values, close, timeperiod=14
    )
    return df


def compute_pynance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    close_df = df[["Close"]]
    growth_df = pn_simple.growth(
        close_df, selection="Close", n_sessions=5, outputcol="PN_GROWTH_5"
    )
    vol_df = pn_movave.volatility(
        close_df, selection="Close", window=20, outputcol="PN_VOL_20"
    )
    movave_df = pn_movave.sma(
        close_df, selection="Close", window=10, outputcol="PN_MOVAVE_10"
    )
    df["PN_GROWTH_5"] = growth_df.reindex(df.index)["PN_GROWTH_5"]
    df["PN_VOL_20"] = vol_df.reindex(df.index)["PN_VOL_20"]
    df["PN_MOVAVE_10"] = movave_df.reindex(df.index)["PN_MOVAVE_10"]
    return df


def build_visualizations(df: pd.DataFrame, ticker: str) -> None:
    latest = df.tail(180).copy()
    apds = [
        mpf.make_addplot(latest["SMA_20"], color="tab:blue"),
        mpf.make_addplot(latest["EMA_50"], color="tab:orange"),
        mpf.make_addplot(
            latest["RSI_14"], panel=1, color="purple", ylabel="RSI", ylim=(0, 100)
        ),
        mpf.make_addplot(
            latest["MACD"], panel=2, color="green", ylabel="MACD"
        ),
        mpf.make_addplot(
            latest["MACD_SIGNAL"], panel=2, color="red",
        ),
        mpf.make_addplot(
            latest["MACD_HIST"], panel=2, type="bar", color="grey"
        ),
    ]
    fig_path = FIG_DIR / f"{ticker}_technicals.png"
    mpf.plot(
        latest,
        type="candle",
        mav=(),
        volume=True,
        addplot=apds,
        style="yahoo",
        figratio=(14, 9),
        figscale=1.2,
        title=f"{ticker} – 6-Month Technicals",
        savefig=fig_path,
    )
    plt.close("all")


def summarize_metrics(df: pd.DataFrame, ticker: str) -> dict[str, float]:
    latest = df.dropna().iloc[-1]
    return {
        "ticker": ticker,
        "close": latest["Close"],
        "sma_20": latest.get("SMA_20"),
        "ema_50": latest.get("EMA_50"),
        "rsi_14": latest.get("RSI_14"),
        "macd_hist": latest.get("MACD_HIST"),
        "atr_14": latest.get("ATR_14"),
        "pn_growth_5": latest.get("PN_GROWTH_5"),
        "pn_vol_20": latest.get("PN_VOL_20"),
    }


def main() -> None:
    summaries: list[dict[str, float]] = []
    for price_file in PRICE_DIR.glob("*.csv"):
        ticker = price_file.stem.upper()
        logging.info("Processing %s", ticker)
        df = load_price_data(price_file)
        df = compute_talib_indicators(df)
        df = compute_pynance_metrics(df)
        output_path = TECH_DIR / f"{ticker}_technicals.csv"
        df.to_csv(output_path, index=True)
        summaries.append(summarize_metrics(df, ticker))
        build_visualizations(df, ticker)

    summary_df = pd.DataFrame(summaries).set_index("ticker")
    summary_path = TECH_DIR / "technical_summary.csv"
    summary_df.to_csv(summary_path)
    logging.info(
        "Saved detailed indicator grids to %s and summary to %s",
        TECH_DIR,
        summary_path,
    )


if __name__ == "__main__":
    main()

