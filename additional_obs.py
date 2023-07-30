import numpy as np
import pandas as pd
import yfinance as yf


def yf_downloader(tickers: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers, end=end
    )
    df = df['Adj Close'].pct_change()
    return df


kodex_3yr_b = yf_downloader(tickers='114260.KS', end='2023-07-28')
ks200 = yf_downloader(tickers='^KS200', end='2023-07-28')
