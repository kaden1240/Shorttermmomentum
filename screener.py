import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
import time
import yfinance as yf
import pandas as pd
import numpy as np

# === CONFIGURATION ===
SHEET_NAME = "Short Term Momentum Engine"

# ... keep all your other functions as-is ...

# === Run Screener ===

from io import StringIO
import requests
import pandas as pd

from io import StringIO
import requests
import pandas as pd

def load_sp500():
    print("Loading S&P 500 tickers...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text

    # Use StringIO to fix the FutureWarning
    tables = pd.read_html(StringIO(html))

    df = None
    for table in tables:
        table.columns = table.columns.map(str)
        if any("symbol" in c.lower() or "ticker" in c.lower() for c in table.columns):
            df = table
            break

    if df is None:
        raise ValueError("No valid S&P 500 table found. Wikipedia page structure may have changed.")

    symbol_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
    tickers = df[symbol_col].astype(str).tolist()
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    print(f"Loaded {len(tickers)} tickers.")
    return tickers

def safe_float(x):
    if hasattr(x, 'iloc'):
        return float(x.iloc[0])
    return float(x)



def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_stock_data(symbol):
    df = yf.download(symbol,
                     period="60d",
                     interval="1d",
                     auto_adjust=False,
                     progress=False)
    if df.empty or len(df) < 35:
        print(f"No or insufficient data for {symbol}")
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

    df['RSI'] = calculate_rsi(df['Close'], 14)
    rsi = df['RSI'].iloc[-1]
    prev_rsi = df['RSI'].iloc[-2]

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd = macd_line.iloc[-1]
    macd_signal = signal_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    prev_macd_signal = signal_line.iloc[-2]
    macd_hist = macd - macd_signal
    prev_hist = prev_macd - prev_macd_signal

    low14 = df['Low'].rolling(window=14).min().iloc[-1]
    high14 = df['High'].rolling(window=14).max().iloc[-1]
    close = latest['Close']

    if isinstance(high14, pd.Series):
        high14 = high14.item()
    if isinstance(low14, pd.Series):
        low14 = low14.item()

    if high14 != low14:
        stoch_k = ((close - low14) / (high14 - low14)) * 100
    else:
        stoch_k = 0

    stoch_k_series = ((df['Close'] - df['Low'].rolling(window=14).min()) /
                      (df['High'].rolling(window=14).max() -
                       df['Low'].rolling(window=14).min())) * 100
    stoch_d = stoch_k_series.rolling(window=3).mean().iloc[-1]

    volume = latest['Volume']
    avg_volume = df['Volume'].rolling(window=10).mean().iloc[-1]

    float_shares = 2_000_000_000  # replace with real float if available
    high = latest['High']
    low = latest['Low']
    open_price = latest['Open']

    return {
        "symbol": symbol,
        "change_pct": safe_float(change_pct),
        "close": safe_float(close),
        "open": safe_float(open_price),
        "high": safe_float(high),
        "low": safe_float(low),
        "volume": safe_float(volume),
        "avg_volume": safe_float(avg_volume),
        "float": float(float_shares),
        "rsi": safe_float(rsi),
        "prev_rsi":
        safe_float(df['RSI'].iloc[-2]),  # Yesterday's RSI (for bounce logic
        "macd_line": safe_float(macd),
        "macd_signal": safe_float(macd_signal),
        "macd_hist": safe_float(macd - macd_signal),
        "prev_hist": safe_float(prev_macd - prev_macd_signal),
        "stoch_k": safe_float(stoch_k),
        "stoch_d": safe_float(stoch_d)
    }

def passes_momentum_criteria(data):
    return (data["volume"] > 0.9 * data["avg_volume"]
            and data["close"] >= 0.9 * data["high"] and 50 < data["rsi"] < 65
            and 60 < data["stoch_k"] < 75 and data["stoch_k"] > data["stoch_d"]
            and data["macd_line"] > data["macd_signal"]
            and data["macd_hist"] > data["prev_hist"])


def run_screener():
    """Run the stock screener and update Google Sheets"""
    print("Starting stock screener...")
    
    # --- Google Sheets Authentication using environment variable ---
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Load credentials JSON from environment variable
    creds_json = os.environ.get("GS_CREDENTIALS")
    if not creds_json:
        raise ValueError("GS_CREDENTIALS environment variable is not set!")
    
    creds_dict = json.loads(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    
    sheet = client.open(SHEET_NAME).sheet1
    sheet2 = client.open(SHEET_NAME).get_worksheet(1)
    sheet3 = client.open(SHEET_NAME).get_worksheet(2)
    
    # Load tickers
    SP500_TICKERS = load_sp500()
    
    # Clear and set up headers
    sheet.clear()
    sheet.append_row([
        "High Momentum Stocks", "Change %", "Close", "Volume", "RSI", "MACD Diff",
        "Stoch K"
    ])
    
    
    # Screen all tickers
    for ticker in SP500_TICKERS:
        print(f"Checking {ticker}...")
        data = get_stock_data(ticker)
        if data:
            print(f"Pulled data: {data}")
            if passes_momentum_criteria(data):
                print(f"✅ {ticker} PASSES")
                sheet.append_row([
                    data["symbol"],
                    round(data["change_pct"], 2), data["close"], data["volume"],
                    round(data["rsi"], 1),
                    round(data["macd_line"] - data["macd_signal"], 2),
                    round(data["stoch_k"], 1)
                ])
            else:
                print(f"❌ {ticker} does NOT meet criteria")
        else:
            print(f"❌ No data for {ticker}")
        time.sleep(0.5)  # avoid API rate limits
    
    sheet2.clear()
    
    print("✅ Done scanning!")
    print(f"Successfully connected to sheet: {sheet.title}")
    return "Screener completed successfully!"

if __name__ == "__main__":
    run_screener()
