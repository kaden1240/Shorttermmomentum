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

def load_sp500():
    print("Loading S&P 500 tickers...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    tables = pd.read_html(html)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers
    
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
