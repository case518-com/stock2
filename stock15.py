import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import os

# =========================
# 讀取 stock.txt
# =========================
def load_stock_list(filename="stock.txt"):
    if not os.path.exists(filename):
        return {}
    stock_list = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                stock_list[parts[0]] = parts[0]
            else:
                stock_list[parts[0]] = " ".join(parts[1:])
    return stock_list

START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
WIN_THRESHOLD_PCT = 0.05
LOOKAHEAD_DAYS = 30
MIN_BARS = 30

# API 抓取台股資料
def get_stock_data(stock_id, start_date=START_DATE, end_date=END_DATE):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date
    }
    try:
        res = requests.get(url, params=params, timeout=15).json()
    except Exception:
        return pd.DataFrame()
    if "data" not in res or len(res["data"]) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(res["data"])
    df = df.rename(columns={
        "Trading_Volume": "volume",
        "max": "high",
        "min": "low",
    })
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

# 加技術指標
def add_indicators(df):
    if df.empty or len(df) < MIN_BARS:
        return df, None, None
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    kdj = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df = pd.concat([df, kdj], axis=1)
    bb = ta.bbands(df["close"], length=20)
    df = pd.concat([df, bb], axis=1)
    upper = next((c for c in df.columns if "BBU_" in c), None)
    lower = next((c for c in df.columns if "BBL_" in c), None)
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd_line"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    return df, upper, lower

# =========================
# 四大規則
# =========================
def rule_kd_low_cross(df, i):
    if i < 1: return False
    if "STOCHk_14_3_3" not in df or "STOCHd_14_3_3" not in df: return False
    k_prev, d_prev = df["STOCHk_14_3_3"].iloc[i-1], df["STOCHd_14_3_3"].iloc[i-1]
    k, d = df["STOCHk_14_3_3"].iloc[i], df["STOCHd_14_3_3"].iloc[i]
    return (k_prev < d_prev) and (k > d) and (k < 20) and (d < 20)

def rule_rsi_oversold_rebound(df, i):
    if i < 1: return False
    r_prev = df["RSI_14"].iloc[i-1]
    r = df["RSI_14"].iloc[i]
    return (r < 30) and (r > r_prev)

def rule_macd_turning_up(df, i):
    if i < 1: return False
    h_prev = df["macd_hist"].iloc[i-1]
    h = df["macd_hist"].iloc[i]
    return (h_prev < 0 and h >= 0) or (h_prev < 0 and h > h_prev)

def rule_bollinger_lower_rebound(df, lower_col, i):
    if i < 1: return False
    close_prev = df["close"].iloc[i-1]
    close_now = df["close"].iloc[i]
    return (close_prev < df[lower_col].iloc[i-1]) and (close_now > df[lower_col].iloc[i])

# 回測
def evaluate_entry_runup_and_final(df, i):
    entry = df["close"].iloc[i]
    end_index = min(i + LOOKAHEAD_DAYS, len(df) - 1)
    seg = df.iloc[i+1:end_index+1]
    if seg.empty: return None, None
    max_price = seg["high"].max()
    max_runup = (max_price - entry) / entry
    final_return = (seg["close"].iloc[-1] - entry) / entry
    return max_runup, final_return

# 回測統計
def backtest_rules_full(df, lower_col):
    rules = {
        "RSI 超賣翻升": lambda i: rule_rsi_oversold_rebound(df, i),
        "布林下軌反彈": lambda i: rule_bollinger_lower_rebound(df, lower_col, i),
        "MACD 直方圖拐頭": lambda i: rule_macd_turning_up(df, i),
        "KD 低檔黃金交叉": lambda i: rule_kd_low_cross(df, i)
    }
    stats = {}
    for name, fn in rules.items():
        trades=wins=0
        runups=[]; finals=[]
        for i in range(MIN_BARS, len(df)-1):
            if fn(i):
                trades += 1
                maxr, fin = evaluate_entry_runup_and_final(df, i)
                if maxr is None: continue
                runups.append(maxr); finals.append(fin)
                if maxr >= WIN_THRESHOLD_PCT: wins += 1
        stats[name] = {
            "trades": trades,
            "win_rate_pct": (wins/trades*100) if trades>0 else 0,
            "avg_max_runup_pct": (np.mean(runups)*100) if runups else 0,
            "avg_final_return_pct": (np.mean(finals)*100) if finals else 0
        }
    return stats

# 目標價
def compute_target_price(close, macd_avg):
    return close * (1 + macd_avg * 0.8)

# Streamlit 開始
st.title("📈 AI 技術面選股工具 (Streamlit 版)")
st.write("本工具依四大技術分析規則進行回測，並依 MACD 平均最大漲幅 ×0.8 計算目標價")

stocks = load_stock_list("stock.txt")
if not stocks:
    st.error("找不到 stock.txt")
else:
    stock_id = st.selectbox("選擇股票代號", list(stocks.keys()))
    if st.button("開始分析"):
        df = get_stock_data(stock_id)
        if df.empty:
            st.error("無法取得資料")
        else:
            df, upper, lower = add_indicators(df)

            stats = backtest_rules_full(df, lower)
            macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100.0
            close = df["close"].iloc[-1]
            target = compute_target_price(close, macd_avg)

            st.subheader(f"📌 {stocks[stock_id]} ({stock_id}) 最新資料：{df['date'].iloc[-1]}")
            st.write(f"收盤價：{close:.2f}")
            st.write(f"MACD 平均最大漲幅：{macd_avg*100:.2f}%")
            st.write(f"🔥 計算目標價：**{target:.2f}**")

            st.subheader("📊 規則回測結果")
            st.dataframe(pd.DataFrame(stats).T)

# ==============================
# 🎨 使用 Streamlit 原生圖表，不依賴 plotly/matplotlib
# ==============================

# 回測參數調整區
st.sidebar.header("回測參數設定")
lookahead_days = st.sidebar.slider("回測觀察天數 (Lookahead)", 10, 120, 30)
win_threshold = st.sidebar.slider("勝率判定門檻 (%)", 1, 20, 5) / 100

# 日期顯示轉為 index（方便 Streamlit chart）
df_chart = df.copy()
df_chart = df_chart.set_index("date")

# K 線 (簡易版：用 open/high/low/close 多線圖)
if st.checkbox("顯示 K 線（簡易折線版）"):
    st.subheader("K 線（Streamlit 無套件版）")
    st.line_chart(df_chart[["open", "high", "low", "close"]])

# RSI
if st.checkbox("顯示 RSI 圖"):
    st.subheader("RSI 14")
    st.line_chart(df_chart[["RSI_14"]])

# MACD
if st.checkbox("顯示 MACD 圖"):
    st.subheader("MACD Line & Signal")
    st.line_chart(df_chart[["macd_line", "macd_signal"]])
    st.subheader("MACD Histogram")
    st.bar_chart(df_chart[["macd_hist"]])
