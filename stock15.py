import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# 原有參數
START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
WIN_THRESHOLD_PCT = 0.05
LOOKAHEAD_DAYS = 30
MIN_BARS = 30

# 取得股價

def get_stock_data(stock_id, start_date=START_DATE, end_date=END_DATE):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {"dataset": "TaiwanStockPrice", "data_id": stock_id, "start_date": start_date, "end_date": end_date}
    try:
        res = requests.get(url, params=params, timeout=15).json()
    except Exception:
        return pd.DataFrame()

    if "data" not in res or len(res["data"]) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(res["data"])
    rename_map = {
        "Trading_Volume": "volume",
        "open": "open",
        "max": "high",
        "min": "low",
        "close": "close",
        "date": "date",
    }
    df = df.rename(columns=rename_map)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df

# 指標

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

# 技術規則

def rule_kd_low(df, i):
    if i < 1:
        return False
    return (
        df["STOCHk_14_3_3"].iloc[i-1] < df["STOCHd_14_3_3"].iloc[i-1]
        and df["STOCHk_14_3_3"].iloc[i] > df["STOCHd_14_3_3"].iloc[i]
        and df["STOCHk_14_3_3"].iloc[i] < 20
        and df["STOCHd_14_3_3"].iloc[i] < 20
    )

def rule_rsi(df, i):
    if i < 1:
        return False
    return df["RSI_14"].iloc[i] > df["RSI_14"].iloc[i-1] and df["RSI_14"].iloc[i] < 30

def rule_macd(df, i):
    return i>0 and df["macd_hist"].iloc[i-1] < 0 <= df["macd_hist"].iloc[i]

def rule_boll(df, lower, i):
    if i < 1 or lower is None:
        return False
    return df["close"].iloc[i-1] < df[lower].iloc[i-1] and df["close"].iloc[i] > df[lower].iloc[i]

# 回測

def evaluate_entry(df, i):
    entry = df["close"].iloc[i]
    end_i = min(i + LOOKAHEAD_DAYS, len(df) - 1)
    seg = df.iloc[i+1:end_i+1]
    if seg.empty:
        return None, None
    max_run = (seg["high"].max() - entry) / entry
    final = (seg["close"].iloc[-1] - entry) / entry
    return max_run, final


def backtest(df, lower):
    rules = {
        "RSI 超賣翻升": lambda i: rule_rsi(df, i),
        "布林下軌反彈": lambda i: rule_boll(df, lower, i),
        "MACD 直方圖拐頭": lambda i: rule_macd(df, i),
        "KD 低檔黃金交叉": lambda i: rule_kd_low(df, i),
    }
    stats = {}
    for name, fn in rules.items():
        trades = wins = 0
        runups = []
        finals = []
        trig = []
        for i in range(MIN_BARS, len(df)-1):
            if not fn(i): continue
            trades += 1
            trig.append(df["date"].iloc[i])

            max_r, final_r = evaluate_entry(df, i)
            if max_r is None: continue
            runups.append(max_r)
            finals.append(final_r)
            if max_r >= WIN_THRESHOLD_PCT: wins += 1

        stats[name] = {
            "trades": trades,
            "wins": wins,
            "win_rate_pct": wins/trades*100 if trades else 0,
            "avg_max_runup_pct": np.mean(runups)*100 if runups else 0,
            "avg_final_return_pct": np.mean(finals)*100 if finals else 0,
            "trigger_dates": trig,
        }
    return stats

# 目標價

def compute_tp(close, macd_avg):
    return close * (1 + macd_avg * 0.6)


def check_current(df, upper, lower, stats):
    macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100
    if macd_avg <= 0: return []

    i = len(df) - 1
    close = df["close"].iloc[i]
    out = []

    if rule_kd_low(df, i): out.append(("KD 低檔黃金交叉", compute_tp(close, macd_avg)))
    if rule_rsi(df, i): out.append(("RSI 超賣翻升", compute_tp(close, macd_avg)))
    if rule_macd(df, i): out.append(("MACD 直方圖拐頭", compute_tp(close, macd_avg)))
    if rule_boll(df, lower, i): out.append(("布林下軌反彈", compute_tp(close, macd_avg)))

    return out

# 排序

def rank_stats(stats):
    return sorted(stats.items(), key=lambda x:(x[1]["trades"], x[1]["win_rate_pct"], x[1]["avg_max_runup_pct"]), reverse=True)


# ============================== Streamlit UI ==============================
st.title("📈 多股票技術分析 + 回測系統 (Streamlit)")

user_input = st.sidebar.text_input("請輸入股票代號（用逗號分隔）", "2330,2603,2317")
run_btn = st.sidebar.button("開始分析")

if run_btn:
    stock_ids = [s.strip() for s in user_input.split(",") if s.strip()]

    for sid in stock_ids:
        st.header(f"📌 股票 {sid}")

        df = get_stock_data(sid)
        if df.empty:
            st.error("⚠️ 抓不到資料")
            continue

        df, upper, lower = add_indicators(df)
        if lower is None or len(df) < MIN_BARS:
            st.warning("資料不足")
            continue

        stats = backtest(df, lower)
        macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100
        current = check_current(df, upper, lower, stats)

        st.subheader(f"最新日期：{df['date'].iloc[-1]}")
        st.write(f"收盤價：{df['close'].iloc[-1]:.2f}")

        # ------ 最新 K 線規則 ------
        st.markdown("### 🔍 最新一根K線規則（目標價 = MACD 回測 × 0.6）")
        if current:
            for rule, tp in current:
                last_date = stats[rule]["trigger_dates"][-1] if stats[rule]["trigger_dates"] else "無"
                st.write(f"- **{rule}** | 目標價：{tp:.2f} | 最後觸發：{last_date}")
        else:
            st.write("（本日無規則觸發）")

        # ------ 回測結果 ------
        st.markdown("### 📊 回測結果")
        ranked = rank_stats(stats)

        df_table = []
        for name, s in ranked:
            last_dates = ", ".join(s["trigger_dates"][-3:]) if s["trigger_dates"] else "無"
            df_table.append([
                name,
                s["trades"],
                f"{s['win_rate_pct']:.2f}%",
                f"{s['avg_max_runup_pct']:.2f}%",
                f"{s['avg_final_return_pct']:.2f}%",
                last_dates,
            ])

        st.table(pd.DataFrame(df_table, columns=["規則","觸發次數","勝率","平均最大漲幅","平均最終報酬","觸發日期（最後3筆）"]))

        st.write(f"**MACD 平均最大漲幅：{macd_avg*100:.2f}%**")
