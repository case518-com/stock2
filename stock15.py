# stock14.py  (股票清單改成寫在程式裡，不讀 stock.txt)

import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# =========================
# 直接寫死股票清單
# =========================
STOCK_LIST = {
    "2330": "台積電",
    "2603": "長榮",
    "2609": "陽明",
}

# =========================
# 基本設定
# =========================
START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

WIN_THRESHOLD_PCT = 0.05   # +5% 判定勝利
LOOKAHEAD_DAYS = 30
MIN_BARS = 30

# =========================
# 抓台股資料
# =========================
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
    rename_map = {
        "Trading_Volume": "volume",
        "open": "open",
        "max": "high",
        "min": "low",
        "close": "close",
        "date": "date"
    }
    df = df.rename(columns=rename_map)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df

# =========================
# 加指標
# =========================
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
# 規則定義
# =========================
def rule_kd_low_cross(df, i):
    if i < 1:
        return False
    k_prev = df["STOCHk_14_3_3"].iloc[i-1]
    d_prev = df["STOCHd_14_3_3"].iloc[i-1]
    k = df["STOCHk_14_3_3"].iloc[i]
    d = df["STOCHd_14_3_3"].iloc[i]
    return (k_prev < d_prev) and (k > d) and (k < 20) and (d < 20)

def rule_rsi_oversold_rebound(df, i):
    if i < 1:
        return False
    r_prev = df["RSI_14"].iloc[i-1]
    r = df["RSI_14"].iloc[i]
    return (r < 30) and (r > r_prev)

def rule_macd_turning_up(df, i):
    if i < 1:
        return False
    h_prev = df["macd_hist"].iloc[i-1]
    h = df["macd_hist"].iloc[i]
    return (h_prev < 0 and h >= 0) or (h_prev < 0 and h > h_prev)

def rule_bollinger_lower_rebound(df, lower_col, i):
    if i < 1 or lower_col is None:
        return False
    return (
        df["close"].iloc[i-1] < df[lower_col].iloc[i-1] and
        df["close"].iloc[i] > df[lower_col].iloc[i]
    )

# =========================
# 回測
# =========================
def evaluate_entry_runup_and_final(df, i):
    entry = df["close"].iloc[i]
    end_index = min(i + LOOKAHEAD_DAYS, len(df) - 1)
    seg = df.iloc[i+1:end_index+1]
    if seg.empty:
        return None, None

    max_price = seg["high"].max()
    max_runup = (max_price - entry) / entry
    final_return = (seg["close"].iloc[-1] - entry) / entry
    return max_runup, final_return

def backtest_rules_full(df, lower_col):
    rules = {
        "RSI 超賣翻升": lambda i: rule_rsi_oversold_rebound(df, i),
        "布林下軌反彈": lambda i: rule_bollinger_lower_rebound(df, lower_col, i),
        "MACD 直方圖拐頭": lambda i: rule_macd_turning_up(df, i),
        "KD 低檔黃金交叉": lambda i: rule_kd_low_cross(df, i),
    }

    stats = {}
    for name, rule_fn in rules.items():
        trades = wins = 0
        runups, finals = [], []

        for i in range(MIN_BARS, len(df)-1):
            if rule_fn(i):
                trades += 1
                max_run, final_ret = evaluate_entry_runup_and_final(df, i)
                if max_run is None:
                    continue

                runups.append(max_run)
                finals.append(final_ret)

                if max_run >= WIN_THRESHOLD_PCT:
                    wins += 1

        stats[name] = {
            "trades": trades,
            "wins": wins,
            "win_rate_pct": (wins / trades * 100) if trades > 0 else 0,
            "avg_max_runup_pct": (np.mean(runups) * 100) if runups else 0,
            "avg_final_return_pct": (np.mean(finals) * 100) if finals else 0,
        }

    return stats

# =========================
# 目標價
# =========================
def compute_target_price(close, macd_avg):
    return close * (1 + macd_avg * 0.6)

def check_current(df, lower_col, stats):
    macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100.0
    if macd_avg <= 0:
        return []

    i = len(df) - 1
    close = df["close"].iloc[i]
    matched = []

    rules = [
        ("KD 低檔黃金交叉", rule_kd_low_cross),
        ("RSI 超賣翻升", rule_rsi_oversold_rebound),
        ("MACD 直方圖拐頭", rule_macd_turning_up),
        ("布林下軌反彈", lambda df, i: rule_bollinger_lower_rebound(df, lower_col, i)),
    ]

    for name, rule_fn in rules:
        if rule_fn(df, i):
            tp = compute_target_price(close, macd_avg)
            matched.append(f"{name} | 潛在目標價 {tp:.2f}")

    return matched

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    for sid, name in STOCK_LIST.items():
        df = get_stock_data(sid)
        if df.empty:
            continue

        df, upper_col, lower_col = add_indicators(df)
        if lower_col is None or len(df) < MIN_BARS:
            continue

        stats = backtest_rules_full(df, lower_col)

        macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100.0
        if macd_avg <= 0:
            continue

        current = check_current(df, lower_col, stats)
        if not current:
            continue

        print("="*60)
        print(f"{name} ({sid}) 最新日期 {df['date'].iloc[-1]}")
        print(f"收盤價: {df['close'].iloc[-1]:.2f}\n")

        print("— 最新一根K線規則 —")
        for c in current:
            print(f"- {c}")
        print("")

        ranked = sorted(
            stats.items(),
            key=lambda x: (x[1]["trades"], x[1]["win_rate_pct"], x[1]["avg_max_runup_pct"]),
            reverse=True
        )

        print("— 規則回測結果 —")
        for rule, s in ranked:
            print(f"- {rule} | 觸發 {s['trades']} | 勝率 {s['win_rate_pct']:.2f}% | 平均最大漲幅 {s['avg_max_runup_pct']:.2f}%")

        best = ranked[0]
        print("\n— 最佳規則 —")
        print(f"{best[0]} | 勝率 {best[1]['win_rate_pct']:.2f}% | 平均最大漲幅 {best[1]['avg_max_runup_pct']:.2f}%")

        print(f"\nMACD 用於目標價計算的平均最大漲幅: {macd_avg*100:.2f}%")
