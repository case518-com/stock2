# stock14.py
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
        print(f"找不到 {filename}")
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

# =========================
# 基本設定（你選擇的參數）
# =========================
START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

WIN_THRESHOLD_PCT = 0.05   # +5% 判定勝利
LOOKAHEAD_DAYS = 30        # A=30
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
    if i < 1 or "STOCHk_14_3_3" not in df.columns or "STOCHd_14_3_3" not in df.columns:
        return False
    k_prev, d_prev = df["STOCHk_14_3_3"].iloc[i-1], df["STOCHd_14_3_3"].iloc[i-1]
    k, d = df["STOCHk_14_3_3"].iloc[i], df["STOCHd_14_3_3"].iloc[i]
    return (k_prev < d_prev) and (k > d) and (k < 20) and (d < 20)

def rule_rsi_oversold_rebound(df, i):
    if i < 1 or "RSI_14" not in df.columns:
        return False
    r_prev = df["RSI_14"].iloc[i-1]
    r = df["RSI_14"].iloc[i]
    return (r < 30) and (r > r_prev)

def rule_macd_turning_up(df, i):
    if i < 1 or "macd_hist" not in df.columns:
        return False
    h_prev = df["macd_hist"].iloc[i-1]
    h = df["macd_hist"].iloc[i]
    return (h_prev < 0 and h >= 0) or (h_prev < 0 and h > h_prev)

def rule_bollinger_lower_rebound(df, lower_col, i):
    if i < 1 or lower_col is None:
        return False
    close_prev = df["close"].iloc[i-1]
    close_now = df["close"].iloc[i]
    lower_prev = df[lower_col].iloc[i-1]
    lower_now = df[lower_col].iloc[i]
    return (close_prev < lower_prev) and (close_now > lower_now)

# =========================
# 回測相關
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
        trades = 0
        wins = 0
        runups = []
        finals = []
        for i in range(MIN_BARS, len(df)-1):
            try:
                triggered = rule_fn(i)
            except Exception:
                triggered = False
            if triggered:
                trades += 1
                max_run, final_ret = evaluate_entry_runup_and_final(df, i)
                if max_run is None:
                    continue
                runups.append(max_run)
                finals.append(final_ret)
                if max_run >= WIN_THRESHOLD_PCT:
                    wins += 1
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        avg_max_runup_pct = (np.mean(runups) * 100.0) if runups else 0.0
        avg_final_return_pct = (np.mean(finals) * 100.0) if finals else 0.0
        stats[name] = {
            "trades": trades,
            "wins": wins,
            "win_rate_pct": win_rate,
            "avg_max_runup_pct": avg_max_runup_pct,
            "avg_final_return_pct": avg_final_return_pct
        }
    return stats

# =========================
# 計算目標價（全部以 MACD 的 avg_max_runup 作為基準，並打 8 折）
# 公式： target = close * (1 + macd_avg_runup * 0.8)
# macd_avg_runup 為小數（ex: 0.1909）
# =========================
def compute_target_price_from_macd_avg(close, macd_avg_runup):
    return close * (1 + macd_avg_runup * 0.6)

# =========================
# 即時檢查（使用回測結果中的 MACD 平均最大漲幅）
# =========================
def check_current_with_targets_using_macd(df, upper_col, lower_col, stats):
    # 取 MACD avg_max_runup（小數）
    macd_stats_entry = stats.get("MACD 直方圖拐頭") or stats.get("MACD") or {}
    # If key name differs, try to compute from stats mapping:
    macd_avg = None
    if "MACD 直方圖拐頭" in stats:
        macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100.0
    else:
        # try find MACD-like key
        for k in stats:
            if "MACD" in k:
                macd_avg = stats[k]["avg_max_runup_pct"] / 100.0
                break
    if macd_avg is None or macd_avg <= 0:
        # 若 MACD 回測無樣本或非正向，回傳空（當天不列出）
        return []

    i = len(df) - 1
    close = df["close"].iloc[i]
    matched = []

    if rule_kd_low_cross(df, i):
        tp = compute_target_price_from_macd_avg(close, macd_avg)
        matched.append(f"KD 低檔黃金交叉 | 潛在目標價 {tp:.2f}")

    if rule_rsi_oversold_rebound(df, i):
        tp = compute_target_price_from_macd_avg(close, macd_avg)
        matched.append(f"RSI 超賣翻升 | 潛在目標價 {tp:.2f}")

    if rule_macd_turning_up(df, i):
        tp = compute_target_price_from_macd_avg(close, macd_avg)
        matched.append(f"MACD 直方圖拐頭 | 潛在目標價 {tp:.2f}")

    if rule_bollinger_lower_rebound(df, lower_col, i):
        tp = compute_target_price_from_macd_avg(close, macd_avg)
        matched.append(f"布林下軌反彈 | 潛在目標價 {tp:.2f}")

    return matched

# =========================
# 排序：你選 B=3 => 以觸發次數由多到少排序輸出回測結果
# 若同交易次數則以勝率再排序
# =========================
def rank_rules_by_trades(stats_dict):
    items = []
    for rule_name, s in stats_dict.items():
        items.append({
            "rule": rule_name,
            "trades": s["trades"],
            "win_rate_pct": s["win_rate_pct"],
            "avg_max_runup_pct": s["avg_max_runup_pct"],
            "avg_final_return_pct": s["avg_final_return_pct"]
        })
    # sort by trades desc, then win_rate desc, then avg_max_runup desc
    ranked = sorted(items, key=lambda x: (x["trades"], x["win_rate_pct"], x["avg_max_runup_pct"]), reverse=True)
    return ranked

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    stock_list = load_stock_list("stock.txt")
    if not stock_list:
        print("stock.txt 無股票清單或檔案不存在。")
        raise SystemExit

    for sid, name in stock_list.items():
        df = get_stock_data(sid)
        if df.empty:
            continue

        df, upper_col, lower_col = add_indicators(df)
        if upper_col is None or lower_col is None or len(df) < MIN_BARS:
            continue

        # 回測（完整統計）
        stats = backtest_rules_full(df, lower_col)

        # 取得 MACD 的 avg_max_runup（小數）
        macd_avg = None
        if "MACD 直方圖拐頭" in stats:
            macd_avg = stats["MACD 直方圖拐頭"]["avg_max_runup_pct"] / 100.0
        else:
            # fallback
            for k in stats:
                if "MACD" in k:
                    macd_avg = stats[k]["avg_max_runup_pct"] / 100.0
                    break

        # 若 MACD 平均最大漲幅 <= 0，視為不列出（保守）
        if macd_avg is None or macd_avg <= 0:
            continue

        # 即時規則（使用 MACD avg 計算所有目標價）
        current = check_current_with_targets_using_macd(df, upper_col, lower_col, stats)
        if not current:
            continue

        # 輸出區塊
        print("="*60)
        print(f"{name} ({sid}) 最新日期 {df['date'].iloc[-1]}")
        print(f"收盤價: {df['close'].iloc[-1]:.2f}\n")

        print("— 最新一根K線規則（統一目標價：MACD 回測 avg_max_runup ×0.8）—")
        for c in current:
            print(f"- {c}")
        print("")

        # 規則回測結果（按觸發次數由多到少排序）
        ranked = rank_rules_by_trades(stats)
        print("— 規則回測結果 —")
        for r in ranked:
            print(f"- {r['rule']} | 觸發次數: {r['trades']} | 勝率: {r['win_rate_pct']:.2f}% | 平均最大漲幅: {r['avg_max_runup_pct']:.2f}% | 平均最終報酬: {r['avg_final_return_pct']:.2f}%")

        # 最佳規則（觸發次數最高者）
        if ranked:
            best = ranked[0]
            print("\n— 最佳規則 —")
            print(f"{best['rule']} | 勝率 {best['win_rate_pct']:.2f}% | 平均最大漲幅 {best['avg_max_runup_pct']:.2f}% | 觸發 {best['trades']} 次")

        # 顯示 MACD 用於計算的平均最大漲幅
        print(f"\nMACD 用於目標價計算的平均最大漲幅: {macd_avg*100:.2f}%")
