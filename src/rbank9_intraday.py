# -*- coding: utf-8 -*-
"""
R-BANK9 intraday snapshot (equal-weight vs prev close, %)
- 5分足優先 / 失敗時は15分足へフォールバック
- 当日JSTセッションの共通グリッドにreindex+ffill
- CSVが空のときもゼロ埋めでグリッドを書き出し（更新検知用）
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

JST_TZ = "Asia/Tokyo"
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_rbank9.txt"

CSV_PATH  = os.path.join(OUT_DIR, "rbank9_intraday.csv")
IMG_PATH  = os.path.join(OUT_DIR, "rbank9_intraday.png")
POST_PATH = os.path.join(OUT_DIR, "rbank9_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "rbank9_stats.json")

PRIMARY_INTERVALS  = ["5m", "15m"]
PRIMARY_PERIODS    = ["3d", "7d"]

PCT_CLIP_LOW  = -20.0
PCT_CLIP_HIGH =  20.0

SESSION_START = time(9, 0)
SESSION_END   = time(15, 30)


def jst_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=JST_TZ)


def _floor_code(freq: str) -> str:
    # pandasのfloorは '5m' を「5 * MonthEnd」と誤解するため '5min' / '15min' を渡す
    return "5min" if freq == "5m" else "15min"


def session_bounds(day: datetime.date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(pd.Timestamp(day), SESSION_START).tz_localize(JST_TZ)
    end   = pd.Timestamp.combine(pd.Timestamp(day), SESSION_END).tz_localize(JST_TZ)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end


def make_grid(day: datetime.date, until: Optional[pd.Timestamp], freq: str) -> pd.DatetimeIndex:
    start, end = session_bounds(day)
    if until is not None:
        end = min(end, until.floor(_floor_code(freq)))
    return pd.date_range(start=start, end=end, freq=freq, tz=JST_TZ)


def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s.split()[0])
    if not xs:
        raise RuntimeError("No tickers found")
    return xs


def _to_series_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    if isinstance(close, pd.Series):
        return pd.to_numeric(close, errors="coerce").dropna()
    d = close.apply(pd.to_numeric, errors="coerce")
    mask = d.notna().any(axis=0)
    d = d.loc[:, mask]
    s = d.iloc[:, 0] if d.shape[1] == 1 else d[d.count(axis=0).idxmax()]
    return s.dropna().astype(float)


def last_trading_day(ts_index: pd.DatetimeIndex) -> datetime.date:
    idx = pd.to_datetime(ts_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST_TZ)
    return idx[-1].date()


def fetch_prev_close(ticker: str, day: datetime.date) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False, prepost=False)
    if d.empty:
        raise RuntimeError(f"prev close empty for {ticker}")
    s = _to_series_1d_close(d)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s.tz_convert(JST_TZ)
    s_before = s[s.index.date < day]
    return float((s_before.iloc[-1] if not s_before.empty else s.iloc[-1]))


def _try_download(ticker: str, period: str, interval: str) -> pd.Series:
    d = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, prepost=False, threads=True)
    if d.empty:
        return pd.Series(dtype=float)
    s = _to_series_1d_close(d)
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST_TZ)
    return pd.Series(s.values, index=idx)


def fetch_intraday_series_smart(ticker: str) -> Tuple[pd.Series, str, str]:
    last_err: Optional[Exception] = None
    for iv in PRIMARY_INTERVALS:
        for pdv in PRIMARY_PERIODS:
            try:
                s = _try_download(ticker, pdv, iv)
                if not s.empty:
                    return s, pdv, iv
            except Exception as e:
                last_err = e
    if last_err:
        print(f"[WARN] all intraday attempts failed for {ticker}: {last_err!r}")
    return pd.Series(dtype=float), "", ""


def _first_probe(tickers: List[str]) -> Tuple[str, pd.Series, str]:
    for t in tickers:
        s, _, iv = fetch_intraday_series_smart(t)
        if not s.empty:
            return t, s, iv
    raise RuntimeError("no available intraday series for probe")


def build_equal_weight_pct(tickers: List[str]) -> Tuple[pd.Series, str]:
    indiv: Dict[str, pd.Series] = {}

    probe_t, probe_s, probe_iv = _first_probe(tickers)
    day = last_trading_day(probe_s.index)
    grid_freq = "5m" if probe_iv == "5m" else "15m"
    print(f"[INFO] target day: {day} (probe={probe_t}, iv={probe_iv})")

    def pick_day(s: pd.Series) -> pd.Series:
        x = s[(s.index.date == day)]
        if x.empty:
            d2 = last_trading_day(s.index)
            x = s[(s.index.date == d2)]
        return x

    grid = make_grid(day, until=jst_now(), freq=grid_freq)

    for t in tickers:
        try:
            s = probe_s if t == probe_t else fetch_intraday_series_smart(t)[0]
            s = pick_day(s)
            if s.empty:
                print(f"[WARN] {t}: empty on target day, skip")
                continue
            prev = fetch_prev_close(t, day)
            pct = (s / prev - 1.0) * 100.0
            pct = pct.clip(PCT_CLIP_LOW, PCT_CLIP_HIGH)
            pct = pct.reindex(grid).ffill()
            indiv[t] = pct.rename(t)
        except Exception as e:
            print(f"[WARN] skip {t}: {e}")

    if not indiv:
        return pd.Series(dtype=float), grid_freq

    df = pd.concat(indiv.values(), axis=1)
    series = df.mean(axis=1, skipna=True).astype(float)
    series.name = "R_BANK9"
    return series, grid_freq


def save_csv(series: pd.Series, path: str, grid_freq: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if series is None or series.empty:
        # ゼロ埋めの当日グリッド
        today = jst_now().date()
        grid = make_grid(today, until=jst_now(), freq=grid_freq)
        out = pd.DataFrame({"ts": grid.strftime("%Y-%m-%dT%H:%M:%S%z"), "pct": 0.0})
        out.to_csv(path, index=False)
        print(f"[INFO] wrote ZERO grid rows={len(out)}")
        return
    s = series.dropna()
    out = pd.DataFrame({
        "ts": s.index.tz_convert(JST_TZ).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pct": s.round(4).values,
    })
    out.to_csv(path, index=False)
    print(f"[INFO] wrote rows={len(out)}")


def plot_debug(series: Optional[pd.Series], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.grid(True, color="#2a2a2a", alpha=0.5, linestyle="--", linewidth=0.7)
    title = f"R-BANK9 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})"
    if series is None or series.empty:
        ax.set_title(title + " (no data)", color="white")
        ax.axhline(0, color="#666666", linewidth=1.0)
    else:
        ax.plot(series.index, series.values, color="#f87171", linewidth=2.0)
        ax.axhline(0, color="#666666", linewidth=1.0)
        ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_post_and_stats(series: Optional[pd.Series], post_path: str, stats_path: str):
    if series is None or series.empty:
        last = 0.0
    else:
        s = series.dropna()
        last = float(s.iloc[-1]) if not s.empty else 0.0
    sign = "+" if last >= 0 else ""
    text = (
        f"▲ R-BANK9 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
        f"{sign}{last:.2f}%（前日終値比）\n"
        f"※ 構成9銘柄の等ウェイト\n"
        f"#地方銀行  #R_BANK9 #日本株\n"
    )
    os.makedirs(os.path.dirname(post_path), exist_ok=True)
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(text)

    import json
    meta = {
        "index_key": "rbank9",
        "label": "R-BANK9",
        "pct_intraday": round(last, 4),
        "basis": "prev_close",
        "updated_at": jst_now().isoformat(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    tickers = load_tickers(TICKER_FILE)
    print(f"[INFO] tickers: {', '.join(tickers)}")
    series, grid_freq = build_equal_weight_pct(tickers)
    save_csv(series, CSV_PATH, grid_freq)
    plot_debug(series, IMG_PATH)
    save_post_and_stats(series, POST_PATH, STATS_PATH)
    if series is not None and not series.empty:
        print(pd.DataFrame({"ts": series.index[-5:], "pct": series[-5:]}))

if __name__ == "__main__":
    main()
