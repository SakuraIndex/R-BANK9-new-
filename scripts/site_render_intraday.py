# -*- coding: utf-8 -*-
"""
最初期デザインの intraday.png を生成（公式サイト用）
- x軸 HH:MM
- 0% 基準線
- 線＋下側を淡く塗りつぶし
- 右肩に最新値をラベル
入力: docs/outputs/rbank9_intraday.csv (ts,pct)
出力: site/docs/charts/R_BANK9/intraday.png
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

BG   = "#0b1220"
FG   = "#d1d5db"
GRID = "#2b3243"
LINE = "#f2a3ad"
FILL = "#f2a3ad"

def read_ts_pct(csv: Path) -> pd.DataFrame:
    if not csv.exists():
        return pd.DataFrame(columns=["ts","pct"])
    df = pd.read_csv(csv)
    if not {"ts","pct"}.issubset(df.columns):
        return pd.DataFrame(columns=["ts","pct"])
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo")
    pct = pd.to_numeric(df["pct"], errors="coerce")
    d = pd.DataFrame({"ts": ts, "pct": pct}).dropna().sort_values("ts")
    return d

def render(df: pd.DataFrame, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.8)
    ax.tick_params(colors=FG, labelcolor=FG)
    ax.set_xlabel("Time", color=FG); ax.set_ylabel("Change vs Prev Close (%)", color=FG)
    ax.set_title("R-BANK9 Intraday Snapshot (JST)", color=FG)

    if not df.empty:
        ax.plot(df["ts"], df["pct"], color=LINE, linewidth=1.8)
        ax.fill_between(df["ts"], df["pct"], 0, alpha=0.12, color=FILL)
        ax.axhline(0, color="#6b7280", linewidth=1.0, alpha=0.8)
        last_ts = df["ts"].iloc[-1]; last_v = df["pct"].iloc[-1]
        ax.annotate(f"{last_v:+.2f}%", xy=(last_ts, last_v), xytext=(6, 6),
                    textcoords="offset points", color=FG, fontsize=11)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    else:
        ax.axhline(0, color="#6b7280", linewidth=1.0, alpha=0.8)
        ax.text(0.5,0.5,"no data", color=FG, alpha=0.6, ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)

def main():
    csv = Path("docs/outputs/rbank9_intraday.csv")
    out = Path("site/docs/charts/R_BANK9/intraday.png")
    df = read_ts_pct(csv)
    render(df, out)

if __name__ == "__main__":
    main()
