# R-BANK9 (Next)

- 9銘柄の等ウェイト・日中スナップショット（前日終値比 %）
- 5分足優先（失敗時は 15分足へ自動フォールバック）
- 公式サイト用（最初期デザイン）のPNG/テキスト/JSONも自動生成＆反映

## 出力（このリポ）
- `docs/outputs/rbank9_intraday.csv` … `ts,pct`
- `docs/outputs/rbank9_intraday.png` … デバッグ用
- `docs/outputs/rbank9_post_intraday.txt`
- `docs/outputs/rbank9_stats.json`

## 公式サイトへの反映
- 生成先（別リポ）: `docs/charts/R_BANK9/`
  - `intraday.png`（最初期デザイン）
  - `post_intraday.txt`（ポスト文）
  - `stats.json`（メタ）
- `SITE_REPO` と `SITE_TOKEN` シークレットが必要（下記）

## 必要シークレット（このリポ）
- `SITE_REPO`  … 例: `SakuraIndex/Sakura-Index-Site`
- `SITE_TOKEN` … `repo` 権限の PAT（サイト側に push するため）

