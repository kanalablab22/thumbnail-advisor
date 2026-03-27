# 楽天サムネイル アドバイザー

楽天市場の商品サムネイル画像をアップロードするだけで、自動解析してスコアと改善アドバイスを表示するWebアプリです。

**公開URL**: https://thumbnail-advisor-6qspbn26sdqwlcryashtjn.streamlit.app/

## できること

- サムネイル画像の自動解析（7項目・100点満点スコア）
- ジャンル別の売れ筋店舗に基づいた改善アドバイス
- 素材に応じたアドバイスの自動切り替え（木製品・革製品・食品など）
- 楽天の実際の検索結果に自分の商品を並べたプレビュー（PC版・スマホ版）

## 評価項目

| 項目 | 内容 |
|------|------|
| 余白（呼吸感） | 商品周囲に十分なスペースがあるか |
| テキスト量 | テキスト面積が適切か（20%以下推奨） |
| 背景のシンプルさ | 背景が商品を引き立てているか |
| 色使い・トーン | 彩度が適切か、色数が絞られているか |
| 写真のクオリティ | コントラスト・シャープネスは十分か |
| 構図・レイアウト | 商品が中央に配置されているか |
| カラバリ表示 | 色展開が表示されているか |

## ジャンル別アドバイス

検索キーワードを入力すると、以下の8ジャンル + 汎用 から自動判別してアドバイスを出します。

- バッグ・財布（参考: LASIEM / スタイルオンバッグ（LIZDAYS））
- ファッション（参考: Dark Angel / HUG.U / 神戸レタス）
- 食品・グルメ（参考: 甲羅組 / LeTAO / 松屋フーズ）
- インテリア・家具（参考: LOWYA / Re:CENO / エア・リゾーム）
- コスメ・美容（参考: SK-II / アテニア / VT COSMETICS）
- 家電・日用品（参考: アイリスオーヤマ / MTG(ReFa)）
- ペット用品
- キッズ・ベビー

木製品（キャットタワーなど）には「木の質感を活かす」、革製品には「革の表情を引き出す」など、素材に合わせたアドバイスに自動で切り替わります。

## セットアップ

### 必要なもの

- Python 3.10以上
- 楽天ウェブサービスのアプリID・アクセスキー（検索結果プレビュー機能を使う場合）

### macOS

```bash
cd thumbnail_advisor
pip install -r requirements.txt
streamlit run app.py
```

### Windows

```bash
cd thumbnail_advisor
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloudにデプロイする場合

1. GitHubにリポジトリをプッシュ
2. [Streamlit Cloud](https://share.streamlit.io/) でデプロイ
3. Secrets に以下を設定:

```toml
RAKUTEN_APP_ID = "your-application-id"
RAKUTEN_ACCESS_KEY = "pk_your-access-key"
```

## ファイル構成

```
thumbnail_advisor/
├── app.py                 # メインUI（Streamlit）
├── image_checker.py       # 画像解析・スコアリングエンジン
├── genre_advisor.py       # ジャンル判別・ジャンル別アドバイス
├── rakuten_api_sim.py     # 楽天API検索 + 検索結果HTML生成
├── rakuten_html_sim.py    # HTML版検索結果シミュレーション
├── rakuten_search_sim.py  # Selenium版検索シミュレーション
├── pdf_report.py          # PDFレポート生成
├── requirements.txt
├── .streamlit/
│   └── config.toml        # テーマ設定
└── .gitignore
```

## 技術スタック

- **フロントエンド**: Streamlit
- **画像解析**: NumPy + Pillow（OpenCV不使用、Cloud対応）
- **検索結果プレビュー**: 楽天商品検索API（Streamlit Cloud）/ Selenium（ローカル）
- **ジャンル判別**: キーワードベースのルールマッチング（APIコストゼロ）

## 注意事項

- 画像解析はAIではなくルールベースのため、判定精度には限界があります
- ジャンル判別は検索キーワード入力が必要です（画像からの自動判別は非対応）
- 楽天APIには利用制限があります（短時間に大量リクエストすると429エラー）
