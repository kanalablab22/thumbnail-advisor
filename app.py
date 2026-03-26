#!/usr/bin/env python3
"""
🎨 楽天サムネイル アドバイザー
サムネイル画像をアップロード → 自動解析 → スコア＆改善アドバイス
AI不要・完全無料
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image, ImageStat, ImageFilter
import cv2
import io
import base64

# ===== ページ設定 =====
st.set_page_config(
    page_title="楽天サムネイル アドバイザー",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== カスタムCSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700;900&display=swap');
    .stApp { font-family: 'Noto Sans JP', sans-serif; }
    .block-container { max-width: 1100px; padding-top: 1.5rem; }

    .app-header {
        text-align: center; padding: 16px 0 24px;
        border-bottom: 2px solid #f0f0f0; margin-bottom: 24px;
    }
    .app-title { font-size: 2rem; font-weight: 900; color: #2C3E50; margin: 0; }
    .app-desc { font-size: 1rem; color: #95A5A6; margin-top: 4px; font-weight: 300; }

    .score-card-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px; padding: 28px; color: white;
        text-align: center; margin: 12px 0;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .score-big { font-size: 3.5rem; font-weight: 900; line-height: 1; }
    .score-max { font-size: 1.3rem; font-weight: 300; opacity: 0.8; }
    .score-grade { font-size: 1.1rem; font-weight: 500; margin-top: 6px; opacity: 0.9; }

    .meter-row {
        display: flex; align-items: center; padding: 10px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .meter-label { font-weight: 600; color: #2C3E50; min-width: 180px; font-size: 0.9rem; }
    .meter-track {
        flex: 1; height: 10px; background: #E8E8E8;
        border-radius: 5px; overflow: hidden; margin: 0 12px;
    }
    .meter-fill { height: 100%; border-radius: 5px; transition: width 0.5s ease; }
    .meter-score { font-weight: 700; min-width: 35px; text-align: right; font-size: 0.95rem; }
    .meter-ref {
        font-size: 0.75rem; color: #8B8B8B; min-width: 30px; text-align: center;
    }

    .advice-box { border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; }
    .advice-high {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        border-left: 4px solid #E53E3E;
    }
    .advice-medium {
        background: linear-gradient(135deg, #FFFFF0 0%, #FEFCBF 100%);
        border-left: 4px solid #ECC94B;
    }
    .advice-low {
        background: linear-gradient(135deg, #F0FFF4 0%, #C6F6D5 100%);
        border-left: 4px solid #38A169;
    }
    .advice-title { font-weight: 700; font-size: 0.93rem; color: #2D3748; }
    .advice-detail { font-size: 0.85rem; color: #4A5568; margin-top: 4px; line-height: 1.6; }

    .detail-card {
        background: #F8F9FA; border-radius: 10px; padding: 14px 18px;
        margin-bottom: 8px;
    }
    .detail-title { font-weight: 700; color: #2C3E50; font-size: 0.9rem; }
    .detail-value { color: #555; font-size: 0.85rem; margin-top: 2px; }

    .quick-win {
        background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
        border-radius: 12px; padding: 20px 24px; margin-top: 16px;
        border-left: 4px solid #0284C7;
    }
    .quick-win-title { font-weight: 800; color: #0C4A6E; font-size: 1rem; }
    .quick-win-detail { color: #1E3A5F; font-size: 0.9rem; margin-top: 6px; line-height: 1.6; }

</style>
""", unsafe_allow_html=True)


# ====================================================================
# 画像解析エンジン
# ====================================================================

def analyze_image(pil_img):
    """PIL画像を解析して各スコアと詳細データを返す"""
    # numpy / OpenCV 用に変換
    img_rgb = np.array(pil_img.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, w = img_gray.shape

    results = {}

    # ----- 1. 余白率（白〜ライトグレー領域の割合）-----
    whitespace_mask = img_gray > 235
    whitespace_ratio = np.sum(whitespace_mask) / (h * w)
    # 端（上下左右10%）の余白も重視
    border = int(min(h, w) * 0.1)
    border_region = np.concatenate([
        img_gray[:border, :].flatten(),
        img_gray[-border:, :].flatten(),
        img_gray[:, :border].flatten(),
        img_gray[:, -border:].flatten(),
    ])
    border_white = np.sum(border_region > 235) / len(border_region)
    effective_whitespace = whitespace_ratio * 0.6 + border_white * 0.4

    results["whitespace"] = {
        "ratio": round(whitespace_ratio * 100, 1),
        "border_ratio": round(border_white * 100, 1),
        "effective": round(effective_whitespace * 100, 1),
    }

    # ----- 2. テキスト量推定（エッジ密度で近似）-----
    # Cannyエッジ検出 → エッジ密度が高い = テキスト/要素が多い
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)

    # 高周波成分（テキストは高周波）
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    high_freq = np.mean(np.abs(laplacian))

    # テキスト領域をMSER等で推定
    # エッジが集中している領域をテキストとみなす
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    text_area_ratio = np.sum(dilated > 0) / (h * w)

    results["text_amount"] = {
        "edge_density": round(edge_density * 100, 1),
        "high_freq": round(high_freq, 1),
        "estimated_text_area": round(text_area_ratio * 100, 1),
    }

    # ----- 3. 背景のシンプルさ -----
    # 画像を5x5ブロックに分割、各ブロックの標準偏差を計算
    block_h, block_w = h // 5, w // 5
    block_stds = []
    for bi in range(5):
        for bj in range(5):
            block = img_gray[bi*block_h:(bi+1)*block_h, bj*block_w:(bj+1)*block_w]
            block_stds.append(np.std(block))
    # 低いstdのブロック数 = シンプルな背景ブロック
    simple_blocks = sum(1 for s in block_stds if s < 20)
    avg_block_std = np.mean(block_stds)

    # 枠線チェック: 画像端1pxの色が均一＝枠線あり
    top_edge = img_gray[0, :]
    bottom_edge = img_gray[-1, :]
    left_edge = img_gray[:, 0]
    right_edge = img_gray[:, -1]
    has_border = (
        np.std(top_edge) < 5 and np.std(bottom_edge) < 5 and
        np.mean(top_edge) < 200 and  # 白以外の均一な色＝枠線
        abs(np.mean(top_edge) - np.mean(bottom_edge)) < 10
    )

    results["background"] = {
        "simple_blocks": simple_blocks,
        "avg_std": round(avg_block_std, 1),
        "has_border": has_border,
    }

    # ----- 4. 色使い・彩度 -----
    saturation = img_hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    high_sat_ratio = np.sum(saturation > 150) / (h * w)  # 派手な色の面積割合
    # 色相のばらつき
    hue = img_hsv[:, :, 0].flatten()
    hue_of_saturated = hue[saturation.flatten() > 50]  # 彩度がある部分のみ
    if len(hue_of_saturated) > 100:
        hue_std = np.std(hue_of_saturated)
        n_dominant_colors = len(set(np.round(hue_of_saturated / 15)))  # 15度刻み
    else:
        hue_std = 0
        n_dominant_colors = 0

    results["color_tone"] = {
        "avg_saturation": round(avg_saturation, 1),
        "high_sat_ratio": round(high_sat_ratio * 100, 1),
        "hue_variety": round(hue_std, 1),
        "n_colors": n_dominant_colors,
    }

    # ----- 5. コントラスト・写真品質 -----
    contrast = np.std(img_gray)
    brightness = np.mean(img_gray)
    # シャープネス（ラプラシアンの分散）
    sharpness = np.var(laplacian)

    results["photo_quality"] = {
        "contrast": round(contrast, 1),
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
    }

    # ----- 6. 情報密度（画像の複雑さ）-----
    # 非白領域の密度と分布
    content_mask = img_gray < 230
    content_ratio = np.sum(content_mask) / (h * w)

    # 中央 vs 周辺の密度差（中央集中度）
    center_region = content_mask[h//4:3*h//4, w//4:3*w//4]
    center_density = np.sum(center_region) / center_region.size
    overall_density = np.sum(content_mask) / content_mask.size
    center_focus = center_density / max(overall_density, 0.01)

    results["composition"] = {
        "content_ratio": round(content_ratio * 100, 1),
        "center_focus": round(center_focus, 2),
    }

    # ----- 7. カラバリ表示の検出 -----
    # 画像下部20%に小さな色の塊がある＝カラバリ表示の可能性
    bottom_region = img_hsv[int(h*0.8):, :, :]
    if bottom_region.size > 0:
        bottom_sat = bottom_region[:, :, 1]
        bottom_has_colors = np.sum(bottom_sat > 60) / bottom_sat.size
    else:
        bottom_has_colors = 0

    # 小さな丸や四角が並んでいるか（カラースウォッチ）
    # 横方向に等間隔の色の変化を検出
    bottom_gray = img_gray[int(h*0.85):, :]
    if bottom_gray.size > 0:
        col_means = np.mean(bottom_gray, axis=0)
        col_diff = np.abs(np.diff(col_means))
        n_transitions = np.sum(col_diff > 30)
        has_swatches = n_transitions > 6
    else:
        has_swatches = False

    results["color_variation"] = {
        "bottom_color_ratio": round(bottom_has_colors * 100, 1),
        "has_swatches": has_swatches,
    }

    return results


def compute_scores(analysis):
    """解析結果からスコアを計算"""
    scores = {}

    # 1. 余白スコア
    ws = analysis["whitespace"]["effective"]
    if ws >= 25:
        scores["whitespace"] = 5
    elif ws >= 18:
        scores["whitespace"] = 4
    elif ws >= 12:
        scores["whitespace"] = 3
    elif ws >= 6:
        scores["whitespace"] = 2
    else:
        scores["whitespace"] = 1

    # 2. テキスト量スコア（エッジ密度ベース: 低い=テキスト少ない=良い）
    ta = analysis["text_amount"]["estimated_text_area"]
    if ta <= 25:
        scores["text_amount"] = 5
    elif ta <= 35:
        scores["text_amount"] = 4
    elif ta <= 45:
        scores["text_amount"] = 3
    elif ta <= 55:
        scores["text_amount"] = 2
    else:
        scores["text_amount"] = 1

    # 3. 背景のシンプルさ
    sb = analysis["background"]["simple_blocks"]
    has_border = analysis["background"]["has_border"]
    if has_border:
        scores["background"] = min(2, sb // 5)  # 枠線あるとペナルティ
    elif sb >= 12:
        scores["background"] = 5
    elif sb >= 8:
        scores["background"] = 4
    elif sb >= 5:
        scores["background"] = 3
    elif sb >= 3:
        scores["background"] = 2
    else:
        scores["background"] = 1

    # 4. 色使い
    avg_sat = analysis["color_tone"]["avg_saturation"]
    high_sat = analysis["color_tone"]["high_sat_ratio"]
    if avg_sat <= 50 and high_sat <= 5:
        scores["color_tone"] = 5  # 落ち着いたトーン
    elif avg_sat <= 70 and high_sat <= 10:
        scores["color_tone"] = 4
    elif avg_sat <= 90 and high_sat <= 20:
        scores["color_tone"] = 3
    elif avg_sat <= 110:
        scores["color_tone"] = 2
    else:
        scores["color_tone"] = 1

    # 5. 写真品質（コントラスト + シャープネス）
    contrast = analysis["photo_quality"]["contrast"]
    sharpness = analysis["photo_quality"]["sharpness"]
    quality_score = 0
    if 40 <= contrast <= 80:
        quality_score += 2  # 適度なコントラスト
    elif 30 <= contrast <= 90:
        quality_score += 1
    if sharpness >= 500:
        quality_score += 3  # 高シャープネス
    elif sharpness >= 200:
        quality_score += 2
    elif sharpness >= 50:
        quality_score += 1
    scores["photo_quality"] = min(5, max(1, quality_score))

    # 6. 構図（中央集中度 + コンテンツ密度）
    cf = analysis["composition"]["center_focus"]
    cr = analysis["composition"]["content_ratio"]
    if 1.1 <= cf <= 1.8 and 40 <= cr <= 75:
        scores["composition"] = 5  # 商品が中央に、適度な密度
    elif 1.0 <= cf <= 2.0 and 35 <= cr <= 80:
        scores["composition"] = 4
    elif cr <= 85:
        scores["composition"] = 3
    else:
        scores["composition"] = 2

    # 7. カラバリ表示
    has_sw = analysis["color_variation"]["has_swatches"]
    bottom_color = analysis["color_variation"]["bottom_color_ratio"]
    if has_sw or bottom_color > 15:
        scores["color_variation"] = 4
    elif bottom_color > 5:
        scores["color_variation"] = 3
    else:
        scores["color_variation"] = 2  # 検出できない＝やや低め（でもない場合もある）

    return scores


# ===== 評価項目の情報 =====
CRITERIA_INFO = {
    "whitespace": {
        "icon": "⬜", "name": "余白（呼吸感）",
        "weight": 1.2,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "余白を大幅に増やす", "d": "テキスト要素を1/3に絞り、残りは画像の一角にまとめましょう。画像の20%以上を何も置かない空間にすることで、高級感と視認性が一気にアップします"},
            2: {"p": "high", "t": "テキストを整理して余白を確保", "d": "「送料無料」「ポイント○倍」はサムネから外し、最も伝えたい1メッセージだけ残しましょう。商品写真の周囲に呼吸できるスペースを"},
            3: {"p": "medium", "t": "もう少し余白を意識する", "d": "テキストのサイズを見直して小さい文字は思い切って削除。商品の周囲に最低10%の余白があると見やすくなります"},
            4: {"p": "low", "t": "良い余白バランスです", "d": "適度な余白が確保できています。他の商品サムネにも同じバランスを展開しましょう"},
            5: {"p": "low", "t": "理想的な余白です！", "d": "高級感と見やすさを両立する素晴らしい余白設計です"},
        },
    },
    "text_amount": {
        "icon": "🔤", "name": "テキスト量",
        "weight": 1.2,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "テキストを大幅に減らす", "d": "情報を3つだけに絞りましょう：「一番の訴求ワード」「商品名」「スペック1つ」。残りは商品ページに任せて、サムネは商品写真を主役に"},
            2: {"p": "high", "t": "テキストを整理・削減する", "d": "一番伝えたいワード1つだけを大きく残し、それ以外のテキストは小さくするか削除。テキスト面積を全体の30%以下に抑えましょう"},
            3: {"p": "medium", "t": "もう少しテキストを絞る", "d": "優先度の低い情報（型番・送料情報など）を外して余白を増やしましょう。文字サイズの大・中・小にメリハリをつけると読みやすくなります"},
            4: {"p": "low", "t": "良いテキスト量です", "d": "バランスの取れたテキスト量です。文字サイズのメリハリ（大・中・小の3段階）を意識するとさらに効果的"},
            5: {"p": "low", "t": "理想的なテキスト量！", "d": "商品写真が主役になる素晴らしいテキスト設計です"},
        },
    },
    "background": {
        "icon": "🖼️", "name": "背景のシンプルさ",
        "weight": 1.0,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "背景をシンプルにする", "d": "爆発マーク・柄・枠線をすべて削除してオフホワイト（#F5F5F5）の無地背景にしましょう。楽天ガイドラインでも枠線は禁止されています"},
            2: {"p": "high", "t": "背景の装飾を減らす", "d": "背景のパターンや装飾を外しましょう。白〜ライトグレーの無地にするだけで商品が際立ち、ブランド感もアップします"},
            3: {"p": "medium", "t": "もう少しシンプルに", "d": "背景の色味を抑えてニュートラルトーン（白〜ライトグレー）に。全商品で同じトーンに揃えると一覧での統一感が出ます"},
            4: {"p": "low", "t": "ほぼ理想的な背景です", "d": "きれいな背景処理です。全商品で同じトーンに統一できればさらに良くなります"},
            5: {"p": "low", "t": "完璧な背景処理！", "d": "クリーンで商品が映える背景です"},
        },
    },
    "color_tone": {
        "icon": "🎨", "name": "色使い・トーン",
        "weight": 0.8,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "色使いを見直す", "d": "蛍光色・原色を外し、テキスト色は黒〜ダークグレーに統一。アクセント色は「ランキング1位」バッジなどピンポイントだけに使いましょう"},
            2: {"p": "medium", "t": "派手な色を抑える", "d": "赤・黄色の面積を減らし、使うなら小さなバッジ程度に。テキスト全体を黒系に統一するだけで上品になります"},
            3: {"p": "medium", "t": "トーンを統一する", "d": "使う色を3-4色に絞りましょう。テキスト＝黒系、背景＝白系、アクセント＝1色だけが理想的です"},
            4: {"p": "low", "t": "落ち着いた色使いです", "d": "基本トーンが良いです。全商品で同じカラーパレットを使うとブランド感が増します"},
            5: {"p": "low", "t": "上品な色使いです！", "d": "洗練されたカラーパレット。ブランドカラーとして定着させましょう"},
        },
    },
    "photo_quality": {
        "icon": "📸", "name": "写真のクオリティ",
        "weight": 0.8,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "写真撮影を改善する", "d": "プロカメラマンに依頼するか、ディフューズボックスで柔らかい光をあてて撮影しましょう。窓際の自然光＋白い布でも改善できます"},
            2: {"p": "medium", "t": "ライティングを改善", "d": "直射光を避けてディフューズ光で撮影。撮影後に彩度をやや下げてマット仕上げにするとプロっぽい仕上がりに"},
            3: {"p": "medium", "t": "写真トーンを調整", "d": "彩度をやや下げてマット感を出し、コントラストを少し上げつつハイライトを抑えると上品な仕上がりに"},
            4: {"p": "low", "t": "良い写真品質です", "d": "基本は良い写真です。全商品で写真トーン（明るさ・彩度）を統一するとブランド感が上がります"},
            5: {"p": "low", "t": "プロレベルの写真品質！", "d": "素晴らしい撮影品質です。このクオリティを維持しましょう"},
        },
    },
    "composition": {
        "icon": "📐", "name": "構図・レイアウト",
        "weight": 1.0,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "構図を見直す", "d": "商品を画像の中央〜やや左下に配置し、テキスト情報は右側か上部にまとめましょう。商品が主役になる構図を意識してください"},
            2: {"p": "medium", "t": "レイアウトを整理する", "d": "要素が散らばっている可能性があります。商品写真を中央に大きく、テキストは端にまとめてスッキリさせましょう"},
            3: {"p": "medium", "t": "もう少し整理できます", "d": "左上に訴求ワード→中央に商品→右下にロゴという配置にすると、自然な視線の流れ（Z型）が生まれます"},
            4: {"p": "low", "t": "良い構図です", "d": "商品が中心にしっかり配置されています。テキスト配置のルールを固定化すると統一感もアップ"},
            5: {"p": "low", "t": "素晴らしい構図！", "d": "商品が映える理想的な構図です"},
        },
    },
    "color_variation": {
        "icon": "🌈", "name": "カラバリ表示",
        "weight": 0.7,
        "ref_score": 4,
        "advice": {
            1: {"p": "medium", "t": "カラバリ表示を追加", "d": "色展開がある商品は必ずサムネに表示を。画像下部にカラードット（小さな色丸）を横並びにするのが一番簡単です"},
            2: {"p": "medium", "t": "カラバリをもう少し目立たせる", "d": "カラードットを少し大きくするか、別色の商品写真を小さく添えましょう。「選べる楽しさ」がクリック率を上げます"},
            3: {"p": "low", "t": "カラバリ表示あり", "d": "色展開の表示が確認できます。複数手法（ドット＋ミニ写真等）を組み合わせるとさらに効果的"},
            4: {"p": "low", "t": "良いカラバリ表示です", "d": "色展開がわかりやすく表示されています"},
            5: {"p": "low", "t": "完璧なカラバリ表示！", "d": "色の豊富さが一目で伝わります"},
        },
    },
}

def render_phone_mockup(pil_img):
    """スクショをiPhone風フレーム内にスクロール表示するHTMLを返す"""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ display:flex; justify-content:center; background:transparent; padding:10px 0; }}
  .phone {{
    width: 280px; height: 580px;
    border: 6px solid #1a1a1a;
    border-radius: 36px;
    overflow: hidden;
    position: relative;
    background: #fff;
    box-shadow: 0 8px 40px rgba(0,0,0,0.15), inset 0 0 0 2px #333;
  }}
  .phone::before {{
    content: "";
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 90px; height: 22px;
    background: #1a1a1a;
    border-radius: 0 0 16px 16px;
    z-index: 10;
  }}
  .phone-screen {{
    width: 100%; height: 100%;
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
  }}
  .phone-screen::-webkit-scrollbar {{ display: none; }}
  .phone-screen img {{
    width: 100%; display: block;
  }}
  .scroll-hint {{
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.5); color: #fff;
    font-size: 11px; padding: 4px 12px; border-radius: 12px;
    pointer-events: none;
    animation: fadeout 3s forwards;
    font-family: sans-serif;
  }}
  @keyframes fadeout {{
    0%,60% {{ opacity: 1; }}
    100% {{ opacity: 0; }}
  }}
</style>
</head><body>
  <div class="phone">
    <div class="phone-screen">
      <img src="data:image/jpeg;base64,{b64}" />
    </div>
    <div class="scroll-hint">↕ スクロールできます</div>
  </div>
</body></html>'''
    return html


PRIORITY_CSS = {"high": "advice-high", "medium": "advice-medium", "low": "advice-low"}
PRIORITY_LABEL = {"high": "🔴 最優先", "medium": "🟡 改善推奨", "low": "🟢 Good"}

# ===== 検索結果プレビュー（リアルタイムで楽天検索→スクショ→合成） =====
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time as _time
import os as _os
import urllib.parse


def capture_rakuten_search(keyword, mobile=False):
    """楽天で検索してスクショ＋サムネ座標を返す"""
    encoded = urllib.parse.quote(keyword)
    url = f"https://search.rakuten.co.jp/search/mall/{encoded}/"

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--lang=ja")

    if mobile:
        opts.add_argument("--window-size=393,852")
    else:
        opts.add_argument("--window-size=1400,900")
        opts.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

    driver = webdriver.Chrome(options=opts)
    try:
        if mobile:
            # CDPでモバイルエミュレーション
            driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 393, "height": 852, "deviceScaleFactor": 1, "mobile": True,
            })
            driver.execute_cdp_cmd("Emulation.setUserAgentOverride", {
                "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            })

        driver.get(url)
        _time.sleep(4)

        if mobile:
            # 通知バナー・邪魔な要素を削除
            driver.execute_script("""
                document.querySelectorAll('[class*="notification"], [class*="banner"], [class*="notice"], [class*="shippingInfo"], [class*="shipping-info"]').forEach(function(el) { el.remove(); });
                document.querySelectorAll('button').forEach(function(btn) {
                    var aria = btn.getAttribute('aria-label') || '';
                    if (aria === 'close' || aria === '閉じる') {
                        var parent = btn.closest('[class]');
                        if (parent && parent.offsetHeight < 150) parent.remove();
                    }
                });
            """)
            _time.sleep(1)
            # グリッド表示に切り替え
            driver.execute_script("""
                var icon = document.querySelector("i.dui-icon.-gridview");
                if (icon) { icon.click(); }
            """)
            _time.sleep(3)
        else:
            # PC版: サイドバーを非表示
            driver.execute_script("""
                var sidebar = document.querySelector('.dui-container--sidebar');
                if (sidebar) sidebar.style.display = 'none';
                var main = document.querySelector('.dui-container--main');
                if (main) main.style.maxWidth = '100%';
            """)
            _time.sleep(0.5)

        # サムネイル画像の要素と座標を取得（PR枠を除外、一般商品のみ）
        min_size = 80 if mobile else 80
        # まず .searchresultitem 内の画像だけ取得（PR除外）
        item_imgs = driver.find_elements(By.CSS_SELECTOR, ".searchresultitem:not([data-card-type='cpc']) img")
        if len(item_imgs) < 3:
            # フォールバック: 全イメージから
            item_imgs = driver.find_elements(By.CSS_SELECTOR, "img")
        thumb_data = []
        for img in item_imgs:
            src = img.get_attribute("src") or ""
            if "tshop.r10s.jp" in src or "thumbnail.image.rakuten" in src:
                rect = img.rect
                if rect["width"] > min_size and rect["height"] > min_size:
                    thumb_data.append({
                        "x": int(rect["x"]),
                        "y": int(rect["y"]),
                        "w": int(rect["width"]),
                        "h": int(rect["height"]),
                    })

        # 商品エリアだけを撮影
        if thumb_data:
            max_bottom = max(t["y"] + t["h"] for t in thumb_data[:15])
            capture_h = min(max_bottom + 300, 4000)
        else:
            capture_h = 1800

        if mobile:
            driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 393, "height": capture_h, "deviceScaleFactor": 1, "mobile": True,
            })
        else:
            driver.set_window_size(1400, capture_h)
        _time.sleep(1)

        screenshot_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_screenshot.png")
        driver.save_screenshot(screenshot_path)

        screenshot = Image.open(screenshot_path).convert("RGB")
    finally:
        driver.quit()

    return screenshot, thumb_data


def composite_on_screenshot(screenshot, thumb_data, user_img, slot_index=4):
    """スクショのサムネ位置にユーザー画像を合成"""
    result = screenshot.copy()

    if slot_index >= len(thumb_data):
        slot_index = min(4, len(thumb_data) - 1)
    if slot_index < 0 or not thumb_data:
        return result

    slot = thumb_data[slot_index]
    x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]

    # ユーザー画像をアスペクト比を保ってリサイズ → 中央配置
    src_w, src_h = user_img.size
    scale = min(w / src_w, h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    user_thumb = user_img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)

    # 背景を白で埋めて中央に配置
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    offset_x = (w - new_w) // 2
    offset_y = (h - new_h) // 2
    bg.paste(user_thumb, (offset_x, offset_y))
    result.paste(bg, (x, y))

    return result


def get_grade(score):
    if score >= 85:
        return "S", "お手本レベル！この品質を維持しましょう"
    elif score >= 70:
        return "A", "かなり良いサムネです！細部の改善でさらに良く"
    elif score >= 55:
        return "B", "基本OK。いくつかの改善で大幅レベルアップ可能"
    elif score >= 40:
        return "C", "改善の余地あり。優先度高の項目から着手を"
    else:
        return "D", "大幅な改善チャンス！まず余白とテキスト量から"


def calc_total(scores):
    total = 0
    max_total = 0
    for cid, info in CRITERIA_INFO.items():
        total += scores.get(cid, 0) * info["weight"]
        max_total += 5 * info["weight"]
    return round(total / max_total * 100) if max_total > 0 else 0


# ===== メインUI =====

st.markdown("""
<div class="app-header">
    <div class="app-title">🎨 楽天サムネイル アドバイザー</div>
    <div class="app-desc">画像をアップロードするだけ → 自動解析 → スコア＆改善アドバイス</div>
</div>
""", unsafe_allow_html=True)

# アップロードエリア
col_up, col_kw = st.columns([2, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "サムネイル画像をドラッグ＆ドロップ（複数OK）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
with col_kw:
    search_keyword = st.text_input(
        "🔍 検索キーワード（検索結果プレビュー用）",
        placeholder="例：財布 レディース 本革",
        help="楽天で実際に検索して、あなたのサムネが他の商品と並んだときのイメージを表示します",
    )

if uploaded_files:
    for file_idx, uploaded_file in enumerate(uploaded_files):
        pil_img = Image.open(uploaded_file)

        if len(uploaded_files) > 1:
            st.markdown(f"---")
            st.markdown(f"### 📊 {uploaded_file.name}")

        col_img, col_result = st.columns([1, 1.6])

        with col_img:
            st.image(pil_img, caption=uploaded_file.name, use_container_width=True)

            # 画像の基本情報
            w_px, h_px = pil_img.size
            st.caption(f"📐 {w_px} x {h_px} px")

        with col_result:
            with st.spinner("解析中..."):
                analysis = analyze_image(pil_img)
                scores = compute_scores(analysis)
                total = calc_total(scores)
                grade, grade_msg = get_grade(total)

            # 総合スコア
            st.markdown(f"""
            <div class="score-card-main">
                <div style="font-size:0.85rem; opacity:0.8;">総合スコア</div>
                <div class="score-big">{total}<span class="score-max"> / 100</span></div>
                <div class="score-grade">グレード {grade} — {grade_msg}</div>
            </div>
            """, unsafe_allow_html=True)

            # 項目別スコアバー
            st.markdown("#### 📋 項目別スコア")
            for cid, info in CRITERIA_INFO.items():
                s = scores.get(cid, 3)
                ref = info["ref_score"]
                pct = s / 5 * 100

                if s >= ref:
                    color = "#38A169"
                elif s >= ref - 1:
                    color = "#ECC94B"
                else:
                    color = "#E53E3E"

                st.markdown(f"""
                <div class="meter-row">
                    <span class="meter-label">{info['icon']} {info['name']}</span>
                    <div class="meter-track">
                        <div class="meter-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                    <span class="meter-score" style="color:{color}">{"★" * s}{"☆" * (5-s)}</span>
                </div>
                """, unsafe_allow_html=True)

        # 改善アドバイス（フル幅で表示）
        improvements = []
        good_points = []
        for cid, info in CRITERIA_INFO.items():
            s = scores.get(cid, 3)
            adv = info["advice"][s]
            item = {**adv, "icon": info["icon"], "name": info["name"], "score": s}
            if adv["p"] in ("high", "medium"):
                improvements.append(item)
            else:
                good_points.append(item)

        # 優先度順
        improvements.sort(key=lambda x: 0 if x["p"] == "high" else 1)

        col_good, col_improve = st.columns([1, 1.3])

        with col_good:
            st.markdown("#### 👍 良い点")
            for item in good_points:
                st.markdown(f"""
                <div class="advice-box advice-low">
                    <span class="advice-title">{item['icon']} {item['name']}：{item['t']}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_improve:
            if improvements:
                st.markdown("#### 🛠️ 改善ポイント")
                for item in improvements:
                    css = PRIORITY_CSS[item["p"]]
                    label = PRIORITY_LABEL[item["p"]]
                    st.markdown(f"""
                    <div class="advice-box {css}">
                        <span class="advice-title">{label} {item['icon']} {item['name']}：{item['t']}</span>
                        <div class="advice-detail">{item['d']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Quick Win
        high_items = [i for i in improvements if i["p"] == "high"]
        if high_items:
            qw = high_items[0]
            st.markdown(f"""
            <div class="quick-win">
                <div class="quick-win-title">⚡ まずこれだけやろう（Quick Win）</div>
                <div class="quick-win-detail">{qw['icon']} <strong>{qw['t']}</strong> — {qw['d']}</div>
            </div>
            """, unsafe_allow_html=True)

        # 楽天検索結果プレビュー（一番下に表示）
        if search_keyword:
            st.markdown("---")
            st.markdown("#### 🔍 実際の検索結果でのイメージ")
            tab_pc, tab_sp = st.tabs(["🖥️ PC版", "📱 スマホ版"])

            with tab_pc:
                with st.spinner(f"PC版「{search_keyword}」を検索中..."):
                    try:
                        screenshot, thumb_data = capture_rakuten_search(search_keyword, mobile=False)
                        if thumb_data:
                            preview = composite_on_screenshot(screenshot, thumb_data, pil_img, slot_index=0)
                            st.image(preview, use_container_width=True)
                        else:
                            st.warning("サムネイルの検出に失敗しました。キーワードを変えて試してみてください。")
                    except Exception as e:
                        st.error(f"検索結果の取得に失敗しました: {e}")

            with tab_sp:
                with st.spinner(f"スマホ版「{search_keyword}」を検索中..."):
                    try:
                        screenshot_sp, thumb_data_sp = capture_rakuten_search(search_keyword, mobile=True)
                        if thumb_data_sp:
                            preview_sp = composite_on_screenshot(screenshot_sp, thumb_data_sp, pil_img, slot_index=0)
                            # iPhoneモックアップで表示
                            phone_html = render_phone_mockup(preview_sp)
                            components.html(phone_html, height=620, scrolling=False)
                        else:
                            st.warning("サムネイルの検出に失敗しました。キーワードを変えて試してみてください。")
                    except Exception as e:
                        st.error(f"検索結果の取得に失敗しました: {e}")

        # 解析詳細（折りたたみ）
        with st.expander("🔬 解析データの詳細を見る"):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-title">⬜ 余白率</div>
                    <div class="detail-value">全体: {analysis['whitespace']['ratio']}%<br>
                    端部分: {analysis['whitespace']['border_ratio']}%<br>
                    総合: {analysis['whitespace']['effective']}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-title">🔤 テキスト量推定</div>
                    <div class="detail-value">エッジ密度: {analysis['text_amount']['edge_density']}%<br>
                    推定テキスト面積: {analysis['text_amount']['estimated_text_area']}%</div>
                </div>
                """, unsafe_allow_html=True)
            with d2:
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-title">🖼️ 背景</div>
                    <div class="detail-value">シンプルブロック: {analysis['background']['simple_blocks']}/25<br>
                    枠線検出: {'あり ⚠️' if analysis['background']['has_border'] else 'なし ✅'}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-title">🎨 色使い</div>
                    <div class="detail-value">平均彩度: {analysis['color_tone']['avg_saturation']}<br>
                    高彩度面積: {analysis['color_tone']['high_sat_ratio']}%<br>
                    使用色数: 約{analysis['color_tone']['n_colors']}色</div>
                </div>
                """, unsafe_allow_html=True)
            with d3:
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-title">📸 写真品質</div>
                    <div class="detail-value">コントラスト: {analysis['photo_quality']['contrast']}<br>
                    明るさ: {analysis['photo_quality']['brightness']}<br>
                    シャープネス: {analysis['photo_quality']['sharpness']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="detail-card">
                    <div class="detail-title">📐 構図</div>
                    <div class="detail-value">コンテンツ率: {analysis['composition']['content_ratio']}%<br>
                    中央集中度: {analysis['composition']['center_focus']}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👆 上のエリアにサムネイル画像をドロップしてください（複数OK）")

# フッター
st.markdown("---")
st.caption("🎨 楽天サムネイル アドバイザー v2.0 — トップ店舗の分析ナレッジに基づく自動評価ツール")
