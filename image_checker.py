"""
楽天サムネイルアドバイザー - 画像解析ロジック
楽天サムネイルのガイドライン＋社内基準に基づいてチェック
"""

import numpy as np
from PIL import Image, ImageDraw, ImageStat
from dataclasses import dataclass


@dataclass
class CheckResult:
    """チェック結果を格納するデータクラス"""
    name: str           # チェック項目ID（CRITERIA_INFOのキー）
    passed: bool        # True=OK, False=NG
    value: str          # 数値や結果の表示用文字列
    detail: str         # 詳細・改善アドバイス
    level: str = "ok"   # "ok"=low, "warn"=medium, "ng"=high
    score_value: int = 3  # 1-5のスコア値


@dataclass
class ImageCheckReport:
    """画像1枚のチェックレポート"""
    filename: str
    width: int
    height: int
    results: list       # CheckResult のリスト
    annotated_image: Image.Image  # 注釈付き画像
    score: int = 0      # 総合スコア（100点満点）
    analysis_data: dict = None  # 解析生データ

    @staticmethod
    def calc_score(results: list) -> int:
        """チェック結果から100点満点のスコアを算出"""
        if not results:
            return 0
        total = len(results)
        points = 0
        ng_count = 0
        for r in results:
            if r.level == "ok":
                points += 1.0
            elif r.level == "warn":
                points += 0.2
            else:
                ng_count += 1
        base = points / total * 100
        penalty = ng_count * 5
        return max(0, round(base - penalty))


# ===== 解析エンジン =====

def _analyze_image(pil_img: Image.Image) -> dict:
    """PIL画像を解析して各種データを返す（numpy/PILのみ、OpenCV不使用）"""
    img_rgb = np.array(pil_img.convert("RGB"))
    # グレースケール変換（手動計算）
    img_gray = (
        img_rgb[:, :, 0].astype(float) * 0.299
        + img_rgb[:, :, 1].astype(float) * 0.587
        + img_rgb[:, :, 2].astype(float) * 0.114
    ).astype(np.uint8)

    # HSV変換（手動計算）
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    diff_rgb = max_rgb - min_rgb

    # 彩度（0-255スケール）
    saturation = np.zeros_like(max_rgb)
    nonzero = max_rgb > 0
    saturation[nonzero] = (diff_rgb[nonzero] / max_rgb[nonzero]) * 255

    # 色相（0-180スケール、OpenCV互換）
    hue = np.zeros_like(max_rgb)
    mask_r = (max_rgb == r) & (diff_rgb > 0)
    mask_g = (max_rgb == g) & (diff_rgb > 0)
    mask_b = (max_rgb == b) & (diff_rgb > 0)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff_rgb[mask_r]) % 360) / 2
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff_rgb[mask_g]) + 120) / 2
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff_rgb[mask_b]) + 240) / 2

    h, w = img_gray.shape
    results = {}

    # ----- 1. 余白率（白〜ライトグレー領域の割合）-----
    whitespace_mask = img_gray > 235
    whitespace_ratio = np.sum(whitespace_mask) / (h * w)
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
    # 簡易Cannyエッジ（ソーベルフィルタで近似）
    gray_f = img_gray.astype(float)
    # 水平方向ソーベル
    sobel_x = np.zeros_like(gray_f)
    sobel_x[:, 1:-1] = gray_f[:, 2:] - gray_f[:, :-2]
    # 垂直方向ソーベル
    sobel_y = np.zeros_like(gray_f)
    sobel_y[1:-1, :] = gray_f[2:, :] - gray_f[:-2, :]
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = (edge_mag > 30).astype(np.uint8)
    edge_density = np.sum(edges > 0) / (h * w)

    # ラプラシアン（高周波成分）
    laplacian = np.zeros_like(gray_f)
    laplacian[1:-1, 1:-1] = (
        gray_f[:-2, 1:-1] + gray_f[2:, 1:-1]
        + gray_f[1:-1, :-2] + gray_f[1:-1, 2:]
        - 4 * gray_f[1:-1, 1:-1]
    )
    high_freq = np.mean(np.abs(laplacian))

    # テキスト領域推定（エッジを膨張して面積を計算）
    # 簡易膨張（3回繰り返し）
    dilated = edges.copy()
    for _ in range(3):
        new_dilated = dilated.copy()
        new_dilated[1:, :] |= dilated[:-1, :]
        new_dilated[:-1, :] |= dilated[1:, :]
        new_dilated[:, 1:] |= dilated[:, :-1]
        new_dilated[:, :-1] |= dilated[:, 1:]
        # 斜め方向も
        new_dilated[1:, 1:] |= dilated[:-1, :-1]
        new_dilated[:-1, :-1] |= dilated[1:, 1:]
        new_dilated[1:, :-1] |= dilated[:-1, 1:]
        new_dilated[:-1, 1:] |= dilated[1:, :-1]
        dilated = new_dilated
    text_area_ratio = np.sum(dilated > 0) / (h * w)

    results["text_amount"] = {
        "edge_density": round(edge_density * 100, 1),
        "high_freq": round(high_freq, 1),
        "estimated_text_area": round(text_area_ratio * 100, 1),
    }

    # ----- 3. 背景のシンプルさ -----
    block_h, block_w = h // 5, w // 5
    block_stds = []
    for bi in range(5):
        for bj in range(5):
            block = img_gray[bi * block_h:(bi + 1) * block_h, bj * block_w:(bj + 1) * block_w]
            block_stds.append(np.std(block))
    simple_blocks = sum(1 for s in block_stds if s < 20)
    avg_block_std = np.mean(block_stds)

    # 枠線チェック: 画像端1pxの色が均一＝枠線あり
    top_edge = img_gray[0, :]
    bottom_edge = img_gray[-1, :]
    left_edge = img_gray[:, 0]
    right_edge = img_gray[:, -1]
    has_border = (
        np.std(top_edge) < 5 and np.std(bottom_edge) < 5
        and np.mean(top_edge) < 200
        and abs(np.mean(top_edge) - np.mean(bottom_edge)) < 10
    )

    results["background"] = {
        "simple_blocks": simple_blocks,
        "avg_std": round(avg_block_std, 1),
        "has_border": has_border,
    }

    # ----- 4. 色使い・彩度 -----
    avg_saturation = np.mean(saturation)
    high_sat_ratio = np.sum(saturation > 150) / (h * w)
    hue_flat = hue.flatten()
    sat_flat = saturation.flatten()
    hue_of_saturated = hue_flat[sat_flat > 50]
    if len(hue_of_saturated) > 100:
        hue_std = np.std(hue_of_saturated)
        n_dominant_colors = len(set(np.round(hue_of_saturated / 15).astype(int)))
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
    sharpness = np.var(laplacian)

    results["photo_quality"] = {
        "contrast": round(float(contrast), 1),
        "brightness": round(float(brightness), 1),
        "sharpness": round(float(sharpness), 1),
    }

    # ----- 6. 構図（中央集中度）-----
    content_mask = img_gray < 230
    content_ratio = np.sum(content_mask) / (h * w)
    center_region = content_mask[h // 4:3 * h // 4, w // 4:3 * w // 4]
    center_density = np.sum(center_region) / center_region.size
    overall_density = np.sum(content_mask) / content_mask.size
    center_focus = center_density / max(overall_density, 0.01)

    results["composition"] = {
        "content_ratio": round(content_ratio * 100, 1),
        "center_focus": round(center_focus, 2),
    }

    # ----- 7. カラバリ表示の検出 -----
    bottom_region_sat = saturation[int(h * 0.8):, :]
    if bottom_region_sat.size > 0:
        bottom_has_colors = np.sum(bottom_region_sat > 60) / bottom_region_sat.size
    else:
        bottom_has_colors = 0

    bottom_gray = img_gray[int(h * 0.85):, :]
    if bottom_gray.size > 0:
        col_means = np.mean(bottom_gray, axis=0)
        col_diff = np.abs(np.diff(col_means.astype(float)))
        n_transitions = np.sum(col_diff > 30)
        has_swatches = n_transitions > 6
    else:
        has_swatches = False

    results["color_variation"] = {
        "bottom_color_ratio": round(bottom_has_colors * 100, 1),
        "has_swatches": has_swatches,
    }

    return results


def _compute_scores(analysis: dict) -> dict:
    """解析結果からスコアを計算（各項目5点満点）"""
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

    # 2. テキスト量スコア（低い=テキスト少ない=良い）
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
        scores["background"] = max(1, min(2, sb // 5))
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
        scores["color_tone"] = 5
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
        quality_score += 2
    elif 30 <= contrast <= 90:
        quality_score += 1
    if sharpness >= 500:
        quality_score += 3
    elif sharpness >= 200:
        quality_score += 2
    elif sharpness >= 50:
        quality_score += 1
    scores["photo_quality"] = min(5, max(1, quality_score))

    # 6. 構図（中央集中度 + コンテンツ密度）
    cf = analysis["composition"]["center_focus"]
    cr = analysis["composition"]["content_ratio"]
    if 1.1 <= cf <= 1.8 and 40 <= cr <= 75:
        scores["composition"] = 5
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
        scores["color_variation"] = 2

    return scores


# ===== 評価項目の情報（楽天版） =====
CRITERIA_INFO = {
    "whitespace": {
        "icon": "⬜", "name": "whitespace", "display_name": "余白（呼吸感）",
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
        "icon": "🔤", "name": "text_amount", "display_name": "テキスト量",
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
        "icon": "🖼️", "name": "background", "display_name": "背景のシンプルさ",
        "weight": 1.0,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "背景を整理する", "d": "派手な装飾や枠線は外しましょう。布・ドライフラワーなどおしゃれな小物を使ったスタイリング背景はOKです。楽天ガイドラインでは枠線は禁止されています"},
            2: {"p": "medium", "t": "背景の装飾を見直す", "d": "背景がやや騒がしいかもしれません。商品が主役になるよう、背景の色味やパターンを抑えるか、スタイリング小物でおしゃれに演出しましょう"},
            3: {"p": "medium", "t": "背景をもう少し統一する", "d": "布や小物を使ったスタイリング背景でOKです。全商品で同じトーン・雰囲気に揃えるとブランドの統一感が出ます"},
            4: {"p": "low", "t": "ほぼ理想的な背景です", "d": "きれいな背景処理です。全商品で同じトーンに統一できればさらに良くなります"},
            5: {"p": "low", "t": "完璧な背景処理！", "d": "商品が映える素敵な背景です"},
        },
    },
    "color_tone": {
        "icon": "🎨", "name": "color_tone", "display_name": "色使い・トーン",
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
        "icon": "📸", "name": "photo_quality", "display_name": "写真のクオリティ",
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
        "icon": "📐", "name": "composition", "display_name": "構図・レイアウト",
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
        "icon": "🌈", "name": "color_variation", "display_name": "カラバリ表示",
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
    "border": {
        "icon": "🚫", "name": "border", "display_name": "枠線検出",
        "weight": 0, "ref_score": 5,
        "advice": {},
    },
}


def _calc_total(scores: dict) -> int:
    """加重平均で100点満点のスコアを算出"""
    total = 0
    max_total = 0
    for cid, info in CRITERIA_INFO.items():
        total += scores.get(cid, 0) * info["weight"]
        max_total += 5 * info["weight"]
    return round(total / max_total * 100) if max_total > 0 else 0


def _get_grade(score: int) -> tuple:
    """スコアからグレードを返す"""
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


def check_image(image: Image.Image, filename: str = "image.jpg") -> "ImageCheckReport":
    """画像を解析して ImageCheckReport を返す"""
    analysis = _analyze_image(image)
    scores = _compute_scores(analysis)
    total_score = _calc_total(scores)
    grade, grade_msg = _get_grade(total_score)

    # チェック結果をリスト化
    results = []
    for cid, info in CRITERIA_INFO.items():
        if not info.get("advice"):
            continue
        s = scores.get(cid, 3)
        adv = info["advice"][s]
        level = adv["p"]  # "low", "medium", "high"
        results.append(CheckResult(
            name=cid,
            passed=(level != "high"),
            value=adv["t"],
            detail=adv["d"],
            level=level,
            score_value=s,
        ))

    # 枠線チェック（楽天ガイドラインで禁止）
    if analysis["background"]["has_border"]:
        results.append(CheckResult(
            name="border",
            passed=False,
            value="枠線が検出されました",
            detail="楽天ガイドラインで枠線の使用は禁止されています。枠線を削除してください",
            level="high",
            score_value=1,
        ))

    # annotated_image（元画像をそのまま使用）
    annotated = image.copy().convert("RGB")
    max_dim = 1200
    if max(annotated.size) > max_dim:
        scale = max_dim / max(annotated.size)
        new_size = (int(annotated.width * scale), int(annotated.height * scale))
        annotated = annotated.resize(new_size, Image.LANCZOS)

    score = total_score

    # analysis_dataをフラットにまとめる（app.pyの解析詳細表示用）
    flat_analysis = {
        "whitespace_ratio": analysis["whitespace"]["ratio"],
        "whitespace_effective": analysis["whitespace"]["effective"],
        "edge_density": analysis["text_amount"]["edge_density"],
        "text_area": analysis["text_amount"]["estimated_text_area"],
        "simple_blocks": analysis["background"]["simple_blocks"],
        "has_border": analysis["background"]["has_border"],
        "avg_saturation": analysis["color_tone"]["avg_saturation"],
        "high_sat_ratio": analysis["color_tone"]["high_sat_ratio"],
        "n_colors": analysis["color_tone"]["n_colors"],
        "contrast": analysis["photo_quality"]["contrast"],
        "brightness": analysis["photo_quality"]["brightness"],
        "sharpness": analysis["photo_quality"]["sharpness"],
        "content_ratio": analysis["composition"]["content_ratio"],
        "center_focus": analysis["composition"]["center_focus"],
    }

    report = ImageCheckReport(
        filename=filename,
        width=image.width,
        height=image.height,
        results=results,
        annotated_image=annotated,
        score=score,
        analysis_data=flat_analysis,
    )

    return report
