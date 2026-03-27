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

    # ----- 1. 余白率（均一な空間の割合）-----
    # 従来: 白ピクセル(>235)だけカウント → ベージュ・グレー背景が不当に低評価
    # 改善: 白ピクセル + 均一領域（低分散ブロック）も余白としてカウント
    whitespace_mask = img_gray > 235
    whitespace_ratio = np.sum(whitespace_mask) / (h * w)

    # 均一領域の検出（8x8ブロックで分散が低い＝シンプルな空間）
    block_size = max(h, w) // 8
    uniform_pixels = 0
    for bi in range(8):
        for bj in range(8):
            block = img_gray[bi * block_size:min((bi + 1) * block_size, h),
                             bj * block_size:min((bj + 1) * block_size, w)]
            if block.size > 0 and np.std(block) < 15:
                uniform_pixels += block.size
    uniform_ratio = uniform_pixels / (h * w)

    border = int(min(h, w) * 0.1)
    border_region = np.concatenate([
        img_gray[:border, :].flatten(),
        img_gray[-border:, :].flatten(),
        img_gray[:, :border].flatten(),
        img_gray[:, -border:].flatten(),
    ])
    border_white = np.sum(border_region > 235) / len(border_region)
    # 均一領域の端部分もチェック
    border_uniform = np.std(border_region) < 25

    # 白ベースの余白 + 均一領域ベースの余白を統合
    white_effective = whitespace_ratio * 0.6 + border_white * 0.4
    uniform_effective = uniform_ratio * 0.7 + (0.3 if border_uniform else 0.0)
    effective_whitespace = max(white_effective, uniform_effective)

    results["whitespace"] = {
        "ratio": round(whitespace_ratio * 100, 1),
        "border_ratio": round(border_white * 100, 1),
        "uniform_ratio": round(uniform_ratio * 100, 1),
        "effective": round(effective_whitespace * 100, 1),
    }

    # ----- 2. テキスト量推定（背景エリアのエッジで判定）-----
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

    # 背景エリアのエッジ密度でテキスト有無を判定
    # 商品テクスチャ（革のシボ等）とテキストを区別するため、
    # 背景（均一な領域）上のエッジだけをカウントする
    # 背景マスク: 白ピクセル or 均一領域（商品以外のエリア）
    bg_mask = whitespace_mask.copy()  # 白ピクセル(>235)
    # 均一ブロックも背景に追加
    for bi in range(8):
        for bj in range(8):
            block_slice = (
                slice(bi * block_size, min((bi + 1) * block_size, h)),
                slice(bj * block_size, min((bj + 1) * block_size, w)),
            )
            block = img_gray[block_slice]
            if block.size > 0 and np.std(block) < 15:
                bg_mask[block_slice] = True

    # 周辺エリアマスク（テキストが配置されやすい上部・下部・左右端）
    # 楽天サムネではスタイリング背景の上にテキストを重ねることが多い
    peripheral_mask = np.zeros((h, w), dtype=bool)
    peripheral_mask[:int(h * 0.20), :] = True   # 上部20%
    peripheral_mask[int(h * 0.75):, :] = True    # 下部25%
    peripheral_mask[:, :int(w * 0.10)] = True    # 左端10%
    peripheral_mask[:, int(w * 0.90):] = True    # 右端10%

    # 背景エリアにあるエッジ＝テキストの可能性が高い
    bg_edges = edges & bg_mask.astype(np.uint8)
    bg_edge_ratio = np.sum(bg_edges > 0) / max(np.sum(bg_mask), 1)

    # 周辺エリアにあるエッジ＝テキストの可能性が高い（背景が白でなくても検出）
    peripheral_edges = edges & peripheral_mask.astype(np.uint8)
    peripheral_edge_ratio = np.sum(peripheral_edges > 0) / max(np.sum(peripheral_mask), 1)

    # 周辺エリアの膨張エッジ面積（テキスト面積の推定に使う）
    peripheral_dilated = dilated & peripheral_mask.astype(np.uint8)
    peripheral_text_area = np.sum(peripheral_dilated > 0) / (h * w)

    # テキスト有無の総合判定（どちらかで検出できればOK）
    # bg_edge_ratio: 白/均一背景上のエッジ → 白抜き画像のテキスト検出
    # peripheral_edge_ratio: 周辺エリアのエッジ → スタイリング背景上のテキスト検出
    has_text_on_bg = bg_edge_ratio > 0.02
    has_text_on_peripheral = peripheral_edge_ratio > 0.08
    has_text_overlay = has_text_on_bg or has_text_on_peripheral

    if has_text_on_bg and bg_mask.sum() > h * w * 0.3:
        # 白/均一背景が広い場合: 背景上のエッジ＝テキストなので従来通り
        effective_text_area = text_area_ratio
    elif has_text_overlay:
        # スタイリング背景の場合: 周辺エリアのエッジだけでテキスト面積を推定
        # （商品テクスチャのエッジを除外する）
        effective_text_area = peripheral_text_area
    else:
        # テキストなし
        effective_text_area = bg_edge_ratio * 100  # ほぼ0になる

    results["text_amount"] = {
        "edge_density": round(edge_density * 100, 1),
        "high_freq": round(high_freq, 1),
        "estimated_text_area": round(effective_text_area * 100, 1),
        "bg_edge_ratio": round(bg_edge_ratio * 100, 2),
        "peripheral_edge_ratio": round(peripheral_edge_ratio * 100, 2),
        "has_text_overlay": has_text_overlay,
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
    # 下部20%の彩度チェック
    bottom_region_sat = saturation[int(h * 0.8):, :]
    if bottom_region_sat.size > 0:
        bottom_has_colors = np.sum(bottom_region_sat > 60) / bottom_region_sat.size
    else:
        bottom_has_colors = 0

    # 下部15%で色の切り替わり（カラードット等）を検出
    bottom_gray = img_gray[int(h * 0.85):, :]
    bottom_hue = hue[int(h * 0.85):, :]
    bottom_sat_region = saturation[int(h * 0.85):, :]
    has_swatches = False
    n_distinct_hues = 0

    if bottom_gray.size > 0:
        col_means = np.mean(bottom_gray, axis=0)
        col_diff = np.abs(np.diff(col_means.astype(float)))
        n_transitions = np.sum(col_diff > 30)

        # カラバリ判定: 色の切り替わりだけでなく、複数の異なる色相が必要
        # 商品が1色だけ下部にはみ出してるケースを除外
        sat_mask = bottom_sat_region > 50
        if np.sum(sat_mask) > 100:
            hues_in_bottom = bottom_hue[sat_mask]
            # 色相を15度刻みでグルーピングして何色あるか数える
            hue_bins = np.round(hues_in_bottom / 15).astype(int)
            n_distinct_hues = len(set(hue_bins))

        # カラバリあり = 色の切り替わりが多い AND 3色以上の異なる色相がある
        has_swatches = n_transitions > 6 and n_distinct_hues >= 3

    results["color_variation"] = {
        "bottom_color_ratio": round(bottom_has_colors * 100, 1),
        "has_swatches": has_swatches,
        "n_distinct_hues": n_distinct_hues,
    }

    return results


def _compute_scores(analysis: dict) -> dict:
    """解析結果からスコアを計算（各項目5点満点）— 楽天市場基準"""
    scores = {}

    # 1. 余白バランス（楽天基準: 多すぎもマイナス。商品50-70%が最適）
    ws = analysis["whitespace"]["effective"]
    ur = analysis["whitespace"].get("uniform_ratio", 0)
    if 15 <= ws <= 35:
        scores["whitespace"] = 5  # 適度な余白（商品＋テキストのバランスが取れる）
    elif 10 <= ws <= 45:
        scores["whitespace"] = 4
    elif 8 <= ws <= 55:
        scores["whitespace"] = 3
    elif ws > 55:
        scores["whitespace"] = 2  # 余白が多すぎ（Amazon風白抜き＝楽天では訴求不足）
    else:
        scores["whitespace"] = 2  # 余白なさすぎ

    # 2. テキスト最適度（楽天基準: ゼロも多すぎもNG。8-15%が最適）
    # edge-based推定なので実際のテキスト面積とは異なるが傾向は合う
    ta = analysis["text_amount"]["estimated_text_area"]
    ed = analysis["text_amount"]["edge_density"]
    if 20 <= ta <= 40:
        scores["text_amount"] = 5  # 最適ゾーン（適度なテキスト訴求あり）
    elif 15 <= ta <= 45:
        scores["text_amount"] = 4
    elif 10 <= ta <= 55:
        scores["text_amount"] = 3
    elif ta < 10:
        scores["text_amount"] = 2  # テキストがほぼない（楽天では訴求力不足）
    else:
        scores["text_amount"] = 1  # テキスト過多

    # 3. 背景の適切さ（楽天基準: 白もスタイリングも同等に評価）
    sb = analysis["background"]["simple_blocks"]
    has_border = analysis["background"]["has_border"]
    avg_std = analysis["background"]["avg_std"]
    if has_border:
        scores["background"] = 1  # 枠線はガイドライン違反
    elif sb >= 8 and avg_std < 50:
        scores["background"] = 5  # きれいな白背景 or 統一感あるスタイリング背景
    elif sb >= 5:
        scores["background"] = 4  # おおむね良い背景
    elif sb >= 3:
        scores["background"] = 3  # やや雑多だが許容範囲
    elif avg_std < 60:
        scores["background"] = 3  # スタイリング背景（多少複雑でもOK）
    else:
        scores["background"] = 2  # 背景がごちゃごちゃ

    # 4. 色使い・トーン
    avg_sat = analysis["color_tone"]["avg_saturation"]
    high_sat = analysis["color_tone"]["high_sat_ratio"]
    n_colors = analysis["color_tone"]["n_colors"]
    if avg_sat <= 60 and high_sat <= 8 and n_colors <= 6:
        scores["color_tone"] = 5  # 2-3色で統一された上品な色使い
    elif avg_sat <= 80 and high_sat <= 15:
        scores["color_tone"] = 4
    elif avg_sat <= 100 and high_sat <= 25:
        scores["color_tone"] = 3
    elif avg_sat <= 120:
        scores["color_tone"] = 2
    else:
        scores["color_tone"] = 1  # 蛍光色・原色多用

    # 5. 写真品質（コントラスト + シャープネス + 明るさ）
    contrast = analysis["photo_quality"]["contrast"]
    sharpness = analysis["photo_quality"]["sharpness"]
    brightness = analysis["photo_quality"]["brightness"]
    quality_score = 0
    # コントラスト
    if 40 <= contrast <= 80:
        quality_score += 2
    elif 30 <= contrast <= 90:
        quality_score += 1
    # シャープネス
    if sharpness >= 500:
        quality_score += 2
    elif sharpness >= 200:
        quality_score += 1.5
    elif sharpness >= 50:
        quality_score += 1
    # 明るさ（暗すぎ・明るすぎをチェック）
    if 100 <= brightness <= 200:
        quality_score += 1
    elif 80 <= brightness <= 220:
        quality_score += 0.5
    scores["photo_quality"] = min(5, max(1, round(quality_score)))

    # 6. 構図・レイアウト（楽天基準: 商品が中央に大きく、テキストスペースも確保）
    cf = analysis["composition"]["center_focus"]
    cr = analysis["composition"]["content_ratio"]
    if 1.0 <= cf <= 1.8 and 40 <= cr <= 80:
        scores["composition"] = 5  # 商品中央＋テキスト配置の余地あり
    elif 0.9 <= cf <= 2.0 and 35 <= cr <= 85:
        scores["composition"] = 4
    elif cr <= 90:
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
        "icon": "⬜", "name": "whitespace", "display_name": "余白バランス",
        "weight": 0.8,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "余白がほぼない", "d": "商品とテキストが詰まりすぎています。スマホ表示では特に窮屈に見えるので、要素を整理して商品の周りに呼吸できるスペースを確保しましょう"},
            2: {"p": "medium", "t": "余白のバランスを調整", "d": "余白が少なすぎるか多すぎるかもしれません。商品が画像の50-70%を占め、テキスト＋余白で残り30-50%が楽天の理想バランスです"},
            3: {"p": "medium", "t": "もう少し余白を調整", "d": "商品・テキスト・余白のバランスをもう少し最適化できそうです。商品を中央に大きく、テキストは上部か下部にまとめてみましょう"},
            4: {"p": "low", "t": "良い余白バランスです", "d": "商品とテキストが程よく配置されています。この配置ルールを全商品に展開しましょう"},
            5: {"p": "low", "t": "理想的な余白バランス！", "d": "商品・テキスト・余白の三位一体が取れた素晴らしいレイアウトです"},
        },
    },
    "text_amount": {
        "icon": "🔤", "name": "text_amount", "display_name": "テキスト最適度",
        "weight": 1.2,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "テキスト量を見直す", "d": "テキストが多すぎるか、ほぼゼロです。楽天ではガイドライン上限20%以内で訴求ワードを入れるのが効果的。「ランキング1位」「送料無料」など一言添えるだけでCTRが大きく変わります"},
            2: {"p": "medium", "t": "訴求テキストを追加する", "d": "テキストがほとんどない（Amazon風）か、逆に多すぎます。楽天では「商品の一番の強み」を1フレーズ入れるのが正解。全角6〜10文字のキャッチコピーを検討しましょう"},
            3: {"p": "medium", "t": "テキスト量をもう少し調整", "d": "あと少し最適化できそうです。テキストは画像面積の8-15%が最適ゾーン。大きさのメリハリ（主訴求を大きく、補足は小さく）も意識してみましょう"},
            4: {"p": "low", "t": "良いテキストバランスです", "d": "適度なテキスト量で訴求力があります。文字サイズの大・中・小にメリハリをつけるとさらに効果的"},
            5: {"p": "low", "t": "理想的なテキスト量！", "d": "商品写真を活かしつつ、訴求テキストも効いている素晴らしいバランスです"},
        },
    },
    "background": {
        "icon": "🖼️", "name": "background", "display_name": "背景の適切さ",
        "weight": 0.9,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "背景を見直す（枠線はNG）", "d": "枠線が検出された場合、楽天ガイドライン違反です。白背景かスタイリング写真（布・木目・小物を一緒に撮影）に変更しましょう"},
            2: {"p": "medium", "t": "背景を整える", "d": "背景がやや雑然としています。白背景 or スタイリング写真のどちらかに統一して、商品が一番目立つようにしましょう"},
            3: {"p": "medium", "t": "背景の統一感を上げる", "d": "背景は悪くないですが、全商品で同じトーンに揃えるとブランド感がアップします。白でもスタイリングでもOKです"},
            4: {"p": "low", "t": "良い背景です", "d": "商品を引き立てる背景になっています。全商品で同じ雰囲気を展開しましょう"},
            5: {"p": "low", "t": "素晴らしい背景！", "d": "商品の魅力を最大限に引き出す背景処理です"},
        },
    },
    "color_tone": {
        "icon": "🎨", "name": "color_tone", "display_name": "色使い・トーン",
        "weight": 0.7,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "色使いを見直す", "d": "蛍光色・原色が多すぎます。メイン2-3色に絞り、テキスト色は黒〜ダークグレーに統一。アクセントは「ランキング1位」バッジなどピンポイントだけに"},
            2: {"p": "medium", "t": "色数を絞る", "d": "色が多すぎて統一感が薄いかもしれません。商品色＋テキスト色＋アクセント1色の3色ルールを試してみましょう"},
            3: {"p": "medium", "t": "トーンを統一する", "d": "使う色を3色に絞りましょう。テキスト＝黒系、背景＝白系、アクセント＝ブランドカラー1色が理想です"},
            4: {"p": "low", "t": "落ち着いた色使いです", "d": "色数が抑えられて上品です。全商品で同じカラーパレットを使ってブランド感を高めましょう"},
            5: {"p": "low", "t": "洗練された色使い！", "d": "統一感のあるカラーパレットです。ブランドカラーとして定着させていきましょう"},
        },
    },
    "photo_quality": {
        "icon": "📸", "name": "photo_quality", "display_name": "写真のクオリティ",
        "weight": 1.0,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "写真撮影を改善する", "d": "暗い・ぼやけ・ピンボケが見られます。ディフューズボックスか窓際の自然光＋白い布で柔らかい光をあてて撮影し直しましょう"},
            2: {"p": "medium", "t": "ライティングを改善", "d": "直射光を避けてディフューズ光（柔らかい光）で撮影。自然光＋レフ板で素材本来の質感が伝わる仕上がりにしましょう"},
            3: {"p": "medium", "t": "写真トーンを調整", "d": "素材の質感が伝わるトーンに調整しましょう。コントラストを少し上げつつハイライトを抑えると上品な仕上がりに"},
            4: {"p": "low", "t": "良い写真品質です", "d": "しっかり撮れています。全商品で写真トーン（明るさ・彩度）を統一するとブランド感がアップします"},
            5: {"p": "low", "t": "プロレベルの写真品質！", "d": "商品の質感がしっかり伝わる素晴らしい撮影です"},
        },
    },
    "composition": {
        "icon": "📐", "name": "composition", "display_name": "構図・レイアウト",
        "weight": 1.1,
        "ref_score": 5,
        "advice": {
            1: {"p": "high", "t": "構図を見直す", "d": "商品を中央〜やや上に大きく配置し、テキストは上部か下部の帯状エリアにまとめましょう。スマホの小さい画面でも一目で何の商品かわかることが最優先です"},
            2: {"p": "medium", "t": "レイアウトを整理する", "d": "要素が散らばっている可能性があります。「テキスト帯（上部15%）→ 商品（中央大きく）→ カラバリ（下部）」の構成を試してみましょう"},
            3: {"p": "medium", "t": "もう少し整理できます", "d": "商品が中心ですが、テキスト配置がもう少し整理できそうです。Z型の視線の流れ（左上→右上→左下→右下）を意識してみましょう"},
            4: {"p": "low", "t": "良い構図です", "d": "商品が中心にしっかり配置されています。テキスト配置のルールを固定化して全商品に展開しましょう"},
            5: {"p": "low", "t": "素晴らしい構図！", "d": "商品とテキストの配置バランスが理想的です"},
        },
    },
    "color_variation": {
        "icon": "🌈", "name": "color_variation", "display_name": "カラバリ表示",
        "weight": 0.6,
        "ref_score": 4,
        "advice": {
            1: {"p": "medium", "t": "カラバリ表示を追加", "d": "色展開がある商品なら、画像下部にカラードット（小さな色丸）を横並びにするのが簡単で効果的です。「選べる楽しさ」がクリック率を上げます"},
            2: {"p": "medium", "t": "カラバリをもう少し目立たせる", "d": "カラードットを少し大きくするか、別色の商品写真を小さく添えましょう"},
            3: {"p": "low", "t": "カラバリ表示あり", "d": "色展開の表示が確認できます。商品の邪魔にならない配置でGoodです"},
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
