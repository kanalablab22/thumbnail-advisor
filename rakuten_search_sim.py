"""
楽天検索結果シミュレーション
キーワードで検索した結果のサムネイル画像を取得し、
指定位置にユーザーの商品画像を挿入したシミュレーション画像を生成する
（PDFレポート用）
"""

import os
import requests
import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import urllib.parse

# フォントキャッシュ
_font_cache = {}


def _get_japanese_font(size: int) -> ImageFont.FreeTypeFont:
    """日本語フォントを取得（Noto Sans JPをダウンロード）"""
    if size in _font_cache:
        return _font_cache[size]

    # ローカルのフォントを試す（macOS）
    local_fonts = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for path in local_fonts:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except Exception:
            continue

    # Noto Sans JPをダウンロード（Streamlit Cloud用）
    font_dir = os.path.join(os.path.dirname(__file__), ".fonts")
    font_path = os.path.join(font_dir, "NotoSansJP-Regular.ttf")

    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)
        try:
            url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(resp.content)
        except Exception:
            pass

    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, size)
            _font_cache[size] = font
            return font
        except Exception:
            pass

    # 最終フォールバック
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def fetch_rakuten_thumbnails(keyword: str, count: int = 8) -> list:
    """
    楽天でキーワード検索し、商品サムネイル画像URLを取得する
    Returns: PIL Image のリスト
    """
    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    session.headers.update(headers)

    try:
        # まずトップページにアクセスしてCookieを取得
        session.get("https://www.rakuten.co.jp/", timeout=10)

        # 検索実行
        encoded = urllib.parse.quote(keyword)
        search_url = f"https://search.rakuten.co.jp/search/mall/{encoded}/"
        resp = session.get(search_url, timeout=10)
        resp.raise_for_status()
        html = resp.text

        # 楽天の商品画像URLパターン
        patterns = [
            r'https://tshop\.r10s\.jp/[^"\'>\s]+\.(?:jpg|jpeg|png|gif|webp)',
            r'https://thumbnail\.image\.rakuten\.co\.jp/[^"\'>\s]+',
            r'https://shop\.r10s\.jp/[^"\'>\s]+\.(?:jpg|jpeg|png|gif|webp)',
        ]

        image_urls = []
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for url in matches:
                if url not in image_urls:
                    image_urls.append(url)
            if len(image_urls) >= count + 10:
                break

        # 画像をダウンロード
        images = []
        seen_bases = set()
        for img_url in image_urls:
            if len(images) >= count:
                break
            # ベースURLで重複排除
            base_url = img_url.split("?")[0]
            if base_url in seen_bases:
                continue
            seen_bases.add(base_url)

            try:
                img_resp = session.get(img_url, timeout=5)
                if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                    img = Image.open(BytesIO(img_resp.content)).convert("RGBA")
                    # 極端に小さい画像（アイコンなど）を除外
                    if img.width >= 50 and img.height >= 50:
                        images.append(img)
            except Exception:
                continue

        return images

    except Exception:
        return []


def _build_grid(
    keyword: str,
    all_items: list,
    cols: int = 4,
    thumb_size: int = 200,
    header_height: int = 70,
    mobile: bool = False,
) -> Image.Image:
    """グリッドレイアウトの画像を生成する（PC/スマホ共通ロジック）"""
    padding = 12 if mobile else 15
    label_height = 20 if mobile else 25
    cell_w = thumb_size + padding * 2
    cell_h = thumb_size + padding * 2 + label_height

    total_items = max(cols * 2, len(all_items))
    rows = (total_items + cols - 1) // cols

    canvas_w = cols * cell_w + padding
    canvas_h = header_height + rows * cell_h + padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ヘッダー（楽天風の検索バー）
    draw.rectangle([0, 0, canvas_w, header_height], fill=(191, 0, 0))  # 楽天レッド

    bar_x = padding
    bar_y = (header_height - 36) // 2 if mobile else 15
    bar_w = canvas_w - padding * 2
    bar_h = 32 if mobile else 40
    draw.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        radius=8,
        fill=(255, 255, 255),
    )

    font_size = 11 if mobile else 14
    font_small = _get_japanese_font(font_size)
    font_label = _get_japanese_font(9 if mobile else 11)

    draw.text((bar_x + 8, bar_y + (bar_h - font_size) // 2), keyword, fill=(0, 0, 0), font=font_small)

    # グリッドに画像を配置
    for idx in range(min(total_items, len(all_items))):
        row = idx // cols
        col = idx % cols

        x = padding + col * cell_w
        y = header_height + 10 + row * cell_h

        item = all_items[idx]
        is_user = isinstance(item, tuple) and item[0] == "USER"
        img = item[1] if is_user else item

        draw.rectangle(
            [x, y, x + cell_w - padding, y + cell_h - 5],
            outline=(230, 230, 230),
            width=1,
        )

        img_rgb = img.convert("RGB") if img.mode != "RGB" else img
        img_rgb.thumbnail((thumb_size - 10, thumb_size - 10), Image.Resampling.LANCZOS)
        paste_x = x + (cell_w - padding - img_rgb.width) // 2
        paste_y = y + (thumb_size - img_rgb.height) // 2 + padding // 2
        canvas.paste(img_rgb, (paste_x, paste_y))

        # ダミー価格ライン
        price_y = y + cell_h - label_height - 5
        line_w = min(55, cell_w // 3)
        draw.rectangle([x + 5, price_y + 2, x + line_w, price_y + 4], fill=(200, 200, 200))
        draw.rectangle([x + 5, price_y + 10, x + line_w + 30, price_y + 12], fill=(220, 220, 220))

    # プレースホルダー
    if len(all_items) < total_items:
        for idx in range(len(all_items), total_items):
            row = idx // cols
            col = idx % cols
            x = padding + col * cell_w
            y = header_height + 10 + row * cell_h
            draw.rectangle(
                [x + 5, y + 5, x + cell_w - padding - 5, y + thumb_size + padding],
                fill=(245, 245, 245),
                outline=(220, 220, 220),
            )
            draw.text(
                (x + cell_w // 3, y + thumb_size // 2),
                "No Image",
                fill=(180, 180, 180),
                font=font_label,
            )

    return canvas


def create_search_simulation(
    keyword: str,
    user_image: Image.Image,
    position: int = 5,
    competitor_images: list = None,
) -> Image.Image:
    """PC版（5列）の検索結果シミュレーション"""
    if competitor_images is None:
        competitor_images = fetch_rakuten_thumbnails(keyword, count=14)

    all_items = list(competitor_images)
    insert_idx = min(position - 1, len(all_items))
    all_items.insert(insert_idx, ("USER", user_image))

    return _build_grid(keyword, all_items, cols=5, thumb_size=180)


def create_mobile_simulation(
    keyword: str,
    user_image: Image.Image,
    position: int = 5,
    competitor_images: list = None,
) -> Image.Image:
    """スマホ版（2列）の検索結果シミュレーション"""
    if competitor_images is None:
        competitor_images = fetch_rakuten_thumbnails(keyword, count=5)

    all_items = list(competitor_images)
    insert_idx = min(position - 1, len(all_items))
    all_items.insert(insert_idx, ("USER", user_image))

    return _build_grid(keyword, all_items, cols=2, thumb_size=160, header_height=55, mobile=True)
