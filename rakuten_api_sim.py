"""
楽天商品検索API を使った検索結果シミュレーション
Streamlit Cloud対応（ボットブロック回避）
"""

import requests
import base64
import urllib.parse
from io import BytesIO
from PIL import Image


def search_rakuten_items(keyword: str, app_id: str, hits: int = 12, access_key: str = "") -> list:
    """
    楽天商品検索APIで商品データを取得
    Returns: [{name, price, image_url, shop_name, review_avg, review_count}, ...]
    """
    url = "https://openapi.rakuten.co.jp/ichibams/api/IchibaItem/Search/20220601"
    params = {
        "applicationId": app_id,
        "keyword": keyword,
        "hits": hits,
        "sort": "standard",
        "imageFlag": 1,
    }
    if access_key:
        params["accessKey"] = access_key
    headers = {
        "Referer": "https://thumbnail-advisor-6qspbn26sdqwlcryashtjn.streamlit.app/",
        "Origin": "https://thumbnail-advisor-6qspbn26sdqwlcryashtjn.streamlit.app",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    items = []
    for item_data in data.get("Items", []):
        item = item_data.get("Item", {})
        images = item.get("mediumImageUrls", [])
        image_url = images[0]["imageUrl"] if images else ""
        # 楽天APIの画像URLは128x128。大きい画像に変換
        image_url = image_url.replace("?_ex=128x128", "?_ex=300x300")

        items.append({
            "name": item.get("itemName", ""),
            "price": item.get("itemPrice", 0),
            "image_url": image_url,
            "shop_name": item.get("shopName", ""),
            "review_avg": item.get("reviewAverage", "0"),
            "review_count": item.get("reviewCount", 0),
            "item_url": item.get("itemUrl", "#"),
        })
    return items


def _format_price(price: int) -> str:
    """価格を日本円フォーマットに"""
    return f"{price:,}"


def _user_image_to_data_url(user_image: Image.Image) -> str:
    """PIL画像をdata URLに変換"""
    buf = BytesIO()
    user_image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _render_stars(avg: str) -> str:
    """レビュー平均を星表示に変換"""
    try:
        score = float(avg)
    except (ValueError, TypeError):
        return ""
    full = int(score)
    half = 1 if score - full >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("☆" if half else "") + "☆" * empty


def build_pc_html(keyword: str, items: list, user_image: Image.Image, position: int = 3,
                   user_product_name: str = "", user_product_price: str = "") -> str:
    """PC版の楽天検索結果風HTMLを生成"""
    user_data_url = _user_image_to_data_url(user_image)
    display_name = user_product_name or f"【送料無料】{keyword} 人気 おすすめ"
    display_price = user_product_price or "3,980"

    # position番目にユーザー画像を挿入
    cards_html = ""
    item_idx = 0
    total = len(items) + 1  # +1 for user image

    for i in range(min(total, 20)):
        if i == position - 1:
            # ユーザーの商品画像（ハイライト付き）
            cards_html += f'''
            <div class="product-card user-card">
                <div class="user-badge">あなたの商品</div>
                <div class="product-image">
                    <img src="{user_data_url}" alt="">
                </div>
                <div class="product-info">
                    <div class="product-name">{display_name}</div>
                    <div class="product-price">¥{display_price}</div>
                    <div class="product-review"><span class="stars">★★★★☆</span> (128)</div>
                    <div class="product-shop">あなたのショップ</div>
                </div>
            </div>'''
        else:
            if item_idx < len(items):
                item = items[item_idx]
                stars = _render_stars(item["review_avg"])
                cards_html += f'''
            <div class="product-card">
                <div class="product-image">
                    <img src="{item['image_url']}" alt="{item['name'][:30]}">
                </div>
                <div class="product-info">
                    <div class="product-name">{item['name'][:60]}</div>
                    <div class="product-price">¥{_format_price(item['price'])}</div>
                    <div class="product-review"><span class="stars">{stars}</span> ({item['review_count']})</div>
                    <div class="product-shop">{item['shop_name']}</div>
                </div>
            </div>'''
                item_idx += 1

    return f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', sans-serif; background: #f5f5f5; }}

.header {{
    background: #bf0000; padding: 10px 20px; display: flex; align-items: center;
}}
.header .logo {{ color: white; font-weight: bold; font-size: 18px; margin-right: 15px; }}
.search-bar {{
    flex: 1; display: flex; height: 36px;
}}
.search-bar input {{
    flex: 1; border: none; padding: 0 12px; font-size: 14px; border-radius: 4px 0 0 4px;
}}
.search-bar button {{
    background: #f5f5f5; border: none; padding: 0 16px; font-size: 14px; cursor: pointer;
    border-radius: 0 4px 4px 0; color: #333;
}}

.breadcrumb {{ padding: 8px 20px; font-size: 12px; color: #666; background: #fff; border-bottom: 1px solid #eee; }}

.results-header {{
    padding: 10px 20px; font-size: 13px; color: #333; background: #fff;
    border-bottom: 1px solid #eee;
}}

.grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    padding: 0; background: #eee; margin: 0;
}}

.product-card {{
    background: #fff; padding: 12px; display: flex; flex-direction: column;
    position: relative;
}}
.product-card.user-card {{
    box-shadow: inset 0 0 0 3px #667eea;
}}
.user-badge {{
    position: absolute; top: 8px; left: 8px; z-index: 2;
    background: #667eea; color: white; font-size: 10px; font-weight: bold;
    padding: 2px 8px; border-radius: 10px;
}}
.product-image {{
    width: 100%; aspect-ratio: 1; display: flex; align-items: center;
    justify-content: center; margin-bottom: 8px; overflow: hidden;
}}
.product-image img {{
    max-width: 100%; max-height: 100%; object-fit: contain;
}}
.product-info {{ flex: 1; }}
.product-name {{
    font-size: 12px; color: #333; line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    overflow: hidden; margin-bottom: 6px;
}}
.product-price {{
    font-size: 16px; font-weight: bold; color: #bf0000; margin-bottom: 4px;
}}
.product-review {{ font-size: 11px; color: #c90; margin-bottom: 2px; }}
.product-review .stars {{ letter-spacing: -1px; }}
.product-shop {{ font-size: 11px; color: #999; }}
</style>
</head>
<body>
<div class="header">
    <div class="logo">楽天市場</div>
    <div class="search-bar">
        <input type="text" value="{keyword}" readonly>
        <button>検索</button>
    </div>
</div>
<div class="breadcrumb">楽天市場 &gt; 検索結果「{keyword}」</div>
<div class="results-header">検索結果（シミュレーション表示）</div>
<div class="grid">
    {cards_html}
</div>
</body>
</html>'''


def build_mobile_html(keyword: str, items: list, user_image: Image.Image, position: int = 3,
                      user_product_name: str = "", user_product_price: str = "") -> str:
    """スマホ版の楽天検索結果風HTMLを生成"""
    user_data_url = _user_image_to_data_url(user_image)
    display_name = user_product_name or f"【送料無料】{keyword} 人気 おすすめ"
    display_price = user_product_price or "3,980"

    cards_html = ""
    item_idx = 0
    total = len(items) + 1

    for i in range(min(total, 12)):
        if i == position - 1:
            cards_html += f'''
            <div class="product-card user-card">
                <div class="user-badge">あなたの商品</div>
                <div class="product-image">
                    <img src="{user_data_url}" alt="">
                </div>
                <div class="product-name">{display_name[:40]}</div>
                <div class="product-price">¥{display_price}</div>
            </div>'''
        else:
            if item_idx < len(items):
                item = items[item_idx]
                cards_html += f'''
            <div class="product-card">
                <div class="product-image">
                    <img src="{item['image_url']}" alt="">
                </div>
                <div class="product-name">{item['name'][:40]}</div>
                <div class="product-price">¥{_format_price(item['price'])}</div>
                <div class="product-shop">{item['shop_name']}</div>
            </div>'''
                item_idx += 1

    return f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=375, initial-scale=1">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Hiragino Sans', sans-serif; background: #f5f5f5; max-width: 375px; margin: 0 auto; }}

.header {{
    background: #bf0000; padding: 8px 10px; display: flex; align-items: center;
}}
.header .logo {{ color: white; font-weight: bold; font-size: 14px; margin-right: 8px; }}
.search-bar {{
    flex: 1; height: 30px; display: flex;
}}
.search-bar input {{
    flex: 1; border: none; padding: 0 8px; font-size: 13px; border-radius: 4px;
}}

.grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px;
    background: #eee;
}}

.product-card {{
    background: #fff; padding: 8px; position: relative;
}}
.product-card.user-card {{
    box-shadow: inset 0 0 0 2px #667eea;
}}
.user-badge {{
    position: absolute; top: 4px; left: 4px; z-index: 2;
    background: #667eea; color: white; font-size: 9px; font-weight: bold;
    padding: 1px 6px; border-radius: 8px;
}}
.product-image {{
    width: 100%; aspect-ratio: 1; display: flex; align-items: center;
    justify-content: center; margin-bottom: 6px; overflow: hidden;
}}
.product-image img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
.product-name {{
    font-size: 11px; color: #333; line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; margin-bottom: 4px;
}}
.product-price {{ font-size: 14px; font-weight: bold; color: #bf0000; margin-bottom: 2px; }}
.product-shop {{ font-size: 10px; color: #999; }}
</style>
</head>
<body>
<div class="header">
    <div class="logo">楽天</div>
    <div class="search-bar">
        <input type="text" value="{keyword}" readonly>
    </div>
</div>
<div class="grid">
    {cards_html}
</div>
</body>
</html>'''
