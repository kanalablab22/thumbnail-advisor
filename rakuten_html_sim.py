"""
楽天検索結果のHTMLを取得し、指定位置の商品画像をユーザー画像に差し替える
Selenium不使用 - requestsでHTML取得
"""

import requests
import re
import base64
import urllib.parse
from io import BytesIO
from PIL import Image


def fetch_rakuten_search_html(keyword: str, user_image: Image.Image, position: int = 5) -> str:
    """
    楽天のPC版検索結果HTMLを取得し、position番目の商品画像をuser_imageに差し替える
    Returns: 表示用HTML文字列
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    session.headers.update(headers)

    try:
        # 検索結果を直接取得（Cookie取得ステップは不要）
        encoded = urllib.parse.quote(keyword)
        url = f"https://search.rakuten.co.jp/search/mall/{encoded}/"
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text

        # ユーザー画像をbase64に変換
        buf = BytesIO()
        user_image.convert("RGB").save(buf, format="JPEG", quality=90)
        user_b64 = base64.b64encode(buf.getvalue()).decode()
        user_data_url = f"data:image/jpeg;base64,{user_b64}"

        # 楽天の商品サムネイル画像URLパターン
        img_patterns = [
            r'(https://tshop\.r10s\.jp/[^"\'>\s]+\.(?:jpg|jpeg|png|gif|webp))',
            r'(https://thumbnail\.image\.rakuten\.co\.jp/[^"\'>\s]+)',
            r'(https://shop\.r10s\.jp/[^"\'>\s]+\.(?:jpg|jpeg|png|gif|webp))',
        ]

        # すべてのパターンでマッチを収集
        all_matches = []
        for pattern in img_patterns:
            matches = list(re.finditer(pattern, html))
            all_matches.extend(matches)

        # ユニークな画像URLを追跡して、position番目を差し替え
        seen_urls = set()
        product_count = 0
        target_base = None

        for m in all_matches:
            img_url = m.group(1)
            # クエリパラメータを除いたベースURLで重複排除
            base_url = img_url.split("?")[0]
            if base_url in seen_urls:
                continue
            seen_urls.add(base_url)
            product_count += 1

            if product_count == position:
                # このURLのベース部分を取得（パスの最後のファイル名を除いたディレクトリ）
                target_base = base_url
                break

        # 対象商品の画像URLを差し替え
        if target_base:
            # 完全一致で差し替え（クエリパラメータ付きも含む）
            escaped = re.escape(target_base)
            html = re.sub(
                escaped + r'[^"\'>\s]*',
                user_data_url,
                html
            )

        # 相対パスを絶対パスに変換
        html = html.replace('href="/', 'href="https://search.rakuten.co.jp/')
        html = html.replace("href='/", "href='https://search.rakuten.co.jp/")

        # リンクのクリックを無効化
        html = html.replace('<a ', '<a onclick="return false;" ')

        # 不要な要素を非表示にするCSS注入
        style_inject = """
        <style>
            body { margin: 0; padding: 0; overflow-x: hidden; }
            * { max-width: 100% !important; }
            /* ナビゲーション・フッター等を非表示 */
            header, footer, nav,
            .header, .footer,
            #header, #footer,
            .dui-header, .dui-footer,
            .dui-container--sidebar,
            .dui-breadcrumb,
            [class*="notification"],
            [class*="banner-"],
            [class*="campaign"],
            .searchConditionDisp,
            #ratRanking,
            .dui-pagination { display: none !important; }
            /* メインコンテンツを広げる */
            .dui-container--main { max-width: 100% !important; }
        </style>
        """
        html = html.replace('</head>', style_inject + '</head>')

        return html

    except Exception as e:
        return f"<html><body><p>検索結果の取得に失敗しました: {e}</p></body></html>"


def fetch_rakuten_mobile_html(keyword: str, user_image: Image.Image, position: int = 5) -> str:
    """
    楽天のモバイル版検索結果HTMLを取得し、position番目の商品画像を差し替える
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        "Accept-Language": "ja-JP,ja;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    session.headers.update(headers)

    try:
        encoded = urllib.parse.quote(keyword)
        url = f"https://search.rakuten.co.jp/search/mall/{encoded}/"
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text

        # ユーザー画像をbase64に変換
        buf = BytesIO()
        user_image.convert("RGB").save(buf, format="JPEG", quality=90)
        user_b64 = base64.b64encode(buf.getvalue()).decode()
        user_data_url = f"data:image/jpeg;base64,{user_b64}"

        # 楽天の商品サムネイル画像URLパターン
        img_patterns = [
            r'(https://tshop\.r10s\.jp/[^"\'>\s]+\.(?:jpg|jpeg|png|gif|webp))',
            r'(https://thumbnail\.image\.rakuten\.co\.jp/[^"\'>\s]+)',
            r'(https://shop\.r10s\.jp/[^"\'>\s]+\.(?:jpg|jpeg|png|gif|webp))',
        ]

        all_matches = []
        for pattern in img_patterns:
            matches = list(re.finditer(pattern, html))
            all_matches.extend(matches)

        seen_urls = set()
        product_count = 0
        target_base = None

        for m in all_matches:
            img_url = m.group(1)
            base_url = img_url.split("?")[0]
            if base_url in seen_urls:
                continue
            seen_urls.add(base_url)
            product_count += 1

            if product_count == position:
                target_base = base_url
                break

        if target_base:
            escaped = re.escape(target_base)
            html = re.sub(
                escaped + r'[^"\'>\s]*',
                user_data_url,
                html
            )

        # URL修正
        html = html.replace('href="/', 'href="https://search.rakuten.co.jp/')
        html = html.replace("href='/", "href='https://search.rakuten.co.jp/")
        html = html.replace('<a ', '<a onclick="return false;" ')

        # モバイル用スタイル調整（2列グリッド強制）
        style_inject = """
        <style>
            body { margin: 0; padding: 0; overflow-x: hidden; font-size: 13px; }
            * { max-width: 100% !important; }
            /* ナビ・フッター非表示 */
            header, footer, nav,
            .header, .footer,
            #header, #footer,
            .dui-header, .dui-footer,
            [class*="notification"],
            [class*="banner-"],
            [class*="campaign"],
            .dui-breadcrumb,
            .dui-pagination { display: none !important; }

            /* 検索結果を2列グリッドに強制 */
            .searchresultitems,
            [class*="searchresultitem"] {
                display: inline-block !important;
                vertical-align: top !important;
            }
            .searchresultitems {
                display: flex !important;
                flex-wrap: wrap !important;
                font-size: 13px !important;
            }
            .searchresultitem,
            [class*="searchresultitem"]:not(.searchresultitems) {
                width: 48% !important;
                margin: 1% !important;
                box-sizing: border-box !important;
            }
            /* 商品画像を中央寄せ */
            .searchresultitem img,
            [class*="searchresultitem"] img {
                display: block !important;
                margin: 0 auto !important;
                max-height: 140px !important;
                width: auto !important;
                max-width: 100% !important;
            }
        </style>
        """
        html = html.replace('</head>', style_inject + '</head>')

        return html

    except Exception as e:
        return f"<html><body><p>取得失敗: {e}</p></body></html>"
