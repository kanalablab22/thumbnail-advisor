"""
🎨 楽天サムネイル アドバイザー
サムネイル画像をアップロード → 自動解析 → スコア＆改善アドバイス
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import io
import base64
from image_checker import check_image, ImageCheckReport, CRITERIA_INFO

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
        background: #F8F9FA; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
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


# ===== Selenium検索プレビュー（ローカル用、Cloud時はHTML版にフォールバック） =====
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time as _time
    import urllib.parse
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

# 楽天API版（Streamlit Cloud対応、ボットブロック回避）
from rakuten_api_sim import search_rakuten_items, build_pc_html, build_mobile_html
from genre_advisor import get_genre_advice, adjust_advice_for_genre


def capture_rakuten_search(keyword, mobile=False):
    """楽天で検索してスクショ＋サムネ座標を返す"""
    import os
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
            driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 393, "height": 852, "deviceScaleFactor": 1, "mobile": True,
            })
            driver.execute_cdp_cmd("Emulation.setUserAgentOverride", {
                "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            })

        driver.get(url)
        _time.sleep(4)

        if mobile:
            driver.execute_script("""
                document.querySelectorAll('button').forEach(function(btn) {
                    var aria = btn.getAttribute('aria-label') || '';
                    if (aria === 'close' || aria === '閉じる') {
                        var parent = btn.closest('[class]');
                        if (parent && parent.offsetHeight < 150) parent.remove();
                    }
                });
            """)
            _time.sleep(1)
            driver.execute_script("""
                var icon = document.querySelector("i.dui-icon.-gridview");
                if (icon) { icon.click(); }
            """)
            _time.sleep(3)
        else:
            driver.execute_script("""
                var sidebar = document.querySelector('.dui-container--sidebar');
                if (sidebar) sidebar.style.display = 'none';
                var main = document.querySelector('.dui-container--main');
                if (main) main.style.maxWidth = '100%';
            """)
            _time.sleep(0.5)

        min_size = 100 if mobile else 80
        if mobile:
            imgs = driver.find_elements(By.CSS_SELECTOR, "img")
        else:
            imgs = driver.find_elements(By.CSS_SELECTOR, ".searchresultitem:not([data-card-type='cpc']) img")
            if len(imgs) < 3:
                imgs = driver.find_elements(By.CSS_SELECTOR, "img")
        thumb_data = []
        for img in imgs:
            src = img.get_attribute("src") or ""
            if "tshop.r10s.jp" in src or "thumbnail.image.rakuten" in src:
                rect = img.rect
                if rect["width"] > min_size and rect["height"] > min_size:
                    thumb_data.append({
                        "x": int(rect["x"]), "y": int(rect["y"]),
                        "w": int(rect["width"]), "h": int(rect["height"]),
                    })

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

        screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_screenshot.png")
        driver.save_screenshot(screenshot_path)
        screenshot = Image.open(screenshot_path).convert("RGB")
    finally:
        driver.quit()
    return screenshot, thumb_data


def composite_on_screenshot(screenshot, thumb_data, user_img, slot_index=2):
    """スクショのサムネ位置にユーザー画像を合成"""
    result = screenshot.copy()
    if slot_index >= len(thumb_data):
        slot_index = min(0, len(thumb_data) - 1)
    if slot_index < 0 or not thumb_data:
        return result
    slot = thumb_data[slot_index]
    x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
    src_w, src_h = user_img.size
    scale = min(w / src_w, h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    user_thumb = user_img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    bg.paste(user_thumb, ((w - new_w) // 2, (h - new_h) // 2))
    result.paste(bg, (x, y))
    return result


def render_phone_mockup(pil_img):
    """iPhoneフレーム内スクロール表示HTML"""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{display:flex;justify-content:center;background:transparent;padding:10px 0}}
.phone{{width:280px;height:580px;border:6px solid #1a1a1a;border-radius:36px;overflow:hidden;position:relative;background:#fff;box-shadow:0 8px 40px rgba(0,0,0,0.15),inset 0 0 0 2px #333}}
.phone::before{{content:"";position:absolute;top:0;left:50%;transform:translateX(-50%);width:90px;height:22px;background:#1a1a1a;border-radius:0 0 16px 16px;z-index:10}}
.phone-screen{{width:100%;height:100%;overflow-y:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}}
.phone-screen::-webkit-scrollbar{{display:none}}
.phone-screen img{{width:100%;display:block}}
.scroll-hint{{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.5);color:#fff;font-size:11px;padding:4px 12px;border-radius:12px;pointer-events:none;animation:fadeout 3s forwards;font-family:sans-serif}}
@keyframes fadeout{{0%,60%{{opacity:1}}100%{{opacity:0}}}}
</style></head><body>
<div class="phone"><div class="phone-screen"><img src="data:image/jpeg;base64,{b64}"/></div>
<div class="scroll-hint">↕ スクロールできます</div></div></body></html>'''


# ===== ヘルパー関数 =====
PRIORITY_CSS = {"high": "advice-high", "medium": "advice-medium", "low": "advice-low"}
PRIORITY_LABEL = {"high": "🔴 最優先", "medium": "🟡 改善推奨", "low": "🟢 Good"}
REF_SCORES = {
    "whitespace": 5, "text_amount": 5, "background": 5,
    "color_tone": 5, "photo_quality": 5, "composition": 5, "color_variation": 4,
}


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


# ===== メインUI =====
st.markdown("""
<div class="app-header">
    <div class="app-title">🎨 楽天サムネイル アドバイザー</div>
    <div class="app-desc">画像をアップロードするだけ → 自動解析 → スコア＆改善アドバイス</div>
</div>
""", unsafe_allow_html=True)

# アップロード＆キーワード入力
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
    )

if not uploaded_files:
    st.info("👆 上のエリアにサムネイル画像をドロップしてください（複数OK）")
    st.stop()

for file_idx, uploaded_file in enumerate(uploaded_files):
    pil_img = Image.open(uploaded_file)

    if len(uploaded_files) > 1:
        st.markdown(f"---")
        st.markdown(f"### 📊 {uploaded_file.name}")

    col_img, col_result = st.columns([1, 1.6])

    with col_img:
        st.image(pil_img, caption=uploaded_file.name, use_container_width=True)
        w_px, h_px = pil_img.size
        st.caption(f"📐 {w_px} x {h_px} px")

    with col_result:
        with st.spinner("解析中..."):
            try:
                report = check_image(pil_img, uploaded_file.name)
                total = report.score
                grade, grade_msg = get_grade(total)
            except Exception as e:
                st.error(f"解析エラー: {type(e).__name__}: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

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
        for result in report.results:
            cid = result.name
            info = CRITERIA_INFO.get(cid, {})
            s = result.score_value
            ref = REF_SCORES.get(cid, 5)
            pct = s / 5 * 100

            if s >= ref:
                color = "#38A169"
            elif s >= ref - 1:
                color = "#ECC94B"
            else:
                color = "#E53E3E"

            display_name = info.get("display_name", cid)
            icon = info.get("icon", "📊")

            st.markdown(f"""
            <div class="meter-row">
                <span class="meter-label">{icon} {display_name}</span>
                <div class="meter-track">
                    <div class="meter-fill" style="width:{pct}%; background:{color};"></div>
                </div>
                <span class="meter-score" style="color:{color}">{"★" * s}{"☆" * (5-s)}</span>
            </div>
            """, unsafe_allow_html=True)

    # 改善アドバイス（フル幅で表示）
    # ジャンル・素材に応じてアドバイスを差し替え
    adjusted_results = adjust_advice_for_genre(report.results, search_keyword) if search_keyword else report.results
    improvements = []
    good_points = []
    for result in adjusted_results:
        item = {
            "icon": CRITERIA_INFO.get(result.name, {}).get("icon", "📊"),
            "name": CRITERIA_INFO.get(result.name, {}).get("display_name", result.name),
            "t": result.value,
            "d": result.detail,
            "p": result.level,
            "score": result.score_value,
        }
        if result.level in ("high", "medium"):
            improvements.append(item)
        else:
            good_points.append(item)

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

    # 楽天検索結果プレビュー
    if search_keyword:
        st.markdown("---")
        st.markdown("#### 🔍 実際の検索結果でのイメージ")
        tab_pc, tab_sp = st.tabs(["🖥️ PC版", "📱 スマホ版"])

        if HAS_SELENIUM:
            # ローカル環境: Seleniumでスクリーンショット
            with tab_pc:
                with st.spinner(f"PC版「{search_keyword}」を検索中..."):
                    try:
                        screenshot, thumb_data = capture_rakuten_search(search_keyword, mobile=False)
                        if thumb_data:
                            preview = composite_on_screenshot(screenshot, thumb_data, pil_img, slot_index=2)
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
                            preview_sp = composite_on_screenshot(screenshot_sp, thumb_data_sp, pil_img, slot_index=2)
                            phone_html = render_phone_mockup(preview_sp)
                            components.html(phone_html, height=620, scrolling=False)
                        else:
                            st.warning("サムネイルの検出に失敗しました。キーワードを変えて試してみてください。")
                    except Exception as e:
                        st.error(f"検索結果の取得に失敗しました: {e}")
        else:
            # Streamlit Cloud: 楽天API版（実際の商品データを表示）
            rakuten_app_id = st.secrets.get("RAKUTEN_APP_ID", "")
            rakuten_access_key = st.secrets.get("RAKUTEN_ACCESS_KEY", "")
            if not rakuten_app_id:
                st.warning("検索結果プレビューを表示するには楽天APIキーの設定が必要です。")
            else:
                with st.spinner(f"「{search_keyword}」を検索中..."):
                    try:
                        items = search_rakuten_items(search_keyword, rakuten_app_id, hits=12, access_key=rakuten_access_key)
                    except Exception as e:
                        items = []
                        st.error(f"楽天API取得エラー: {e}")

                if items:
                    with tab_pc:
                        pc_html = build_pc_html(search_keyword, items, pil_img, position=8)
                        components.html(pc_html, height=800, scrolling=True)

                    with tab_sp:
                        sp_html = build_mobile_html(search_keyword, items, pil_img, position=5)
                        components.html(sp_html, height=900, scrolling=True)

    # ===== ジャンル別アドバイス =====
    if search_keyword:
        genre_info = get_genre_advice(search_keyword)
        st.markdown("---")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f4f8 0%, #f0e6f6 100%);
                    border-radius: 16px; padding: 20px; margin: 12px 0;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #2C3E50; margin-bottom: 4px;">
                {genre_info['icon']} 「{genre_info['genre']}」ジャンルのサムネイル戦略
            </div>
            <div style="font-size: 0.85rem; color: #7f8c8d; margin-bottom: 16px;">
                参考店舗: {genre_info['reference_shops']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        for i, tip in enumerate(genre_info["tips"]):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: white; border-radius: 12px; padding: 16px; margin-bottom: 12px;
                            border-left: 4px solid {'#667eea' if i % 2 == 0 else '#764ba2'};
                            box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                    <div style="font-weight: 700; color: #2C3E50; margin-bottom: 6px; font-size: 0.95rem;">
                        💡 {tip['title']}
                    </div>
                    <div style="color: #555; font-size: 0.85rem; line-height: 1.6;">
                        {tip['detail']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 解析詳細（折りたたみ）
    with st.expander("🔬 解析データの詳細を見る"):
        d1, d2, d3 = st.columns(3)
        analysis = report.analysis_data
        with d1:
            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-title">⬜ 余白率</div>
                <div class="detail-value">全体: {analysis.get('whitespace_ratio', 0):.1f}%<br>
                総合: {analysis.get('whitespace_effective', 0):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-title">🔤 テキスト量推定</div>
                <div class="detail-value">エッジ密度: {analysis.get('edge_density', 0):.1f}%<br>
                推定テキスト面積: {analysis.get('text_area', 0):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-title">🖼️ 背景</div>
                <div class="detail-value">シンプルブロック: {analysis.get('simple_blocks', 0)}/25<br>
                枠線検出: {'あり ⚠️' if analysis.get('has_border', False) else 'なし ✅'}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-title">🎨 色使い</div>
                <div class="detail-value">平均彩度: {analysis.get('avg_saturation', 0):.1f}<br>
                高彩度面積: {analysis.get('high_sat_ratio', 0):.1f}%<br>
                使用色数: 約{analysis.get('n_colors', 0)}色</div>
            </div>
            """, unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-title">📸 写真品質</div>
                <div class="detail-value">コントラスト: {analysis.get('contrast', 0):.1f}<br>
                明るさ: {analysis.get('brightness', 0):.1f}<br>
                シャープネス: {analysis.get('sharpness', 0):.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="detail-card">
                <div class="detail-title">📐 構図</div>
                <div class="detail-value">コンテンツ率: {analysis.get('content_ratio', 0):.1f}%<br>
                中央集中度: {analysis.get('center_focus', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

# フッター
st.markdown("---")
st.caption("🎨 楽天サムネイル アドバイザー — トップ店舗の分析ナレッジに基づく自動評価ツール")
