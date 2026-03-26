"""
🎨 楽天サムネイル アドバイザー
サムネイル画像をアップロード → 自動解析 → スコア＆改善アドバイス
"""

import streamlit as st
import json
import base64
import requests
import io
from PIL import Image
from io import BytesIO
from image_checker import check_image, ImageCheckReport, CRITERIA_INFO
from pdf_report import generate_pdf_report
from rakuten_search_sim import create_search_simulation, fetch_rakuten_thumbnails
from rakuten_html_sim import fetch_rakuten_search_html, fetch_rakuten_mobile_html
import streamlit.components.v1 as components
import os


# --- カスタムデータの永続化（GitHub API / 汎用） ---

def _github_headers():
    token = st.secrets.get("github", {}).get("token", "")
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def _github_repo():
    return st.secrets.get("github", {}).get("repo", "kanalablab22/thumbnail-advisor")


def _has_github_secrets() -> bool:
    try:
        return len(st.secrets.get("github", {}).get("token", "")) > 0
    except Exception:
        return False


def _load_data(filename: str):
    """GitHub（data branch）またはローカルからJSONを読み込む"""
    if _has_github_secrets():
        try:
            repo = _github_repo()
            url = f"https://api.github.com/repos/{repo}/contents/{filename}?ref=data"
            resp = requests.get(url, headers=_github_headers(), timeout=5)
            if resp.status_code == 200:
                content = base64.b64decode(resp.json()["content"]).decode("utf-8")
                return json.loads(content)
        except Exception:
            pass
    # ローカルフォールバック
    local_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_data(filename: str, data, commit_msg: str = "データ更新"):
    """GitHub（data branch）+ ローカルにJSONを保存"""
    local_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if not _has_github_secrets():
        return
    try:
        repo = _github_repo()
        url = f"https://api.github.com/repos/{repo}/contents/{filename}"
        sha = None
        resp = requests.get(url + "?ref=data", headers=_github_headers(), timeout=5)
        if resp.status_code == 200:
            sha = resp.json()["sha"]
        payload = {
            "message": commit_msg,
            "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii"),
            "branch": "data",
        }
        if sha:
            payload["sha"] = sha
        put_resp = requests.put(url, headers=_github_headers(), json=payload, timeout=5)
        if put_resp.status_code in (200, 201):
            st.toast("✅ 保存しました", icon="✅")
        else:
            st.toast(f"⚠️ GitHub保存失敗（{put_resp.status_code}）", icon="⚠️")
    except Exception:
        st.toast("⚠️ GitHub接続エラー。ローカルには保存済み", icon="⚠️")


# 便利関数
def load_custom_guidelines() -> list:
    return _load_data("custom_guidelines.json")


def save_custom_guidelines(guidelines: list):
    _save_data("custom_guidelines.json", guidelines, "ガイドライン更新")


def load_examples(kind: str) -> list:
    return _load_data(f"examples_{kind}.json")


def save_examples(kind: str, examples: list):
    label = "OK例集" if kind == "ok" else "NG例集"
    _save_data(f"examples_{kind}.json", examples, f"{label}更新")


def load_comments() -> dict:
    return _load_data("comments.json") or {}


def save_comments(comments: dict):
    _save_data("comments.json", comments, "コメント更新")


# ===== ページ設定 =====
st.set_page_config(
    page_title="楽天サムネイル アドバイザー",
    page_icon="🎨",
    layout="wide",
)

# ===== カスタムCSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700;900&display=swap');
    .stApp { font-family: 'Noto Sans JP', sans-serif; }
    .block-container { max-width: 1100px; padding-top: 1.5rem; }

    /* 削除ボタン（✕）をミニマルに */
    button[kind="secondary"]:has(p) { all: unset !important; }
    div[data-testid="stColumn"]:last-child button {
        background: none !important; border: none !important;
        box-shadow: none !important; padding: 0 !important;
        min-height: 0 !important; font-size: 0.7em !important;
        color: #aaa !important; cursor: pointer !important;
    }
    div[data-testid="stColumn"]:last-child button:hover { color: #e53935 !important; }

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


# ===== サイドバー =====
PRIORITY_CSS = {"high": "advice-high", "medium": "advice-medium", "low": "advice-low"}
PRIORITY_LABEL = {"high": "🔴 最優先", "medium": "🟡 改善推奨", "low": "🟢 Good"}

with st.sidebar:
    st.markdown("## 📋 楽天公式ガイドライン")
    st.caption("楽天市場 商品画像ガイドラインより")

    rakuten_guidelines = [
        "テキスト要素の占有率は画像の20%以内",
        "枠線の使用は禁止",
        "アニメーションGIFは禁止",
        "商品と関連性の低い画像やテキストの使用は禁止",
    ]

    for g in rakuten_guidelines:
        st.checkbox(g, value=False, key=g)

    st.divider()
    with st.expander("🏢 **社内ガイドライン**", expanded=False):
        default_guidelines = [
            ("テキストは3層構造に", "訴求ワード特大→商品名中→スペック小"),
            ("余白を20%以上確保する", "引き算のデザイン。情報を削ることで高級感を"),
            ("1枚1メッセージの原則", "最も伝えたいことを1つだけ特大で"),
            ("背景はオフホワイト〜ライトグレーで統一", "全商品で同じトーンを維持"),
            ("フォントの使い分け", "和文明朝体×欧文サンセリフ体"),
            ("カラーバリエーションの見せ方を工夫する", "カラードット、ミニ写真など"),
            ("ブランドロゴは控えめに配置", "画像の3-5%程度にとどめる"),
            ("写真は柔らかいディフューズ光で撮影", "直射光を避けて自然な陰影に"),
        ]

        custom_guidelines = load_custom_guidelines()

        for title, desc in default_guidelines:
            st.checkbox(f"**{title}**", value=False, key=f"internal_{title}")
            st.markdown(f"<p style='margin-top: -15px; margin-bottom: 8px; padding-left: 32px; font-size: 0.78em; color: #888;'>{desc}</p>", unsafe_allow_html=True)

        for i, g in enumerate(custom_guidelines):
            st.checkbox(f"**{g['title']}**", value=False, key=f"custom_{i}_{g['title']}")
            desc_text = g.get("desc", "")
            desc_part = f'<span style="color: #888;">{desc_text}</span>　' if desc_text else ""
            st.markdown(
                f"<p style='margin-top: -15px; margin-bottom: 8px; padding-left: 32px; font-size: 0.78em;'>"
                f"{desc_part}</p>",
                unsafe_allow_html=True,
            )
            if st.button("削除", key=f"del_{i}", type="secondary"):
                custom_guidelines.pop(i)
                save_custom_guidelines(custom_guidelines)
                st.rerun()

        st.markdown("---")
        if not _has_github_secrets():
            st.caption("⚠️ GitHub未接続（ローカル保存モード）")
        with st.form("add_guideline_form", clear_on_submit=True):
            new_title = st.text_input("チェック項目を追加", placeholder="例: 背景に余計なものを入れない")
            new_desc = st.text_input("補足説明（任意）", placeholder="例: 商品以外の小道具やテキストはNG")
            submitted = st.form_submit_button("➕ 追加", type="primary")
            if submitted and new_title.strip():
                custom_guidelines.append({
                    "title": new_title.strip(),
                    "desc": new_desc.strip() if new_desc.strip() else "",
                })
                save_custom_guidelines(custom_guidelines)
                st.rerun()

    # ブランド一覧
    st.markdown("---")
    DEFAULT_BRANDS = [
        "GRAV", "CAMP GREEB", "sopoa",
        "mura", "oeuf_soleil", "hugmotti", "shop_channel",
        "hugmin", "riceking", "pin_eagle", "ponbaby",
        "hacono", "kameto", "qp", "nocor",
        "turfmate", "forest_pellet", "chotplus", "baby_potage",
    ]
    custom_brands = _load_data("custom_brands.json")
    all_brands = DEFAULT_BRANDS + [b for b in custom_brands if b not in DEFAULT_BRANDS]

    st.markdown("### 🏷️ ブランド別 OK / NG例集")
    selected_brand = st.selectbox(
        "ブランドを選択",
        ["（選択してください）"] + all_brands,
        key="brand_filter",
    )

    with st.expander("ブランドを追加"):
        with st.form("add_brand_form", clear_on_submit=True):
            new_brand = st.text_input("ブランド名", placeholder="例: 新ブランド名")
            brand_submitted = st.form_submit_button("➕ 追加")
            if brand_submitted and new_brand.strip() and new_brand.strip() not in all_brands:
                custom_brands.append(new_brand.strip())
                _save_data("custom_brands.json", custom_brands, "ブランド追加")
                st.rerun()

    ok_examples = load_examples("ok")
    ng_examples = load_examples("ng")

    if selected_brand != "（選択してください）":
        filtered_ok = [ex for ex in ok_examples if ex.get("brand", "") == selected_brand]
        filtered_ng = [ex for ex in ng_examples if ex.get("brand", "") == selected_brand]

        st.markdown(f"#### ✅ {selected_brand} の OK例")
        if not filtered_ok:
            st.caption("まだ登録されていません")
        for i, ex in enumerate(filtered_ok):
            orig_idx = ok_examples.index(ex)
            col_txt, col_del = st.columns([8, 1])
            with col_txt:
                desc_part = f' <span style="font-size:0.78em;color:#888;">— {ex["desc"]}</span>' if ex.get("desc") else ""
                st.markdown(f"**・{ex['title']}**{desc_part}", unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_ok_{orig_idx}", help="削除"):
                    ok_examples.pop(orig_idx)
                    save_examples("ok", ok_examples)
                    st.rerun()

        st.markdown("---")

        st.markdown(f"#### ❌ {selected_brand} の NG例")
        if not filtered_ng:
            st.caption("まだ登録されていません")
        for i, ex in enumerate(filtered_ng):
            orig_idx = ng_examples.index(ex)
            col_txt, col_del = st.columns([8, 1])
            with col_txt:
                desc_part = f' <span style="font-size:0.78em;color:#888;">— {ex["desc"]}</span>' if ex.get("desc") else ""
                st.markdown(f"**・{ex['title']}**{desc_part}", unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_ng_{orig_idx}", help="削除"):
                    ng_examples.pop(orig_idx)
                    save_examples("ng", ng_examples)
                    st.rerun()
    else:
        st.caption("👆 ブランドを選ぶとOK例・NG例が表示されます")

    st.markdown("---")
    st.markdown("### ➕ 例を追加")
    with st.form("add_example_form", clear_on_submit=True):
        ex_brand = st.selectbox("ブランド", ["（選択してください）"] + all_brands, key="ex_brand",
                                index=(all_brands.index(selected_brand) + 1) if selected_brand in all_brands else 0)
        ex_type = st.radio("種類", ["✅ OK例", "❌ NG例"], horizontal=True)
        ex_title = st.text_input("内容", placeholder="例: 影が自然に入っていて立体的")
        ex_desc = st.text_input("補足（任意）", placeholder="例: 左上からの光で高級感がある", key="ex_desc")
        ex_submitted = st.form_submit_button("➕ 追加", type="primary")
        if ex_submitted and ex_title.strip() and ex_brand != "（選択してください）":
            entry = {"title": ex_title.strip(), "desc": ex_desc.strip() if ex_desc.strip() else "", "brand": ex_brand}
            if "OK" in ex_type:
                ok_examples = load_examples("ok")
                ok_examples.append(entry)
                save_examples("ok", ok_examples)
            else:
                ng_examples = load_examples("ng")
                ng_examples.append(entry)
                save_examples("ng", ng_examples)
            st.rerun()


# ===== メインエリア =====
st.markdown("# 🎨 楽天サムネイル アドバイザー")
st.markdown("作成したサムネイル画像をアップロードすると、楽天ガイドラインに基づいて自動チェックします")

with st.expander("📌 **楽天サムネイルの技術的要件**", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- **形式**: JPEG / PNG / GIF
- **テキスト占有率**: 20%以内
- **枠線**: 使用禁止
        """)
    with col2:
        st.markdown("""
- **アニメーションGIF**: 禁止
- **関連性**: 商品と無関係な画像・テキスト禁止
- **推奨サイズ**: 700×700px以上
        """)

# --- キーワード入力（必須） ---
st.markdown("### 🔍 狙っているキーワード")
target_keyword = st.text_input(
    "この商品が狙っている楽天検索キーワードを入力してください（必須）",
    placeholder="例: 財布 レディース 本革",
    key="target_keyword",
)

if not target_keyword.strip():
    st.warning("⬆️ まずキーワードを入力してください")
    st.stop()

# ファイルアップロード
uploaded_files = st.file_uploader(
    "サムネイル画像をドラッグ＆ドロップ（複数枚OK）",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 チェックしたい商品画像をアップロードしてください")
    st.stop()

# --- チェック実行 ---
all_reports = []
all_analyses = []
all_scores_list = []

for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    report, analysis, scores, total_score, grade, grade_msg = check_image(image, uploaded_file.name)
    all_reports.append(report)
    all_analyses.append(analysis)
    all_scores_list.append(scores)

# --- 複数枚サマリー ---
if len(all_reports) > 1:
    st.markdown("---")
    st.markdown("## 📊 一括チェック結果")

    for report in all_reports:
        ng_count = sum(1 for r in report.results if r.level == "ng")
        warn_count = sum(1 for r in report.results if r.level == "warn")
        status = "✅ 合格" if ng_count == 0 and warn_count == 0 else f"❌ NG {ng_count}件" if ng_count else f"⚠️ 注意 {warn_count}件"
        st.markdown(f"**`{report.filename}`** → **{report.score}点** {status}")
        result_text = " | ".join([
            f"{'✅' if r.level == 'ok' else ('⚠️' if r.level == 'warn' else '❌')}{r.name}"
            for r in report.results
        ])
        st.caption(result_text)

    st.markdown("---")

# --- 画像ごとの詳細結果 ---
for file_idx, (report, analysis, scores) in enumerate(zip(all_reports, all_analyses, all_scores_list)):
    if len(all_reports) > 1:
        st.markdown(f"## 📸 {report.filename}")
    else:
        st.markdown("## 📸 チェック結果")

    # 総合スコア計算
    total_score = report.score
    grade_map = {85: ("S", "お手本レベル！"), 70: ("A", "かなり良い！"), 55: ("B", "基本OK"), 40: ("C", "改善余地あり")}
    grade, grade_msg = "D", "大幅な改善チャンス！"
    for threshold, (g, m) in sorted(grade_map.items(), reverse=True):
        if total_score >= threshold:
            grade, grade_msg = g, m
            break

    col_img, col_result = st.columns([1, 1.6])

    with col_img:
        st.image(report.annotated_image, caption=f"{report.filename} ({report.width}x{report.height}px)", use_container_width=True)

    with col_result:
        # 総合スコア
        st.markdown(f"""
        <div class="score-card-main">
            <div style="font-size:0.85rem; opacity:0.8;">総合スコア</div>
            <div class="score-big">{total_score}<span class="score-max"> / 100</span></div>
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
                <span class="meter-score" style="color:{color}">{"★" * s}{"☆" * (5 - s)}</span>
            </div>
            """, unsafe_allow_html=True)

    # 改善アドバイス
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

    # --- コメント欄 ---
    comments = load_comments()
    if not isinstance(comments, dict):
        comments = {}
    file_key = report.filename
    file_comments = comments.get(file_key, [])

    if file_comments:
        st.markdown("**💬 コメント**")
        for ci, c in enumerate(file_comments):
            col_comment, col_del = st.columns([20, 1])
            with col_comment:
                st.markdown(f"<span style='font-size:0.9em;'>{c.get('text', '')}</span>", unsafe_allow_html=True)
            with col_del:
                if st.button("×", key=f"del_comment_{file_idx}_{ci}", help="削除", type="secondary"):
                    file_comments.pop(ci)
                    comments[file_key] = file_comments
                    save_comments(comments)
                    st.rerun()

    with st.expander("💬 コメントを追加", expanded=False):
        comment_text = st.text_area("コメント", key=f"comment_text_{file_idx}", placeholder="例: 影をもう少し強くしたほうが良さそう", height=80)
        if st.button("💾 保存", key=f"save_comment_{file_idx}"):
            if comment_text.strip():
                file_comments.append({"text": comment_text.strip()})
                comments[file_key] = file_comments
                save_comments(comments)
                st.success("コメントを保存しました！")
                st.rerun()
            else:
                st.warning("コメントを入力してください")

    # 解析データの詳細（折りたたみ）
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

    if file_idx < len(all_reports) - 1:
        st.markdown("---")


# --- 検索結果シミュレーション ---
st.markdown("---")
st.markdown("## 🔍 検索結果シミュレーション")
st.caption(f"「{target_keyword}」で楽天検索した場合のイメージ")

sim_pc = None
with st.spinner("楽天検索結果を取得中..."):
    try:
        uploaded_files[0].seek(0)
        user_img = Image.open(uploaded_files[0])

        tab_pc, tab_sp = st.tabs(["🖥️ PC版", "📱 スマホ版"])
        with tab_pc:
            # 楽天検索結果HTMLを丸ごと取得して商品画像を差し替え
            rakuten_html = fetch_rakuten_search_html(target_keyword, user_img, position=5)
            components.html(rakuten_html, height=800, scrolling=True)

            # PDF用にサムネグリッドも生成（非表示）
            competitors = fetch_rakuten_thumbnails(target_keyword, count=14)
            sim_pc = create_search_simulation(
                keyword=target_keyword,
                user_image=user_img,
                position=5,
                competitor_images=competitors,
            )
        with tab_sp:
            # モバイル版HTMLを取得
            mobile_html = fetch_rakuten_mobile_html(target_keyword, user_img, position=5)

            # iPhoneフレームのCSSをHTML自体に注入
            phone_frame_css = """
            <style>
                html { background: #1a1a1a !important; }
                body {
                    max-width: 284px !important;
                    margin: 28px auto 20px !important;
                    border-radius: 30px !important;
                    overflow-x: hidden !important;
                    background: #fff !important;
                    position: relative !important;
                }
            </style>
            """
            mobile_html = mobile_html.replace('</head>', phone_frame_css + '</head>')

            # iPhoneフレーム外枠 + 中にモバイルHTMLを直接表示
            phone_wrapper = f'''
            <div style="display:flex; justify-content:center;">
                <div style="
                    width: 300px; height: 640px;
                    border-radius: 45px; border: 8px solid #1a1a1a;
                    background: #1a1a1a; position: relative;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute; top: 0; left: 50%; transform: translateX(-50%);
                        width: 120px; height: 28px;
                        background: #1a1a1a; border-radius: 0 0 18px 18px; z-index: 10;
                    "></div>
                    <div style="
                        width: 100%; height: 100%;
                        border-radius: 37px; overflow-y: auto;
                        -webkit-overflow-scrolling: touch; background: #fff;
                    ">
                        {mobile_html}
                    </div>
                    <div style="
                        position:absolute; bottom:6px; left:50%; transform:translateX(-50%);
                        width:100px; height:4px; background:#666; border-radius:2px;
                    "></div>
                </div>
            </div>'''

            components.html(phone_wrapper, height=680)
    except Exception as e:
        st.warning(f"検索結果の取得に失敗しました: {e}")
        st.info("楽天側のアクセス制限の可能性があります。時間をおいて再度お試しください。")


# --- PDFレポート ---
st.markdown("---")
st.markdown("## 📄 PDFレポート")

with st.spinner("PDF生成中..."):
    original_images = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        original_images.append(Image.open(uploaded_file))

    all_comments = load_comments()
    if not isinstance(all_comments, dict):
        all_comments = {}
    pdf_bytes = generate_pdf_report(all_reports, original_images, comments=all_comments, sim_image=sim_pc)

st.download_button(
    label="📥 PDFレポートをダウンロード",
    data=pdf_bytes,
    file_name="rakuten_thumbnail_report.pdf",
    mime="application/pdf",
    type="primary",
)

# フッター
st.markdown("---")
st.caption("🎨 楽天サムネイル アドバイザー v2.0 — トップ店舗の分析ナレッジに基づく自動評価ツール")
