# app.py
# -*- coding: utf-8 -*-

import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="미샵 상세페이지 이미지 추출기",
    page_icon="🧩",
    layout="wide",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# -----------------------------
# 데이터 구조
# -----------------------------
@dataclass
class CutItem:
    idx: int
    pil: Image.Image
    excluded_auto: bool = False
    excluded_manual: bool = False
    reason: str = ""


# -----------------------------
# 유틸
# -----------------------------
def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120] if len(name) > 120 else name


def pil_to_bytes_jpg(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def resize_keep(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    if w == target_w:
        return img
    new_h = int(round(h * (target_w / float(w))))
    return img.resize((target_w, new_h), Image.LANCZOS)


def center_crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    """왜곡 없이 가운데 기준으로 비율 맞추기(잘라내기)."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    cur_aspect = w / float(h)
    if abs(cur_aspect - target_aspect) < 1e-6:
        return img

    if cur_aspect > target_aspect:
        # 가로가 더 넓음 -> 좌우를 잘라서 맞춤
        new_w = int(round(h * target_aspect))
        new_w = max(1, min(new_w, w))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        # 세로가 더 김 -> 상하를 잘라서 맞춤
        new_h = int(round(w / target_aspect))
        new_h = max(1, min(new_h, h))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def trim_white_margin(img: Image.Image, white_thr: int = 245, pad: int = 2) -> Image.Image:
    """
    흰 배경 여백 제거:
    - RGB에서 각 채널이 white_thr 이상이면 흰색으로 간주
    - 남는 부분 bbox로 crop
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img)
    # 흰색 판정: 모든 채널이 threshold 이상
    is_white = (arr[:, :, 0] >= white_thr) & (arr[:, :, 1] >= white_thr) & (arr[:, :, 2] >= white_thr)
    non_white = ~is_white

    if not np.any(non_white):
        return img  # 전부 흰색이면 그대로

    ys, xs = np.where(non_white)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)

    return img.crop((x0, y0, x1 + 1, y1 + 1))


def row_nonwhite_ratio(arr_rgb: np.ndarray, white_thr: int = 245) -> np.ndarray:
    """각 row에서 '흰색이 아닌 픽셀' 비율."""
    is_white = (arr_rgb[:, :, 0] >= white_thr) & (arr_rgb[:, :, 1] >= white_thr) & (arr_rgb[:, :, 2] >= white_thr)
    non_white = ~is_white
    return non_white.mean(axis=1).astype(np.float32)


def smooth_1d(x: np.ndarray, k: int = 21) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    k = k if k % 2 == 1 else k + 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="valid")


def find_separator_gaps(ratio: np.ndarray, gap_thr: float = 0.006, min_gap: int = 18) -> List[Tuple[int, int]]:
    """
    row nonwhite ratio가 아주 낮은(=거의 흰 여백) 구간을 separator로 봄.
    """
    low = ratio <= gap_thr
    gaps = []
    start = None
    for i, v in enumerate(low):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            end = i - 1
            if end - start + 1 >= min_gap:
                gaps.append((start, end))
            start = None
    if start is not None:
        end = len(low) - 1
        if end - start + 1 >= min_gap:
            gaps.append((start, end))
    return gaps


def segment_long_detail_image(img: Image.Image) -> List[Image.Image]:
    """
    긴 상세페이지 JPG(여러 컷 세로 배열)에서 컷 분리:
    1) row별 non-white 비율로 흰 여백 구간(Separator) 찾음
    2) separator 사이를 segment로 추출
    3) 각 segment는 여백 trim(상하좌우)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img)
    r = row_nonwhite_ratio(arr, white_thr=245)
    r = smooth_1d(r, k=31)

    gaps = find_separator_gaps(r, gap_thr=0.006, min_gap=20)

    # segment 범위 만들기
    h = arr.shape[0]
    cuts = []
    prev_end = -1
    for (g0, g1) in gaps:
        seg_top = prev_end + 1
        seg_bot = g0 - 1
        if seg_bot - seg_top + 1 >= 80:  # 너무 작은 조각 제외
            cuts.append((seg_top, seg_bot))
        prev_end = g1

    # 마지막 구간
    if prev_end < h - 1:
        seg_top = prev_end + 1
        seg_bot = h - 1
        if seg_bot - seg_top + 1 >= 80:
            cuts.append((seg_top, seg_bot))

    out: List[Image.Image] = []
    w = img.size[0]
    for (t, b) in cuts:
        seg = img.crop((0, t, w, b + 1))
        seg = trim_white_margin(seg, white_thr=245, pad=2)
        # 너무 얇은 것 제외(오작동 방지)
        if seg.size[1] < 120 or seg.size[0] < 200:
            continue
        out.append(seg)

    return out


def looks_like_text_card(img: Image.Image) -> Tuple[bool, str]:
    """
    텍스트/타이틀 컷 자동 제외용 휴리스틱.
    - 배경이 대부분 흰색/연회색인데
    - 어두운(검정) 픽셀이 '어느 정도' 있고
    - 색상 다양성이 낮고(거의 단색),
    - 세그먼트 높이가 너무 작거나(띠 형태) 글자만 있는 경우가 많음
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.uint8)

    # 흰색 비율
    white = (arr[:, :, 0] >= 245) & (arr[:, :, 1] >= 245) & (arr[:, :, 2] >= 245)
    white_ratio = float(white.mean())

    # 어두운 픽셀 비율(글자/로고)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    dark_ratio = float((gray < 80).mean())

    # 색상 다양성(표준편차)
    std = float(arr.reshape(-1, 3).std(axis=0).mean())

    # 매우 얇은 띠(상단/하단 안내문) 제거
    if h < 220 and white_ratio > 0.75 and dark_ratio > 0.002:
        return True, "텍스트 안내(얇은 띠)로 추정"

    # 타이틀/문구 카드
    if white_ratio > 0.70 and 0.002 < dark_ratio < 0.18 and std < 35:
        return True, "텍스트/타이틀 컷으로 추정"

    # 로고/아이콘만 크게 있는 경우(예: 인스타 로고)
    # 색 다양성은 높을 수 있으나 '실사 대비 형태'가 단순한 케이스가 있어
    # 여기서는 크기가 작거나(짧은 높이) 내용이 단순할 때만 걸러줌
    if (h < 500 and w < 900) and white_ratio > 0.40 and dark_ratio < 0.10:
        return True, "아이콘/로고성 이미지로 추정"

    return False, ""


def apply_crop_mode(img: Image.Image, mode: str) -> Image.Image:
    """
    모드:
    - 그대로: 여백 제거된 컷을 그대로 저장(사이즈 유지)
    - 인스타그램 피드 규격: 4:5 (1080x1350)
    - 정방형: 1:1 (1080x1080)
    - 숏폼규격 900*1600: 9:16 (900x1600)
    """
    # 항상 먼저 흰 여백 제거
    base = trim_white_margin(img, white_thr=245, pad=2)

    if mode == "이미지 그대로 자르기":
        return base

    if mode == "인스타그램 피드 규격(4:5)":
        target_aspect = 4 / 5
        out = center_crop_to_aspect(base, target_aspect)
        out = out.resize((1080, 1350), Image.LANCZOS)
        return out

    if mode == "정방형(1:1)":
        target_aspect = 1.0
        out = center_crop_to_aspect(base, target_aspect)
        out = out.resize((1080, 1080), Image.LANCZOS)
        return out

    if mode == "숏폼규격(900x1600)":
        target_aspect = 900 / 1600
        out = center_crop_to_aspect(base, target_aspect)
        out = out.resize((900, 1600), Image.LANCZOS)
        return out

    return base


# -----------------------------
# URL에서 "본문 상세이미지" 후보만 찾기 (Cafe24 대응)
# -----------------------------
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def is_image_url(u: str) -> bool:
    u_low = u.lower()
    return any(u_low.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"])


def absolutize(base_url: str, src: str) -> str:
    src = src.strip()
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if src.startswith("/"):
        m = re.match(r"^(https?://[^/]+)", base_url)
        return (m.group(1) if m else base_url.rstrip("/")) + src
    return base_url.rstrip("/") + "/" + src.lstrip("/")


def pick_body_image_urls_from_html(product_url: str, html: str) -> List[str]:
    """
    '본문 상세페이지 이미지에서만' 후보 추출:
    1) Cafe24에서 흔한 본문 영역 id/class 우선 탐색
    2) 그 내부 img src만 수집
    3) 그래도 없으면, 전체 img 중 '상세이미지로 보이는 것(긴 세로, 큰 사이즈)' 후보만
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) 본문 영역 후보
    selectors = [
        "#prdDetail", "#prdDetailContent", "#prdDetailCont",
        "#productDetail", "#product_detail", "#contents",
        ".prdDetail", ".prdDetailContent", ".productDetail",
        "#tabProductDetail", "#tabDetail",
        "div[id*='prdDetail']", "div[class*='prdDetail']",
    ]

    img_urls: List[str] = []
    for sel in selectors:
        node = soup.select_one(sel)
        if not node:
            continue
        for img in node.select("img"):
            src = img.get("src") or img.get("data-src") or img.get("ec-data-src")
            if not src:
                continue
            src = absolutize(product_url, src)
            if is_image_url(src):
                img_urls.append(src)

    # 중복 제거
    img_urls = list(dict.fromkeys(img_urls))

    # 2) fallback: 전체 img 중, "상세이미지로 의심"만
    if not img_urls:
        all_imgs = soup.select("img")
        tmp = []
        for img in all_imgs:
            src = img.get("src") or img.get("data-src") or img.get("ec-data-src")
            if not src:
                continue
            src = absolutize(product_url, src)
            if not is_image_url(src):
                continue

            # 흔한 아이콘/스프라이트 제외
            s_low = src.lower()
            if any(k in s_low for k in ["icon", "logo", "sprite", "common", "btn", "banner"]):
                continue

            tmp.append(src)
        img_urls = list(dict.fromkeys(tmp))

    return img_urls


def download_image(url: str, timeout: int = 20) -> Optional[Image.Image]:
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        return img.convert("RGB")
    except Exception:
        return None


def fetch_detail_images_from_product_url(product_url: str) -> List[Image.Image]:
    """
    URL 입력 시:
    - HTML에서 본문 상세이미지 후보 url들을 찾고
    - 실제로 다운로드 후 '긴 세로 상세이미지'에 해당하는 것만 선별
    """
    html = fetch_html(product_url)
    candidates = pick_body_image_urls_from_html(product_url, html)

    downloaded: List[Tuple[str, Image.Image]] = []
    for u in candidates:
        img = download_image(u)
        if img is None:
            continue
        downloaded.append((u, img))

    # 상세페이지용 긴 이미지 우선(세로가 매우 긴 것)
    # 기준: height > width*2 AND height > 2000
    long_imgs = []
    for _, im in downloaded:
        w, h = im.size
        if h > w * 2 and h > 2000:
            long_imgs.append(im)

    # 그래도 없으면, 큰 실사 이미지(본문 컷이 여러장 개별로 박혀있는 형태)도 허용
    if not long_imgs:
        big_imgs = []
        for _, im in downloaded:
            w, h = im.size
            if min(w, h) >= 700 and (h >= 900 or w >= 900):
                big_imgs.append(im)
        # 너무 많은 경우를 대비해 상위 30장까지만
        long_imgs = big_imgs[:30]

    return long_imgs


def guess_base_name_from_url(url: str) -> str:
    # product_no=12345 우선
    m = re.search(r"product_no=(\d+)", url)
    if m:
        return f"product_{m.group(1)}"
    # 마지막 경로
    base = re.sub(r"[?#].*$", "", url).rstrip("/").split("/")[-1]
    return safe_filename(base or "misharp_detail")


# -----------------------------
# 컷 생성 파이프라인
# -----------------------------
def build_cuts_from_long_image(
    long_img: Image.Image,
    auto_exclude_text: bool = True,
) -> List[CutItem]:
    segs = segment_long_detail_image(long_img)

    items: List[CutItem] = []
    for i, seg in enumerate(segs, start=1):
        ex = False
        reason = ""
        if auto_exclude_text:
            ex, reason = looks_like_text_card(seg)
        items.append(CutItem(idx=i, pil=seg, excluded_auto=ex, reason=reason))
    return items


def flatten_cuts_from_sources(
    source_images: List[Image.Image],
    auto_exclude_text: bool = True,
) -> List[CutItem]:
    """
    source_images가
    - 긴 상세페이지 1장일 수도 있고,
    - 본문이 여러 이미지로 쪼개져 있을 수도 있음
    처리:
    - '긴 이미지'는 segment
    - '단일 컷' 형태는 trim 후 item으로 추가(하지만 텍스트 자동 제외 적용)
    """
    all_items: List[CutItem] = []
    global_idx = 1

    for img in source_images:
        w, h = img.size
        if h > w * 2 and h > 2000:
            items = build_cuts_from_long_image(img, auto_exclude_text=auto_exclude_text)
            for it in items:
                it.idx = global_idx
                global_idx += 1
                all_items.append(it)
        else:
            seg = trim_white_margin(img, white_thr=245, pad=2)
            ex = False
            reason = ""
            if auto_exclude_text:
                ex, reason = looks_like_text_card(seg)
            all_items.append(CutItem(idx=global_idx, pil=seg, excluded_auto=ex, reason=reason))
            global_idx += 1

    # 너무 작은 찌꺼기 제거(최종 안전장치)
    cleaned = []
    for it in all_items:
        w, h = it.pil.size
        if w < 300 or h < 200:
            # 작은 텍스트/아이콘 조각 가능성이 높음
            it.excluded_auto = True
            it.reason = it.reason or "너무 작은 이미지(조각)로 제외"
        cleaned.append(it)
    return cleaned


def make_download_zip(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files:
            zf.writestr(fname, data)
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1, h2, h3 { letter-spacing: -0.2px; }
.small-note { font-size: 12px; color: #666; }
.card { border:1px solid #eee; border-radius:12px; padding:14px; background:#fff; }
.hr { height:1px; background:#eee; margin:14px 0; }
.footer-note { font-size: 11px; color:#777; line-height: 1.5; padding-top: 18px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧩 상세페이지 이미지 추출기")
st.caption("미샵 상품 URL 또는 상세페이지 JPG(긴 이미지)를 넣으면, 본문 상품컷만 자동 분리/크롭하여 다운로드합니다.")

tab1, tab2 = st.tabs(["업로드", "미리보기 · 제외 · 다운로드"])

with tab1:
    st.subheader("1) 입력 방식")
    colA, colB = st.columns([1.2, 1])

    with colA:
        mode_input = st.radio(
            "입력 선택",
            ["상품 URL", "상세페이지 JPG 업로드"],
            horizontal=True,
        )

        product_url = ""
        uploaded_files = None

        if mode_input == "상품 URL":
            product_url = st.text_input(
                "미샵 상품 URL",
                placeholder="https://misharp.co.kr/product/detail.html?product_no=XXXXX ...",
            )
            st.markdown(
                '<div class="small-note">※ URL 입력 시: <b>상품 상세 HTML에서 본문 상세이미지 후보만</b> 선별 → 다운로드 → 컷 분리합니다.</div>',
                unsafe_allow_html=True,
            )
        else:
            uploaded_files = st.file_uploader(
                "상세페이지 JPG(여러 장 가능)",
                type=["jpg", "jpeg"],
                accept_multiple_files=True,
                help="긴 상세페이지 이미지 1장을 넣어도 되고, 여러 장을 한 번에 넣어도 됩니다.",
            )

    with colB:
        st.subheader("2) 자르기 옵션")
        crop_mode = st.selectbox(
            "자르기 모드",
            [
                "이미지 그대로 자르기",
                "인스타그램 피드 규격(4:5)",
                "정방형(1:1)",
                "숏폼규격(900x1600)",
            ],
            index=0,
        )
        auto_exclude_text = st.checkbox("텍스트/타이틀 컷 자동 제외", value=True)
        st.markdown('<div class="small-note">※ 자동 제외는 오탐이 있을 수 있어, 다음 탭에서 수동 체크로 조정할 수 있어요.</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    run = st.button("✅ 본문 상품컷 추출하기", type="primary", use_container_width=True)

    if run:
        with st.spinner("이미지 수집/분석 중..."):
            base_name = "misharp_detail"
            source_images: List[Image.Image] = []

            if mode_input == "상품 URL":
                product_url = normalize_url(product_url)
                if not product_url:
                    st.error("상품 URL을 입력해 주세요.")
                    st.stop()

                base_name = guess_base_name_from_url(product_url)
                imgs = fetch_detail_images_from_product_url(product_url)

                if not imgs:
                    st.error("본문 상세이미지를 찾지 못했어요. (상품 상세 HTML 내 본문 이미지가 없거나 접근 제한일 수 있습니다)")
                    st.stop()

                source_images = imgs

            else:
                if not uploaded_files:
                    st.error("상세페이지 JPG를 1장 이상 업로드해 주세요.")
                    st.stop()

                # 파일명 기반 base_name
                first_name = uploaded_files[0].name
                base_name = safe_filename(re.sub(r"\.(jpg|jpeg)$", "", first_name, flags=re.I)) or "misharp_detail"

                for f in uploaded_files:
                    try:
                        im = Image.open(f).convert("RGB")
                        source_images.append(im)
                    except Exception:
                        continue

                if not source_images:
                    st.error("업로드한 파일을 이미지로 읽지 못했어요.")
                    st.stop()

            cuts = flatten_cuts_from_sources(source_images, auto_exclude_text=auto_exclude_text)

            if not cuts:
                st.error("추출 결과가 없습니다.")
                st.stop()

            # 세션 저장
            st.session_state["cuts_base_name"] = base_name
            st.session_state["cuts_crop_mode"] = crop_mode
            st.session_state["cuts_items"] = cuts

        st.success(f"추출 완료! (총 {len(cuts)}개 후보) → 다음 탭에서 미리보기/제외/다운로드를 진행하세요.")


with tab2:
    st.subheader("결과 미리보기 · 제외 · 다운로드")

    if "cuts_items" not in st.session_state:
        st.info("먼저 **업로드 탭**에서 ‘본문 상품컷 추출하기’를 실행해 주세요.")
        st.stop()

    base_name = st.session_state.get("cuts_base_name", "misharp_detail")
    crop_mode = st.session_state.get("cuts_crop_mode", "이미지 그대로 자르기")
    cuts: List[CutItem] = st.session_state.get("cuts_items", [])

    # 상단 요약
    total = len(cuts)
    auto_ex = sum(1 for c in cuts if c.excluded_auto)
    st.markdown(
        f"""
<div class="card">
<b>현재 상태</b><br/>
- 추출 후보: <b>{total}개</b><br/>
- 자동 제외(텍스트/아이콘 추정): <b>{auto_ex}개</b><br/>
- 자르기 모드: <b>{crop_mode}</b>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # 수동 제외 UI
    st.write("### 2) 제외할 컷 선택")
    st.caption("자동 제외가 오탐이면 체크를 해제해 주세요. 반대로 빼고 싶은 컷은 체크하면 됩니다.")

    # 그리드 미리보기
    cols = st.columns(4)
    manual_key_prefix = f"manual_ex_{base_name}_{crop_mode}"

    for i, item in enumerate(cuts):
        col = cols[i % 4]

        # 기본 체크 상태: 자동 제외는 체크 True(제외), 아니면 False
        default_exclude = bool(item.excluded_auto)
        key = f"{manual_key_prefix}_{item.idx}"

        if key not in st.session_state:
            st.session_state[key] = default_exclude

        # 미리보기는 너무 커지면 느리니, 축소 썸네일 표시
        thumb = item.pil.copy()
        thumb.thumbnail((360, 360))

        with col:
            st.image(thumb, caption=f"#{item.idx} ({item.pil.size[0]}x{item.pil.size[1]})", use_container_width=True)
            label = "이 컷 제외"
            if item.excluded_auto and item.reason:
                label += f" (자동: {item.reason})"
            st.session_state[key] = st.checkbox(label, value=st.session_state[key], key=key)

    # 제외 적용
    for item in cuts:
        key = f"{manual_key_prefix}_{item.idx}"
        item.excluded_manual = bool(st.session_state.get(key, False))

    final_items = [c for c in cuts if not c.excluded_manual]

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # 다운로드 생성
    st.write("### 3) 다운로드")
    st.caption("다운로드는 ‘최종 포함’된 컷만 생성합니다.")
    st.write(f"최종 포함: **{len(final_items)}개** / 제외: **{total - len(final_items)}개**")

    if len(final_items) == 0:
        st.warning("포함된 컷이 0개입니다. 제외 체크를 해제해 주세요.")
        st.stop()

    # 생성 버튼
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📦 ZIP 만들기(전체)", use_container_width=True):
            with st.spinner("ZIP 생성 중..."):
                files: List[Tuple[str, bytes]] = []
                for n, it in enumerate(final_items, start=1):
                    out = apply_crop_mode(it.pil, crop_mode)
                    fname = f"{safe_filename(base_name)}_{n:03d}.jpg"
                    files.append((fname, pil_to_bytes_jpg(out, quality=95)))

                zip_bytes = make_download_zip(files)
                st.session_state["dl_zip"] = zip_bytes
                st.session_state["dl_zip_name"] = f"{safe_filename(base_name)}_cuts.zip"

    with col2:
        # 대표 1장 JPG 다운로드(첫 번째 컷)
        out0 = apply_crop_mode(final_items[0].pil, crop_mode)
        st.download_button(
            "⬇️ 대표 1장 JPG 다운로드(첫 컷)",
            data=pil_to_bytes_jpg(out0, quality=95),
            file_name=f"{safe_filename(base_name)}_001.jpg",
            mime="image/jpeg",
            use_container_width=True,
            key=f"download_first_{base_name}_{crop_mode}",
        )

    # ZIP 다운로드 버튼(생성 후)
    if "dl_zip" in st.session_state and st.session_state.get("dl_zip"):
        st.download_button(
            "⬇️ ZIP 다운로드",
            data=st.session_state["dl_zip"],
            file_name=st.session_state.get("dl_zip_name", f"{safe_filename(base_name)}_cuts.zip"),
            mime="application/zip",
            use_container_width=True,
            key=f"download_zip_{base_name}_{crop_mode}",
        )

    st.markdown(
        """
<div class="footer-note">
<hr/>
<b>저작권 / 보안 안내</b><br/>
- (KR) 본 프로그램의 저작권은 <b>misharpcompany</b>에 있으며, 무단 복제·배포·사용을 금합니다.<br/>
- (KR) 본 프로그램은 <b>미샵컴퍼니 직원 전용</b>이며, 외부로 유출하거나 제3자에게 제공할 수 없습니다.<br/><br/>
- (EN) Copyright of this program belongs to <b>misharpcompany</b>. Unauthorized copying, distribution, or use is prohibited.<br/>
- (EN) This program is <b>for misharpcompany staff only</b> and must not be shared externally or provided to third parties.
</div>
""",
        unsafe_allow_html=True,
    )
