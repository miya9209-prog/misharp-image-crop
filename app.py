# app.py
# -*- coding: utf-8 -*-

import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# ✅ OpenCV (사각 상품컷 오브젝트 감지용)
import cv2

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="MISHARP 이미지 추출생성기",
    page_icon="🧩",
    layout="wide",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

WHITE_THR = 245  # 흰색 판정 임계값
EDGE_WHITE_STRIP_MIN_PX = 5  # ✅ 5px 이상 흰줄은 여백으로 처리


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
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120] if len(name) > 120 else name


def pil_to_bytes_jpg(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def make_download_zip(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files:
            zf.writestr(fname, data)
    return buf.getvalue()


def center_crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    """왜곡 없이 가운데 기준으로 비율 맞추기(잘라내기)."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    cur_aspect = w / float(h)
    if abs(cur_aspect - target_aspect) < 1e-6:
        return img

    if cur_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        new_w = max(1, min(new_w, w))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))

    new_h = int(round(w / target_aspect))
    new_h = max(1, min(new_h, h))
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


# -----------------------------
# ✅ 5px 이상 흰줄 제거 (상/하/좌/우)
# -----------------------------
def _is_row_white(arr_rgb: np.ndarray, y: int, thr: int = WHITE_THR, ratio: float = 0.995) -> bool:
    row = arr_rgb[y, :, :]
    white = (row[:, 0] >= thr) & (row[:, 1] >= thr) & (row[:, 2] >= thr)
    return float(white.mean()) >= ratio


def _is_col_white(arr_rgb: np.ndarray, x: int, thr: int = WHITE_THR, ratio: float = 0.995) -> bool:
    col = arr_rgb[:, x, :]
    white = (col[:, 0] >= thr) & (col[:, 1] >= thr) & (col[:, 2] >= thr)
    return float(white.mean()) >= ratio


def trim_edge_white_strips(img: Image.Image, thr: int = WHITE_THR, min_strip: int = EDGE_WHITE_STRIP_MIN_PX) -> Image.Image:
    """가장자리(상/하/좌/우)에서 '연속된 흰줄'이 min_strip px 이상이면 그 구간을 잘라냄."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    top = 0
    while top < h and _is_row_white(arr, top, thr=thr):
        top += 1
    if top < min_strip:
        top = 0

    bottom = h - 1
    while bottom >= 0 and _is_row_white(arr, bottom, thr=thr):
        bottom -= 1
    # bottom은 "마지막 비-흰줄" 인덱스
    bottom_cut = h - 1 - bottom
    if bottom_cut < min_strip:
        bottom = h - 1

    left = 0
    while left < w and _is_col_white(arr, left, thr=thr):
        left += 1
    if left < min_strip:
        left = 0

    right = w - 1
    while right >= 0 and _is_col_white(arr, right, thr=thr):
        right -= 1
    right_cut = w - 1 - right
    if right_cut < min_strip:
        right = w - 1

    # 유효 범위 체크
    if right - left < 10 or bottom - top < 10:
        return img

    return img.crop((left, top, right + 1, bottom + 1))


def trim_white_margin_tight(img: Image.Image, thr: int = WHITE_THR) -> Image.Image:
    """흰 배경 여백 제거(타이트). pad=0 느낌으로 최대한 여백 없이."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img)
    is_white = (arr[:, :, 0] >= thr) & (arr[:, :, 1] >= thr) & (arr[:, :, 2] >= thr)
    non_white = ~is_white

    if not np.any(non_white):
        return img

    ys, xs = np.where(non_white)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    out = img.crop((x0, y0, x1 + 1, y1 + 1))
    # ✅ 가장자리 흰줄(5px 이상) 추가 제거
    out = trim_edge_white_strips(out, thr=thr, min_strip=EDGE_WHITE_STRIP_MIN_PX)
    return out


# -----------------------------
# ✅ 사각 "상품컷 오브젝트" 감지 → 각각 잘라내기
# -----------------------------
def detect_rect_photo_boxes(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    흰 배경 기반 상세페이지에서 '사진(사각 오브젝트)'로 보이는 영역을 검출.
    반환: (x0,y0,x1,y1) 리스트
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img)
    h, w = arr.shape[:2]

    # 흰색 배경 vs 비-흰색 분리
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # thr보다 밝으면 배경(흰색)로 보고, 어두운 쪽을 전경으로
    # -> 전경(상품컷, 모델컷, 텍스트 포함)을 1로 만들기
    _, bin_inv = cv2.threshold(gray, WHITE_THR, 255, cv2.THRESH_BINARY_INV)

    # 작은 점 노이즈 제거 / 사각 내부 채움
    kernel = np.ones((5, 5), np.uint8)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    min_area = max(200 * 200, int((w * h) * 0.01))  # 너무 작은 건 제외
    max_area = int((w * h) * 0.98)                  # 화면 거의 전체는 제외(롱 이미지 전체 등)

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < min_area or area > max_area:
            continue

        # 너무 얇은 띠/선형 제거
        if ww < 220 or hh < 220:
            continue

        # 화면 가장자리 테두리만 잡히는 경우 완화
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)

        boxes.append((x0, y0, x1, y1))

    # ✅ 정렬: 위→아래, 좌→우
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # ✅ 겹침 제거(큰 박스가 작은 박스를 집어삼키는 케이스)
    filtered: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        x0, y0, x1, y1 = b
        keep = True
        for bb in filtered:
            xx0, yy0, xx1, yy1 = bb
            # 포함 관계(거의 완전 포함)이면 작은 것을 우선 살리고 큰 것을 버림
            if x0 >= xx0 and y0 >= yy0 and x1 <= xx1 and y1 <= yy1:
                keep = False
                break
        if keep:
            filtered.append(b)

    return filtered


def split_into_photo_objects(img: Image.Image) -> List[Image.Image]:
    """
    입력 이미지(롱 세그먼트 포함)에서 '사각 상품컷 오브젝트'를 모두 찾아
    각각 여백 없이 잘라 반환.
    - 감지가 안 되면: 원본(여백 제거) 1장 반환
    """
    base = trim_edge_white_strips(img, thr=WHITE_THR, min_strip=EDGE_WHITE_STRIP_MIN_PX)

    boxes = detect_rect_photo_boxes(base)

    # 감지된 사각 오브젝트가 없으면 기존 방식(타이트 트림)으로 1장
    if not boxes:
        only = trim_white_margin_tight(base, thr=WHITE_THR)
        return [only]

    out: List[Image.Image] = []
    for (x0, y0, x1, y1) in boxes:
        crop = base.crop((x0, y0, x1, y1))
        crop = trim_white_margin_tight(crop, thr=WHITE_THR)
        # 너무 작은 조각은 제외
        if crop.size[0] < 300 or crop.size[1] < 240:
            continue
        out.append(crop)

    # 그래도 너무 적게 나오면(오검출) fallback
    if len(out) == 0:
        only = trim_white_margin_tight(base, thr=WHITE_THR)
        return [only]

    return out


# -----------------------------
# 긴 상세페이지(롱 이미지) 대략 분할(가로줄 여백 기준) + 각 구간에서 오브젝트 추출
# -----------------------------
def row_nonwhite_ratio(arr_rgb: np.ndarray, white_thr: int = WHITE_THR) -> np.ndarray:
    is_white = (arr_rgb[:, :, 0] >= white_thr) & (arr_rgb[:, :, 1] >= white_thr) & (arr_rgb[:, :, 2] >= white_thr)
    non_white = ~is_white
    return non_white.mean(axis=1).astype(np.float32)


def smooth_1d(x: np.ndarray, k: int = 31) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    k = k if k % 2 == 1 else k + 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, kernel, mode="valid")


def find_separator_gaps(ratio: np.ndarray, gap_thr: float = 0.006, min_gap: int = 20) -> List[Tuple[int, int]]:
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
    1) 롱 이미지를 흰 여백 구간 기준으로 큰 덩어리로 분리
    2) 각 덩어리에서 '사각 상품컷 오브젝트'를 모두 찾아 각각 저장
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 먼저 가장자리 흰줄 제거
    img = trim_edge_white_strips(img, thr=WHITE_THR, min_strip=EDGE_WHITE_STRIP_MIN_PX)

    arr = np.array(img)
    r = row_nonwhite_ratio(arr, white_thr=WHITE_THR)
    r = smooth_1d(r, k=31)

    gaps = find_separator_gaps(r, gap_thr=0.006, min_gap=20)

    h = arr.shape[0]
    cuts = []
    prev_end = -1
    for (g0, g1) in gaps:
        seg_top = prev_end + 1
        seg_bot = g0 - 1
        if seg_bot - seg_top + 1 >= 120:
            cuts.append((seg_top, seg_bot))
        prev_end = g1

    if prev_end < h - 1:
        seg_top = prev_end + 1
        seg_bot = h - 1
        if seg_bot - seg_top + 1 >= 120:
            cuts.append((seg_top, seg_bot))

    out: List[Image.Image] = []
    w = img.size[0]
    for (t, b) in cuts:
        seg = img.crop((0, t, w, b + 1))
        seg = trim_edge_white_strips(seg, thr=WHITE_THR, min_strip=EDGE_WHITE_STRIP_MIN_PX)

        # ✅ 핵심: 세그먼트 안에서 '사각 상품컷'을 각각 추출
        objs = split_into_photo_objects(seg)
        for o in objs:
            if o.size[1] < 220 or o.size[0] < 300:
                continue
            out.append(o)

    return out


# -----------------------------
# 텍스트/로고 자동 제외(기존 로직 유지, 입력은 더 타이트해진 컷 기준)
# -----------------------------
def looks_like_text_card(img: Image.Image) -> Tuple[bool, str]:
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    arr = np.array(img).astype(np.uint8)

    white = (arr[:, :, 0] >= WHITE_THR) & (arr[:, :, 1] >= WHITE_THR) & (arr[:, :, 2] >= WHITE_THR)
    white_ratio = float(white.mean())

    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    dark_ratio = float((gray < 80).mean())

    std = float(arr.reshape(-1, 3).std(axis=0).mean())

    if h < 220 and white_ratio > 0.75 and dark_ratio > 0.002:
        return True, "텍스트 안내(얇은 띠)로 추정"

    if white_ratio > 0.70 and 0.002 < dark_ratio < 0.18 and std < 35:
        return True, "텍스트/타이틀 컷으로 추정"

    if (h < 500 and w < 900) and white_ratio > 0.40 and dark_ratio < 0.10:
        return True, "아이콘/로고성 이미지로 추정"

    return False, ""


def apply_crop_mode(img: Image.Image, mode: str) -> Image.Image:
    """4가지 자르기 모드 적용."""
    base = trim_white_margin_tight(img, thr=WHITE_THR)

    if mode == "이미지 그대로 자르기":
        return base

    if mode == "인스타그램 피드 규격(4:5)":
        out = center_crop_to_aspect(base, 4 / 5)
        return out.resize((1080, 1350), Image.LANCZOS)

    if mode == "정방형(1:1)":
        out = center_crop_to_aspect(base, 1.0)
        return out.resize((1080, 1080), Image.LANCZOS)

    if mode == "숏폼규격(900x1600)":
        out = center_crop_to_aspect(base, 900 / 1600)
        return out.resize((900, 1600), Image.LANCZOS)

    return base


# -----------------------------
# URL에서 "본문 상세이미지" 후보만 찾기
# -----------------------------
def normalize_url(url: str) -> str:
    url = (url or "").strip()
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
    src = (src or "").strip()
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if src.startswith("/"):
        m = re.match(r"^(https?://[^/]+)", base_url)
        return (m.group(1) if m else base_url.rstrip("/")) + src
    return base_url.rstrip("/") + "/" + src.lstrip("/")


def pick_body_image_urls_from_html(product_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")

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

    img_urls = list(dict.fromkeys(img_urls))

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
    html = fetch_html(product_url)
    candidates = pick_body_image_urls_from_html(product_url, html)

    downloaded: List[Image.Image] = []
    for u in candidates:
        im = download_image(u)
        if im is not None:
            downloaded.append(im)

    # 긴 상세이미지 우선
    long_imgs = []
    for im in downloaded:
        w, h = im.size
        if h > w * 2 and h > 2000:
            long_imgs.append(im)

    if not long_imgs:
        big_imgs = []
        for im in downloaded:
            w, h = im.size
            if min(w, h) >= 700 and (h >= 900 or w >= 900):
                big_imgs.append(im)
        long_imgs = big_imgs[:30]

    return long_imgs


def guess_base_name_from_url(url: str) -> str:
    m = re.search(r"product_no=(\d+)", url)
    if m:
        return f"product_{m.group(1)}"
    base = re.sub(r"[?#].*$", "", url).rstrip("/").split("/")[-1]
    return safe_filename(base or "misharp_detail")


# -----------------------------
# 컷 생성 파이프라인
# -----------------------------
def build_items_from_sources(source_images: List[Image.Image], auto_exclude_text: bool = True) -> List[CutItem]:
    all_items: List[CutItem] = []
    global_idx = 1

    for img in source_images:
        w, h = img.size

        # ✅ 롱 이미지면: (1) 세그먼트 → (2) 세그먼트 안의 사각 오브젝트 각각 추출
        if h > w * 2 and h > 2000:
            extracted = segment_long_detail_image(img)
        else:
            # ✅ 롱이 아니어도: 이미지 안에 사각 오브젝트가 여러 개면 각각 분리
            extracted = split_into_photo_objects(img)

        for cut in extracted:
            cut = trim_white_margin_tight(cut, thr=WHITE_THR)

            ex = False
            reason = ""
            if auto_exclude_text:
                ex, reason = looks_like_text_card(cut)

            all_items.append(CutItem(idx=global_idx, pil=cut, excluded_auto=ex, reason=reason))
            global_idx += 1

    # 안전장치: 너무 작은 조각 자동 제외
    for it in all_items:
        ww, hh = it.pil.size
        if ww < 300 or hh < 240:
            it.excluded_auto = True
            it.reason = it.reason or "너무 작은 이미지(조각)로 제외"

    return all_items


# -----------------------------
# UI 스타일 (심플/세련/직관 + 제목 잘림 방지)
# -----------------------------
st.markdown(
    """
<style>
/* 전체 여백(상단 제목 잘림 방지) */
.block-container { padding-top: 2.2rem; padding-bottom: 2.6rem; max-width: 1280px; }

/* 타이포 */
h1 { margin-top: 0 !important; letter-spacing: -0.3px; }
h2 { margin-top: 1.4rem; letter-spacing: -0.2px; }
p, label, div, span { letter-spacing: -0.1px; }

/* 카드/구분선 */
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 16px; background: rgba(255,255,255,0.02); }
.hr { height: 1px; background: rgba(255,255,255,0.10); margin: 18px 0; }

.small-note { font-size: 12px; opacity: 0.72; line-height: 1.6; }
.footer-note { font-size: 11px; opacity: 0.72; line-height: 1.65; padding-top: 18px; }

/* 버튼/입력 간격 */
.stButton button { padding: 0.8rem 1rem; border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧩 MISHARP 이미지 추출생성기")
st.caption("MISHARP IMAGE GENERATOR V1")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# 1) 입력
# -----------------------------
st.subheader("1) 입력")
colA, colB = st.columns([1.25, 1])

with colA:
    input_type = st.radio("입력 선택", ["상품 URL", "상세페이지 JPG 업로드"], horizontal=True)

    product_url = ""
    uploaded_files = None

    if input_type == "상품 URL":
        product_url = st.text_input(
            "미샵 상품 URL",
            placeholder="https://misharp.co.kr/product/detail.html?product_no=XXXXX ...",
        )
        st.markdown(
            '<div class="small-note">※ URL 입력 시: <b>상품 상세 HTML에서 본문 상세이미지 후보만</b> 선별해 처리합니다.</div>',
            unsafe_allow_html=True,
        )
    else:
        uploaded_files = st.file_uploader(
            "상세페이지 JPG 업로드 (여러 장 가능)",
            type=["jpg", "jpeg"],
            accept_multiple_files=True,
            help="긴 상세페이지 이미지 1장 또는 여러 장을 올릴 수 있습니다.",
        )

with colB:
    # -----------------------------
    # 2) 자르기 옵션
    # -----------------------------
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

    auto_exclude_text = st.checkbox("텍스트/타이틀/로고 컷 자동 제외", value=True)
    st.markdown(
        '<div class="small-note">※ 아래 3)에서 수동 체크로 제외/포함을 조정할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

run = st.button("✅ 본문 상품컷 추출하기", type="primary", use_container_width=True)

if run:
    with st.spinner("이미지 수집/분석 중..."):
        base_name = "misharp_detail"
        source_images: List[Image.Image] = []

        if input_type == "상품 URL":
            product_url = normalize_url(product_url)
            if not product_url:
                st.error("상품 URL을 입력해 주세요.")
                st.stop()

            base_name = guess_base_name_from_url(product_url)
            imgs = fetch_detail_images_from_product_url(product_url)

            if not imgs:
                st.error("본문 상세이미지를 찾지 못했어요. (접근 제한/본문 이미지가 다른 방식일 수 있음)")
                st.stop()

            source_images = imgs

        else:
            if not uploaded_files or len(uploaded_files) == 0:
                st.error("상세페이지 JPG를 1장 이상 업로드해 주세요.")
                st.stop()

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

        items = build_items_from_sources(source_images, auto_exclude_text=auto_exclude_text)

        if not items:
            st.error("추출 결과가 없습니다.")
            st.stop()

        st.session_state["cuts_base_name"] = base_name
        st.session_state["cuts_crop_mode"] = crop_mode
        st.session_state["cuts_items"] = items

        # zip 캐시 제거
        st.session_state.pop("dl_zip", None)
        st.session_state.pop("dl_zip_name", None)

    st.success(f"추출 완료! (총 {len(items)}개 후보) 아래 3)에서 미리보기/제외/다운로드를 진행하세요.")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# 3) 미리보기, 제외, 다운로드
# -----------------------------
st.subheader("3) 미리보기 · 제외 · 다운로드")

if "cuts_items" not in st.session_state:
    st.info("먼저 위에서 **‘본문 상품컷 추출하기’**를 실행해 주세요.")
else:
    base_name = st.session_state.get("cuts_base_name", "misharp_detail")
    crop_mode = st.session_state.get("cuts_crop_mode", "이미지 그대로 자르기")
    cuts: List[CutItem] = st.session_state.get("cuts_items", [])

    total = len(cuts)
    auto_ex = sum(1 for c in cuts if c.excluded_auto)

    st.markdown(
        f"""
<div class="card">
<b>현재 상태</b><br/>
- 추출 후보: <b>{total}개</b><br/>
- 자동 제외(텍스트/로고 추정): <b>{auto_ex}개</b><br/>
- 자르기 모드: <b>{crop_mode}</b><br/>
- 흰줄 처리: <b>{EDGE_WHITE_STRIP_MIN_PX}px 이상</b>은 여백으로 제거
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.write("### 제외할 컷 선택")
    st.caption("자동 제외가 오탐이면 체크를 해제하고, 빼고 싶은 컷은 체크하세요.")

    cols = st.columns(4)
    manual_key_prefix = f"manual_ex_{base_name}_{crop_mode}"

    for i, item in enumerate(cuts):
        col = cols[i % 4]
        key = f"{manual_key_prefix}_{item.idx}"

        if key not in st.session_state:
            st.session_state[key] = bool(item.excluded_auto)

        thumb = item.pil.copy()
        thumb.thumbnail((420, 420))

        with col:
            st.image(thumb, caption=f"#{item.idx} ({item.pil.size[0]}x{item.pil.size[1]})", use_container_width=True)

            label = "이 컷 제외"
            if item.excluded_auto and item.reason:
                label += f" (자동: {item.reason})"

            st.checkbox(label, key=key)

    # 수동 제외 적용
    for item in cuts:
        key = f"{manual_key_prefix}_{item.idx}"
        item.excluded_manual = bool(st.session_state.get(key, False))

    final_items = [c for c in cuts if not c.excluded_manual]

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.write("### 다운로드")
    st.caption("다운로드는 ‘최종 포함’된 컷만 생성합니다.")
    st.write(f"최종 포함: **{len(final_items)}개** / 제외: **{total - len(final_items)}개**")

    if len(final_items) == 0:
        st.warning("포함된 컷이 0개입니다. 제외 체크를 해제해 주세요.")
    else:
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
            out0 = apply_crop_mode(final_items[0].pil, crop_mode)
            st.download_button(
                "⬇️ 대표 1장 JPG 다운로드(첫 컷)",
                data=pil_to_bytes_jpg(out0, quality=95),
                file_name=f"{safe_filename(base_name)}_001.jpg",
                mime="image/jpeg",
                use_container_width=True,
                key=f"download_first_{base_name}_{crop_mode}",
            )

        if st.session_state.get("dl_zip"):
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
