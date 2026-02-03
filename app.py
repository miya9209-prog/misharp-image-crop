import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image


# =========================
# 기본 설정
# =========================
DEFAULT_TIMEOUT = 20
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

# IG 규격
IG_W, IG_H = 1080, 1350  # 4:5
SQ_W, SQ_H = 1080, 1080  # 1:1
SF_W, SF_H = 900, 1600   # 9:16 (요청)


# =========================
# 유틸
# =========================
def safe_base(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^\w\-.가-힣]+", "_", base).strip("_")
    return base[:120] if base else "detail"


def pil_open_rgb(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, headers=UA, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.content


def normalize_url(src: str, base_url: str) -> str:
    # //cdn... 형태 처리
    if src.startswith("//"):
        return "https:" + src
    # 절대
    if src.startswith("http://") or src.startswith("https://"):
        return src
    # 상대
    if src.startswith("/"):
        # base domain
        m = re.match(r"^(https?://[^/]+)", base_url)
        if m:
            return m.group(1) + src
    # 기타 상대
    if base_url.endswith("/"):
        return base_url + src
    return base_url.rsplit("/", 1)[0] + "/" + src


def is_probably_detail_image(url: str) -> bool:
    """
    URL 문자열 기반 1차 필터:
    - icon, logo, common 등 제외
    - product/detail류 우선
    """
    u = url.lower()
    bad = ["icon", "logo", "common", "btn", "banner", "sprite", "loading", "blank", "youtube", "kakao", "naver"]
    if any(x in u for x in bad):
        return False
    # 상세/상품 이미지일 가능성 키워드
    good = ["product", "upload", "data", "image", "img", "detail", "editor", "contents"]
    return any(x in u for x in good)


def score_detail_candidate(img: Image.Image) -> int:
    """
    본문 상세(세로 긴)일수록 점수↑
    """
    w, h = img.size
    score = 0
    if w >= 600:
        score += 2
    if h >= 1200:
        score += 3
    if h >= w * 2:
        score += 3
    if h >= w * 3:
        score += 2
    return score


# =========================
# URL → 본문 상세이미지(세로 JPG) 찾기
# =========================
def extract_detail_image_urls_from_product_page(product_url: str) -> List[str]:
    html = requests.get(product_url, headers=UA, timeout=DEFAULT_TIMEOUT).text
    soup = BeautifulSoup(html, "html.parser")

    # 1) 모든 img 후보 수집
    candidates = []
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("data-original") or img.get("src")
        if not src:
            continue
        src = src.strip()
        full = normalize_url(src, product_url)

        if is_probably_detail_image(full):
            candidates.append(full)

    # 2) 중복 제거
    uniq = []
    seen = set()
    for u in candidates:
        if u not in seen:
            uniq.append(u)
            seen.add(u)

    return uniq


def pick_body_detail_images(product_url: str, max_fetch: int = 12) -> List[Tuple[str, Image.Image]]:
    """
    상품 URL에서 '본문 상세페이지 이미지'만 최대한 골라오기
    - 여러 장이면: '세로로 긴 이미지' 위주로 선별
    - 썸네일/아이콘 제외 목적
    """
    urls = extract_detail_image_urls_from_product_page(product_url)

    # 너무 많으면 앞에서 조금만 fetch
    urls = urls[:max_fetch]

    images: List[Tuple[str, Image.Image]] = []
    for u in urls:
        try:
            b = fetch_bytes(u)
            im = pil_open_rgb(b)
            images.append((u, im))
        except Exception:
            continue

    if not images:
        return []

    # "본문 상세"는 보통 세로 긴 이미지를 포함
    scored = sorted(images, key=lambda x: score_detail_candidate(x[1]), reverse=True)

    # 점수 상위만 추리되, 너무 공격적이면 놓칠 수 있으니 최소 1장~최대 4장
    # (상세가 여러 장인 경우 대비)
    top = []
    for u, im in scored:
        if score_detail_candidate(im) >= 5:
            top.append((u, im))

    if not top:
        top = [scored[0]]

    # 상위 4장 제한 (너무 많으면 품질 떨어짐)
    return top[:4]


# =========================
# 흰 여백 기반 컷 분리 + 트리밍
# =========================
@dataclass
class SplitConfig:
    white_thr: int = 245        # 픽셀 흰색 판정 기준
    row_white_ratio: float = 0.985  # 한 줄이 흰색으로 간주될 비율
    min_gap: int = 25           # 컷 사이 여백 최소 길이(px)
    edge_pad: int = 0           # 분리 후 여백 조금 남길지(기본 0)


def _find_white_runs(gray: np.ndarray, cfg: SplitConfig) -> List[Tuple[int, int]]:
    """
    gray: HxW
    return: 흰줄(여백) 구간들의 (start_row, end_row) 리스트
    """
    white_mask = (gray >= cfg.white_thr)
    ratios = white_mask.mean(axis=1)
    is_white_row = ratios >= cfg.row_white_ratio

    runs = []
    in_run = False
    s = 0
    for i, v in enumerate(is_white_row):
        if v and not in_run:
            in_run = True
            s = i
        elif (not v) and in_run:
            e = i - 1
            if (e - s + 1) >= cfg.min_gap:
                runs.append((s, e))
            in_run = False
    if in_run:
        e = len(is_white_row) - 1
        if (e - s + 1) >= cfg.min_gap:
            runs.append((s, e))
    return runs


def split_by_white_gaps(img: Image.Image, cfg: SplitConfig) -> List[Image.Image]:
    """
    세로로 긴 상세페이지 이미지에서
    '가로 전체가 흰색에 가까운 여백 구간'을 기준으로 컷을 분리.
    """
    arr = np.array(img.convert("RGB"))
    gray = arr.mean(axis=2).astype(np.uint8)
    H, W = gray.shape

    runs = _find_white_runs(gray, cfg)

    # 여백이 거의 없으면 "통 이미지 1장" 처리
    if len(runs) == 0:
        return [img]

    # runs를 경계로 구간 생성
    segments = []
    prev_end = 0

    for (s, e) in runs:
        seg_top = prev_end
        seg_bottom = s - 1
        if seg_bottom - seg_top + 1 > 30:  # 너무 얇은 건 제외
            crop_top = max(0, seg_top + cfg.edge_pad)
            crop_bottom = min(H - 1, seg_bottom - cfg.edge_pad)
            seg = img.crop((0, crop_top, W, crop_bottom + 1))
            segments.append(seg)
        prev_end = e + 1

    # 마지막 구간
    if H - prev_end > 30:
        crop_top = max(0, prev_end + cfg.edge_pad)
        seg = img.crop((0, crop_top, W, H))
        segments.append(seg)

    return segments if segments else [img]


def trim_white_margins(img: Image.Image, white_thr: int = 245, ratio_thr: float = 0.98) -> Image.Image:
    """
    컷 단위 이미지에서 상하좌우 흰 여백을 최대한 제거.
    (피사체가 흰 배경일 때 과하게 잘릴 수 있으니 ratio_thr로 안전장치)
    """
    arr = np.array(img.convert("RGB"))
    gray = arr.mean(axis=2)

    H, W = gray.shape

    # 행/열별 "흰색 비율"
    row_ratio = (gray >= white_thr).mean(axis=1)
    col_ratio = (gray >= white_thr).mean(axis=0)

    # top
    top = 0
    while top < H and row_ratio[top] >= ratio_thr:
        top += 1
    # bottom
    bottom = H - 1
    while bottom >= 0 and row_ratio[bottom] >= ratio_thr:
        bottom -= 1
    # left
    left = 0
    while left < W and col_ratio[left] >= ratio_thr:
        left += 1
    # right
    right = W - 1
    while right >= 0 and col_ratio[right] >= ratio_thr:
        right -= 1

    # 방어: 너무 작아지면 원본 반환
    if bottom - top < 50 or right - left < 50:
        return img

    return img.crop((left, top, right + 1, bottom + 1))


# =========================
# 크롭 모드 (비율/규격)
# =========================
def center_crop_to_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    왜곡 없이 중앙 기준으로 비율 맞추기(필요한 만큼만 좌/우 또는 상/하 잘라냄)
    """
    w, h = img.size
    target_ratio = target_w / target_h
    src_ratio = w / h

    if abs(src_ratio - target_ratio) < 1e-6:
        return img

    if src_ratio > target_ratio:
        # 너무 넓다 -> 좌우 자르기
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        # 너무 높다 -> 상하 자르기
        new_h = int(round(w / target_ratio))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def crop_mode_apply(img: Image.Image, mode: str) -> Image.Image:
    """
    mode:
      - "이미지 그대로 자르기" => trim만 수행(리사이즈/추가 크롭 없음)
      - "인스타그램 피드 규격(4:5)"
      - "정방형(1:1)"
      - "숏폼규격(900x1600)"
    """
    if mode == "이미지 그대로 자르기":
        return img

    if mode == "인스타그램 피드 규격(4:5)":
        c = center_crop_to_ratio(img, IG_W, IG_H)
        return c.resize((IG_W, IG_H), Image.LANCZOS)

    if mode == "정방형(1:1)":
        c = center_crop_to_ratio(img, SQ_W, SQ_H)
        return c.resize((SQ_W, SQ_H), Image.LANCZOS)

    if mode == "숏폼규격(900x1600)":
        c = center_crop_to_ratio(img, SF_W, SF_H)
        return c.resize((SF_W, SF_H), Image.LANCZOS)

    return img


# =========================
# 출력(저장/ZIP)
# =========================
def img_to_jpg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_zip(files: List[Tuple[str, bytes]]) -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return zbuf.getvalue()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="misharp-image-extractor", layout="wide")
st.title("상세페이지 이미지 추출기")
st.caption("미샵 상품 URL 또는 상세페이지 JPG를 넣으면, 본문 상품컷을 흰 여백 없이 분리/트리밍 후 규격별로 저장합니다.")

# ---- Sidebar: 옵션 ----
with st.sidebar:
    st.header("설정")

    crop_mode = st.radio(
        "자르기 방식",
        [
            "이미지 그대로 자르기",
            "인스타그램 피드 규격(4:5)",
            "정방형(1:1)",
            "숏폼규격(900x1600)",
        ],
        index=0,
    )

    st.divider()
    st.subheader("분리/트리밍 민감도")

    white_thr = st.slider("흰색 판정 기준(높을수록 엄격)", 230, 255, 245, 1)
    row_ratio = st.slider("여백(흰줄) 판정 비율", 0.950, 0.999, 0.985, 0.001)
    min_gap = st.slider("컷 사이 여백 최소 높이(px)", 10, 120, 25, 1)
    trim_ratio = st.slider("트리밍(흰여백 제거) 비율", 0.90, 0.999, 0.98, 0.001)

    cfg = SplitConfig(white_thr=white_thr, row_white_ratio=row_ratio, min_gap=min_gap, edge_pad=0)

    st.divider()
    st.caption("※ ‘이미지 그대로’는 **리사이즈/크롭 없이** 분리+트리밍만 합니다.")


tab1, tab2 = st.tabs(["URL 입력", "JPG 업로드"])


# -------------------------
# 1) URL 입력
# -------------------------
with tab1:
    st.subheader("미샵 상품 URL 입력")
    url = st.text_input("상품 URL", placeholder="https://miyawa.cafe24.com/product/detail.html?product_no=xxxx")

    colA, colB = st.columns([1, 1])
    with colA:
        do_run = st.button("본문 상품컷 추출하기", type="primary", use_container_width=True)
    with colB:
        st.info("URL 입력 시: **상품 상세 HTML에서 본문 상세이미지 후보만 선별**합니다.", icon="ℹ️")

    if do_run:
        if not url.strip():
            st.error("URL을 입력해주세요.")
        else:
            with st.spinner("상품 페이지 분석 → 본문 상세이미지 다운로드 중..."):
                try:
                    detail_imgs = pick_body_detail_images(url.strip(), max_fetch=12)
                except Exception as e:
                    detail_imgs = []
                    st.error(f"URL 처리 중 오류: {e}")

            if not detail_imgs:
                st.error("본문 상세이미지를 찾지 못했습니다. (URL이 올바른지 / 접근이 가능한지 확인)")
            else:
                st.success(f"본문 상세이미지 후보 {len(detail_imgs)}장 찾음 (세로 상세 위주)")
                # 각 상세이미지에서 컷 분리
                all_cuts: List[Image.Image] = []
                for idx, (img_url, im) in enumerate(detail_imgs, start=1):
                    segs = split_by_white_gaps(im, cfg)
                    for s in segs:
                        cut = trim_white_margins(s, white_thr=white_thr, ratio_thr=trim_ratio)
                        all_cuts.append(cut)

                if not all_cuts:
                    st.error("컷 분리에 실패했습니다. (여백 판정 값을 조정해보세요)")
                else:
                    # 모드 적용
                    processed = [crop_mode_apply(c, crop_mode) for c in all_cuts]

                    # 파일명
                    base = safe_base(url.strip())
                    jpg_files: List[Tuple[str, bytes]] = []
                    for i, img in enumerate(processed, start=1):
                        fn = f"{base}_cut_{i:03d}.jpg"
                        jpg_files.append((fn, img_to_jpg_bytes(img, quality=95)))

                    # 미리보기
                    st.divider()
                    st.subheader("결과 미리보기")
                    st.write(f"- 추출된 컷: **{len(processed)}장**")
                    st.caption("미리보기는 일부만 보일 수 있습니다.")
                    preview = processed[:18]
                    st.image(preview, width=160)

                    # 다운로드
                    st.divider()
                    st.subheader("다운로드")

                    # 1장 JPG(첫 컷)
                    st.download_button(
                        "첫 번째 컷 JPG 다운로드",
                        data=jpg_files[0][1],
                        file_name=jpg_files[0][0],
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"dl_first_url_{base}_{crop_mode}",
                    )

                    # ZIP
                    zip_bytes = build_zip(jpg_files)
                    st.download_button(
                        "전체 컷 ZIP 다운로드",
                        data=zip_bytes,
                        file_name=f"{base}_cuts.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key=f"dl_zip_url_{base}_{crop_mode}",
                    )


# -------------------------
# 2) JPG 업로드
# -------------------------
with tab2:
    st.subheader("상세페이지 JPG 업로드")
    up = st.file_uploader("상세페이지 JPG 파일", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)

    do_run2 = st.button("업로드 이미지에서 상품컷 추출하기", type="primary", use_container_width=True)

    if do_run2:
        if not up:
            st.error("상세페이지 JPG를 업로드해주세요.")
        else:
            try:
                im = Image.open(up).convert("RGB")
            except Exception as e:
                st.error(f"이미지 열기 실패: {e}")
                im = None

            if im:
                segs = split_by_white_gaps(im, cfg)
                cuts = [trim_white_margins(s, white_thr=white_thr, ratio_thr=trim_ratio) for s in segs]
                processed = [crop_mode_apply(c, crop_mode) for c in cuts]

                base = safe_base(up.name)
                jpg_files: List[Tuple[str, bytes]] = []
                for i, img in enumerate(processed, start=1):
                    fn = f"{base}_cut_{i:03d}.jpg"
                    jpg_files.append((fn, img_to_jpg_bytes(img, quality=95)))

                st.divider()
                st.subheader("결과 미리보기")
                st.write(f"- 추출된 컷: **{len(processed)}장**")
                st.image(processed[:18], width=160)

                st.divider()
                st.subheader("다운로드")

                st.download_button(
                    "첫 번째 컷 JPG 다운로드",
                    data=jpg_files[0][1],
                    file_name=jpg_files[0][0],
                    mime="image/jpeg",
                    use_container_width=True,
                    key=f"dl_first_upload_{base}_{crop_mode}",
                )

                zip_bytes = build_zip(jpg_files)
                st.download_button(
                    "전체 컷 ZIP 다운로드",
                    data=zip_bytes,
                    file_name=f"{base}_cuts.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key=f"dl_zip_upload_{base}_{crop_mode}",
                )


# =========================
# 하단 공지 (이전 프로그램과 동일)
# =========================
st.markdown(
    """
    <hr style="margin-top:40px; margin-bottom:10px;">
    <div style="font-size:11px; color:#888; line-height:1.5; text-align:center;">
        ⓒ misharpcompany. All rights reserved.<br>
        본 프로그램의 저작권은 미샵컴퍼니(misharpcompany)에 있으며, 무단 복제·배포·사용을 금합니다.<br>
        본 프로그램은 미샵컴퍼니 내부 직원 전용으로, 외부 유출 및 제3자 제공을 엄격히 금합니다.
        <br><br>
        ⓒ misharpcompany. All rights reserved.<br>
        This program is the intellectual property of misharpcompany.
        Unauthorized copying, distribution, or use is strictly prohibited.<br>
        This program is for internal use by misharpcompany employees only
        and must not be disclosed or shared externally.
    </div>
    """,
    unsafe_allow_html=True
)
