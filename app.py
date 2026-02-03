import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image


# =========================
# 기본 설정
# =========================
DEFAULT_TIMEOUT = 25
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

# 출력 규격
IG_W, IG_H = 1080, 1350  # 인스타 피드 4:5
SQ_W, SQ_H = 1080, 1080  # 정방형 1:1
SF_W, SF_H = 900, 1600   # 숏폼 9:16 (요청 규격)


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
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if src.startswith("/"):
        m = re.match(r"^(https?://[^/]+)", base_url)
        if m:
            return m.group(1) + src
    if base_url.endswith("/"):
        return base_url + src
    return base_url.rsplit("/", 1)[0] + "/" + src


# =========================
# URL → 본문 상세이미지 URL 수집(강화버전)
# =========================
def extract_detail_image_urls_from_product_page(product_url: str) -> List[str]:
    """
    ✅ 핵심 개선
    - 가능한 한 '본문 상세영역' 컨테이너 내부의 img만 수집
    - lazy-load 속성(ec-data-src / data-src / data-original 등) 지원
    - 상세가 iframe으로 분리된 경우 iframe src 따라가서 img 수집
    - 인스타/로고/아이콘/공용 이미지 1차 제거
    """
    html = requests.get(product_url, headers=UA, timeout=DEFAULT_TIMEOUT).text
    soup = BeautifulSoup(html, "html.parser")

    detail_selectors = [
        "#prdDetail",
        "#prdDetailContent",
        "#prdDetailCont",
        "#detailArea",
        "#productDetail",
        "#product_detail",
        "#contents",
        ".xans-product-detail",
        ".xans-product-detaildesign",
        ".detailArea",
        ".cont",
        "#tabProductDetail",
        "#prdDetailContentLazy",
    ]

    def collect_imgs_from(node, base_for_norm: str) -> List[str]:
        urls = []
        for img in node.find_all("img"):
            src = (
                img.get("ec-data-src")
                or img.get("data-src")
                or img.get("data-original")
                or img.get("data-lazy")
                or img.get("src")
            )
            if not src:
                continue
            full = normalize_url(src.strip(), base_for_norm)
            urls.append(full)
        return urls

    candidates: List[str] = []

    # 1) 상세 컨테이너 우선
    detail_node = None
    for sel in detail_selectors:
        found = soup.select_one(sel)
        if found:
            detail_node = found
            break

    if detail_node is not None:
        candidates.extend(collect_imgs_from(detail_node, product_url))
    else:
        # 2) iframe 상세 시도
        iframe = soup.find("iframe")
        iframe_src = None
        if iframe:
            iframe_src = iframe.get("src") or iframe.get("data-src")

        if iframe_src:
            try:
                iframe_url = normalize_url(iframe_src, product_url)
                html2 = requests.get(iframe_url, headers=UA, timeout=DEFAULT_TIMEOUT).text
                soup2 = BeautifulSoup(html2, "html.parser")

                detail_node2 = None
                for sel in detail_selectors:
                    f2 = soup2.select_one(sel)
                    if f2:
                        detail_node2 = f2
                        break

                if detail_node2 is not None:
                    candidates.extend(collect_imgs_from(detail_node2, iframe_url))
                else:
                    candidates.extend(collect_imgs_from(soup2, iframe_url))
            except Exception:
                candidates.extend(collect_imgs_from(soup, product_url))
        else:
            # 3) 최후: 전체 문서
            candidates.extend(collect_imgs_from(soup, product_url))

    # 4) 중복 제거 + 강한 제외 키워드
    filtered = []
    seen = set()
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)

        low = u.lower()
        bad = ["instagram", "facebook", "kakao", "naver", "logo", "icon", "btn", "sprite", "share", "common"]
        if any(x in low for x in bad):
            continue

        filtered.append(u)

    return filtered


def pick_body_detail_images(product_url: str, max_fetch: int = 120) -> List[Tuple[str, Image.Image]]:
    """
    ✅ 본문 상세이미지를 가능한 한 모두 가져오기
    - 실제 이미지 크기 기반으로 로고/아이콘 제거
    """
    urls = extract_detail_image_urls_from_product_page(product_url)
    if not urls:
        return []

    urls = urls[:max_fetch]

    images: List[Tuple[str, Image.Image]] = []
    for u in urls:
        try:
            b = fetch_bytes(u)
            im = pil_open_rgb(b)
            w, h = im.size

            # ✅ 아이콘/로고 제거 필터
            if w < 650 and h < 650:
                continue
            if (abs(w - h) <= 20) and (w < 800) and (h < 800):
                continue

            images.append((u, im))
        except Exception:
            continue

    return images


# =========================
# 흰 여백 기반 컷 분리 + 트리밍
# =========================
@dataclass
class SplitConfig:
    white_thr: int = 245
    row_white_ratio: float = 0.985
    min_gap: int = 25
    edge_pad: int = 0


def _find_white_runs(gray: np.ndarray, cfg: SplitConfig) -> List[Tuple[int, int]]:
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
    arr = np.array(img.convert("RGB"))
    gray = arr.mean(axis=2).astype(np.uint8)
    H, W = gray.shape

    runs = _find_white_runs(gray, cfg)
    if len(runs) == 0:
        return [img]

    segments = []
    prev_end = 0

    for (s, e) in runs:
        seg_top = prev_end
        seg_bottom = s - 1
        if seg_bottom - seg_top + 1 > 30:
            crop_top = max(0, seg_top + cfg.edge_pad)
            crop_bottom = min(H - 1, seg_bottom - cfg.edge_pad)
            seg = img.crop((0, crop_top, W, crop_bottom + 1))
            segments.append(seg)
        prev_end = e + 1

    if H - prev_end > 30:
        crop_top = max(0, prev_end + cfg.edge_pad)
        seg = img.crop((0, crop_top, W, H))
        segments.append(seg)

    return segments if segments else [img]


def trim_white_margins(img: Image.Image, white_thr: int = 245, ratio_thr: float = 0.98) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = arr.mean(axis=2)

    H, W = gray.shape
    row_ratio = (gray >= white_thr).mean(axis=1)
    col_ratio = (gray >= white_thr).mean(axis=0)

    top = 0
    while top < H and row_ratio[top] >= ratio_thr:
        top += 1

    bottom = H - 1
    while bottom >= 0 and row_ratio[bottom] >= ratio_thr:
        bottom -= 1

    left = 0
    while left < W and col_ratio[left] >= ratio_thr:
        left += 1

    right = W - 1
    while right >= 0 and col_ratio[right] >= ratio_thr:
        right -= 1

    if bottom - top < 50 or right - left < 50:
        return img

    return img.crop((left, top, right + 1, bottom + 1))


# =========================
# 크롭 모드
# =========================
def center_crop_to_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    target_ratio = target_w / target_h
    src_ratio = w / h

    if abs(src_ratio - target_ratio) < 1e-6:
        return img

    if src_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(round(w / target_ratio))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def crop_mode_apply(img: Image.Image, mode: str) -> Image.Image:
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
# 저장/ZIP
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


def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, Image.Image]]:
    """
    ZIP 내부 이미지들을 파일명 순서로 읽어서 반환
    """
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        # 파일명 정렬(실무에서 순서 유지에 유리)
        names.sort()

        for n in names:
            low = n.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png") or low.endswith(".webp")):
                continue
            try:
                b = zf.read(n)
                im = pil_open_rgb(b)
                out.append((os.path.basename(n), im))
            except Exception:
                continue
    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="misharp-image-extractor", layout="wide")
st.title("상세페이지 이미지 추출기")
st.caption("미샵 상품 URL 또는 상세페이지 이미지를 넣으면, 본문 상품컷을 흰 여백 없이 분리/트리밍 후 규격별로 저장합니다.")

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
    st.caption("※ ‘이미지 그대로’는 **리사이즈/크롭 없이** 분리 + 트리밍만 합니다.")


tab1, tab2 = st.tabs(["URL 입력", "이미지/ZIP 업로드"])


# -------------------------
# URL 입력 탭
# -------------------------
with tab1:
    st.subheader("미샵 상품 URL 입력")
    url = st.text_input("상품 URL", placeholder="https://misharp.co.kr/product/detail.html?product_no=xxxxx")

    colA, colB = st.columns([1, 1])
    with colA:
        do_run = st.button("본문 상품컷 추출하기", type="primary", use_container_width=True)
    with colB:
        st.info("URL 입력 시: 상품 상세 HTML에서 **본문 상세이미지 후보만 선별**합니다.", icon="ℹ️")

    if do_run:
        if not url.strip():
            st.error("URL을 입력해주세요.")
        else:
            with st.spinner("상품 페이지 분석 → 본문 상세이미지 다운로드 중..."):
                try:
                    detail_imgs = pick_body_detail_images(url.strip(), max_fetch=120)
                except Exception as e:
                    detail_imgs = []
                    st.error(f"URL 처리 중 오류: {e}")

            if not detail_imgs:
                st.error("본문 상세이미지를 찾지 못했습니다. (URL이 올바른지 / 접근 가능한지 확인)")
            else:
                st.success(f"본문 상세이미지 후보 {len(detail_imgs)}장 찾음")

                all_cuts: List[Image.Image] = []

                # ✅ 이미지가 여러 장일 수 있으므로: 긴 합본만 분리, 단독컷은 그대로
                for idx, (img_url, im) in enumerate(detail_imgs, start=1):
                    w, h = im.size
                    if h >= 1600 and h >= w * 1.8:
                        segs = split_by_white_gaps(im, cfg)
                        for s in segs:
                            cut = trim_white_margins(s, white_thr=white_thr, ratio_thr=trim_ratio)
                            all_cuts.append(cut)
                    else:
                        cut = trim_white_margins(im, white_thr=white_thr, ratio_thr=trim_ratio)
                        all_cuts.append(cut)

                if not all_cuts:
                    st.error("컷 분리에 실패했습니다. (여백 판정 값을 조정해보세요)")
                else:
                    processed = [crop_mode_apply(c, crop_mode) for c in all_cuts]

                    base = safe_base(url.strip())
                    jpg_files: List[Tuple[str, bytes]] = []
                    for i, img in enumerate(processed, start=1):
                        fn = f"{base}_cut_{i:03d}.jpg"
                        jpg_files.append((fn, img_to_jpg_bytes(img, quality=95)))

                    st.divider()
                    st.subheader("결과 미리보기")
                    st.write(f"- 추출된 컷: **{len(processed)}장**")
                    st.caption("미리보기는 일부만 보일 수 있습니다.")
                    st.image(processed[:18], width=160)

                    st.divider()
                    st.subheader("다운로드")

                    st.download_button(
                        "첫 번째 컷 JPG 다운로드",
                        data=jpg_files[0][1],
                        file_name=jpg_files[0][0],
                        mime="image/jpeg",
                        use_container_width=True,
                        key=f"dl_first_url_{base}_{crop_mode}",
                    )

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
# 이미지/ZIP 업로드 탭
# -------------------------
with tab2:
    st.subheader("상세페이지 이미지 또는 ZIP 업로드")

    ups = st.file_uploader(
        "상세페이지 이미지 업로드 (여러 장 가능)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    up_zip = st.file_uploader(
        "ZIP 업로드 (ZIP 안에 이미지가 들어있어야 함)",
        type=["zip"],
        accept_multiple_files=False
    )

    do_run2 = st.button("업로드 파일에서 상품컷 추출하기", type="primary", use_container_width=True)

    if do_run2:
        if (not ups) and (not up_zip):
            st.error("이미지(1장 이상) 또는 ZIP 파일을 업로드해주세요.")
        else:
            all_sources: List[Tuple[str, Image.Image]] = []

            # 1) 이미지 여러장
            if ups:
                for up in ups:
                    try:
                        im = Image.open(up).convert("RGB")
                        all_sources.append((up.name, im))
                    except Exception as e:
                        st.warning(f"{up.name} 이미지 열기 실패: {e}")

            # 2) ZIP 내부 이미지
            if up_zip is not None:
                try:
                    zbytes = up_zip.read()
                    zimgs = extract_images_from_zip(zbytes)
                    # zip 내 파일명과 이미지
                    all_sources.extend(zimgs)
                except Exception as e:
                    st.warning(f"ZIP 처리 실패: {e}")

            if not all_sources:
                st.error("업로드된 파일에서 이미지를 읽지 못했습니다.")
            else:
                all_cuts: List[Image.Image] = []
                base_names: List[str] = [safe_base(n) for n, _ in all_sources]
                base = base_names[0] if base_names else "detail"

                for name, im in all_sources:
                    w, h = im.size
                    if h >= 1600 and h >= w * 1.8:
                        segs = split_by_white_gaps(im, cfg)
                        for s in segs:
                            cut = trim_white_margins(s, white_thr=white_thr, ratio_thr=trim_ratio)
                            all_cuts.append(cut)
                    else:
                        cut = trim_white_margins(im, white_thr=white_thr, ratio_thr=trim_ratio)
                        all_cuts.append(cut)

                if not all_cuts:
                    st.error("컷 추출에 실패했습니다. (여백 판정 값을 조정해보세요)")
                else:
                    processed = [crop_mode_apply(c, crop_mode) for c in all_cuts]

                    # 파일이 여러 개면 multi 표기
                    if len(all_sources) > 1:
                        base = f"{base}_multi"

                    jpg_files: List[Tuple[str, bytes]] = []
                    for i, img in enumerate(processed, start=1):
                        fn = f"{base}_cut_{i:03d}.jpg"
                        jpg_files.append((fn, img_to_jpg_bytes(img, quality=95)))

                    st.divider()
                    st.subheader("결과 미리보기")
                    st.write(f"- 입력 이미지: **{len(all_sources)}개**")
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
