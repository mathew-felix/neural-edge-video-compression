"""ROI and full-frame quality metrics for codec benchmark (BGR uint8 frames)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio
    from skimage.metrics import structural_similarity as ssim_2d
except ImportError as exc:
    peak_signal_noise_ratio = None  # type: ignore[assignment]
    ssim_2d = None  # type: ignore[assignment]
    _SKIMAGE_IMPORT_ERROR = exc
else:
    _SKIMAGE_IMPORT_ERROR = None

_MS_SSIM_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)


def require_skimage() -> None:
    if peak_signal_noise_ratio is None or ssim_2d is None:
        raise RuntimeError(
            "Quality metrics require scikit-image. Install with: pip install scikit-image"
        ) from _SKIMAGE_IMPORT_ERROR


def psnr_full_bgr(ref: np.ndarray, test: np.ndarray) -> float:
    require_skimage()
    ref_f = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tst_f = cv2.cvtColor(test, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return float(peak_signal_noise_ratio(ref_f, tst_f, data_range=1.0))


def ms_ssim_full_bgr(ref: np.ndarray, test: np.ndarray) -> float:
    require_skimage()
    ref_f = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tst_f = cv2.cvtColor(test, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return float(_multiscale_ssim_rgb(ref_f, tst_f))


def _multiscale_ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """MS-SSIM on RGB float [0,1] HWC; renormalize weights if fewer scales are used."""
    x, y = img1.astype(np.float64), img2.astype(np.float64)
    contribs: list[float] = []
    for i in range(5):
        if min(x.shape[0], x.shape[1]) < 7:
            break
        cs = []
        for c in range(3):
            cs.append(ssim_2d(x[..., c], y[..., c], data_range=1.0))
        contribs.append(float(_MS_SSIM_WEIGHTS[i]) * float(np.mean(cs)))
        nh, nw = max(1, x.shape[0] // 2), max(1, x.shape[1] // 2)
        if nh < 16 or nw < 16:
            break
        x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
        y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_AREA)
    if not contribs:
        return float("nan")
    n = len(contribs)
    wnorm = float(_MS_SSIM_WEIGHTS[:n].sum())
    return float(sum(contribs) / wnorm)


def _bbox_from_mask(mask: np.ndarray, margin: int = 2) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("empty mask")
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    return x0, y0, x1, y1


def psnr_roi_bgr(ref: np.ndarray, test: np.ndarray, mask_u8: np.ndarray) -> float:
    """Mean MSE restricted to ROI mask (any channel masked)."""
    m = mask_u8 > 0
    if not np.any(m):
        return float("nan")
    ref_f = ref.astype(np.float64)
    tst_f = test.astype(np.float64)
    sse = 0.0
    cnt = 0
    for c in range(3):
        d = ref_f[..., c] - tst_f[..., c]
        sse += float(np.sum((d * d)[m]))
        cnt += int(np.sum(m))
    if cnt <= 0:
        return float("nan")
    mse = sse / cnt
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def ms_ssim_roi_bgr(ref: np.ndarray, test: np.ndarray, mask_u8: np.ndarray) -> float:
    """MS-SSIM on tight RGB crop around ROI mask."""
    require_skimage()
    if not np.any(mask_u8 > 0):
        return float("nan")
    x0, y0, x1, y1 = _bbox_from_mask(mask_u8)
    cr = ref[y0:y1, x0:x1]
    ct = test[y0:y1, x0:x1]
    cm = mask_u8[y0:y1, x0:x1]
    if cr.size == 0 or not np.any(cm > 0):
        return float("nan")
    # Pad tiny crops so SSIM window works
    min_side = max(cr.shape[0], cr.shape[1])
    pad_total = max(0, 64 - min_side)
    if pad_total > 0:
        p = pad_total // 2 + 1
        cr = cv2.copyMakeBorder(cr, p, p, p, p, cv2.BORDER_REFLECT_101)
        ct = cv2.copyMakeBorder(ct, p, p, p, p, cv2.BORDER_REFLECT_101)
        cm = cv2.copyMakeBorder(cm, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    rf = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tf = cv2.cvtColor(ct, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    mf = cm > 0
    # If crop is mostly non-ROI due to bbox, SSIM still compares full crop (ROI-focused crop is tight)
    if rf.shape[0] < 7 or rf.shape[1] < 7:
        return float("nan")
    return float(_multiscale_ssim_rgb(rf, tf))


def aggregate_metrics(samples: Dict[str, list]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, vals in samples.items():
        clean = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        if not clean:
            out[f"{k}_mean"] = float("nan")
            continue
        out[f"{k}_mean"] = float(np.mean(clean))
    return out
