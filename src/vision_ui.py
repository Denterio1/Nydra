"""Vision Lab payload builder (single-image mode). JSON-safe, never raises."""
import base64, io, time
from pathlib import Path

_MAX_THUMB = 256


def _clean(v):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, float):
        return None if v != v or v in (float("inf"), float("-inf")) else round(v, 4)
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_clean(x) for x in v]
    return str(v)


def _thumb(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((_MAX_THUMB, _MAX_THUMB))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _issues(r):
    out = []
    def add(title, sev, desc):
        out.append({"title": title, "severity": sev, "description": desc})
    if r.get("is_blurry"):
        add("Blurry image", "high", "Blur type: %s" % r.get("blur_type"))
    if r.get("is_noisy"):
        add("Noisy image", "medium", "Noise type: %s" % r.get("noise_type"))
    if r.get("is_underexposed"):
        add("Underexposed", "high", "Too dark: %.0f%% of pixels near black" % (100 * (r.get("underexposed_ratio") or 0)))
    if r.get("is_overexposed"):
        add("Overexposed", "high", "Too bright: %.0f%% of pixels near white" % (100 * (r.get("overexposed_ratio") or 0)))
    if r.get("is_low_resolution"):
        add("Low resolution", "medium", str(r.get("resolution_verdict") or "Too small for ML training"))
    if r.get("has_blocking"):
        add("JPEG blocking artifacts", "low", "Estimated JPEG quality: %s" % r.get("jpeg_quality_est"))
    if r.get("has_ringing"):
        add("Ringing artifacts", "low", "Edge ringing detected")
    if r.get("color_cast"):
        add("Color cast", "low", "Dominant cast: %s" % r.get("color_cast"))
    return out


_METRICS = ["brightness_score", "contrast_score", "sharpness_score", "blur_score",
            "noise_score", "exposure_score", "artifact_score"]


def build_single_report(path, name=None):
    t = time.time()
    path = Path(path)
    name = name or path.name
    try:
        from src.vision_nlp.image_quality import analyze_image_quality
        r = analyze_image_quality(path) or {}
        if r.get("error"):
            return _clean({"status": "ok", "mode": "single", "name": name, "rejected": True,
                           "quality": {"score": 0, "verdict": "reject", "verdict_label": r.get("verdict")},
                           "message": r.get("error"), "issues": [], "metrics": {},
                           "thumbnail": None, "seconds": round(time.time() - t, 2)})
        label = str(r.get("verdict") or "")
        code = label.split(" ")[0].lower() if label else "unknown"
        return _clean({
            "status": "ok", "mode": "single", "name": name, "rejected": code == "reject",
            "quality": {"score": r.get("overall_score"), "verdict": code, "verdict_label": label,
                        "reject_reason": r.get("reject_reason") or None,
                        "suggested_fix": r.get("suggested_fix") or None},
            "image": {"width": r.get("width"), "height": r.get("height"),
                      "exposure": r.get("exposure_type"), "entropy": r.get("entropy"),
                      "edge_density": r.get("edge_density"), "dynamic_range": r.get("dynamic_range")},
            "metrics": {k: r.get(k) for k in _METRICS},
            "issues": _issues(r),
            "thumbnail": _thumb(path),
            "seconds": round(time.time() - t, 2),
        })
    except Exception as e:
        return {"status": "error", "mode": "single", "name": name, "message": type(e).__name__ + ": " + str(e)}
