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
    _et = str(r.get("exposure_type") or "").lower()
    if _et in ("underexposed", "overexposed") and not any(i["title"] in ("Underexposed", "Overexposed") for i in out):
        add(_et.capitalize(), "medium", "Exposure analysis: %s" % _et)
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


# ---------------------------------------------------------------- dataset mode

def _txt(s):
    import re
    return re.sub(r"[^\x20-\x7E\u00C0-\u024F\u0600-\u06FF]+", "", str(s)).strip()


def _txts(lst, n=8):
    out = []
    for x in list(lst or [])[:n]:
        s = _txt(x)
        if s:
            out.append(s)
    return out


def _rel(p, root):
    import os
    try:
        return os.path.relpath(str(p), str(root)).replace("\\", "/")
    except Exception:
        return str(p)


def _g(obj, attr, default=None):
    return getattr(obj, attr, default) if obj is not None else default


def _thumb_n(path, size=96):
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=60)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _build_dataset_report_inner(root, name=None, max_images=300):
    """Dataset mode: folder of images -> JSON-safe report. Never raises."""
    t = time.time()
    root = Path(root)
    name = name or root.name
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import pandas as pd
        from src.vision_nlp.image_loader import load_images
        from src.vision_nlp.image_quality import analyze_dataset_quality
        from src.vision_nlp.image_analyzer import ImageAnalyzer

        df = load_images(str(root), verbose=False)
        found = int(len(df))
        if found == 0:
            return {"status": "error", "mode": "dataset", "name": name, "message": "No images found."}
        truncated = found > max_images
        full = df
        full_valid = full["is_valid"].astype(bool)
        n_valid_all = int(full_valid.sum())
        n_dup = int(full["is_duplicate"].astype(bool).sum())
        unreadable = [_rel(x, root) for x in full.loc[~full_valid, "file_path"].tolist()[:20]]
        if "label" in full.columns:
            full_counts = full["label"].fillna("(unlabeled)").astype(str).value_counts().to_dict()
        else:
            full_counts = {}
        if truncated:
            from src.sampling import stratified_sample
            df = stratified_sample(full.loc[full_valid], max_images)

        valid_mask = df["is_valid"].astype(bool)
        n_valid = int(valid_mask.sum())
        if n_valid < 2:
            return {"status": "error", "mode": "dataset", "name": name,
                    "message": "Need at least 2 readable images (found %d)." % n_valid}

        dfq, qrep = analyze_dataset_quality(df, n_workers=2, verbose=False)
        dfq = dfq.copy()
        dfq["overall_score"] = pd.to_numeric(dfq["overall_score"], errors="coerce").fillna(0.0)

        valid_paths = set(df.loc[valid_mask, "file_path"].tolist())
        sq = dfq.loc[dfq["file_path"].isin(valid_paths), ["file_path", "overall_score"]]
        dfv = df.loc[valid_mask].reset_index(drop=True)

        an = ImageAnalyzer(loader_df=dfv, quality_df=sq, image_dir="", label_column="label",
                           path_column="file_path", dataset_name=name)
        rep = an.analyze_all()

        ml = rep.ml_readiness
        bal = rep.class_balance
        div = rep.diversity_redundancy
        out = rep.outlier_detection
        sz = rep.size_dimension
        cb = rep.color_bias

        readiness = {
            "score": _g(ml, "final_score"),
            "grade": _g(ml, "grade"),
            "verdict": _txt(_g(ml, "verdict", "")),
            "sub_scores": {
                "diversity": _g(ml, "diversity_score"), "balance": _g(ml, "balance_score"),
                "shift": _g(ml, "shift_score"), "quality": _g(ml, "quality_score"),
                "coverage": _g(ml, "coverage_score"), "size": _g(ml, "size_score"),
                "outlier": _g(ml, "outlier_score"),
            },
            "priority_issues": _txts(_g(ml, "priority_issues", [])),
            "ranked": [{"issue": _txt(r.get("issue", "")), "impact": _txt(r.get("impact", ""))}
                       for r in list(_g(ml, "recommendations_ranked", []) or [])[:6] if isinstance(r, dict)],
        }

        quality = {
            "mean": _g(qrep, "mean_overall_score"),
            "high": _g(qrep, "high_quality"), "medium": _g(qrep, "medium_quality"),
            "low": _g(qrep, "low_quality"), "rejected": _g(qrep, "rejected"),
            "blurry": _g(qrep, "blurry_count"), "dark": _g(qrep, "dark_count"),
            "overexposed": _g(qrep, "overexposed_count"), "noisy": _g(qrep, "noisy_count"),
            "artifacts": _g(qrep, "artifact_count"), "low_res": _g(qrep, "low_res_count"),
            "recommendations": _txts(_g(qrep, "recommendations", [])),
            "fix_priority": _txts(_g(qrep, "fix_priority", [])),
        }

        balance = {
            "class_counts": _g(bal, "class_counts", {}), "num_classes": _g(bal, "num_classes"),
            "imbalance_ratio": _g(bal, "imbalance_ratio"), "balance_score": _g(bal, "balance_score"),
            "severely_imbalanced": _g(bal, "is_severely_imbalanced"),
            "majority": _g(bal, "majority_class"), "minority": _g(bal, "minority_class"),
            "suggested_augmentation": _g(bal, "suggested_augmentation", {}),
            "recommendations": _txts(_g(bal, "recommendations", [])),
        }

        if truncated and full_counts:
            vals = list(full_counts.values())
            balance.update({
                "class_counts": full_counts, "num_classes": len(vals),
                "imbalance_ratio": round(max(vals) / max(min(vals), 1), 2),
                "balance_score": None,
                "majority": max(full_counts, key=full_counts.get),
                "minority": min(full_counts, key=full_counts.get),
                "suggested_augmentation": {}, "recommendations": [],
            })

        pairs = []
        for tup in list(_g(div, "near_duplicate_pairs", []) or [])[:15]:
            try:
                pairs.append({"a": _rel(tup[0], root), "b": _rel(tup[1], root), "similarity": tup[2]})
            except Exception:
                continue
        duplicates = {
            "exact_flagged": n_dup, "pairs": pairs,
            "clusters": len(_g(div, "near_duplicate_clusters", []) or []),
            "redundancy_rate": _g(div, "redundancy_rate"), "vendi_score": _g(div, "vendi_score"),
            "diversity_grade": _g(div, "diversity_grade"),
            "effective_size": _g(div, "effective_dataset_size"),
            "recommendations": _txts(_g(div, "recommendations", [])),
        }

        flagged = {}
        for tag, attr in (("isolation_forest", "isolation_forest_outliers"), ("lof", "lof_outliers"),
                          ("knn", "knn_outliers"), ("out_of_distribution", "ood_images")):
            for pth in list(_g(out, attr, []) or []):
                flagged.setdefault(pth, []).append(tag)
        outliers = {
            "rate": _g(out, "outlier_rate"), "method": _txt(_g(out, "outlier_method", "")),
            "images": [{"name": _rel(k, root), "flagged_by": v} for k, v in list(flagged.items())[:20]],
            "recommendations": _txts(_g(out, "recommendations", [])),
        }

        ws = _g(sz, "width_stats", {}) or {}
        hs = _g(sz, "height_stats", {}) or {}
        sizes = {
            "mean_width": ws.get("mean"), "mean_height": hs.get("mean"),
            "min_width": ws.get("min"), "max_width": ws.get("max"),
            "min_height": hs.get("min"), "max_height": hs.get("max"),
            "recommended_target_size": list(_g(sz, "recommended_target_size", []) or []),
            "dominant_aspect_ratio": _txt(_g(sz, "dominant_aspect_ratio", "")),
            "size_outliers": len(_g(sz, "size_outlier_images", []) or []),
            "recommendations": _txts(_g(sz, "recommendations", [])),
        }

        color = {
            "bias_detected": _g(cb, "color_bias_detected"), "severity": _txt(_g(cb, "bias_severity", "")),
            "biased_classes": list(_g(cb, "biased_classes", []) or []),
            "recommendations": _txts(_g(cb, "recommendations", [])),
        }

        has_label = "label" in dfq.columns
        ordered = dfq.sort_values("overall_score", ascending=True)
        pick = list(ordered.head(12).index) + list(ordered.tail(12).index)
        seen, gallery = set(), []
        for ix in pick:
            if ix in seen:
                continue
            seen.add(ix)
            row = dfq.loc[ix]
            fp = str(row["file_path"])
            ok = fp in valid_paths
            gallery.append({
                "name": _rel(fp, root), "label": str(row["label"]) if has_label else "",
                "score": float(row["overall_score"]), "unreadable": not ok,
                "thumbnail": _thumb_n(fp) if ok else None,
            })


        return _clean({
            "status": "ok", "mode": "dataset", "name": name,
            "counts": {"found": found, "analyzed": int(len(df)), "readable": n_valid_all,
                       "unreadable": found - n_valid_all,
                       "classes": (len(full_counts) or _g(bal, "num_classes"))},
            "truncated": truncated,
            "sample_note": ("Analyzed {:,} of {:,} readable images (stratified by class). "
                            "Class counts and exact duplicates cover all {:,} files; "
                            "the readiness balance sub-score comes from the sample.").format(
                                int(len(df)), n_valid_all, found) if truncated else "",
            "unreadable_files": unreadable,
            "readiness": readiness, "quality": quality, "balance": balance,
            "duplicates": duplicates, "outliers": outliers, "sizes": sizes, "color": color,
            "gallery": gallery,
            "warnings": _txts(list(_g(rep, "errors", []) or []) + list(_g(rep, "warnings", []) or [])),
            "seconds": round(time.time() - t, 2),
        })
    except Exception as e:
        return {"status": "error", "mode": "dataset", "name": name,
                "message": type(e).__name__ + ": " + str(e)[:200]}


# ------------------------------------------------ cheap first pass for big datasets
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic"}


def _scan_images(root):
    out = []
    for p in Path(root).rglob("*"):
        try:
            if (p.is_file() and p.suffix.lower() in _IMG_EXTS
                    and not any(x.startswith(".") or x == "__MACOSX" for x in p.relative_to(root).parts)):
                out.append(p)
        except OSError:
            pass
    return out


def _label_of(p, root):
    return p.parent.name if p.parent != Path(root) else "(unlabeled)"


def _probe_file(p):
    """sha256 + cheap readability check (one pass over the file, no full decode)."""
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    ok = True
    if Path(p).suffix.lower() != ".heic":
        try:
            from PIL import Image
            with Image.open(p) as im:
                im.verify()
        except Exception:
            ok = False
    return h.hexdigest(), ok


def build_dataset_report(root, name=None, max_images=300):
    """Small sets: full analysis. Big sets: sha256 + class counts on ALL files, then the
    full load / pHash / quality / analyzer only on a stratified sample. Never raises."""
    t = time.time()
    root = Path(root)
    name = name or root.name
    try:
        files = _scan_images(root)
        if len(files) <= max_images:
            return _build_dataset_report_inner(root, name, max_images)

        import math, os, shutil as _sh, tempfile
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor
        from src.sampling import stratified_sample

        with ThreadPoolExecutor(8) as ex:
            probes = list(ex.map(_probe_file, files))
        full = pd.DataFrame([{"file_path": str(p), "label": _label_of(p, root), "sha": s, "ok": ok}
                             for p, (s, ok) in zip(files, probes)])
        found = int(len(full))
        ok_mask = full["ok"].astype(bool)
        n_valid_all = int(ok_mask.sum())
        if n_valid_all < 2:
            return {"status": "error", "mode": "dataset", "name": name,
                    "message": "Need at least 2 readable images (found %d)." % n_valid_all}
        n_dup = int(full.loc[ok_mask, "sha"].duplicated().sum())
        unreadable = [_rel(x, root) for x in full.loc[~ok_mask, "file_path"].tolist()[:20]]
        full_counts = {str(k): int(v) for k, v in full.loc[ok_mask, "label"].value_counts().to_dict().items()}
        sample = stratified_sample(full.loc[ok_mask], max_images)

        tmp = Path(tempfile.mkdtemp(prefix="nydra_sample_"))
        try:
            for fp in sample["file_path"].tolist():
                src = Path(fp)
                dst = tmp / src.relative_to(root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, dst)
                except OSError:
                    _sh.copy2(src, dst)
            rep = _build_dataset_report_inner(tmp, name, max_images)
        finally:
            _sh.rmtree(tmp, ignore_errors=True)

        if not isinstance(rep, dict) or rep.get("status") != "ok":
            return rep
        analyzed = int((rep.get("counts") or {}).get("analyzed") or len(sample))
        rep["counts"] = {"found": found, "analyzed": analyzed, "readable": n_valid_all,
                         "unreadable": found - n_valid_all, "classes": len(full_counts)}
        rep["truncated"] = True
        rep["unreadable_files"] = unreadable
        rep.setdefault("duplicates", {})["exact_flagged"] = n_dup

        vals = list(full_counts.values())
        ratio = round(max(vals) / max(min(vals), 1), 2)
        bal_score = round(100.0 * min(vals) / max(max(vals), 1), 1)
        rep["balance"] = dict(rep.get("balance") or {}, class_counts=full_counts, num_classes=len(vals),
                              imbalance_ratio=ratio, balance_score=bal_score,
                              majority=max(full_counts, key=full_counts.get),
                              minority=min(full_counts, key=full_counts.get),
                              suggested_augmentation={}, recommendations=[])
        rd = rep.setdefault("readiness", {})
        rd.setdefault("sub_scores", {})["balance"] = bal_score
        penalty = 0.0
        if ratio >= 4 and isinstance(rd.get("score"), (int, float)):
            penalty = min(15.0, 3.0 * math.log2(ratio))
            rd["score"] = round(max(0.0, rd["score"] - penalty), 2)
        rd["balance_penalty"] = round(penalty, 1)
        rep["sample_note"] = ("Analyzed {:,} of {:,} readable images (stratified by class). Class counts, "
                              "exact duplicates and readability cover all {:,} files. The balance sub-score "
                              "uses the full class counts; imbalance of 4x or more lowers the readiness "
                              "score by up to 15 points.").format(analyzed, n_valid_all, found)
        rep["seconds"] = round(time.time() - t, 2)
        return rep
    except Exception as e:
        return {"status": "error", "mode": "dataset", "name": name,
                "message": type(e).__name__ + ": " + str(e)}
