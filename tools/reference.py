"""Python PatchCore reference, parity check and fixture generator.

Usage:  tools/.venv/Scripts/python tools/reference.py

1. Generates procedural woven scenes (good + planted defects).
2. Checks exported ONNX backbones against PyTorch (max abs error <= 1e-3).
3. Runs the full teach/detect reference and asserts every defect is flagged.
4. Writes tests/fixtures/ (math + parity fixtures), public/samples/ and the
   bundled example profile in public/profiles/.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as pc  # noqa: E402
import pcprof  # noqa: E402
from export_backbone import MODELS_DIR, build_module  # noqa: E402
from texture import scene  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
SAMPLES = ROOT / "public" / "samples"
PROFILES = ROOT / "public" / "profiles"
DEFAULT_ROI = {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}
PARITY_TOL = 1e-3

TEACH = [("good", s) for s in range(100, 107)]  # 7 frames -> 5 bank + 2 held out
TESTS = [("good", 200), ("good", 201), ("good", 202),
         ("dropped_tow", 300), ("foreign_patch", 301), ("wrinkle", 302),
         ("dropped_tow", 303), ("foreign_patch", 304), ("wrinkle", 305)]
MATH_TESTS = ["good_200", "dropped_tow_300", "foreign_patch_301", "wrinkle_302"]
PROFILE_TEACH = [("good", s) for s in range(400, 415)]  # 15 frames, the spec default


def name_of(kind, seed):
    return f"{kind}_{seed}"


def save_scene(kind, seed, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name_of(kind, seed)}.jpg"
    Image.fromarray(scene(kind, seed)).save(p, quality=95)
    return p


def crop_resize(img: Image.Image, size: int, roi=DEFAULT_ROI) -> np.ndarray:
    """ROI crop + resize to the backbone input. Returns uint8 HWC."""
    w, h = img.size
    box = (roi["x"] * w, roi["y"] * h, (roi["x"] + roi["w"]) * w, (roi["y"] + roi["h"]) * h)
    return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR, box=box))


def to_chw(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


class Backbone:
    def __init__(self, entry: dict):
        self.entry = entry
        self.size = entry["inputSize"]
        self.grid = entry["grid"]
        self.sess = ort.InferenceSession(str(MODELS_DIR / entry["file"]), providers=["CPUExecutionProvider"])

    def __call__(self, u8: np.ndarray) -> np.ndarray:
        return self.sess.run(["features"], {"image": to_chw(u8)})[0]


def teach(features_per_frame: list[np.ndarray], grid: int, size: int, margin=pc.DEFAULT_MARGIN):
    n_hold = pc.holdout_count(len(features_per_frame))
    bank_src = np.concatenate(features_per_frame[:-n_hold], 0)
    held = features_per_frame[-n_hold:]
    k = pc.coreset_target(bank_src.shape[0])
    idx = pc.greedy_coreset(bank_src, k)
    bank = bank_src[idx].astype(np.float32)
    sq = (bank.astype(np.float64) ** 2).sum(1).astype(np.float32)
    held_scores, held_patch = [], []
    for f in held:
        d = pc.knn_min_dist(f, bank, sq)
        held_patch.append(d)
        held_scores.append(float(pc.score_map(d, grid, size).max()))
    max_held = max(held_scores)
    return dict(idx=idx, bank=bank, sqnorm=sq, heldScores=held_scores,
                maxHeld=max_held, threshold=max_held * margin,
                heatFloor=float(np.median(np.concatenate(held_patch))), nHold=n_hold)


def detect(f, prof, grid, size):
    d = pc.knn_min_dist(f, prof["bank"], prof["sqnorm"])
    return d, float(pc.score_map(d, grid, size).max())


def check_parity(entry: dict, images: list[np.ndarray]) -> float:
    module = build_module(entry["id"])
    if entry["dtype"] == "fp16":
        pass  # compared against the fp32 PyTorch module: weight rounding is what we measure
    bb = Backbone(entry)
    worst = 0.0
    with torch.no_grad():
        for u8 in images:
            ref = module(torch.from_numpy(to_chw(u8))).numpy()
            got = bb(u8)
            assert got.shape == (entry["grid"] ** 2, entry["D"]), got.shape
            worst = max(worst, float(np.abs(ref - got).max()))
    return worst


def f16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float32)


def main():
    manifest = json.loads((MODELS_DIR / "models.json").read_text())
    entries = {m["key"]: m for m in manifest["models"]}
    r18 = entries["r18_224"]

    if FIX.exists():
        shutil.rmtree(FIX)
    (FIX / "scenes").mkdir(parents=True)

    teach_paths = [save_scene(k, s, FIX / "scenes") for k, s in TEACH]
    test_paths = {name_of(k, s): save_scene(k, s, FIX / "scenes") for k, s in TESTS}
    load = lambda p, size: crop_resize(Image.open(p), size)  # noqa: E731

    # ---- 1. ONNX vs PyTorch parity for every exported backbone -----------------
    parity_imgs_scene = [teach_paths[0], test_paths["foreign_patch_301"]]
    parity = {}
    for key, entry in entries.items():
        imgs = [load(p, entry["inputSize"]) for p in parity_imgs_scene]
        err = check_parity(entry, imgs)
        parity[key] = err
        status = "OK" if err <= PARITY_TOL else "FAIL"
        print(f"parity {key}: max abs err {err:.2e} {status}")
        if entry["dtype"] == "fp32":
            assert err <= PARITY_TOL, f"{key} ONNX/PyTorch mismatch {err}"
    (FIX / "parity_report.json").write_text(json.dumps(parity, indent=2))

    # ---- 2. Math fixtures on fp16-rounded r18@224 features --------------------
    bb = Backbone(r18)
    size, grid = bb.size, bb.grid
    teach_feats = [f16(bb(load(p, size))) for p in teach_paths]
    prof = teach(teach_feats, grid, size)
    n_hold = prof["nHold"]
    np.concatenate(teach_feats[:-n_hold], 0).astype("<f2").tofile(FIX / "teach_bank_src.f16")
    np.concatenate(teach_feats[-n_hold:], 0).astype("<f2").tofile(FIX / "heldout.f16")
    prof["idx"].astype("<i4").tofile(FIX / "coreset_idx.i32")

    math_tests = []
    for name in MATH_TESTS:
        f = f16(bb(load(test_paths[name], size)))
        f.astype("<f2").tofile(FIX / f"test_{name}.f16")
        d, s = detect(f, prof, grid, size)
        smap = pc.score_map(d, grid, size)
        math_tests.append({"name": name, "minDist": d.tolist(), "imageScore": s,
                           "argmax": int(np.argmax(smap)),
                           "smoothedSample": [float(smap[y, x]) for y, x in
                                              [(0, 0), (17, 200), (112, 112), (223, 5), (223, 223)]]})

    # ONNX Runtime Web parity input: raw uint8 HWC + fp32 features.
    for tag, p in [("teach0", teach_paths[0]), ("foreign", test_paths["foreign_patch_301"])]:
        u8 = load(p, size)
        u8.tofile(FIX / f"parity_{tag}.u8")
        bb(u8).astype("<f4").tofile(FIX / f"parity_{tag}.f32")

    # ---- 3. Full-precision end-to-end verdicts --------------------------------
    full_feats = [bb(load(p, size)) for p in teach_paths]
    full = teach(full_feats, grid, size)
    verdicts = []
    for k, s in TESTS:
        name = name_of(k, s)
        _, score = detect(bb(load(test_paths[name], size)), full, grid, size)
        flagged = score > full["threshold"]
        verdicts.append({"name": name, "kind": k, "imageScore": score, "defect": bool(flagged)})
        print(f"  {name:20s} score {score:7.3f}  thr {full['threshold']:.3f}  "
              f"{'DEFECT' if flagged else 'PASS'}")
        assert flagged == (k != "good"), f"reference misclassified {name}"

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {k: r18[k] for k in ("key", "id", "file", "inputSize", "grid", "D", "sha256")},
        "roi": DEFAULT_ROI,
        "teach": [p.name for p in teach_paths],
        "tests": [{"file": f"{v['name']}.jpg", **v} for v in verdicts],
        "math": {
            "teachPatches": int(sum(f.shape[0] for f in teach_feats[:-n_hold])),
            "heldoutPatches": int(sum(f.shape[0] for f in teach_feats[-n_hold:])),
            "coresetSeed": pc.CORESET_SEED,
            "coresetK": int(len(prof["idx"])),
            "margin": pc.DEFAULT_MARGIN,
            "heldScores": prof["heldScores"],
            "threshold": prof["threshold"],
            "heatFloor": prof["heatFloor"],
            "tests": math_tests,
        },
        "fullPrecision": {"threshold": full["threshold"], "heldScores": full["heldScores"],
                          "heatFloor": full["heatFloor"]},
    }
    (FIX / "meta.json").write_text(json.dumps(meta) + "\n")

    # ---- 4. Public samples (benchmark workload, attract loop, demo image source)
    if SAMPLES.exists():
        shutil.rmtree(SAMPLES)
    SAMPLES.mkdir(parents=True)
    sample_list = []
    for p in teach_paths:
        shutil.copy(p, SAMPLES / p.name)
        sample_list.append({"file": p.name, "kind": "good", "role": "teach"})
    for v in verdicts:
        shutil.copy(test_paths[v["name"]], SAMPLES / f"{v['name']}.jpg")
        sample_list.append({"file": f"{v['name']}.jpg", "kind": v["kind"], "role": "test"})
    (SAMPLES / "samples.json").write_text(json.dumps({"images": sample_list}, indent=2) + "\n")

    # ---- 5. Bundled example profile (15 frames, spec defaults) -----------------
    PROFILES.mkdir(parents=True, exist_ok=True)
    pfeats, first = [], None
    for k, s in PROFILE_TEACH:
        img = Image.fromarray(scene(k, s))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=95)
        img = Image.open(io.BytesIO(buf.getvalue()))
        u8 = crop_resize(img, size)
        first = first if first is not None else u8
        pfeats.append(bb(u8))
    pr = teach(pfeats, grid, size)
    bank16 = f16(pr["bank"])
    sq16 = (bank16.astype(np.float64) ** 2).sum(1).astype(np.float32)
    tb = io.BytesIO()
    Image.fromarray(first).resize((128, 128)).save(tb, "JPEG", quality=85)
    header = {
        "name": "Carbon twill (demo)",
        "notes": "Procedural 2x2 twill. Teach set generated by tools/reference.py.",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "appVersion": "0.1.0",
        "backboneId": "r18", "modelKey": "r18_224", "inputSize": size, "modelSha256": r18["sha256"],
        "threshold": pr["threshold"], "margin": pc.DEFAULT_MARGIN, "heatFloor": pr["heatFloor"],
        "roi": DEFAULT_ROI,
        "camera": {"label": "synthetic", "width": 448, "height": 448, "settings": {}},
        "rig": "procedural",
        "teachStats": {"frames": len(PROFILE_TEACH), "heldOut": pr["nHold"],
                       "heldScores": pr["heldScores"], "teachPatches": int(sum(f.shape[0] for f in pfeats[:-pr['nHold']]))},
    }
    pcprof.write(PROFILES / "carbon-twill-demo.pcprof", header, bank16, sq16, tb.getvalue())
    (PROFILES / "profiles.json").write_text(json.dumps({"profiles": [
        {"file": "carbon-twill-demo.pcprof", "name": header["name"], "modelKey": "r18_224"}]}, indent=2) + "\n")
    h2, b2, _, _ = pcprof.read(PROFILES / "carbon-twill-demo.pcprof")
    assert h2["M"] == bank16.shape[0] and np.array_equal(b2, bank16)
    print(f"bundled profile: M={h2['M']} threshold={pr['threshold']:.3f}")
    print("fixtures written to", FIX)


if __name__ == "__main__":
    main()
