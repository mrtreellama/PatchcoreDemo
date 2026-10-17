"""Export PatchCore backbones and the kNN distance graph to ONNX.

Usage:  tools/.venv/Scripts/python tools/export_backbone.py [--only r18|wrn50|knn]

Writes public/models/*.onnx and public/models/models.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from onnx import numpy_helper, helper, TensorProto

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "public" / "models"
EXPORT_VERSION = 1
OPSET = 17
PROJECTION_SEED = 1234

BACKBONES = {
    "r18": dict(
        name="ResNet-18",
        ctor=lambda: torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
        channels=128 + 256,
        out_dim=384,
        sizes=[224, 320],
    ),
    "wrn50": dict(
        name="Wide ResNet-50-2",
        ctor=lambda: torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        ),
        channels=512 + 1024,
        out_dim=512,
        sizes=[224, 320],
    ),
}


class PatchFeatures(nn.Module):
    """image [1,3,H,W] in 0..1 RGB -> PatchCore patch features [N, D]."""

    def __init__(self, net: nn.Module, channels: int, out_dim: int):
        super().__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        if channels > out_dim:
            g = torch.Generator().manual_seed(PROJECTION_SEED)
            proj = torch.randn(channels, out_dim, generator=g) / out_dim**0.5
            self.register_buffer("proj", proj)
        else:
            self.proj = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.mean) / self.std
        x = self.stem(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        p2 = self.pool(l2)
        p3 = self.pool(l3)
        p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([p2, p3], dim=1)[0]  # [C, h, w]
        c = feats.shape[0]
        feats = feats.permute(1, 2, 0).reshape(-1, c)  # row-major over grid
        if self.proj is not None:
            feats = feats @ self.proj
        return feats


class KnnMinDist(nn.Module):
    def forward(self, features, bank, bank_sqnorm):
        fsq = (features * features).sum(dim=1, keepdim=True)
        d2 = fsq - 2.0 * (features @ bank.t()) + bank_sqnorm.unsqueeze(0)
        d2 = torch.clamp(d2, min=0.0)
        return torch.sqrt(d2.min(dim=1).values)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_module(bid: str) -> PatchFeatures:
    spec = BACKBONES[bid]
    m = PatchFeatures(spec["ctor"](), spec["channels"], spec["out_dim"])
    return m.eval()


def export_backbone(bid: str, size: int, path: Path) -> None:
    m = build_module(bid)
    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(
        m, (dummy,), str(path), dynamo=False, external_data=False, opset_version=OPSET,
        input_names=["image"], output_names=["features"], do_constant_folding=True,
    )
    onnx.checker.check_model(str(path))


def to_fp16_weights(src: Path, dst: Path, min_elems: int = 1024) -> None:
    """Store large float initializers as fp16 and Cast them back to fp32 in-graph.

    Compute stays fp32 (WASM has no fp16 kernels), only the download shrinks.
    """
    model = onnx.load(str(src))
    g = model.graph
    casts = []
    new_inits = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            arr = numpy_helper.to_array(init).astype(np.float16)
            half_name = init.name + "__fp16"
            new_inits.append(numpy_helper.from_array(arr, half_name))
            casts.append(helper.make_node("Cast", [half_name], [init.name], to=TensorProto.FLOAT))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def export_knn(path: Path) -> None:
    f = torch.rand(10, 8)
    b = torch.rand(20, 8)
    bn = (b * b).sum(1)
    torch.onnx.export(
        KnnMinDist(), (f, b, bn), str(path), dynamo=False, external_data=False, opset_version=OPSET,
        input_names=["features", "bank", "bank_sqnorm"], output_names=["min_dist"],
        dynamic_axes={"features": {0: "N", 1: "D"}, "bank": {0: "M", 1: "D"},
                      "bank_sqnorm": {0: "M"}, "min_dist": {0: "N"}},
    )
    onnx.checker.check_model(str(path))


def load_manifest() -> dict:
    p = MODELS_DIR / "models.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"exportVersion": EXPORT_VERSION, "models": [], "knn": None}


def upsert(manifest: dict, entry: dict) -> None:
    manifest["models"] = [m for m in manifest["models"] if m["key"] != entry["key"]] + [entry]
    order = {"r18": 0, "wrn50": 1}
    manifest["models"].sort(key=lambda m: (order.get(m["id"], 9), m["inputSize"], m["dtype"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["r18", "wrn50", "knn"], action="append")
    args = ap.parse_args()
    todo = args.only or ["r18", "knn"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    if "knn" in todo:
        p = MODELS_DIR / "knn.onnx"
        export_knn(p)
        manifest["knn"] = {"file": "knn.onnx", "sha256": sha256(p)}
        print("knn.onnx", p.stat().st_size, "bytes")

    for bid in [b for b in ("r18", "wrn50") if b in todo]:
        spec = BACKBONES[bid]
        for size in spec["sizes"]:
            fname = f"backbone_{bid}_{size}.onnx"
            p = MODELS_DIR / fname
            export_backbone(bid, size, p)
            variants = [("fp32", p)]
            if bid == "wrn50":
                # FP32 is over the Pages budget; ship FP16 weights if they pass parity in reference.py.
                p16 = MODELS_DIR / f"backbone_{bid}_{size}_fp16.onnx"
                to_fp16_weights(p, p16)
                variants = [("fp32", p), ("fp16", p16)]
            for dtype, vp in variants:
                mb = vp.stat().st_size / 1e6
                print(f"{vp.name}: {mb:.1f} MB")
                upsert(manifest, {
                    "key": vp.stem.replace("backbone_", ""),
                    "id": bid,
                    "name": f"{spec['name']} @ {size}" + (" (FP16 weights)" if dtype == "fp16" else ""),
                    "file": vp.name,
                    "inputSize": size,
                    "grid": size // 8,
                    "D": spec["out_dim"],
                    "dtype": dtype,
                    "bytes": vp.stat().st_size,
                    "sha256": sha256(vp),
                    "exportVersion": EXPORT_VERSION,
                })

    (MODELS_DIR / "models.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote models.json")


if __name__ == "__main__":
    main()
