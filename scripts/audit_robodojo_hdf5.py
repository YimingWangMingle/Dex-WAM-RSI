#!/usr/bin/env python3
"""Audit a few real RoboDojo HDF5 episodes before writing the field map.

Reports structure and evidence only; it never decides which field is RGB,
depth, state or action. Output: JSON (machine) + Markdown (review).

  python scripts/audit_robodojo_hdf5.py --files ep0.hdf5 ep1.hdf5 --out tmp/robodojo_audit
  python scripts/audit_robodojo_hdf5.py --root /path/to/robodojo --max-files 5 --out ... \
      --depth-field obs/depth/head --depth-unit-to-mm 1000

Per dataset: shape, dtype, chunks/compression, attrs, sampled min/max/mean,
first/last rows; 1-D numeric series are checked for monotonicity (timestamp
candidates) and their median step. Datasets are grouped by first dimension
to show which streams share a time axis. Across files, keys and trailing
shapes are compared. With --depth-field, the valid-pixel rate (finite,
> 0, not in --depth-invalid) is reported per sampled frame.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _sample_rows(n: int, k: int = 5):
    if n <= k:
        return list(range(n))
    return sorted({0, n - 1, *np.linspace(0, n - 1, k).astype(int).tolist()})


def _attrs(obj):
    out = {}
    for k, v in obj.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist() if v.size <= 16 else f"<array {v.shape} {v.dtype}>"
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v if isinstance(v, (str, int, float, bool, list, type(None))) else repr(v)
    return out


def audit_dataset(name, ds):
    info = {"path": name, "shape": list(ds.shape), "dtype": str(ds.dtype),
            "chunks": list(ds.chunks) if ds.chunks else None,
            "compression": ds.compression, "attrs": _attrs(ds)}
    if ds.dtype.kind in "OSV":
        first = ds[0] if ds.shape and ds.shape[0] else None
        info["kind"] = "bytes_or_object"
        if first is not None:
            b = bytes(first) if not isinstance(first, (bytes, str)) else first
            info["first_item_len"] = len(b)
            info["first_item_magic"] = b[:4].hex() if isinstance(b, bytes) else None
        return info
    if ds.dtype.kind not in "biuf" or not ds.shape:
        return info
    rows = _sample_rows(int(ds.shape[0]))
    x = np.asarray(ds[rows], dtype=np.float64)
    info["sample_rows"] = rows
    info["sample_min"] = float(np.nanmin(x)) if x.size else None
    info["sample_max"] = float(np.nanmax(x)) if x.size else None
    info["sample_mean"] = float(np.nanmean(x)) if x.size else None
    info["nonfinite_in_sample"] = int((~np.isfinite(x)).sum())
    if ds.ndim == 2 and ds.shape[1] <= 64:
        info["first_row"] = np.asarray(ds[0], dtype=np.float64).round(5).tolist()
        info["last_row"] = np.asarray(ds[-1], dtype=np.float64).round(5).tolist()
    if ds.ndim == 1 and ds.shape[0] > 1:
        full = np.asarray(ds[()], dtype=np.float64)
        d = np.diff(full)
        info["monotonic_increasing"] = bool((d > 0).all())
        info["median_step"] = float(np.median(d))
        info["step_min_max"] = [float(d.min()), float(d.max())]
    if ds.ndim in (3, 4) and ds.dtype == np.uint8:
        info["kind"] = "image_like_uint8"
    return info


def depth_validity(ds, unit_to_mm, invalid):
    rows = _sample_rows(int(ds.shape[0]), 8)
    out = []
    for r in rows:
        d = np.asarray(ds[r], dtype=np.float64)
        ok = np.isfinite(d) & (d > 0)
        for v in invalid:
            ok &= d != v
        mm = d[ok] * unit_to_mm
        out.append({"row": r, "valid_rate": float(ok.mean()),
                    "mm_p01_p50_p99": (np.percentile(mm, [1, 50, 99]).round(1).tolist()
                                       if mm.size else None)})
    return out


def audit_file(path, depth_fields, unit_to_mm, invalid):
    import h5py
    rep = {"file": str(path), "datasets": [], "groups": {}}
    with h5py.File(path, "r") as f:
        rep["root_attrs"] = _attrs(f)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rep["datasets"].append(audit_dataset(name, obj))
            else:
                rep["groups"][name] = _attrs(obj)

        f.visititems(visit)
        by_len = defaultdict(list)
        for d in rep["datasets"]:
            if d["shape"]:
                by_len[d["shape"][0]].append(d["path"])
        rep["first_dim_groups"] = {str(k): v for k, v in sorted(by_len.items())}
        rep["depth"] = {name: depth_validity(f[name], unit_to_mm, invalid)
                        for name in depth_fields if name in f}
    return rep


def cross_file(reports):
    keys = [{d["path"]: (d["dtype"], d["shape"][1:]) for d in r["datasets"]} for r in reports]
    common = set.intersection(*(set(k) for k in keys)) if keys else set()
    union = set.union(*(set(k) for k in keys)) if keys else set()
    shape_diff = sorted(p for p in common if len({k[p] for k in keys}) > 1)
    return {"common_datasets": sorted(common), "not_in_all_files": sorted(union - common),
            "trailing_shape_or_dtype_differs": shape_diff}


def to_markdown(reports, xf):
    lines = ["# RoboDojo HDF5 audit", "", "Evidence only; field roles are decided by a human "
             "in the field map.", ""]
    for r in reports:
        lines += [f"## {r['file']}", "", f"root attrs: `{json.dumps(r['root_attrs'])[:400]}`", "",
                  "| path | shape | dtype | min | max | notes |", "|---|---|---|---|---|---|"]
        for d in r["datasets"]:
            notes = []
            if d.get("monotonic_increasing"):
                notes.append(f"monotonic, step~{d['median_step']:.4g}")
            if d.get("kind"):
                notes.append(d["kind"])
            if d.get("nonfinite_in_sample"):
                notes.append(f"nonfinite={d['nonfinite_in_sample']}")
            fmt = lambda v: "" if v is None else f"{v:.4g}"  # noqa: E731
            lines.append(f"| `{d['path']}` | {d['shape']} | {d['dtype']} | {fmt(d.get('sample_min'))} "
                         f"| {fmt(d.get('sample_max'))} | {'; '.join(notes)} |")
        lines += ["", "first-dim groups (shared time axis candidates):"]
        lines += [f"- {n}: {', '.join(f'`{p}`' for p in ps)}" for n, ps in r["first_dim_groups"].items()]
        for name, rows in r["depth"].items():
            lines += ["", f"depth `{name}` validity:"] + [f"- row {x['row']}: {x['valid_rate']:.3f} "
                                                        f"mm p1/p50/p99 {x['mm_p01_p50_p99']}" for x in rows]
        lines.append("")
    lines += ["## cross-file", "", f"not in all files: {xf['not_in_all_files']}",
              f"trailing shape/dtype differs: {xf['trailing_shape_or_dtype_differs']}", "",
              "## still to confirm by a human (do not infer from this table)", "",
              "- left/right arm order and gripper definition (open/close sign, range)",
              "- absolute vs delta action; joint vs EEF; units",
              "- whether action row t is the command executed between observation t and t+1",
              "- measured state vs control target (separate statistics, never reuse Aloha stats)",
              "- camera ids and the RGB/depth pairing per view", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--root")
    ap.add_argument("--max-files", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-field", action="append", default=[])
    ap.add_argument("--depth-unit-to-mm", type=float, default=1.0)
    ap.add_argument("--depth-invalid", type=float, action="append", default=[])
    a = ap.parse_args()
    files = [Path(p) for p in a.files]
    if a.root:
        files += sorted(Path(a.root).rglob("*.hdf5"))[: a.max_files]
        files += sorted(Path(a.root).rglob("*.h5"))[: max(0, a.max_files - len(files))]
    if not files:
        raise SystemExit("no HDF5 files given")
    reports = [audit_file(p, a.depth_field, a.depth_unit_to_mm, a.depth_invalid) for p in files]
    xf = cross_file(reports)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps({"files": reports, "cross_file": xf}, indent=2,
                                               ensure_ascii=False))
    (out / "audit.md").write_text(to_markdown(reports, xf), encoding="utf-8")
    print(f"wrote {out / 'audit.json'} and {out / 'audit.md'} for {len(files)} files")


if __name__ == "__main__":
    main()
