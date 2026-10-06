"""
rigs.py — windows from two independent laboratory rigs, plus the fixed splits
===============================================================================
Loaders for the Version 2 cross-rig benchmark (docs/VERSION2_CROSS_RIG_PREREG.md):
Mendeley testbed (accelerometer / hydrophone) and the Sheffield rig. Every rig
goes through ONE per-stream path (`stream_windows`): demean, 5 kHz, 2 kHz
low-pass, 0.4 s windows, drop dead windows, keep log10(RMS), scale to unit RMS.

  stream  = one sensor channel of one recording (a Sheffield CSV column, or a
            Mendeley A1/A2 channel). Windows are paired within a stream.
  group   = independence unit for splits and bootstrap (see the pre-registration).

Splits are pure functions of the group names (crc32), with no RNG.
Windows are cached in cache_xrig/ (git-ignored).
"""

import hashlib
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Model_F"))
import bank_f                                             # noqa: E402
import mendeley as M                                      # noqa: E402
import public_data as P                                   # noqa: E402
from _common import DATASETS, REPO_ROOT                   # noqa: E402

SHEFFIELD_ROOT = (DATASETS / "public" / "sheffield" / "extracted"
                  / "Acoustic Data (Leakage Experiments)" / "All_Data")
MENDELEY_ROOTS = {"accelerometer": DATASETS / "Accelerometer" / "Accelerometer",
                  "hydrophone": DATASETS / "Hydrophone" / "Hydrophone"}
MENDELEY_CACHE = REPO_ROOT / "cache_mendeley"
XRIG_CACHE = REPO_ROOT / "cache_xrig"
RIGS = ("mendeley_acc", "mendeley_hyd", "sheffield")
SHEFFIELD_FS = 8192
FIELDS = ("x", "log_rms", "y", "group", "stream", "meta")


@dataclass
class RigWindows:
    x: np.ndarray          # (N, 2000) float32, unit RMS
    log_rms: np.ndarray    # (N,) float32, log10 RMS before unit-RMS scaling
    y: np.ndarray          # (N,) int8, 1 = leak
    group: np.ndarray      # (N,) str, independence group
    stream: np.ndarray     # (N,) str, one sensor channel of one recording
    meta: np.ndarray       # (N,) str, descriptive
    n_dropped_dead: int = 0

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "RigWindows":
        return RigWindows(*(getattr(self, f)[mask] for f in FIELDS), n_dropped_dead=self.n_dropped_dead)

    def summary(self) -> dict:
        g_leak = set(self.group[self.y == 1])
        g_none = set(self.group[self.y == 0])
        return {"n_windows": len(self), "n_streams": len(set(self.stream)),
                "n_groups": len(g_leak | g_none), "n_groups_leak": len(g_leak),
                "n_groups_no_leak": len(g_none), "n_windows_leak": int((self.y == 1).sum()),
                "n_windows_no_leak": int((self.y == 0).sum()),
                "n_dropped_dead": int(self.n_dropped_dead)}


def stream_windows(x: np.ndarray, fs: int):
    """One stream -> (unit-RMS windows float32, log10 RMS float32, n dropped dead)."""
    w = P.windows_of(P.to_common(x, fs)).astype(np.float64)
    keep = bank_f.usable(w) if len(w) else np.zeros(0, bool)
    n_drop = int((~keep).sum())
    w = w[keep]
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    return (w / rms[:, None]).astype(np.float32), np.log10(rms).astype(np.float32), n_drop


def _pack(parts, n_dropped: int) -> RigWindows:
    """parts: list of (x, log_rms, label, group, stream, meta)."""
    parts = [p for p in parts if len(p[0])]
    return RigWindows(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
        np.concatenate([np.full(len(p[0]), p[2], np.int8) for p in parts]),
        np.concatenate([np.full(len(p[0]), p[3]) for p in parts]),
        np.concatenate([np.full(len(p[0]), p[4]) for p in parts]),
        np.concatenate([np.full(len(p[0]), p[5]) for p in parts]), n_dropped)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sheffield(root: Path = SHEFFIELD_ROOT, verbose: bool = True) -> RigWindows:
    """All_Data only (not Coherence). Duplicate no-leak files are skipped and printed."""
    import pandas as pd
    folders = sorted(d for d in Path(root).iterdir() if d.is_dir() and re.match(r"\(test#\d+\)", d.name))
    seen, skipped, parts, dropped = {}, [], [], 0
    for folder in folders:
        k = int(re.match(r"\(test#(\d+)\)", folder.name).group(1))
        for f in sorted(folder.glob("*.csv")):
            if "noleak" in f.stem.lower():
                h = _md5(f)
                if h in seen:
                    skipped.append(f"{folder.name}/{f.name} duplicates {seen[h]}")
                    continue
                seen[h] = f"{folder.name}/{f.name}"
                group, label = f"sh:noleak:{f.stem}", 0
            else:
                group, label = f"sh:test{k}", 1
            df = pd.read_csv(f, dtype=np.float32)
            t = df.iloc[:, 0].to_numpy(np.float64)
            fs = 1.0 / np.median(np.diff(t[:1000]))
            assert abs(fs - SHEFFIELD_FS) < 1, f"{f.name}: fs {fs:.1f} Hz, expected {SHEFFIELD_FS}"
            for col in df.columns:
                if not re.fullmatch(r"Acc\d+", col):
                    continue
                w, lr, nd = stream_windows(df[col].to_numpy(), SHEFFIELD_FS)
                dropped += nd
                parts.append((w, lr, label, group, f"{group}|{f.stem}|{col}", f"pos={int(col[3:])}"))
            del df
    assert len(seen) == 3, (f"expected 3 distinct no-leak recordings in Sheffield All_Data, "
                            f"found {len(seen)}: {sorted(seen.values())}")
    if verbose:
        print("Sheffield: skipped duplicate no-leak files:")
        for s in skipped:
            print("  ", s)
    return _pack(parts, dropped)


def load_mendeley(sensor: str = "accelerometer") -> RigWindows:
    """All recordings of both topologies; both channels share the group."""
    parts, dropped = [], 0
    for rec in M.discover(MENDELEY_ROOTS[sensor], sensor):
        x1, x2 = M.load_recording(rec, fs_out=P.FS, cache_dir=MENDELEY_CACHE)
        group = f"md:{rec.rec_id}"
        meta = f"{rec.topology}/{rec.condition}/{M.flow_condition(rec.rec_id)}"
        for tag, x in (("ch1", x1), ("ch2", x2)):
            w, lr, nd = stream_windows(x, P.FS)
            dropped += nd
            parts.append((w, lr, rec.label, group, f"{group}|{tag}", meta))
    return _pack(parts, dropped)


def _loader(name: str) -> RigWindows:
    if name == "sheffield":
        return load_sheffield()
    return load_mendeley("accelerometer" if name == "mendeley_acc" else "hydrophone")


def load_rig(name: str, cache: bool = True) -> RigWindows:
    assert name in RIGS, f"unknown rig {name!r}"
    path = XRIG_CACHE / f"{name}.npz"
    if cache and path.exists():
        z = np.load(path, allow_pickle=False)
        rw = RigWindows(*(z[f] for f in FIELDS), n_dropped_dead=int(z["n_dropped_dead"]))
    else:
        rw = _loader(name)
        if cache:
            XRIG_CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(path, **{f: getattr(rw, f) for f in FIELDS}, n_dropped_dead=rw.n_dropped_dead)
    print(f"{name}: {rw.summary()}")
    return rw


# ── splits (deterministic, no RNG) ─────────────────────────────────────────────

def _group_class(groups, y) -> dict:
    cls = {}
    for g, c in zip(np.asarray(groups), np.asarray(y)):
        assert cls.setdefault(str(g), int(c)) == int(c), f"group {g} has both classes"
    return cls


def _ordered(cls: dict, c: int) -> list:
    return sorted((g for g, v in cls.items() if v == c), key=lambda g: (zlib.crc32(g.encode()), g))


def val_split(groups, y) -> np.ndarray:
    """Boolean mask over windows: first ceil(0.2 n) groups per class, ordered by crc32."""
    cls = _group_class(groups, y)
    val = set()
    for c in (0, 1):
        order = _ordered(cls, c)
        if order:
            val.update(order[:max(1, math.ceil(0.2 * len(order)))])
    return np.isin(np.asarray(groups).astype(str), list(val))


def fold_assign(groups, y, k: int = 3) -> dict:
    """group -> fold; groups of each class dealt round-robin in crc32 order."""
    cls = _group_class(groups, y)
    out = {}
    for c in (0, 1):
        for i, g in enumerate(_ordered(cls, c)):
            out[g] = i % k
    return out


def youden_threshold(y, s) -> float:
    """Score maximising balanced accuracy for the rule score >= t (ties: smallest t)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
    cand = np.unique(s)
    srt_p, srt_n = np.sort(s[y == 1]), np.sort(s[y == 0])
    tpr = 1 - np.searchsorted(srt_p, cand, side="left") / max(len(srt_p), 1)
    tnr = np.searchsorted(srt_n, cand, side="left") / max(len(srt_n), 1)
    return float(cand[int(np.argmax(np.round((tpr + tnr) / 2, 12)))])   # rounding: exact ties
