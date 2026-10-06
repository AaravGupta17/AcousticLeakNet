"""
xrig_train.py — Model F recipe on ONE laboratory rig (Version 2 cross-rig benchmark)
======================================================================================
Protocol: docs/VERSION2_CROSS_RIG_PREREG.md. This is train_f.py with --real-frac 1.0:
same architecture, loss, optimiser, schedule, AMP, pairing [F5], EQ [F4] and joint
z-score [F2], but the only data is one rig's labelled windows. No synthetic cache and
no other rig is ever opened: `prepare_data` loads the training rig once and nothing else.

  --fold -1   cross-rig mode: all groups of the rig; train/val by rigs.val_split
  --fold f    within-rig mode: groups with rigs.fold_assign == f are held out as the
              TEST groups (never seen here); val_split is applied to the rest

Checkpoint score = window AUROC (logits) on the training rig's validation groups;
strict improvement only, so ties keep the earlier epoch. Afterwards the balanced-
accuracy (Youden) threshold on those validation windows is frozen into a JSON.

    python Model_F/xrig_train.py --train-rig mendeley_acc --seed 0
    python Model_F/xrig_train.py --train-rig sheffield --smoke
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))
import augment_f as A                                      # noqa: E402
import rigs                                                # noqa: E402
import train_f                                             # noqa: E402
from _common import MODELS_DIR, REPO_ROOT, _jsonable, build_model, record_run   # noqa: E402

TRAIN_RIGS = ("mendeley_acc", "sheffield")


class RigBank:
    """Same interface as train_f.Bank, built from RigWindows. The partner of a
    window is another window of the SAME STREAM (one sensor of one recording)."""

    def __init__(self, rw: "rigs.RigWindows"):
        self.single = rw.x.astype(np.float32)
        self.y = rw.y.astype(int)
        self.group = rw.group
        self.stream = rw.stream
        order = np.argsort(self.stream, kind="stable")
        s_sorted = self.stream[order]
        starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
        self._members = np.split(order, starts[1:])
        self._stream_of = np.empty(len(self.stream), int)
        for k, m in enumerate(self._members):
            self._stream_of[m] = k
        self.no_leak = np.flatnonzero(self.y == 0)
        self.leak = np.flatnonzero(self.y == 1)

    def partner(self, i: int, rng) -> int:
        m = self._members[self._stream_of[i]]
        return int(m[rng.integers(len(m))])

    def two_channel(self, i: int, rng) -> np.ndarray:
        return A.pair_channels(self.single[i], self.single[self.partner(i, rng)], rng)


class RealRowDataset(Dataset):
    """Real-row branch of train_f.MixDataset: 50/50 leak, uniform within the class."""

    def __init__(self, bank: RigBank, seed: int, length: int):
        if len(bank.leak) == 0 or len(bank.no_leak) == 0:
            raise ValueError("training needs both classes")
        self.bank, self.seed, self.length, self.epoch = bank, seed, length, 0

    def __len__(self):
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        want_leak = rng.random() < 0.5
        pool = self.bank.leak if want_leak else self.bank.no_leak
        x = self.bank.two_channel(int(rng.choice(pool)), rng)
        x = A.finish(x.astype(np.float64), rng)
        lab = np.array([float(want_leak), 0.0, 0.0, 0.0], np.float32)
        return torch.from_numpy(x), torch.from_numpy(lab), torch.tensor(0.0)


def prepare_data(args) -> dict:
    """Loads ONLY args.train_rig and splits it. Returns banks and group lists."""
    assert args.train_rig in TRAIN_RIGS, args.train_rig
    rw = rigs.load_rig(args.train_rig)
    test_groups = []
    if args.fold >= 0:
        fold = rigs.fold_assign(rw.group, rw.y)
        held = np.array([fold[str(g)] == args.fold for g in rw.group])
        test_groups = sorted(set(rw.group[held].tolist()))
        rw = rw.subset(~held)
    is_val = rigs.val_split(rw.group, rw.y)
    tr, va = rw.subset(~is_val), rw.subset(is_val)
    assert not (set(tr.group) & set(va.group)) and not (set(rw.group) & set(test_groups))
    return {"bank_tr": RigBank(tr), "bank_va": RigBank(va),
            "train_groups": sorted(set(tr.group.tolist())),
            "val_groups": sorted(set(va.group.tolist())), "test_groups": test_groups}


def val_tensor(bank_va: RigBank, seed: int) -> torch.Tensor:
    """Fixed validation inputs, built exactly as train_f builds xr_val."""
    rng = np.random.default_rng(seed + 2)
    return torch.from_numpy(np.stack([A.finish(bank_va.two_channel(i, rng).astype(np.float64), rng)
                                      for i in range(len(bank_va.single))]))


def youden(y, s):
    return rigs.youden_threshold(y, s)


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        return None


def _hardware(max_alloc_mb):
    hw = {"torch": torch.__version__, "cuda": torch.version.cuda,
          "max_memory_allocated_mb": max_alloc_mb}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        hw.update(gpu=p.name, gpu_total_mem_mb=p.total_memory / 2 ** 20)
    try:
        import psutil
        hw["system_ram_total_mb"] = psutil.virtual_memory().total / 2 ** 20
    except ImportError:
        pass
    return hw


def _mem_line(device):
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    rss = _rss_mb()
    return peak, f"gpu peak {peak:.0f} MB" + ("" if rss is None else f" | rss {rss:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rig", required=True, choices=TRAIN_RIGS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1, help="-1 = cross-rig; f>=0 = within-rig, hold out fold f")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1500, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="3 optimiser steps, batch 32, no files written")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "v2_cross_rig"))
    args = ap.parse_args()
    if args.smoke:
        args.batch, args.steps, args.epochs = 32, 3, 1

    fold_tag = "x" if args.fold < 0 else str(args.fold)
    stem = f"xrig_{args.train_rig}_fold{fold_tag}_seed{args.seed}"
    ckpt_path = MODELS_DIR / f"{stem}.pt"
    out_dir = Path(args.out_dir)
    if ckpt_path.exists() and not (args.overwrite or args.smoke):
        sys.exit(f"{ckpt_path.name} exists; pass --overwrite to replace it")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = prepare_data(args)
    bank_tr, bank_va = d["bank_tr"], d["bank_va"]
    print(f"Device: {device} | rig {args.train_rig} fold {args.fold} seed {args.seed}")
    print(f"Train: {len(bank_tr.single)} windows ({len(bank_tr.no_leak)} no-leak) in "
          f"{len(d['train_groups'])} groups | val: {len(bank_va.single)} windows in "
          f"{len(d['val_groups'])} groups | held-out test groups: {len(d['test_groups'])}")

    train_ds = RealRowDataset(bank_tr, args.seed, args.steps * args.batch)
    xv, yv = val_tensor(bank_va, args.seed), bank_va.y
    loader = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    cfg = {"base_channels": args.base_channels, "dropout": args.dropout, "fusion": "cca",
           "input_norm": "zscore", "band_hz": A.BAND_HZ, **vars(args),
           "recipe": "model_f_real_only", "fold": args.fold,
           "train_groups": d["train_groups"], "val_groups": d["val_groups"]}
    cfg = _jsonable(cfg)                    # repo-relative paths: no user folder names in records
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history, best, best_epoch = [], -1.0, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds.epoch = epoch                  # workers are re-spawned each epoch
        t0, losses = time.time(), []
        for step, (x, lab, is_syn) in enumerate(loader):
            x, lab, is_syn = x.to(device), lab.to(device), is_syn.to(device)
            if args.smoke:
                print(f"  batch {step}: x {tuple(x.shape)} {x.dtype} | lab {tuple(lab.shape)} "
                      f"{lab.dtype} | leak share {lab[:, 0].mean():.2f}")
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                det, pos, sev = model(x, torch.zeros(len(x), 11, device=device))
            loss, _ = train_f.loss_fn(det.float(), pos.float(), sev.float(), lab, is_syn)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.smoke:
                assert torch.isfinite(loss).item(), "loss is not finite"
                gok = all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                          if p.grad is not None)
                # on CUDA the GradScaler may legitimately see inf on early steps and skip them
                assert gok or device.type == "cuda", "non-finite gradients"
                print(f"  step {step}: loss {loss.item():.4f} | grads finite: {gok}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        auc = float(roc_auc_score(yv, train_f.logits_of(model, xv, device)))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auroc": auc})
        peak, mem = _mem_line(device)
        print(f"Ep {epoch:3d}/{args.epochs} | loss {np.mean(losses):.4f} | val AUROC {auc:.4f} | "
              f"{time.time() - t0:.0f}s | {mem}")
        if args.smoke:
            continue
        if auc > best:
            best, best_epoch = auc, epoch
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_score": auc, "history": history}, ckpt_path)
            print(f"  saved {ckpt_path.name}")

    if args.smoke:
        print("Smoke run finished: no checkpoint and no run record written.")
        return

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    s_val = train_f.logits_of(model, xv, device)
    thr = youden(yv, s_val)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    result = {"checkpoint": ckpt_path.name, "train_rig": args.train_rig, "fold": args.fold,
              "seed": args.seed, "best_epoch": best_epoch, "val_auroc": float(roc_auc_score(yv, s_val)),
              "threshold_val_youden": thr, "history": history, "train_groups": d["train_groups"],
              "val_groups": d["val_groups"], "cfg": cfg, "hardware": _hardware(peak),
              "wall_time_s": time.time() - t_start}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(_jsonable(result), indent=1, default=str))
    record_run(f"v2_xrig_train_{args.train_rig}_fold{args.fold}_seed{args.seed}", cfg, result,
               f"best epoch {best_epoch}: val AUROC {result['val_auroc']:.3f} on held-out groups "
               f"of {args.train_rig}; Youden threshold {thr:.3f}")


if __name__ == "__main__":
    main()
