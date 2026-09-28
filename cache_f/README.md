# cache_f

Training and validation cache for Model F, stored with Git LFS. Built by
`python Model_F/bank_f.py` then `python Model_F/pregen_f.py` (default `--split train val`,
seed 0) at commit `2dbc6ae`. Logs: `tests29/logs/02_bank_f.log`, `tests29/logs/03_pregen_f.log`.

`train/leak.npy` (3.9 GB) is over the Git LFS per-file size limit, so it is stored as four parts
(`train/leak.npy.part00` to `part03`, split at 1000 MiB). Rebuild it before training:

```bash
cat cache_f/train/leak.npy.part0* > cache_f/train/leak.npy       # Git Bash / Linux / macOS
```

```bat
copy /b cache_f\train\leak.npy.part00+cache_f\train\leak.npy.part01+cache_f\train\leak.npy.part02+cache_f\train\leak.npy.part03 cache_f\train\leak.npy
```

SHA-256:

| File | SHA-256 |
|---|---|
| `train/leak.npy` (rebuilt) | `c03d86e90a643e720c035dfbba2f835636c27a0e3237a76476627df2b843ba3b` |
| `train/labels.npy` | `e5b5094b38235d42a8f445508d5418693aaae5b1a970ed250a7a7c45899f49a8` |
| `train/leak_row.npy` | `9aa1cba796a7045489e0fddfc189ab8988c4334e1150095ae474b1d9cdcfdbe4` |
| `val/leak.npy` | `44df9c1bb3644cdabdd630bfde95c3ec32548bc4bd9202ba499debe3e6b45df9` |
| `val/labels.npy` | `93f95c3e28cd1303d9d74c481cb3342b92b79b0e4cc434b92c3f22938c23d375` |
| `val/leak_row.npy` | `fd3397e55b960b755209dac31e9354ecff5c55de32ed3f20c2a28cedb73f2ac5` |
| `bank.npz` | `ed3b0c766d6a73e3f9fed58424674f78b733679f7692e8bfe7e9e7c0c4b83bf8` |

`bank.npz` holds band-limited windows from the public datasets and the 6 Mendeley Branched no-leak
recordings (backgrounds only). Its licence terms are those of the source datasets.
