#!/usr/bin/env python3
"""
HSE HPC pipeline: stego (mas_GRDH) vs clean SD (Genimage/ai) using DCT features + DeepSAD.

- Reads images from Yandex Object Storage (S3-compatible) via boto3.
- Avoids /tmp: honors $TMPDIR if set (tempfile will use it).
- Does NOT import diffusers / StableDiffusion wrappers.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from botocore.config import Config
import boto3
from botocore.exceptions import ClientError
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import center_crop, resize, to_tensor
from torch_dct import dct_2d

# DeepSAD implementation vendored in flipad
from sad.DeepSAD import DeepSAD
from sad.base.base_dataset import BaseADDataset


EPS_DCT = 1e-10
EPS_STD = 0.05


def sha1_u32(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def split_of_key(key: str, seed: int) -> str:
    v = sha1_u32(f"{seed}|{key}") % 100
    if v < 80:
        return "train"
    if v < 90:
        return "val"
    return "test"


def stego_source(key: str) -> str:
    # Expected: Stego/<source>/.../identity/00000000.png
    parts = key.split("/")
    if len(parts) > 2 and parts[0] == "Stego":
        return parts[1]
    return "unknown"


def round_robin_mix(keys_by_source: dict[str, list[str]], seed: int) -> list[str]:
    from collections import deque

    qs: list[deque[str]] = []
    for src, keys in sorted(keys_by_source.items()):
        salt = sha1_u32(f"{seed}|{src}")
        keys_sorted = sorted(keys, key=lambda k: hashlib.sha1(f"{salt}|{k}".encode()).hexdigest())
        qs.append(deque(keys_sorted))
    out: list[str] = []
    active = True
    while active:
        active = False
        for q in qs:
            if q:
                out.append(q.popleft())
                active = True
    return out


def make_s3_client(endpoint: str, region: str, access_key: str, secret_key: str):
    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return session.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            max_pool_connections=128,
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )

def s3_upload_file(s3, bucket: str, local_path: Path, key: str, logger: Optional[logging.Logger] = None):
    if logger:
        logger.info("upload -> s3://%s/%s (%s)", bucket, key, local_path.name)
    extra_args = {}
    # Best-effort content type
    if local_path.suffix == ".json":
        extra_args["ContentType"] = "application/json; charset=utf-8"
    elif local_path.suffix in (".log", ".txt", ".jsonl"):
        extra_args["ContentType"] = "text/plain; charset=utf-8"
    try:
        if extra_args:
            s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args)
        else:
            s3.upload_file(str(local_path), bucket, key)
    except ClientError as e:
        if logger:
            logger.error("upload failed for %s: %s", local_path, e)
        raise

def upload_dir(
    *,
    s3,
    bucket: str,
    local_dir: Path,
    prefix: str,
    include_suffixes: tuple[str, ...],
    logger: Optional[logging.Logger] = None,
):
    prefix = prefix.strip("/")
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        if include_suffixes and p.suffix not in include_suffixes:
            continue
        rel = p.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        s3_upload_file(s3, bucket=bucket, local_path=p, key=key, logger=logger)


def iter_keys(
    s3,
    bucket: str,
    prefix: str,
    limit: Optional[int] = None,
) -> Iterator[str]:
    paginator = s3.get_paginator("list_objects_v2")
    seen = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        for obj in page.get("Contents", []):
            yield obj["Key"]
            seen += 1
            if limit is not None and seen >= limit:
                return


def load_png_from_s3(s3, bucket: str, key: str) -> Image.Image:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        # copy to detach from context
        return im.copy()


def preprocess_image_to_tensor(im: Image.Image, size: int) -> torch.Tensor:
    # to square + resize
    w, h = im.size
    m = min(w, h)
    im = center_crop(im, (m, m))
    if (m, m) != (size, size):
        im = resize(im, (size, size))
    x = to_tensor(im)  # [0,1], shape 3xHxW
    x = (x - 0.5) / 0.5  # [-1,1]
    return x


def dct_features(x: torch.Tensor) -> torch.Tensor:
    # x: float tensor 3xHxW
    # Apply log(|DCT|+eps)
    return torch.log(torch.abs(dct_2d(x)) + EPS_DCT)


@dataclass
class SelectedKeys:
    stego: dict[str, list[str]]
    clean: dict[str, list[str]]


def select_keys(
    stego_keys: list[str],
    clean_keys: list[str],
    seed: int,
    n_train: int,
    n_val_stego: int,
    n_test_stego: int,
    n_train_clean: int,
    n_val_clean: int,
    n_test_clean: int,
    extra: int,
) -> SelectedKeys:
    # Split
    stego_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    clean_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for k in stego_keys:
        stego_split[split_of_key(k, seed)].append(k)
    for k in clean_keys:
        clean_split[split_of_key(k, seed)].append(k)

    # Stego stratification inside each split
    stego_sel: dict[str, list[str]] = {}
    for sp in ["train", "val", "test"]:
        by_src: dict[str, list[str]] = {}
        for k in stego_split[sp]:
            by_src.setdefault(stego_source(k), []).append(k)
        stego_sel[sp] = round_robin_mix(by_src, seed=seed)

    # Deterministic order for clean
    clean_sel = {sp: sorted(clean_split[sp]) for sp in ["train", "val", "test"]}

    def take(xs: list[str], n: int) -> list[str]:
        need = n + extra
        if len(xs) < need:
            need = len(xs)
        return xs[:need]

    return SelectedKeys(
        stego={
            "train": take(stego_sel["train"], n_train),
            "val": take(stego_sel["val"], n_val_stego),
            "test": take(stego_sel["test"], n_test_stego),
        },
        clean={
            "train": take(clean_sel["train"], n_train_clean),
            "val": take(clean_sel["val"], n_val_clean),
            "test": take(clean_sel["test"], n_test_clean),
        },
    )


def write_keys(out_dir: Path, keys: SelectedKeys) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls, m in [("stego", keys.stego), ("clean", keys.clean)]:
        for sp, arr in m.items():
            (out_dir / f"{cls}_{sp}.txt").write_text("\n".join(arr) + "\n")


def read_keys(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def open_memmap(path: Path, n: int, h: int, w: int, dtype=np.float16):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        str(path), mode="w+", dtype=dtype, shape=(n, 3, h, w)
    )

def setup_logging(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def compute_mean_std_subset(memmap_path: Path, indices: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(memmap_path, mmap_mode="r")
    # streaming sum / sumsq in float64
    s = None
    ss = None
    n = 0
    for i0 in range(0, len(indices), chunk):
        idx = indices[i0 : i0 + chunk]
        batch = x[idx].astype(np.float32)
        if s is None:
            s = np.zeros_like(batch[0], dtype=np.float64)
            ss = np.zeros_like(batch[0], dtype=np.float64)
        s += batch.sum(axis=0, dtype=np.float64)
        ss += (batch * batch).sum(axis=0, dtype=np.float64)
        n += batch.shape[0]
    mean = (s / max(n, 1)).astype(np.float32)
    var = (ss / max(n, 1) - (mean.astype(np.float64) ** 2)).clip(min=0.0)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


class MemmapX(Dataset):
    """Dataset that yields 4-tuples for DeepSAD: (x, label, semi_target, idx)."""

    def __init__(
        self,
        memmap_path: Path,
        indices: np.ndarray,
        label_value: float,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.x = np.load(memmap_path, mmap_mode="r")
        self.indices = indices.astype(np.int64)
        self.label_value = float(label_value)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        arr = self.x[idx].astype(np.float32)
        arr = (arr - self.mean) / (self.std + EPS_STD)
        x = torch.from_numpy(arr)
        y = torch.tensor(self.label_value, dtype=torch.float32)
        return x, y, y, torch.tensor(i, dtype=torch.int64)


class Concat4(Dataset):
    def __init__(self, a: Dataset, b: Dataset):
        self.a = a
        self.b = b

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, idx):
        if idx < len(self.a):
            return self.a[idx]
        return self.b[idx - len(self.a)]


class ADDataset(BaseADDataset):
    def __init__(self, train_set: Dataset, test_set: Dataset, val_set: Dataset):
        super().__init__(root="")
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return train_loader, test_loader


def eval_distances(net: torch.nn.Module, c: list[float], dl: DataLoader, device: str) -> np.ndarray:
    center = torch.tensor(c, device=device, dtype=torch.float32)
    dists: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device, non_blocking=True)
            out = net(x)
            dist = torch.sum((out - center) ** 2, dim=1).detach().cpu().numpy()
            dists.append(dist)
    return np.concatenate(dists) if dists else np.array([], dtype=np.float32)


def cache_features_for_prefix(
    *,
    s3,
    bucket: str,
    prefix: str,
    key_filter: callable,
    out_dir: Path,
    name: str,
    size: int,
    max_keys: Optional[int],
    workers: int,
    logger: logging.Logger,
) -> dict:
    """
    Cache features for all keys under prefix. Writes:
      - <name>_keys_used.txt
      - <name>_bad_keys.txt
      - <name>_dct<size>.npy (memmap) with shape (N_listed,3,size,size), only first ok_count are valid
      - <name>_meta.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keys = [k for k in iter_keys(s3, bucket, prefix, limit=max_keys) if key_filter(k)]
    t_list = time.time() - t0
    logger.info("Listed %s keys under %s in %.1fs", len(keys), prefix, t_list)
    save_json(out_dir / f"{name}_listed_keys.json", {"prefix": prefix, "count": len(keys), "time_s": t_list})

    feat_path = out_dir / f"{name}_dct{size}.npy"
    mm = open_memmap(feat_path, len(keys), size, size, dtype=np.float16)
    used: list[str] = []
    bad: list[str] = []

    def work(key: str):
        try:
            im = load_png_from_s3(s3, bucket=bucket, key=key)
            x = preprocess_image_to_tensor(im, size=size)
            f = dct_features(x).to(torch.float32).numpy().astype(np.float16, copy=False)
            return True, key, f
        except (UnidentifiedImageError, OSError, ValueError):
            return False, key, None

    ok = 0
    t1 = time.time()
    # bounded submission to avoid huge memory
    in_flight: set[cf.Future] = set()
    it = iter(keys)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        while True:
            while len(in_flight) < workers * 4:
                try:
                    k = next(it)
                except StopIteration:
                    break
                in_flight.add(ex.submit(work, k))
            if not in_flight:
                break
            done, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                ok_flag, k, f = fut.result()
                if ok_flag and f is not None:
                    mm[ok] = f
                    used.append(k)
                    ok += 1
                else:
                    bad.append(k)
            if ok and ok % 10000 == 0:
                dt = time.time() - t1
                logger.info("%s: ok=%d bad=%d rate=%.1f img/s", name, ok, len(bad), ok / max(dt, 1e-6))

    del mm
    t_total = time.time() - t1
    (out_dir / f"{name}_keys_used.txt").write_text("\n".join(used) + "\n")
    (out_dir / f"{name}_bad_keys.txt").write_text("\n".join(bad) + "\n")

    meta = {
        "name": name,
        "prefix": prefix,
        "size": size,
        "listed": len(keys),
        "ok": ok,
        "bad": len(bad),
        "workers": workers,
        "time_s": t_total,
        "rate_img_s": ok / max(t_total, 1e-6),
        "features_path": str(feat_path),
    }
    save_json(out_dir / f"{name}_meta.json", meta)
    logger.info("Cached %s: ok=%d bad=%d time=%.1fs rate=%.1f img/s", name, ok, len(bad), t_total, meta["rate_img_s"])
    return meta


def extract_split_features(
    *,
    s3,
    bucket: str,
    keys: list[str],
    out_path: Path,
    out_keys_path: Path,
    size: int,
    required: int,
    max_retries: int = 3,
) -> tuple[int, int]:
    """
    Extract features for the first `required` valid images from `keys` (keys already include extra).
    Writes:
      - features to out_path (.npy memmap)
      - used keys to out_keys_path
    Returns (ok, skipped_bad).
    """
    mm = open_memmap(out_path, required, size, size, dtype=np.float16)
    used: list[str] = []
    ok = 0
    bad = 0

    for key in keys:
        if ok >= required:
            break
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                im = load_png_from_s3(s3, bucket=bucket, key=key)
                x = preprocess_image_to_tensor(im, size=size)
                f = dct_features(x).to(torch.float32).numpy().astype(np.float16, copy=False)
                mm[ok] = f
                used.append(key)
                ok += 1
                last_err = None
                break
            except (UnidentifiedImageError, OSError, ValueError) as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        if last_err is not None:
            bad += 1

    del mm  # flush to disk
    out_keys_path.write_text("\n".join(used) + "\n")
    return ok, bad


class Memmap4TupleDataset(Dataset):
    def __init__(self, memmap_path: Path, labels: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.x = np.load(memmap_path, mmap_mode="r")
        self.labels = labels.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.x[idx].astype(np.float32)
        x = (x - self.mean) / (self.std + 0.05)
        x_t = torch.from_numpy(x)
        y = torch.tensor(self.labels[idx])
        # DeepSAD code expects: inputs, labels, semi_targets, idx
        return x_t, y, y, torch.tensor(idx)


class MemmapADDataset(BaseADDataset):
    def __init__(self, train_set: Dataset, test_set: Dataset, val_set: Dataset):
        super().__init__(root="")
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return train_loader, test_loader


def compute_mean_std(memmap_paths: list[Path], max_items: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    # Compute mean/std over concatenation of provided memmaps
    count = 0
    mean = None
    m2 = None

    for p in memmap_paths:
        x = np.load(p, mmap_mode="r")
        n = len(x) if max_items is None else min(len(x), max_items - count)
        for i in range(n):
            v = x[i].astype(np.float64)
            if mean is None:
                mean = np.zeros_like(v)
                m2 = np.zeros_like(v)
            count += 1
            delta = v - mean
            mean += delta / count
            delta2 = v - mean
            m2 += delta * delta2
        if max_items is not None and count >= max_items:
            break
    if mean is None or m2 is None or count < 2:
        raise RuntimeError("Not enough data to compute mean/std")
    var = m2 / (count - 1)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def eval_distances(net: torch.nn.Module, c: np.ndarray, dl: DataLoader, device: str) -> np.ndarray:
    center = torch.tensor(c, device=device, dtype=torch.float32)
    dists: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            out = net(x)
            dist = torch.sum((out - center) ** 2, dim=1).detach().cpu().numpy()
            dists.append(dist)
    return np.concatenate(dists)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    cache = sub.add_parser("cache", help="Cache DCT features for full stego+clean datasets from S3.")
    cache.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "https://storage.yandexcloud.net"))
    cache.add_argument("--bucket", default=os.environ.get("BUCKET", "stegopractice"))
    cache.add_argument("--region", default=os.environ.get("AWS_REGION", "ru-central1"))
    cache.add_argument("--access-key", default=os.environ.get("AWS_ACCESS_KEY_ID", ""))
    cache.add_argument("--secret-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    cache.add_argument("--stego-prefix", default="Stego/")
    cache.add_argument("--clean-prefix", default="Genimage/ai/")
    cache.add_argument("--size", type=int, default=32)
    cache.add_argument("--max-stego-keys", type=int, default=None)
    cache.add_argument("--max-clean-keys", type=int, default=None)
    cache.add_argument("--workers", type=int, default=32)
    cache.add_argument("--out-dir", type=Path, required=True)
    cache.add_argument("--upload-prefix", type=str, default=None, help="If set, upload logs/JSON (and optionally features) to s3://<bucket>/<upload-prefix>/")
    cache.add_argument("--upload-features", action="store_true", help="Also upload large .npy feature files (can be many GB).")

    cv = sub.add_parser("cv", help="K-fold cross-validation using cached features.")
    cv.add_argument("--cache-dir", type=Path, required=True)
    cv.add_argument("--out-dir", type=Path, required=True)
    cv.add_argument("--k", type=int, default=5)
    cv.add_argument("--seed", type=int, default=20260320)
    cv.add_argument("--val-size", type=int, default=10_000)
    cv.add_argument("--train-per-class", type=int, default=130_000)
    cv.add_argument("--epochs", type=int, default=50)
    cv.add_argument("--batch-size", type=int, default=256)
    cv.add_argument("--num-workers", type=int, default=8)
    cv.add_argument("--lr", type=float, default=1e-3)
    cv.add_argument("--weight-decay", type=float, default=5e-7)
    cv.add_argument("--tolerable-fnr", type=float, default=0.01)
    cv.add_argument("--net", default="cifar10_LeNet", choices=["cifar10_LeNet", "cifar10_biglenet"])
    cv.add_argument("--upload-prefix", type=str, default=None, help="If set, upload CV logs/results to s3://<bucket>/<upload-prefix>/")

    prep = sub.add_parser("prepare", help="List S3 keys and select deterministic splits.")
    prep.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "https://storage.yandexcloud.net"))
    prep.add_argument("--bucket", default=os.environ.get("BUCKET", "stegopractice"))
    prep.add_argument("--region", default=os.environ.get("AWS_REGION", "ru-central1"))
    prep.add_argument("--access-key", default=os.environ.get("AWS_ACCESS_KEY_ID", ""))
    prep.add_argument("--secret-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    prep.add_argument("--stego-prefix", default="Stego/")
    prep.add_argument("--clean-prefix", default="Genimage/ai/")
    prep.add_argument("--seed", type=int, default=20260320)
    prep.add_argument("--extra", type=int, default=5000)
    prep.add_argument("--out-dir", type=Path, required=True)
    # selection sizes
    prep.add_argument("--train-stego", type=int, default=130_000)
    prep.add_argument("--val-stego", type=int, default=10_000)
    prep.add_argument("--test-stego", type=int, default=20_000)
    prep.add_argument("--train-clean", type=int, default=130_000)
    prep.add_argument("--val-clean", type=int, default=10_000)
    prep.add_argument("--test-clean", type=int, default=166_000)
    prep.add_argument("--limit-stego-list", type=int, default=None)
    prep.add_argument("--limit-clean-list", type=int, default=None)

    ext = sub.add_parser("extract", help="Extract DCT features to memmap files.")
    ext.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "https://storage.yandexcloud.net"))
    ext.add_argument("--bucket", default=os.environ.get("BUCKET", "stegopractice"))
    ext.add_argument("--region", default=os.environ.get("AWS_REGION", "ru-central1"))
    ext.add_argument("--access-key", default=os.environ.get("AWS_ACCESS_KEY_ID", ""))
    ext.add_argument("--secret-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    ext.add_argument("--keys-dir", type=Path, required=True)
    ext.add_argument("--out-dir", type=Path, required=True)
    ext.add_argument("--size", type=int, default=32)
    ext.add_argument("--train-stego", type=int, default=130_000)
    ext.add_argument("--val-stego", type=int, default=10_000)
    ext.add_argument("--test-stego", type=int, default=20_000)
    ext.add_argument("--train-clean", type=int, default=130_000)
    ext.add_argument("--val-clean", type=int, default=10_000)
    ext.add_argument("--test-clean", type=int, default=166_000)

    tr = sub.add_parser("train", help="Train DeepSAD on extracted features and evaluate.")
    tr.add_argument("--feat-dir", type=Path, required=True)
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--net", default="cifar10_LeNet", choices=["cifar10_LeNet", "cifar10_biglenet"])
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--num-workers", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=5e-7)
    tr.add_argument("--tolerable-fnr", type=float, default=0.01)
    tr.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.cmd in ("prepare", "extract", "cache"):
        if not args.access_key or not args.secret_key:
            print("Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY", file=sys.stderr)
            sys.exit(2)
        s3 = make_s3_client(
            endpoint=args.endpoint,
            region=args.region,
            access_key=args.access_key,
            secret_key=args.secret_key,
        )

    if args.cmd == "cache":
        out_dir: Path = args.out_dir
        logger = setup_logging(out_dir / "logs", "cache")
        logger.info("Starting cache: bucket=%s stego_prefix=%s clean_prefix=%s size=%d workers=%d",
                    args.bucket, args.stego_prefix, args.clean_prefix, args.size, args.workers)
        meta = {
            "bucket": args.bucket,
            "endpoint": args.endpoint,
            "region": args.region,
            "size": args.size,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(out_dir / "cache_run.json", meta)

        stego_meta = cache_features_for_prefix(
            s3=s3,
            bucket=args.bucket,
            prefix=args.stego_prefix,
            key_filter=lambda k: ("/identity/" in k and k.endswith(".png")),
            out_dir=out_dir,
            name="stego",
            size=args.size,
            max_keys=args.max_stego_keys,
            workers=args.workers,
            logger=logger,
        )
        clean_meta = cache_features_for_prefix(
            s3=s3,
            bucket=args.bucket,
            prefix=args.clean_prefix,
            key_filter=lambda k: k.endswith(".png"),
            out_dir=out_dir,
            name="clean",
            size=args.size,
            max_keys=args.max_clean_keys,
            workers=args.workers,
            logger=logger,
        )
        save_json(out_dir / "cache_summary.json", {"stego": stego_meta, "clean": clean_meta})

        if args.upload_prefix:
            up = args.upload_prefix.rstrip("/")
            logger.info("Uploading cache artifacts to s3://%s/%s/", args.bucket, up)
            # upload logs + small artifacts
            upload_dir(
                s3=s3,
                bucket=args.bucket,
                local_dir=out_dir,
                prefix=up,
                include_suffixes=(".json", ".txt", ".log"),
                logger=logger,
            )
            if args.upload_features:
                # upload .npy (large)
                upload_dir(
                    s3=s3,
                    bucket=args.bucket,
                    local_dir=out_dir,
                    prefix=up,
                    include_suffixes=(".npy",),
                    logger=logger,
                )
        return

    if args.cmd == "cv":
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(out_dir / "logs", "cv")
        logger.info("Starting CV: k=%d seed=%d val_size=%d train_per_class=%d epochs=%d fnr=%.4f net=%s",
                    args.k, args.seed, args.val_size, args.train_per_class, args.epochs, args.tolerable_fnr, args.net)

        cache_dir: Path = args.cache_dir
        # Detect feature file by reading meta
        stego_meta = json.loads((cache_dir / "stego_meta.json").read_text())
        clean_meta = json.loads((cache_dir / "clean_meta.json").read_text())
        size = int(stego_meta["size"])
        stego_feat = cache_dir / f"stego_dct{size}.npy"
        clean_feat = cache_dir / f"clean_dct{size}.npy"
        stego_keys = read_keys(cache_dir / "stego_keys_used.txt")
        clean_keys = read_keys(cache_dir / "clean_keys_used.txt")

        n_stego = len(stego_keys)
        n_clean = len(clean_keys)
        logger.info("Cache loaded: stego=%d clean=%d size=%d", n_stego, n_clean, size)

        # Precompute fold ids
        stego_fold = np.array([sha1_u32(f"{args.seed}|{k}") % args.k for k in stego_keys], dtype=np.int32)
        clean_fold = np.array([sha1_u32(f"{args.seed}|{k}") % args.k for k in clean_keys], dtype=np.int32)

        results = []
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for fold in range(args.k):
            t_fold = time.time()
            # Test indices
            stego_test_idx = np.where(stego_fold == fold)[0]
            clean_test_idx = np.where(clean_fold == fold)[0]

            # Train pool indices
            stego_pool = np.where(stego_fold != fold)[0]
            clean_pool = np.where(clean_fold != fold)[0]

            # Val indices from stego_pool: pick by secondary hash, deterministic
            scores = np.array([sha1_u32(f"val|{args.seed}|{stego_keys[i]}") for i in stego_pool], dtype=np.uint32)
            order = np.argsort(scores)
            stego_val_idx = stego_pool[order[: min(args.val_size, len(stego_pool))]]

            # Train indices exclude stego_val
            stego_train_pool = np.setdiff1d(stego_pool, stego_val_idx, assume_unique=False)
            # Balance per class for training
            n_train = min(args.train_per_class, len(stego_train_pool), len(clean_pool))
            # deterministic subsample by hash
            stego_train_scores = np.array([sha1_u32(f"tr|{args.seed}|{stego_keys[i]}") for i in stego_train_pool], dtype=np.uint32)
            clean_train_scores = np.array([sha1_u32(f"tr|{args.seed}|{clean_keys[i]}") for i in clean_pool], dtype=np.uint32)
            stego_train_idx = stego_train_pool[np.argsort(stego_train_scores)[:n_train]]
            clean_train_idx = clean_pool[np.argsort(clean_train_scores)[:n_train]]

            logger.info(
                "fold=%d sizes: train=%d/%d val(stego)=%d test=%d/%d",
                fold, len(stego_train_idx), len(clean_train_idx), len(stego_val_idx), len(stego_test_idx), len(clean_test_idx)
            )

            # Norm from train subsets (both classes) on stego_feat + clean_feat separately, then combine
            mean_s, std_s = compute_mean_std_subset(stego_feat, stego_train_idx)
            mean_c, std_c = compute_mean_std_subset(clean_feat, clean_train_idx)
            # Combine means weighted
            mean = ((mean_s.astype(np.float64) + mean_c.astype(np.float64)) / 2.0).astype(np.float32)
            std = ((std_s.astype(np.float64) + std_c.astype(np.float64)) / 2.0).astype(np.float32)

            ds_stego_train = MemmapX(stego_feat, stego_train_idx, 1.0, mean, std)
            ds_clean_train = MemmapX(clean_feat, clean_train_idx, -1.0, mean, std)
            ds_train = Concat4(ds_stego_train, ds_clean_train)
            ds_val = MemmapX(stego_feat, stego_val_idx, 1.0, mean, std)
            ds_stego_test = MemmapX(stego_feat, stego_test_idx, 1.0, mean, std)
            ds_clean_test = MemmapX(clean_feat, clean_test_idx, -1.0, mean, std)
            ds_test = Concat4(ds_stego_test, ds_clean_test)

            dataset = ADDataset(train_set=ds_train, test_set=ds_test, val_set=ds_val)

            deep = DeepSAD()
            # Channels=3
            deep.set_network(args.net, num_select_channels=3)
            deep.train(
                dataset,
                lr=args.lr,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                n_jobs_dataloader=args.num_workers,
                tolerable_fnr=[args.tolerable_fnr],
            )

            deep.net.eval()
            deep.net.to(device)
            val_loader = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
            dist_val = eval_distances(deep.net, deep.c, val_loader, device=device)
            thr = float(np.quantile(dist_val, q=1 - args.tolerable_fnr))

            stego_test_loader = DataLoader(ds_stego_test, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
            clean_test_loader = DataLoader(ds_clean_test, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
            dist_st = eval_distances(deep.net, deep.c, stego_test_loader, device=device)
            dist_cl = eval_distances(deep.net, deep.c, clean_test_loader, device=device)

            tpr = float(np.mean(dist_st <= thr)) if len(dist_st) else 0.0
            fpr = float(np.mean(dist_cl <= thr)) if len(dist_cl) else 0.0
            y = np.concatenate([np.ones_like(dist_st), np.zeros_like(dist_cl)])
            scores = np.concatenate([-dist_st, -dist_cl])
            auc = float(roc_auc_score(y, scores)) if len(y) else 0.0

            rec = {
                "fold": fold,
                "threshold": thr,
                "tolerable_fnr": args.tolerable_fnr,
                "tpr": tpr,
                "fpr": fpr,
                "auc": auc,
                "sizes": {
                    "train_per_class": int(n_train),
                    "val_stego": int(len(stego_val_idx)),
                    "test_stego": int(len(stego_test_idx)),
                    "test_clean": int(len(clean_test_idx)),
                },
                "time_s": float(time.time() - t_fold),
            }
            results.append(rec)
            (out_dir / "cv_results.jsonl").open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("fold=%d done: tpr=%.4f fpr=%.4f auc=%.4f time=%.1fs", fold, tpr, fpr, auc, rec["time_s"])

        # Summary
        def mean_of(k: str) -> float:
            return float(np.mean([r[k] for r in results])) if results else 0.0

        summary = {
            "k": args.k,
            "seed": args.seed,
            "tolerable_fnr": args.tolerable_fnr,
            "avg": {"tpr": mean_of("tpr"), "fpr": mean_of("fpr"), "auc": mean_of("auc")},
            "folds": results,
        }
        save_json(out_dir / "cv_summary.json", summary)
        logger.info("CV summary avg: tpr=%.4f fpr=%.4f auc=%.4f", summary["avg"]["tpr"], summary["avg"]["fpr"], summary["avg"]["auc"])

        if args.upload_prefix:
            # Use cache bucket/endpoint creds from environment is assumed; upload into same bucket as cache artifacts.
            up = args.upload_prefix.rstrip("/")
            logger.info("Uploading CV artifacts to s3://%s/%s/", os.environ.get("BUCKET", ""), up)
            # We don't know which bucket user wants here; default to BUCKET env. If not set, skip.
            bucket = os.environ.get("BUCKET")
            if not bucket:
                logger.error("BUCKET env is not set; cannot upload. Set BUCKET and re-run or omit --upload-prefix.")
            else:
                s3_u = make_s3_client(
                    endpoint=os.environ.get("ENDPOINT", "https://storage.yandexcloud.net"),
                    region=os.environ.get("AWS_REGION", "ru-central1"),
                    access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                    secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                )
                upload_dir(
                    s3=s3_u,
                    bucket=bucket,
                    local_dir=out_dir,
                    prefix=up,
                    include_suffixes=(".json", ".jsonl", ".log"),
                    logger=logger,
                )
        return

    if args.cmd == "prepare":
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "endpoint": args.endpoint,
            "bucket": args.bucket,
            "region": args.region,
            "stego_prefix": args.stego_prefix,
            "clean_prefix": args.clean_prefix,
            "seed": args.seed,
            "extra": args.extra,
            "sizes": {
                "train_stego": args.train_stego,
                "val_stego": args.val_stego,
                "test_stego": args.test_stego,
                "train_clean": args.train_clean,
                "val_clean": args.val_clean,
                "test_clean": args.test_clean,
            },
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

        # List keys
        print("Listing stego keys...", flush=True)
        stego_all = [
            k
            for k in iter_keys(s3, args.bucket, args.stego_prefix, limit=args.limit_stego_list)
            if "/identity/" in k and k.endswith(".png")
        ]
        print(f"stego_png_keys={len(stego_all)}", flush=True)
        (out_dir / "stego_all.txt").write_text("\n".join(stego_all) + "\n")

        print("Listing clean keys...", flush=True)
        clean_all = [
            k
            for k in iter_keys(s3, args.bucket, args.clean_prefix, limit=args.limit_clean_list)
            if k.endswith(".png")
        ]
        print(f"clean_png_keys={len(clean_all)}", flush=True)
        (out_dir / "clean_all.txt").write_text("\n".join(clean_all) + "\n")

        sel = select_keys(
            stego_keys=stego_all,
            clean_keys=clean_all,
            seed=args.seed,
            n_train=args.train_stego,
            n_val_stego=args.val_stego,
            n_test_stego=args.test_stego,
            n_train_clean=args.train_clean,
            n_val_clean=args.val_clean,
            n_test_clean=args.test_clean,
            extra=args.extra,
        )
        write_keys(out_dir / "selected", sel)
        print("Wrote selected key lists to", out_dir / "selected", flush=True)
        return

    if args.cmd == "extract":
        keys_dir: Path = args.keys_dir
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = out_dir / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)

        tasks = [
            ("stego", "train", args.train_stego),
            ("stego", "val", args.val_stego),
            ("stego", "test", args.test_stego),
            ("clean", "train", args.train_clean),
            ("clean", "val", args.val_clean),
            ("clean", "test", args.test_clean),
        ]

        for cls, sp, req in tasks:
            key_file = keys_dir / f"{cls}_{sp}.txt"
            keys = read_keys(key_file)
            out_path = out_dir / f"{cls}_{sp}_dct{args.size}.npy"
            used_path = out_dir / f"{cls}_{sp}_keys_used.txt"
            print(f"Extracting {cls}/{sp}: required={req} keys_in_file={len(keys)} -> {out_path.name}", flush=True)
            ok, bad = extract_split_features(
                s3=s3,
                bucket=args.bucket,
                keys=keys,
                out_path=out_path,
                out_keys_path=used_path,
                size=args.size,
                required=req,
            )
            (log_dir / f"{cls}_{sp}.json").write_text(
                json.dumps({"ok": ok, "bad": bad, "required": req, "size": args.size}, indent=2) + "\n"
            )
            if ok < req:
                raise RuntimeError(f"Not enough valid images for {cls}/{sp}: ok={ok} required={req}. Increase --extra in prepare.")
        print("Feature extraction complete.", flush=True)
        return

    if args.cmd == "train":
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        feat_dir: Path = args.feat_dir
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Kept for backwards compatibility with the old split-based pipeline.
        # NOTE: Prefer using `cv` with cached features for the main experiment.
        mean, std = compute_mean_std_subset(
            feat_dir / "stego_train_dct32.npy",
            np.arange(np.load(feat_dir / "stego_train_dct32.npy", mmap_mode="r").shape[0]),
        )
        np.savez(out_dir / "norm.npz", mean=mean, std=std)

        # Labels: stego=+1 (inlier), clean=-1 (outlier)
        def labels_for(n: int, v: float) -> np.ndarray:
            return np.full((n,), v, dtype=np.float32)

        x_stego_train = feat_dir / "stego_train_dct32.npy"
        x_clean_train = feat_dir / "clean_train_dct32.npy"
        x_stego_val = feat_dir / "stego_val_dct32.npy"
        x_stego_test = feat_dir / "stego_test_dct32.npy"
        x_clean_test = feat_dir / "clean_test_dct32.npy"

        n_stego_train = np.load(x_stego_train, mmap_mode="r").shape[0]
        n_clean_train = np.load(x_clean_train, mmap_mode="r").shape[0]
        n_stego_val = np.load(x_stego_val, mmap_mode="r").shape[0]
        n_stego_test = np.load(x_stego_test, mmap_mode="r").shape[0]
        n_clean_test = np.load(x_clean_test, mmap_mode="r").shape[0]

        # Train set: concatenate via a wrapper dataset by index mapping
        # For simplicity: create two datasets and merge with ConcatDataset-like logic.
        class Concat4Tuple(Dataset):
            def __init__(self, a: Dataset, b: Dataset):
                self.a = a
                self.b = b
            def __len__(self): return len(self.a) + len(self.b)
            def __getitem__(self, idx):
                if idx < len(self.a):
                    return self.a[idx]
                return self.b[idx - len(self.a)]

        ds_stego_train = Memmap4TupleDataset(x_stego_train, labels_for(n_stego_train, 1.0), mean, std)
        ds_clean_train = Memmap4TupleDataset(x_clean_train, labels_for(n_clean_train, -1.0), mean, std)
        ds_train = Concat4Tuple(ds_stego_train, ds_clean_train)

        # Validation set is stego only (thresholding for FNR)
        ds_val = Memmap4TupleDataset(x_stego_val, labels_for(n_stego_val, 1.0), mean, std)

        # Test set combines stego+clean (for AUC and FPR)
        ds_stego_test = Memmap4TupleDataset(x_stego_test, labels_for(n_stego_test, 1.0), mean, std)
        ds_clean_test = Memmap4TupleDataset(x_clean_test, labels_for(n_clean_test, -1.0), mean, std)
        ds_test = Concat4Tuple(ds_stego_test, ds_clean_test)

        dataset = MemmapADDataset(train_set=ds_train, test_set=ds_test, val_set=ds_val)

        deep = DeepSAD()
        deep.set_network(args.net, num_select_channels=3)
        deep.train(
            dataset,
            lr=args.lr,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            n_jobs_dataloader=args.num_workers,
            tolerable_fnr=[args.tolerable_fnr],
        )

        # Build loaders for eval
        train_loader, test_loader = dataset.loaders(batch_size=args.batch_size, num_workers=args.num_workers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deep.net.eval()
        deep.net.to(device)

        # Threshold from val distances (stego only)
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
        val_dist = eval_distances(deep.net, np.array(deep.c), val_loader, device=device)
        thr = float(np.quantile(val_dist, q=1 - args.tolerable_fnr))

        # Distances on test splits separately
        stego_test_loader = DataLoader(ds_stego_test, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
        clean_test_loader = DataLoader(ds_clean_test, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
        dist_stego = eval_distances(deep.net, np.array(deep.c), stego_test_loader, device=device)
        dist_clean = eval_distances(deep.net, np.array(deep.c), clean_test_loader, device=device)

        # Inlier if dist <= thr
        tpr = float(np.mean(dist_stego <= thr))
        fpr = float(np.mean(dist_clean <= thr))

        # AUROC on combined test (stego=1, clean=0), score=-dist (higher => more inlier)
        y = np.concatenate([np.ones_like(dist_stego), np.zeros_like(dist_clean)])
        scores = np.concatenate([-dist_stego, -dist_clean])
        auc = float(roc_auc_score(y, scores))

        result = {
            "threshold": thr,
            "tolerable_fnr": args.tolerable_fnr,
            "tpr": tpr,
            "fpr": fpr,
            "auc": auc,
            "net": args.net,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "sizes": {
                "stego_train": int(n_stego_train),
                "clean_train": int(n_clean_train),
                "stego_val": int(n_stego_val),
                "stego_test": int(n_stego_test),
                "clean_test": int(n_clean_test),
            },
        }
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return


if __name__ == "__main__":
    main()

