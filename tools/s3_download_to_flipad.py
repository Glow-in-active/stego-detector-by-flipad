#!/usr/bin/env python3
"""
Скачать поднабор изображений из S3 (Yandex Object Storage) в структуру папок,
которую ожидает FLIPAD `flipad/sma_deepsad.py`.

Задача проекта: inlier = mas_GRDH stego, outlier = GenImage (SD).

Пример структуры на выходе:
  data/coco2014train/
    mas_grdh_stego/{train,val,test}/*.png
    genimage/{train,val,test}/*.png

Ключи берутся из бакета по префиксам:
  - stego: Stego/**/identity/*.png
  - clean: Genimage/ai/*.png

Важно: скрипт НЕ использует самописный DCT. Это только загрузка PNG на диск.
"""

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


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


def score_of_key(key: str, seed: int) -> int:
    # используется только для детерминированного отбора "первых N"
    return sha1_u32(f"sel|{seed}|{key}")


def sha_name(key: str) -> str:
    # делаем имя уникальным между подпапками (000000.png коллизит)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24] + ".png"


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


def iter_keys(s3, bucket: str, prefix: str) -> Iterator[str]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        for obj in page.get("Contents", []):
            yield obj["Key"]


class Need:
    def __init__(self, train: int, val: int, test: int):
        self.train = int(train)
        self.val = int(val)
        self.test = int(test)

    def for_split(self, sp: str) -> int:
        if sp == "train":
            return self.train
        if sp == "val":
            return self.val
        if sp == "test":
            return self.test
        raise ValueError(sp)


def _select_smallest_by_score(
    *,
    keys: Iterable[str],
    seed: int,
    need: Need,
    extra: int,
    key_filter: Callable[[str], bool],
) -> Dict[str, List[str]]:
    """
    Детерминированно выбираем (need+extra) ключей для каждого split, независимо от порядка листинга.
    Реализация: для каждого split держим max-heap по score и оставляем самые маленькие score.
    """
    import heapq

    heaps = {"train": [], "val": [], "test": []}  # type: Dict[str, List[Tuple[int, str]]]  # (-score, key)
    limits = {sp: need.for_split(sp) + extra for sp in ["train", "val", "test"]}

    seen = 0
    kept = 0
    for k in keys:
        seen += 1
        if not key_filter(k):
            continue
        sp = split_of_key(k, seed=seed)
        lim = limits[sp]
        if lim <= 0:
            continue
        sc = score_of_key(k, seed=seed)
        h = heaps[sp]
        heapq.heappush(h, (-sc, k))
        kept += 1
        if len(h) > lim:
            heapq.heappop(h)
        # редкий прогресс
        if seen % 100000 == 0:
            pass

    out = {}  # type: Dict[str, List[str]]
    for sp, h in heaps.items():
        # извлекаем, сортируем по score по возрастанию
        arr = [(-neg_sc, k) for (neg_sc, k) in h]
        arr.sort(key=lambda x: x[0])
        out[sp] = [k for _, k in arr]
    return out


def download_one(
    *,
    s3,
    bucket: str,
    key: str,
    out_path: Path,
    retries: int = 5,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            s3.download_file(bucket, key, str(tmp))
            tmp.replace(out_path)
            return True
        except ClientError:
            time.sleep(0.5 * (attempt + 1))
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "https://storage.yandexcloud.net"))
    ap.add_argument("--bucket", default=os.environ.get("BUCKET", "stegopractice"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "ru-central1"))
    ap.add_argument("--access-key", default=os.environ.get("AWS_ACCESS_KEY_ID", ""))
    ap.add_argument("--secret-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY", ""))

    ap.add_argument("--data-path", type=Path, default=Path("data"))
    ap.add_argument("--dataset", default="coco2014train")
    ap.add_argument("--stego-dir", default="mas_grdh_stego")
    ap.add_argument("--real-dir", default="genimage")

    ap.add_argument("--stego-prefix", default="Stego/")
    ap.add_argument("--real-prefix", default="Genimage/ai/")
    ap.add_argument("--seed", type=int, default=20260320)
    ap.add_argument("--extra", type=int, default=200)

    ap.add_argument("--stego-train", type=int, default=5000)
    ap.add_argument("--stego-val", type=int, default=2000)
    ap.add_argument("--stego-test", type=int, default=5000)
    ap.add_argument("--real-train", type=int, default=5000)
    ap.add_argument("--real-val", type=int, default=2000)
    ap.add_argument("--real-test", type=int, default=20000)

    ap.add_argument("--max-stego-list", type=int, default=None, help="Ограничить листинг stego ключей (для отладки).")
    ap.add_argument("--max-real-list", type=int, default=None, help="Ограничить листинг real ключей (для отладки).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    if not args.access_key or not args.secret_key:
        raise SystemExit("Нужны AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY (через env или флаги).")

    s3 = make_s3_client(args.endpoint, args.region, args.access_key, args.secret_key)

    stego_need = Need(args.stego_train, args.stego_val, args.stego_test)
    real_need = Need(args.real_train, args.real_val, args.real_test)

    def limited(it: Iterator[str], limit: Optional[int]) -> Iterator[str]:
        if limit is None:
            yield from it
            return
        n = 0
        for x in it:
            yield x
            n += 1
            if n >= limit:
                return

    print("Listing+selecting stego keys...", flush=True)
    stego_keys = limited(iter_keys(s3, args.bucket, args.stego_prefix), args.max_stego_list)
    stego_sel = _select_smallest_by_score(
        keys=stego_keys,
        seed=args.seed,
        need=stego_need,
        extra=args.extra,
        key_filter=lambda k: ("/identity/" in k and k.endswith(".png")),
    )
    print({k: len(v) for k, v in stego_sel.items()}, flush=True)

    print("Listing+selecting real keys...", flush=True)
    real_keys = limited(iter_keys(s3, args.bucket, args.real_prefix), args.max_real_list)
    real_sel = _select_smallest_by_score(
        keys=real_keys,
        seed=args.seed,
        need=real_need,
        extra=args.extra,
        key_filter=lambda k: k.endswith(".png"),
    )
    print({k: len(v) for k, v in real_sel.items()}, flush=True)

    base = args.data_path / args.dataset
    out_stego = base / args.stego_dir
    out_real = base / args.real_dir

    # сохраняем manifests (удобно для воспроизводимости)
    manifests = base / "_manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (manifests / f"stego_{sp}.txt").write_text("\n".join(stego_sel[sp]) + "\n")
        (manifests / f"real_{sp}.txt").write_text("\n".join(real_sel[sp]) + "\n")
        # список, который реально будем качать на диск (без extra)
        (manifests / f"stego_{sp}_download.txt").write_text(
            "\n".join(stego_sel[sp][: stego_need.for_split(sp)]) + "\n"
        )
        (manifests / f"real_{sp}_download.txt").write_text(
            "\n".join(real_sel[sp][: real_need.for_split(sp)]) + "\n"
        )

    if args.dry_run:
        print("Dry-run: manifests written to", manifests, flush=True)
        return

    # скачиваем параллельно
    import concurrent.futures as cf

    def submit_all(pairs):
        ok = 0
        bad = 0
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_one, s3=s3, bucket=args.bucket, key=k, out_path=p) for k, p in pairs]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                if fut.result():
                    ok += 1
                else:
                    bad += 1
                if i % 1000 == 0:
                    dt = time.time() - t0
                    print(f"progress {i}/{len(futs)} ok={ok} bad={bad} rate={ok/max(dt,1e-6):.1f} file/s", flush=True)
        return ok, bad

    def build_pairs(sel, root):
        pairs = []  # type: List[Tuple[str, Path]]
        for sp in ["train", "val", "test"]:
            # Важно для диска: на диск кладём только "required" (extra остаётся в manifests как запас)
            required = stego_need.for_split(sp) if root == out_stego else real_need.for_split(sp)
            for k in sel[sp][:required]:
                pairs.append((k, root / sp / sha_name(k)))
        return pairs

    stego_pairs = build_pairs(stego_sel, out_stego)
    real_pairs = build_pairs(real_sel, out_real)

    print("Downloading stego...", len(stego_pairs), flush=True)
    ok_s, bad_s = submit_all(stego_pairs)
    print("Downloading real...", len(real_pairs), flush=True)
    ok_r, bad_r = submit_all(real_pairs)

    print(
        "Done.",
        {"stego_ok": ok_s, "stego_bad": bad_s, "real_ok": ok_r, "real_bad": bad_r},
        flush=True,
    )


if __name__ == "__main__":
    main()

