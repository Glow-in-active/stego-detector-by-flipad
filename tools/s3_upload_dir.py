#!/usr/bin/env python3
"""
Загрузить локальную директорию в S3 (Yandex Object Storage).

Ожидает креды в env:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
и опционально:
  ENDPOINT (default https://storage.yandexcloud.net)
  BUCKET   (default stegopractice)
  AWS_REGION (default ru-central1)

Пример:
  python3 tools/s3_upload_dir.py --local-dir results/run123 --s3-prefix Results/stego_detector/run123
"""

import argparse
import mimetypes
import os
from pathlib import Path

import boto3
from botocore.config import Config


def make_s3_client(endpoint, region, access_key, secret_key):
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
            max_pool_connections=64,
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )


def should_skip(rel_posix, skip_cache):
    if skip_cache and (rel_posix == "cache" or rel_posix.startswith("cache/")):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", type=Path, required=True)
    ap.add_argument("--s3-prefix", type=str, required=True, help="Префикс внутри бакета (без s3://bucket/).")
    ap.add_argument("--bucket", default=os.environ.get("BUCKET", "stegopractice"))
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "https://storage.yandexcloud.net"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "ru-central1"))
    ap.add_argument("--skip-cache", action="store_true", help="Не загружать поддиректорию cache/ (если есть).")
    args = ap.parse_args()

    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        raise SystemExit("Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in environment")

    local_dir = args.local_dir.resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        raise SystemExit("local-dir does not exist or is not a directory: %s" % str(local_dir))

    s3 = make_s3_client(args.endpoint, args.region, access_key, secret_key)
    prefix = args.s3_prefix.strip("/")

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    files.sort()

    for p in files:
        rel = p.relative_to(local_dir).as_posix()
        if should_skip(rel, args.skip_cache):
            continue
        key = prefix + "/" + rel if prefix else rel
        ctype, _ = mimetypes.guess_type(p.name)
        extra = {"ContentType": ctype} if ctype else None
        if extra:
            s3.upload_file(str(p), args.bucket, key, ExtraArgs=extra)
        else:
            s3.upload_file(str(p), args.bucket, key)
        print("uploaded", rel, "->", "s3://%s/%s" % (args.bucket, key), flush=True)


if __name__ == "__main__":
    main()

