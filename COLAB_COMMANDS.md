# Команды для Google Colab (пилот FLIPAD: stego vs GAN)

Ниже — последовательность ячеек, которые можно копировать в Colab.  
Цель: **скачать небольшой поднабор** из Yandex Object Storage (S3) и запустить пилот `flipad/sma_deepsad.py`.

## 0) Предпосылки

- S3 бакет: `stegopractice`
- stego изображения: `Stego/**/identity/*.png` (рядом `*.json` — игнорируем)
- чистые (GAN): `gan/ai/*.png` (например `000_biggan_00000.png`)
- Мы **не переносим** данные в бакете. Для пилота скачиваем **только небольшой поднабор**.

Размеры пилота (цель ≤ 1 часа):

- stego: `train=2000`, `val=500`, `test=2000`
- gan: `train=2000`, `val=500`, `test=5000`
- фичи: `--feat dct` (самый быстрый baseline)

## 1) Установка зависимостей и `flipad`

Если ты хочешь брать FLIPAD напрямую из GitHub (рекомендую для чистого Colab-пилота), сначала клонируй:

```bash
%%bash
set -e
git clone https://github.com/MikeLasz/flipad.git
```

Дальше предполагаем, что папка `flipad/` лежит в текущей директории Colab.

```bash
%%bash
set -e
pip -q install -U pip setuptools wheel
# В Colab AWS CLI (pip) часто ломается на Python 3.12. Поэтому S3 читаем через boto3.
# Ставим boto3 с зависимостями (иначе будет падать на отсутствии s3transfer).
pip -q install -U boto3
pip -q install --no-deps torch-dct==0.1.6
# Иногда в Colab/контейнере отсутствует `jedi`, а старый IPython его требует.
pip -q install "jedi>=0.16"
# Важно: для пилота (feat=dct) нам НЕ нужны diffusers/transformers/huggingface_hub,
# а их версии в Colab часто конфликтуют и тянут большой резолвинг. Не ставим их.
```

Установи `flipad`:

```bash
%%bash
set -e
pip -q install -e ./flipad --no-deps
```

## 1.1) Хотфикс для пилота: не импортировать/не загружать Stable Diffusion при `feat=dct`

`flipad/sma_deepsad.py` по умолчанию **импортирует** `StableDiffusionWrapper` на уровне модуля.
В Colab это может падать из‑за конфликтов `diffusers/huggingface_hub`, даже если мы используем только `dct`-признаки.
А ещё оно не должно скачивать/грузить SD для `feat=dct`.

В пилоте нам это не нужно, поэтому делаем локальный хотфикс: **инициализируем wrapper только для `feat=act`**.
Дополнительно делаем lazy import `StableDiffusionWrapper` с заглушкой, чтобы `--model stablediffusion` остался доступен
для ветки `dct` (там wrapper реально не используется).

```python
from pathlib import Path
import re

p = Path("flipad/sma_deepsad.py")
txt = p.read_text()

# 1) Заменяем eager-import StableDiffusionWrapper на try/except + stub
txt = re.sub(
    r"^from flipad\.wrappers\.stablediffusion import StableDiffusionWrapper\s*$",
    "try:\n"
    "    from flipad.wrappers.stablediffusion import StableDiffusionWrapper\n"
    "except Exception:\n"
    "    class StableDiffusionWrapper:\n"
    "        name = \"stablediffusion\"\n"
    "        seed_dim = None\n"
    "        def __init__(self, checkpoint_path):\n"
    "            self.checkpoint_path = checkpoint_path\n",
    txt,
    flags=re.MULTILINE,
)

# 2) Делаем инициализацию wrapper условной и оставляем stub для feat!=act
needle = "    # initialize wrapper\n    wrapper = WRAPPERS[args.model](checkpoint_path=args.checkpoint_path)\n\n    path_fake_train = args.data_path / args.dataset / args.train_dir / \"train\"\n"

if needle in txt:
    replacement = (
        "    # initialize wrapper\n"
        "    # - feat=act: нужен реальный wrapper\n"
        "    # - feat=dct/raw: wrapper не используется, но ниже иногда читается wrapper.name\n"
        "    from types import SimpleNamespace\n"
        "    wrapper = SimpleNamespace(name=args.model, checkpoint_path=args.checkpoint_path)\n"
        "    if args.feat == \"act\":\n"
        "        wrapper = WRAPPERS[args.model](checkpoint_path=args.checkpoint_path)\n\n"
        "    path_fake_train = args.data_path / args.dataset / args.train_dir / \"train\"\n"
    )
    txt = txt.replace(needle, replacement)
    p.write_text(txt)
    print("Patched sma_deepsad.py: stablediffusion import is lazy, wrapper init conditional")
else:
    print("Patch not applied: expected snippet not found (maybe repo version changed).")
```

## 2) Настройка AWS CLI профиля под Yandex Object Storage

Настройка для Yandex Object Storage (S3) через `boto3`.

```python
import os
import boto3
from botocore.config import Config

ENDPOINT = os.environ.get("ENDPOINT", "https://storage.yandexcloud.net")
BUCKET = os.environ.get("BUCKET", "stegopractice")
REGION = os.environ.get("AWS_REGION", "ru-central1")

# ====== ВСТАВЬ СЮДА КЛЮЧИ (плейсхолдеры) ======
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "<YANDEX_S3_ACCESS_KEY_ID>")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "<YANDEX_S3_SECRET_ACCESS_KEY>")

session = boto3.session.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION,
)
s3 = session.client(
    "s3",
    endpoint_url=ENDPOINT,
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        max_pool_connections=64,
        retries={"max_attempts": 10, "mode": "standard"},
    ),
)

# Быстрая проверка доступа: вывести первые 10 ключей из Stego/ (если есть)
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="Stego/", MaxKeys=10)
print("OK, sample keys:", [x["Key"] for x in resp.get("Contents", [])][:5])
```

Если тут ошибка `SignatureDoesNotMatch`, чаще всего проблема в:
- не выставлен `ru-central1`,
- не включён `path-style`,
- неверные ключи,
- сильно неверное время в VM (редко в Colab).

## 3) Сбор списков ключей для пилота (быстро, без полного листинга)

```python
from pathlib import Path

MAX_ITEMS_GAN = 50_000
MAX_ITEMS_STEGO = 80_000

def iter_keys(prefix: str, limit: int):
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        for obj in page.get("Contents", []):
            yield obj["Key"]
            count += 1
            if count >= limit:
                return

manifests = Path("manifests")
manifests.mkdir(exist_ok=True)

gan_keys = [k for k in iter_keys("gan/ai/", MAX_ITEMS_GAN) if k.endswith(".png")]
stego_keys = [
    k for k in iter_keys("Stego/", MAX_ITEMS_STEGO)
    if "/identity/" in k and k.endswith(".png")
]

(manifests / "gan_keys.txt").write_text("\n".join(gan_keys) + "\n")
(manifests / "stego_keys.txt").write_text("\n".join(stego_keys) + "\n")

print("gan keys:", len(gan_keys))
print("stego keys:", len(stego_keys))
```

## 4) Детерминированный split + выбор пилотного поднабора

Эта ячейка:
- делает split по **hash(seed|key)** в `train/val/test`;
- для stego дополнительно старается “перемешать” источники (по первому сегменту после `Stego/`), чтобы не было перекоса;
- сохраняет списки ключей для скачивания в `splits/`.

```python
import hashlib
import os
from collections import defaultdict, deque
from pathlib import Path

SEED = 20260320

PILOT_FAST = {
    "stego": {"train": 2000, "val": 500, "test": 2000},
    "gan": {"train": 2000, "val": 500, "test": 5000},
}
PILOT = PILOT_FAST

# Сколько “запасных” ключей добавлять к каждому split.
# Нужно, потому что иногда в бакете встречаются битые/не-PNG объекты с расширением .png,
# а `flipad` требует иметь >= num_samples валидных файлов в директории.
EXTRA_KEYS_PER_SPLIT = 200

def which_split(key: str) -> str:
    h = hashlib.sha1(f"{SEED}|{key}".encode("utf-8")).digest()
    v = int.from_bytes(h[:4], "big") % 100
    if v < 80:
        return "train"
    if v < 90:
        return "val"
    return "test"

def stego_source(key: str) -> str:
    # ожидаем ключ вида Stego/<source>/.../identity/00000000.png
    parts = key.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "Stego" else "unknown"

def load_keys(p: Path) -> list[str]:
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]

def round_robin_mix(keys_by_source: dict[str, list[str]]) -> list[str]:
    # перемешиваем внутри источника детерминированно и выдаём round-robin
    qs = []
    for src, keys in sorted(keys_by_source.items()):
        # детерминированный порядок через хэш src
        salt = int.from_bytes(hashlib.sha1(f"{SEED}|{src}".encode()).digest()[:4], "big")
        keys_sorted = sorted(keys, key=lambda k: hashlib.sha1(f"{salt}|{k}".encode()).hexdigest())
        qs.append(deque(keys_sorted))
    out = []
    active = True
    while active:
        active = False
        for q in qs:
            if q:
                out.append(q.popleft())
                active = True
    return out

root = Path("manifests")
stego_keys = load_keys(root / "stego_keys.txt")
gan_keys = load_keys(root / "gan_keys.txt")

splits = {"stego": defaultdict(list), "gan": defaultdict(list)}

for k in stego_keys:
    splits["stego"][which_split(k)].append(k)
for k in gan_keys:
    splits["gan"][which_split(k)].append(k)

# Для stego: смешиваем источники внутри каждого split
stego_splits_mixed = {}
for sp in ["train", "val", "test"]:
    by_src = defaultdict(list)
    for k in splits["stego"][sp]:
        by_src[stego_source(k)].append(k)
    stego_splits_mixed[sp] = round_robin_mix(by_src)

# Для GAN: достаточно детерминированного порядка
gan_splits_sorted = {sp: sorted(splits["gan"][sp]) for sp in ["train", "val", "test"]}

out_dir = Path("splits")
out_dir.mkdir(exist_ok=True)

def write_subset(name: str, sp: str, keys: list[str], n: int):
    sel = keys[: n + EXTRA_KEYS_PER_SPLIT]
    (out_dir / f"{name}_{sp}.txt").write_text("\n".join(sel) + "\n")
    return len(sel)

for sp in ["train", "val", "test"]:
    ns = PILOT["stego"][sp]
    ng = PILOT["gan"][sp]
    cs = write_subset("stego", sp, stego_splits_mixed[sp], ns)
    cg = write_subset("gan", sp, gan_splits_sorted[sp], ng)
    print(sp, "stego_keys_written", cs, "gan_keys_written", cg, "required", (ns, ng))

# Примечание: следующая ячейка знает PILOT и может проверить, что файлов скачалось достаточно.
```

## 5) Скачивание пилота из S3 в структуру `flipad`

Скачиваем в:

- `data/coco2014train/mas_grdh_stego/<split>/...png`
- `data/coco2014train/gan_ai/<split>/...png`

Имена файлов делаем **уникальными**: `sha1(key).png` (иначе будут коллизии `00000000.png` из разных папок).

```python
import hashlib
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, UnidentifiedImageError
import time

# Используем уже созданный выше `s3` клиент (boto3)

def sha_name(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".png"

def load_list(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]

def download(keys: list[str], dest_dir: Path, parallel: int = 10):
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _is_valid_png(path: Path) -> bool:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    bad_keys = []

    def _one(key: str):
        out = dest_dir / sha_name(key)
        if out.exists() and _is_valid_png(out):
            return
        # если файл есть, но битый/недокачанный (часто после reconnect) — перекачаем
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass

        tmp = out.with_suffix(out.suffix + ".tmp")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

        # ретраи на сетевые/транспортные сбои
        last_err = None
        for attempt in range(5):
            try:
                s3.download_file(BUCKET, key, str(tmp))
                if not _is_valid_png(tmp):
                    raise UnidentifiedImageError(f"Downloaded file is not a valid PNG: {tmp}")
                tmp.replace(out)  # атомарно (в пределах ФС)
                return
            except Exception as e:
                last_err = e
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
                time.sleep(0.5 * (attempt + 1))
        # объект в бакете может быть реально битым — фиксируем и пропускаем (у нас есть EXTRA_KEYS_PER_SPLIT)
        bad_keys.append(key)
        return

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(_one, k) for k in keys]
        for fut in as_completed(futures):
            fut.result()

    # итоговая проверка: оставляем только валидные PNG
    ok = 0
    for p in dest_dir.glob("*.png"):
        if _is_valid_png(p):
            ok += 1
        else:
            try:
                p.unlink()
            except OSError:
                pass
    if bad_keys:
        (dest_dir / "_bad_keys.txt").write_text("\n".join(bad_keys) + "\n")
    return ok

base = Path("data/coco2014train")
for sp in ["train", "val", "test"]:
    stego = load_list(Path("splits") / f"stego_{sp}.txt")
    gan = load_list(Path("splits") / f"gan_{sp}.txt")
    ok_s = download(stego, base / "mas_grdh_stego" / sp, parallel=10)
    ok_g = download(gan, base / "gan_ai" / sp, parallel=10)
    need_s = PILOT["stego"][sp]
    need_g = PILOT["gan"][sp]
    print(sp, "ok_stego", ok_s, "need", need_s, "| ok_gan", ok_g, "need", need_g)
    assert ok_s >= need_s, f"Not enough valid stego PNGs in {sp}: {ok_s} < {need_s}"
    assert ok_g >= need_g, f"Not enough valid gan PNGs in {sp}: {ok_g} < {need_g}"
```

Проверка, что всё скачалось:

```bash
%%bash
set -e
find data/coco2014train/mas_grdh_stego -type f -name '*.png' | wc -l
find data/coco2014train/gan_ai -type f -name '*.png' | wc -l
```

## 6) Запуск пилота `flipad` (DeepSAD)

Запуск (≤ 1 часа): `feat=dct`

```bash
%%bash
set -e
python3 -u flipad/sma_deepsad.py \
  --data-path data \
  --dataset coco2014train \
  --model stablediffusion \
  --checkpoint-path "stabilityai/stable-diffusion-2-1-base" \
  --train-dir mas_grdh_stego \
  --real-dir gan_ai \
  --feat dct \
  --net-name cifar10_biglenet \
  --num-workers 2 \
  --seed 42 \
  --num-train 2000 \
  --num-val 500 \
  --num-test 2000 \
  --tolerable-fnr 0.01 \
  --batch-size 128 \
  --n-epochs 20 \
  --output-path output
```

## 7) Что смотреть в результате

- порог под `FNR=0.01` (это \(TPR \approx 99\%\)) будет в логах как `Thresholds: {...}`
- дальше в `output/.../results/results.json` будет качество при этом пороге

